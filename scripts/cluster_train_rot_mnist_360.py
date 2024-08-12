import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from scipy.ndimage import rotate
from scipy.sparse.csgraph import dijkstra
from sklearn.datasets import fetch_openml
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from splice.splice import splice_model
from splice.base import decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- DISTRIBUTED MODEL DEFINITIONS ---------------------------------
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def prepare(dataset, rank, world_size, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=20,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader


def cleanup():
    dist.destroy_process_group()


def save_on_master(*args, **kwargs):
    if dist.get_rank() == 0:
        torch.save(*args, **kwargs)
        print("Saved model")


def main(rank, world_size, A, B, a_validation, b_validation):
    setup(rank, world_size)

    dataloader = prepare(torch.hstack((A, B)).to(rank), rank, world_size)
    a_validation = a_validation.to(rank)
    b_validation = b_validation.to(rank)

    # ---------------------------------- SPLICE ----------------------------------------
    n_shared = 30
    n_a = 784
    n_b = 784
    layers = [256, 128, 64, 32]

    # instantiate SPLICE model + training
    model = splice_model(
        n_a, n_b, n_shared, 0, 3, layers, layers[::-1], layers[::-1]
    ).to(rank)
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )

    lr = 0.001
    num_epochs = 50
    msr_iter = 3
    rec_iter = 2
    c = np.linspace(0.1, 0.1, 25001)

    # normal trainers
    optimizer = torch.optim.Adam(model.module.parameters(), lr=lr)
    scheduler = LinearLR(optimizer, total_iters=30, start_factor=1, end_factor=1 / 50)

    # msr trainers
    msr_params = list(model.module.M_b2a.parameters())
    msr_optimizer = torch.optim.Adam(msr_params, lr=lr)
    msr_mean_loss = torch.Tensor([0]).to(rank)[0]

    msr_loss = 1

    # SPLICE TRAINING LOOP
    for epoch in range(num_epochs):
        model.module.train()
        dataloader.sampler.set_epoch(epoch)

        train_source_loss = 0.0
        train_target_loss = 0.0
        source_disent_loss = 0.0

        for step, data in enumerate(dataloader):
            print("rec %d" % (step), end="\r")
            a = data[:, :784]
            b = data[:, 784:]

            # 1) minimize source/target reconstructions
            # a) unfreeze all weights
            for param in model.module.parameters():
                param.requires_grad = True

            for i in range(rec_iter):
                # b) calculate losses
                z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(a, b)
                source_loss = torch.nn.functional.mse_loss(x, x_hat)
                target_loss = torch.nn.functional.mse_loss(y, y_hat)
                step1_loss = source_loss + target_loss

                # c) backpropagate
                optimizer.zero_grad()
                step1_loss.backward()
                optimizer.step()

                train_source_loss += source_loss.item()
                train_target_loss += target_loss.item()

            # 2) disentangling
            # a) unfreeze target encoder, freeze others
            for param in model.module.parameters():
                param.requires_grad = False
            for param in model.module.target_enc.parameters():
                param.requires_grad = True

            # b) calculate loss
            z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(x, y)
            x_loss = m_s_x.var(dim=0).sum() / z_s_x.var(dim=0).sum()
            y_loss = 0
            msr_mean_loss = c[epoch] * (x_loss + y_loss)

            # c) backpropagate
            if epoch > 3:
                optimizer.zero_grad()
                msr_mean_loss.backward()
                optimizer.step()

            source_disent_loss += msr_mean_loss.item() / c[epoch]

        # 3) cold restart measurement networks if needed
        if epoch % 3 == 0:
            model.module.msr_source = decoder(2, n_a, layers[::-1]).to(rank)
            msr_params = list(model.module.msr_source.parameters())
            msr_optimizer = torch.optim.Adam(msr_params, lr=0.001)
            msr_iter = 15
        else:
            msr_iter = 5

        # 4) train measurement networks
        # a) freeze other networks
        for param in model.module.parameters():
            param.requires_grad = False
        for param in model.module.msr_source.parameters():
            param.requires_grad = True

        for iter in range(msr_iter):
            train_msr_loss = 0.0

            for step, data in enumerate(dataloader):
                print("msr%d|%d" % (iter, step), end="\r")
                x = data[:, :784]
                y = data[:, 784:]

                # b) calculate loss
                z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(x, y)
                # msr_loss = nShared * (
                #     torch.nn.functional.mse_loss(z_s_x, m_s_x)
                #     / z_s_x.detach().var(dim=0).sum()
                # )
                msr_loss = (
                    n_a
                    * torch.nn.functional.mse_loss(x, m_s_x)
                    / x.detach().var(dim=0).sum()
                )

                # c) backpropagate
                msr_optimizer.zero_grad()
                msr_loss.backward()
                msr_optimizer.step()

                train_msr_loss += msr_loss.item()

        scheduler.step()

        z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(a_validation, b_validation)
        test_source_loss = torch.nn.functional.mse_loss(a_validation, x_hat)
        test_target_loss = torch.nn.functional.mse_loss(b_validation, y_hat)
        # test_msr_loss = (
        #     nShared
        #     * torch.nn.functional.mse_loss(z_s_x, m_s_x)
        #     / z_s_x.detach().var(dim=0).sum()
        # )
        test_msr_loss = (
            n_a
            * torch.nn.functional.mse_loss(a_validation, m_s_x)
            / a_validation.detach().var(dim=0).sum()
        )
        # test_disent_loss = m_s_x.var(dim=0).sum() / z_s_x.var(dim=0).sum()
        test_disent_loss = m_s_x.var(dim=0).sum() / a_validation.var(dim=0).sum()

        print(
            "EPOCH %d \t source: %.5f | %.5f \t target: %.5f | %.5f \t msr: %.5f | %.5f \t disentangle: %.5f | %.5f"
            % (
                epoch,
                train_source_loss / len(dataloader) / rec_iter,
                test_source_loss.item(),
                train_target_loss / len(dataloader) / rec_iter,
                test_target_loss.item(),
                train_msr_loss / len(dataloader),
                test_msr_loss.item(),
                source_disent_loss / len(dataloader),
                test_disent_loss.item(),
            )
        )

    # save model
    dist.barrier()

    if dist.get_rank() == 0:
        print("Finished training")
        save_on_master(model.module.state_dict(), "./mnist_model_3d.pt")

        A = A.to(rank)
        B = B.to(rank)

        z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(A, B)
        mu = A.mean(dim=0)
        s1 = (torch.linalg.norm(A - mu) ** 2).mean()
        s2 = (torch.linalg.norm(A - x_hat) ** 2).mean()
        print("train X explained: %.4f" % (1 - s2 / s1))

        mu = B.mean(dim=0)
        s1 = (torch.linalg.norm(B - mu) ** 2).mean()
        s2 = (torch.linalg.norm(B - y_hat) ** 2).mean()
        print("train Y explained: %.4f" % (1 - s2 / s1))

        z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(x_test, y_test)
        mu = x_test.mean(dim=0)
        s1 = (torch.linalg.norm(x_test - mu) ** 2).mean()
        s2 = (torch.linalg.norm(x_test - x_hat) ** 2).mean()
        print("test X explained: %.4f" % (1 - s2 / s1))

        mu = y_test.mean(dim=0)
        s1 = (torch.linalg.norm(y_test - mu) ** 2).mean()
        s2 = (torch.linalg.norm(y_test - y_hat) ** 2).mean()
        print("test Y explained: %.4f" % (1 - s2 / s1))

    # ---------------------------------- ISOMAP ----------------------------------------

    cleanup()


if __name__ == "__main__":
    world_size = 5

    fold = int(sys.argv[1])
    if fold >= 7:
        raise ValueError("Fold must be in [0, 6]")

    print(fold)
    data = np.load("../data/mnist/mnist_rotated_360.npz")
    mnist = data["original"]
    rotated_mnist = data["rotated"]

    kf = KFold(n_splits=7, shuffle=False)
    train_idx, test_idx = list(kf.split(np.arange(mnist.shape[0])))[fold]
    val_idx = train_idx[50000:].copy()
    train_idx = train_idx[:50000]

    a_train, a_val, a_test = (
        torch.Tensor(mnist[train_idx]),
        torch.Tensor(mnist[val_idx]),
        torch.Tensor(mnist[test_idx]),
    )
    b_train, b_val, b_test = (
        torch.Tensor(rotated_mnist[train_idx]),
        torch.Tensor(rotated_mnist[val_idx]),
        torch.Tensor(rotated_mnist[test_idx]),
    )

    mp.spawn(
        main,
        args=[world_size, X[:60000], Y[:60000], X[-5000:], Y[-5000:]],
        nprocs=world_size,
    )
