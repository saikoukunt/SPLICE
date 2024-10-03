import numpy as np
from torch_nl3rSetup import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
)
from torch.utils.data import DataLoader
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import dijkstra

from sklearn.datasets import fetch_openml
from scipy.ndimage import rotate
import os
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rot_digit(m, restricted_rotations=True):
    """
    Returns the digit/image "m" by a random angle [-45,45]deg
    clips it to MNIST size
    and returns it flattened into (28*28,) shape
    """
    if restricted_rotations:
        angle = np.random.rand() * 2 * 45 - 45
    else:
        angle = np.random.rand() * 360  # will lead to ambiguities because "6" = "9"

    m = m.reshape((28, 28))
    tmp = rotate(m, angle=angle)
    xs, ys = tmp.shape
    xs = int(xs / 2)
    ys = int(ys / 2)
    rot_m = tmp[xs - 14 : xs + 14, ys - 14 : ys + 14]
    return rot_m.reshape((28 * 28,)), angle


# --------------------------- NETWORK DEFINITIONS --------------------------------------
def carlosPlus(x, logof2):
    return 2 * (F.softplus(x) - logof2)


class encoder(nn.Module):
    def __init__(self, nInputs, nOutputs, layers=None):
        super(encoder, self).__init__()
        self.layers = nn.ModuleList()
        if layers:
            self.layers.append(nn.Linear(nInputs, layers[0]))
            for i in range(len(layers) - 1):
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.Linear(layers[-1], nOutputs))

    def forward(self, x):
        for layer in self.layers:
            x = carlosPlus(layer(x), np.log(2))
            # x = F.relu(layer(x))
        return x


class decoder(nn.Module):
    def __init__(self, nInputs, nOutputs, layers=None):
        super(decoder, self).__init__()
        self.layers = nn.ModuleList()
        if layers:
            self.layers.append(nn.Linear(nInputs, layers[0]))
            for i in range(len(layers) - 1):
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.Linear(layers[-1], nOutputs))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = carlosPlus(layer(x), np.log(2))
                # x = F.relu(layer(x))
        return x


class new_nl(nn.Module):
    def __init__(
        self, nInputs, nOutputs, nShared, nPrivSource, nPrivTarget, layers, layers_msr
    ):
        super().__init__()
        # save params
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.nShared = nShared
        self.nPrivSource = nPrivSource
        self.nPrivTarget = nPrivTarget

        # encoders
        self.shared_source_enc = encoder(self.nInputs, self.nShared, layers)
        self.shared_target_enc = encoder(self.nOutputs, self.nShared, layers)
        self.target_enc = encoder(self.nOutputs, self.nPrivTarget, layers)

        # measurement networks
        # self.msr_source = decoder(self.nPrivTarget, self.nShared, layers_msr[::-1])
        self.msr_source = decoder(self.nPrivTarget, self.nInputs, layers_msr[::-1])

        # decoders
        self.source_dec = decoder(
            self.nPrivSource + self.nShared, self.nInputs, layers[::-1]
        )
        self.target_dec = decoder(
            self.nShared + self.nPrivTarget, self.nOutputs, layers[::-1]
        )

    def forward(self, x, y):
        # calculate latents
        z_s_x = self.shared_source_enc(x)
        z_s_y = self.shared_target_enc(y)
        z_y = self.target_enc(y)

        # estimate shared from private
        m_s_x = self.msr_source(z_y)

        # calculate reconstructions
        x_hat = self.source_dec(z_s_y)
        y_hat = self.target_dec(torch.hstack((z_s_x, z_y)))

        return z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat


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


# ---------------------------------------- MAIN -----------------------------------------
def main(rank, world_size, X, Y, x_test, y_test):
    setup(rank, world_size)

    dataloader = prepare(torch.hstack((X, Y)).to(rank), rank, world_size)
    x_test = x_test.to(rank)
    y_test = y_test.to(rank)

    # ---------------------------------- SPLICE ----------------------------------------
    nShared = 30
    nInputs = 784
    nOutputs = 784
    layers = [256, 128, 64, 32]

    # instantiate SPLICE model + training
    model = new_nl(nInputs, nOutputs, nShared, 0, 2, layers, layers[::-1]).to(rank)
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )

    lr = 0.001
    num_epochs = 100
    msr_iter = 3
    rec_iter = 2
    c = np.linspace(0.1, 0.1, 25001)

    # normal trainers
    optimizer = torch.optim.Adam(model.module.parameters(), lr=lr)
    scheduler = LinearLR(optimizer, total_iters=30, start_factor=1, end_factor=1 / 50)

    # msr trainers
    msr_params = list(model.module.msr_source.parameters())
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
            x = data[:, :784]
            y = data[:, 784:]

            # 1) minimize source/target reconstructions
            # a) unfreeze all weights
            for param in model.module.parameters():
                param.requires_grad = True

            for i in range(rec_iter):
                # b) calculate losses
                z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(x, y)
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
            model.module.msr_source = decoder(2, nInputs, layers[::-1]).to(rank)
            msr_params = list(model.module.msr_source.parameters())
            msr_optimizer = torch.optim.Adam(msr_params, lr=0.001)
            msr_iter = 50
        else:
            msr_iter = 10

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
                    nInputs
                    * torch.nn.functional.mse_loss(x, m_s_x)
                    / x.detach().var(dim=0).sum()
                )

                # c) backpropagate
                msr_optimizer.zero_grad()
                msr_loss.backward()
                msr_optimizer.step()

                train_msr_loss += msr_loss.item()

        scheduler.step()

        z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(x_test, y_test)
        test_source_loss = torch.nn.functional.mse_loss(x_test, x_hat)
        test_target_loss = torch.nn.functional.mse_loss(y_test, y_hat)
        # test_msr_loss = (
        #     nShared
        #     * torch.nn.functional.mse_loss(z_s_x, m_s_x)
        #     / z_s_x.detach().var(dim=0).sum()
        # )
        test_msr_loss = (
            nInputs
            * torch.nn.functional.mse_loss(x_test, m_s_x)
            / x_test.detach().var(dim=0).sum()
        )
        # test_disent_loss = m_s_x.var(dim=0).sum() / z_s_x.var(dim=0).sum()
        test_disent_loss = m_s_x.var(dim=0).sum() / x_test.var(dim=0).sum()

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

        X = X.to(rank)
        Y = Y.to(rank)

        z_s_x, z_s_y, z_y, m_s_x, x_hat, y_hat = model(X, Y)
        mu = X.mean(dim=0)
        s1 = (torch.linalg.norm(X - mu) ** 2).mean()
        s2 = (torch.linalg.norm(X - x_hat) ** 2).mean()
        print("train X explained: %.4f" % (1 - s2 / s1))

        mu = Y.mean(dim=0)
        s1 = (torch.linalg.norm(Y - mu) ** 2).mean()
        s2 = (torch.linalg.norm(Y - y_hat) ** 2).mean()
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

    print("fetching data")
    mnist, mnist_labels = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    print("generating rotated")
    rotated_mnist = np.zeros_like(mnist)
    angles = np.zeros(mnist.shape[0], dtype="float32")
    for i in range(mnist.shape[0]):
        rotated_mnist[i], angles[i] = rot_digit(mnist[i], restricted_rotations=False)

    np.save("./angles.npy", angles)

    X = mnist.astype("int16").astype("float32")
    Y = rotated_mnist.astype("int16").astype("float32")
    Y[Y > 255] = 255
    Y[Y < 0] = 0

    X = torch.Tensor(X) / 255
    Y = torch.Tensor(Y) / 255

    print("spawning DDP")
    mp.spawn(
        main,
        args=[world_size, X[:60000], Y[:60000], X[-5000:], Y[-5000:]],
        nprocs=world_size,
    )
