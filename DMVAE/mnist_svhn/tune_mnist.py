from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time
import random
import torch
import os
import numpy as np
from torchvision.datasets import MNIST, SVHN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# from datasets import getPairedDataset
from model import EncoderA, EncoderB, DecoderA, DecoderB
from classifier import MNIST_Classifier, SVHN_Classifier
from util import unpack_data, apply_poe


from copy import deepcopy

import sys

sys.path.append("../")
import probtorch

# ------------------------------------------------
# training parameters


# Multiview Dateset
class ViewDataset(Dataset):
    def __init__(self, v1, v2):
        self.v1 = torch.tensor(v1).float()
        self.v2 = torch.tensor(v2).float()
        self.data_len = v1.shape[0]

    def __getitem__(self, index):
        return self.v1[index], self.v2[index], index

    def __len__(self):
        return self.data_len


# Get a dataloader
def get_dataloader(view1, view2, batchsize, shuffle):
    dataset = ViewDataset(view1, view2)

    # Dataloader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=shuffle
    )

    return data_loader


def train_dmvae(config):
    n_shared = 30
    n_privateA = 0
    n_privateB = 3
    batch_size = config["batch_size"]
    epochs = 50
    lr = config["lr"]
    beta1 = 10
    beta2 = 10
    run_id = 0
    seed = 0
    wseed = 0
    lambda_text1 = 1.0
    lambda_text2 = 1.0

    EPS = 1e-9
    CUDA = torch.cuda.is_available()

    if CUDA:
        device = "cuda"
        num_workers = 1
    else:
        device = "cpu"
        num_workers = 0

    MODEL_NAME = (
        "mnist-run_id%d-privA%02ddim-privB%02ddim-sh%02ddim-lamb_text1_%s-lamb_text2_%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s"
        % (
            run_id,
            n_privateA,
            n_privateB,
            n_shared,
            lambda_text1,
            lambda_text2,
            beta1,
            beta2,
            seed,
            batch_size,
            wseed,
        )
    )

    # path parameters
    MODEL_NAME = (
        "mnist_svhn_cont2-run_id%d-privA%02ddim-privB%02ddim-sh%02ddim-lamb_text1_%s-lamb_text2_%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s"
        % (
            run_id,
            n_privateA,
            n_privateB,
            n_shared,
            lambda_text1,
            lambda_text2,
            beta1,
            beta2,
            seed,
            batch_size,
            wseed,
        )
    )

    params = [
        n_privateA,
        n_privateB,
        n_shared,
        lambda_text1,
        lambda_text2,
        beta1,
        beta2,
    ]

    # print(
    #     "privateA",
    #     "privateB",
    #     "shared",
    #     "lambda_text1",
    #     "lambda_text2",
    #     "beta1",
    #     "beta2",
    # )
    # print(params)

    # if not os.path.isdir(ckpt_path):
    #     os.makedirs(ckpt_path)

    # if len(run_desc) > 1:
    #     desc_file = os.path.join(ckpt_path, "run_id" + str(run_id) + ".txt")
    #     with open(desc_file, "w") as outfile:
    #         outfile.write(run_desc)

    BETA1 = (1.0, beta1, 1.0)
    BETA2 = (1.0, beta2, 1.0)

    data = np.load(
        r"C:\Users\Harris_Lab\Projects\SPLICE\data\mnist\mnist_rotated_360.npz"
    )

    X = torch.Tensor(data["original"][:50000]).to(device).reshape(-1, 1, 28, 28)
    Y = torch.Tensor(data["rotated"][:50000]).to(device).reshape(-1, 1, 28, 28)

    X_val = (
        torch.Tensor(data["original"][50000:60000]).to(device).reshape(-1, 1, 28, 28)
    )
    Y_val = torch.Tensor(data["rotated"][50000:60000]).to(device).reshape(-1, 1, 28, 28)

    dataset = ViewDataset(X[:50000], Y[:50000])
    val_dataset = ViewDataset(X_val, Y_val)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )

    BIAS_TRAIN = (train_loader.dataset.__len__() - 1) / (batch_size - 1)
    BIAS_TEST = (test_loader.dataset.__len__() - 1) / (batch_size - 1)

    def cuda_tensors(obj):
        for attr in dir(obj):
            value = getattr(obj, attr)
            if isinstance(value, torch.Tensor):
                setattr(obj, attr, value.cuda())

    encA = EncoderA(wseed, zShared_dim=n_shared, zPrivate_dim=n_privateA)
    decA = DecoderA(wseed, zShared_dim=n_shared, zPrivate_dim=n_privateA)
    encB = EncoderB(wseed, zShared_dim=n_shared, zPrivate_dim=n_privateB)
    decB = DecoderB(wseed, zShared_dim=n_shared, zPrivate_dim=n_privateB)

    encA.cuda()
    decA.cuda()
    encB.cuda()
    decB.cuda()
    cuda_tensors(encA)
    cuda_tensors(decA)
    cuda_tensors(encB)
    cuda_tensors(decB)

    optimizer = torch.optim.Adam(
        list(encB.parameters())
        + list(decB.parameters())
        + list(encA.parameters())
        + list(decA.parameters()),
        lr=lr,
        weight_decay=config["weight_decay"],
    )

    def elbo(
        q,
        pA,
        pB,
        lamb1=1.0,
        lamb2=1.0,
        beta1=(1.0, 1.0, 1.0),
        beta2=(1.0, 1.0, 1.0),
        bias=1.0,
    ):
        # from each of modality
        reconst_loss_A, kl_A = probtorch.objectives.mws_tcvae.elbo(
            q,
            pA,
            pA["images1_sharedA"],
            latents=["privateA", "sharedA"],
            sample_dim=0,
            batch_dim=1,
            beta=beta1,
            bias=bias,
        )
        reconst_loss_B, kl_B = probtorch.objectives.mws_tcvae.elbo(
            q,
            pB,
            pB["images2_sharedB"],
            latents=["privateB", "sharedB"],
            sample_dim=0,
            batch_dim=1,
            beta=beta2,
            bias=bias,
        )
        reconst_loss_poeA, kl_poeA = probtorch.objectives.mws_tcvae.elbo(
            q,
            pA,
            pA["images1_poe"],
            latents=["privateA", "poe"],
            sample_dim=0,
            batch_dim=1,
            beta=beta1,
            bias=bias,
        )
        reconst_loss_poeB, kl_poeB = probtorch.objectives.mws_tcvae.elbo(
            q,
            pB,
            pB["images2_poe"],
            latents=["privateB", "poe"],
            sample_dim=0,
            batch_dim=1,
            beta=beta2,
            bias=bias,
        )

        # # by cross
        reconst_loss_crA, kl_crA = probtorch.objectives.mws_tcvae.elbo(
            q,
            pA,
            pA["images1_sharedB"],
            latents=["privateA", "sharedB"],
            sample_dim=0,
            batch_dim=1,
            beta=beta1,
            bias=bias,
        )
        reconst_loss_crB, kl_crB = probtorch.objectives.mws_tcvae.elbo(
            q,
            pB,
            pB["images2_sharedA"],
            latents=["privateB", "sharedA"],
            sample_dim=0,
            batch_dim=1,
            beta=beta2,
            bias=bias,
        )

        # reconst_loss_crA = torch.tensor(0)
        # reconst_loss_crB = torch.tensor(0)

        # reconst_loss_poeA = torch.tensor(0)
        # reconst_loss_poeB = torch.tensor(0)

        loss = (
            (lamb1 * reconst_loss_A - kl_A)
            + (lamb2 * reconst_loss_B - kl_B)
            + (lamb1 * reconst_loss_crA - kl_crA)
            + (lamb2 * reconst_loss_crB - kl_crB)
            + (lamb1 * reconst_loss_poeA - kl_poeA)
            + (lamb2 * reconst_loss_poeB - kl_poeB)
        )

        return (
            -loss,
            [reconst_loss_A, reconst_loss_poeA, reconst_loss_crA],
            [reconst_loss_B, reconst_loss_poeB, reconst_loss_crB],
        )

    def train_nets(encA, decA, encB, decB, optimizer):
        epoch_elbo = 0.0
        epoch_recA = epoch_rec_poeA = epoch_rec_crA = 0.0
        epoch_recB = epoch_rec_poeB = epoch_rec_crB = 0.0
        encA.train()
        encB.train()
        decA.train()
        decB.train()
        N = 0
        # torch.autograd.set_detect_anomaly(True)
        for i, data in enumerate(train_loader):
            # data0, data1 = paired modalA&B
            # data2, data3 = random modalA&B
            if data[0].size()[0] == batch_size:
                N += 1
                images1 = data[0]
                images2 = data[1]

                optimizer.zero_grad()
                # encode
                # print(images.sum())
                q = encA(images1, num_samples=1)
                q = encB(images2, num_samples=1, q=q)

                ## poe ##
                mu_poe, std_poe = apply_poe(
                    CUDA,
                    q["sharedA"].dist.loc,
                    q["sharedA"].dist.scale,
                    q["sharedB"].dist.loc,
                    q["sharedB"].dist.scale,
                )
                q.normal(mu_poe, std_poe, name="poe")

                # decode
                pA = decA(
                    images1,
                    {"sharedA": q["sharedA"], "sharedB": q["sharedB"], "poe": q["poe"]},
                    q=q,
                    num_samples=1,
                )
                pB = decB(
                    images2,
                    {"sharedA": q["sharedA"], "sharedB": q["sharedB"], "poe": q["poe"]},
                    q=q,
                    num_samples=1,
                )

                # pA = decA(images1, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                #           num_samples=NUM_SAMPLES)
                # pB = decB(images2, {'sharedA': q['sharedA'], 'sharedB': q['sharedB']}, q=q,
                #           num_samples=NUM_SAMPLES)

                # loss
                loss, recA, recB = elbo(
                    q,
                    pA,
                    pB,
                    lamb1=lambda_text1,
                    lamb2=lambda_text2,
                    beta1=BETA1,
                    beta2=BETA2,
                    bias=BIAS_TRAIN,
                )

                loss.backward()
                optimizer.step()
                if CUDA:
                    loss = loss.cpu()
                    recA[0] = recA[0].cpu()
                    recB[0] = recB[0].cpu()

                epoch_elbo += loss.item()
                epoch_recA += recA[0].item()
                epoch_recB += recB[0].item()

                if CUDA:
                    for i in range(2):
                        recA[i] = recA[i].cpu()
                        recB[i] = recB[i].cpu()
                epoch_rec_poeA += recA[1].item()
                epoch_rec_crA += recA[2].item()
                epoch_rec_poeB += recB[1].item()
                epoch_rec_crB += recB[2].item()

        return (
            epoch_elbo / N,
            [epoch_recA / N, epoch_rec_poeA / N, epoch_rec_crA / N],
            [epoch_recB / N, epoch_rec_poeB / N, epoch_rec_crB / N],
        )

    # def save_ckpt():
    #     if not os.path.isdir(ckpt_path):
    #         os.mkdir(ckpt_path)
    #     torch.save(
    #         encA.state_dict(),
    #         "%s/%s-encA.rar"
    #         % (
    #             ckpt_path,
    #             MODEL_NAME,
    #         ),
    #     )
    #     torch.save(
    #         decA.state_dict(),
    #         "%s/%s-decA.rar"
    #         % (
    #             ckpt_path,
    #             MODEL_NAME,
    #         ),
    #     )
    #     torch.save(
    #         encB.state_dict(),
    #         "%s/%s-encB"
    #         % (
    #             ckpt_path,
    #             MODEL_NAME,
    #         ),
    #     )
    #     torch.save(
    #         decB.state_dict(),
    #         "%s/%s-decB"
    #         % (
    #             ckpt_path,
    #             MODEL_NAME,
    #         ),
    #     )

    def test(encA, decA, encB, decB, epoch):
        encA.eval()
        decA.eval()
        encB.eval()
        decB.eval()
        epoch_elbo = 0.0
        N = 0
        for i, data in enumerate(test_loader):
            if data[0].size()[0] == batch_size:
                N += 1
                images1 = data[0]
                images2 = data[1]

                # encode
                q = encA(images1, num_samples=1)
                q = encB(images2, num_samples=1, q=q)

                mu_poe, std_poe = apply_poe(
                    CUDA,
                    q["sharedA"].dist.loc,
                    q["sharedA"].dist.scale,
                    q["sharedB"].dist.loc,
                    q["sharedB"].dist.scale,
                )
                q.normal(mu_poe, std_poe, name="poe")

                pA = decA(
                    images1,
                    {"sharedA": q["sharedA"], "sharedB": q["sharedB"], "poe": q["poe"]},
                    q=q,
                    num_samples=1,
                )
                pB = decB(
                    images2,
                    {"sharedB": q["sharedB"], "sharedA": q["sharedA"], "poe": q["poe"]},
                    q=q,
                    num_samples=1,
                )

                batch_elbo, _, _ = elbo(
                    q,
                    pA,
                    pB,
                    lamb1=lambda_text1,
                    lamb2=lambda_text2,
                    beta1=BETA1,
                    beta2=BETA2,
                    bias=BIAS_TEST,
                )

                if CUDA:
                    batch_elbo = batch_elbo.cpu()
                epoch_elbo += batch_elbo.item()

        return epoch_elbo / N

    best_loss = np.inf

    for e in range(epochs):
        print("====> Epoch: %d" % e)
        train_start = time.time()
        train_elbo, rec_lossA, rec_lossB = train_nets(encA, decA, encB, decB, optimizer)
        train_end = time.time()

        test_elbo = test(encA, decA, encB, decB, e)

        if test_elbo < best_loss:
            # print("saving best model")
            # save_ckpt()
            best_loss = test_elbo

        # print(
        #     "[Epoch %d] Train: ELBO %.4e RECA %.4f RECB %.4f (%ds) Test: ELBO %.4e"
        #     % (
        #         e,
        #         train_elbo,
        #         rec_lossA[1],
        #         rec_lossB[1],
        #         train_end - train_start,
        #         test_elbo,
        #     )
        # )
        train.report({"elbo": best_loss})

    return {"elbo": best_loss}


if __name__ == "__main__":
    search_space = {
        # "_lambda": tune.choice([0.001, 0.01, 0.1, 1]),
        "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.choice([0, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([100]),
    }

    results = tune.run(
        train_dmvae,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=25,
        search_alg=HyperOptSearch(search_space, metric="elbo", mode="min"),
        scheduler=ASHAScheduler(metric="elbo", mode="min"),
    )

    print(
        "Best hyperparameters found were: ",
        results.get_best_trial("elbo", "min", "last").last_result,
    )
