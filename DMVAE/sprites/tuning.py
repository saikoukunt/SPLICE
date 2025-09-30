import argparse
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import test
import torch
from model import DecoderA, DecoderB, EncoderA, EncoderB
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.manifold import TSNE

# from datasets import getPairedDataset
from sprites import test as test_sprites
from sprites import train as train_sprites
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, SVHN
from util import apply_poe, unpack_data

sys.path.append("../")
import pickle

import probtorch


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


def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=5, metavar="N", help="run_id")
    parser.add_argument("--run_desc", type=str, default="", help="run_id desc")
    parser.add_argument(
        "--n_shared",
        type=int,
        default=500,
        help="size of the latent embedding of shared",
    )
    parser.add_argument(
        "--n_privateA",
        type=int,
        default=2,
        help="size of the latent embedding of private",
    )
    parser.add_argument(
        "--n_privateB",
        type=int,
        default=2,
        help="size of the latent embedding of private",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        metavar="N",
        help="input batch size for training [default: 100]",
    )
    parser.add_argument(
        "--ckpt_epochs",
        type=int,
        default=0,
        metavar="N",
        help="number of epochs to train [default: 200]",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3000,
        metavar="N",
        help="number of epochs to train [default: 200]",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate [default: 1e-3]",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        metavar="WD",
        help="weight decay [default: 0.0]",
    )

    parser.add_argument(
        "--lambda_text1",
        type=float,
        default=10.0,
        help="multipler for text reconstruction [default: 10]",
    )
    parser.add_argument(
        "--lambda_text2",
        type=float,
        default=10.0,
        help="multipler for text reconstruction [default: 10]",
    )
    parser.add_argument(
        "--beta1", type=float, default=1, help="multipler for TC [default: 10]"
    )
    parser.add_argument(
        "--beta2", type=float, default=1, help="multipler for TC [default: 10]"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="random seed for get_paired_data",
    )
    parser.add_argument(
        "--wseed", type=int, default=0, metavar="N", help="random seed for weight"
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="../weights/sprites",
        help="save and load path for ckpt",
    )
    parser.set_defaults(**config)
    args = parser.parse_known_args()[0]

    return args


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


def train_DMVAE(config):
    args = parse_args(config)
    EPS = 1e-9
    CUDA = torch.cuda.is_available()

    if CUDA:
        print("Using GPU")
        device = "cuda"
        num_workers = 1
    else:
        print("Using CPU")
        device = "cpu"
        num_workers = 0

    if not os.path.isdir(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    if len(args.run_desc) > 1:
        desc_file = os.path.join(args.ckpt_path, "run_id" + str(args.run_id) + ".txt")
        with open(desc_file, "w") as outfile:
            outfile.write(args.run_desc)

    train_data = np.load(
        r"C:\Users\Harris_Lab\Projects\SPLICE\data\sprites\single-pose_private-angle_train.npz"
    )
    test_data = np.load(
        r"C:\Users\Harris_Lab\Projects\SPLICE\data\sprites\single-pose_private-angle_test.npz"
    )

    A_train = torch.Tensor(train_data["view1"].reshape(-1, 64 * 64 * 3)).to(device)
    B_train = torch.Tensor(train_data["view2"].reshape(-1, 64 * 64 * 3)).to(device)
    A_test = torch.Tensor(test_data["view1"].reshape(-1, 64 * 64 * 3)).to(device)
    B_test = torch.Tensor(test_data["view2"].reshape(-1, 64 * 64 * 3)).to(device)

    dataset = ViewDataset(A_train, B_train)
    val_dataset = ViewDataset(A_test, B_test)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False
    )

    MODEL_NAME = f"sprites_lr{args.lr}_bs{args.batch_size}_wd{args.weight_decay}"
    BETA1 = (1.0, args.beta1, 1.0)
    BETA2 = (1.0, args.beta2, 1.0)
    BIAS_TRAIN = (train_loader.dataset.__len__() - 1) / (args.batch_size - 1)
    BIAS_TEST = (test_loader.dataset.__len__() - 1) / (args.batch_size - 1)

    encA = EncoderA(args.wseed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateA)
    decA = DecoderA(args.wseed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateA)
    encB = EncoderB(args.wseed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateB)
    decB = DecoderB(args.wseed, zShared_dim=args.n_shared, zPrivate_dim=args.n_privateB)

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
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_loss = np.inf

    for e in range(args.ckpt_epochs, args.epochs):
        print("====> Epoch: %d" % e)
        train_start = time.time()
        train_elbo, rec_lossA, rec_lossB = train_sprites(
            encA,
            decA,
            encB,
            decB,
            optimizer,
            train_loader,
            BIAS_TRAIN,
            BETA1,
            BETA2,
            args,
        )
        train_end = time.time()

        test_elbo = test_sprites(
            encA, decA, encB, decB, e, test_loader, BIAS_TEST, BETA1, BETA2, args
        )

        tune.report({"epoch": e, "elbo": test_elbo})
        # if test_elbo < best_loss:
        #     print("saving best model")
        #     save_ckpt(MODEL_NAME, args.ckpt_path, encA, decA, encB, decB)
        #     best_loss = test_elbo

        print(
            "[Epoch %d] Train: ELBO %.4e RECA %.4f RECB %.4f (%ds) Test: ELBO %.4e"
            % (
                e,
                train_elbo,
                rec_lossA[1] / 12288,
                rec_lossB[1] / 12288,
                train_end - train_start,
                test_elbo,
            )
        )


if __name__ == "__main__":
    search_space = {
        "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.choice([0, 1e-4, 1e-3, 1e-2, 1e-1]),
    }

    algo = HyperOptSearch(metric="elbo", mode="min")

    scheduler = ASHAScheduler(
        time_attr="epoch",
        max_t=3000,
        metric="elbo",
        mode="min",
        grace_period=10,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            train_DMVAE,
            {"cpu": 1, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            num_samples=20,
            search_alg=algo,
            scheduler=scheduler,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    df_results = results.get_dataframe()
    df_results.to_csv(
        os.path.join("../weights/sprites", "ray_results", "results.csv"), index=False
    )

    best_config = results.get_best_result(metric="elbo", mode="min").config
    print("Best config: ", best_config)
