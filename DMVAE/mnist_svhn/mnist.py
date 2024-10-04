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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=5, metavar="N", help="run_id")
    parser.add_argument("--run_desc", type=str, default="", help="run_id desc")
    parser.add_argument(
        "--n_shared",
        type=int,
        default=30,
        help="size of the latent embedding of shared",
    )
    parser.add_argument(
        "--n_privateA",
        type=int,
        default=0,
        help="size of the latent embedding of private",
    )
    parser.add_argument(
        "--n_privateB",
        type=int,
        default=10,
        help="size of the latent embedding of private",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
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
        default=50,
        metavar="N",
        help="number of epochs to train [default: 200]",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate [default: 1e-3]",
    )

    parser.add_argument(
        "--label_frac", type=float, default=1.0, help="how many labels to use"
    )
    parser.add_argument("--sup_frac", type=float, default=1.0, help="supervision ratio")
    parser.add_argument(
        "--lambda_text1",
        type=float,
        default=1.0,
        help="multipler for text reconstruction [default: 10]",
    )
    parser.add_argument(
        "--lambda_text2",
        type=float,
        default=1.0,
        help="multipler for text reconstruction [default: 10]",
    )
    parser.add_argument(
        "--beta1", type=float, default=1.0, help="multipler for TC [default: 10]"
    )
    parser.add_argument(
        "--beta2", type=float, default=1.0, help="multipler for TC [default: 10]"
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
        default="../weights/mnist_svhn_cont",
        help="save and load path for ckpt",
    )

    args = parser.parse_args()


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
        args.run_id,
        args.n_privateA,
        args.n_privateB,
        args.n_shared,
        args.lambda_text1,
        args.lambda_text2,
        args.beta1,
        args.beta2,
        args.seed,
        args.batch_size,
        args.wseed,
    )
)

# path parameters
MODEL_NAME = (
    "mnist_svhn_cont2-run_id%d-privA%02ddim-privB%02ddim-sh%02ddim-lamb_text1_%s-lamb_text2_%s-beta1%s-beta2%s-seed%s-bs%s-wseed%s"
    % (
        args.run_id,
        args.n_privateA,
        args.n_privateB,
        args.n_shared,
        args.lambda_text1,
        args.lambda_text2,
        args.beta1,
        args.beta2,
        args.seed,
        args.batch_size,
        args.wseed,
    )
)

params = [
    args.n_privateA,
    args.n_privateB,
    args.n_shared,
    args.lambda_text1,
    args.lambda_text2,
    args.beta1,
    args.beta2,
]

print(
    "privateA", "privateB", "shared", "lambda_text1", "lambda_text2", "beta1", "beta2"
)
print(params)

if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)

if len(args.run_desc) > 1:
    desc_file = os.path.join(args.ckpt_path, "run_id" + str(args.run_id) + ".txt")
    with open(desc_file, "w") as outfile:
        outfile.write(args.run_desc)

BETA1 = (1.0, args.beta1, 1.0)
BETA2 = (1.0, args.beta2, 1.0)

data = np.load(r"C:\Users\Harris_Lab\Projects\SPLICE\data\mnist\mnist_rotated_360.npz")

X = torch.Tensor(data["original"][:50000]).to(device).reshape(-1, 1, 28, 28)
Y = torch.Tensor(data["rotated"][:50000]).to(device).reshape(-1, 1, 28, 28)

X_val = torch.Tensor(data["original"][50000:60000]).to(device).reshape(-1, 1, 28, 28)
Y_val = torch.Tensor(data["rotated"][50000:60000]).to(device).reshape(-1, 1, 28, 28)

dataset = ViewDataset(X[:50000], Y[:50000])
val_dataset = ViewDataset(X_val, Y_val)

train_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=args.batch_size, shuffle=False
)

BIAS_TRAIN = (train_loader.dataset.__len__() - 1) / (args.batch_size - 1)
BIAS_TEST = (test_loader.dataset.__len__() - 1) / (args.batch_size - 1)


def cuda_tensors(obj):
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())


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
    weight_decay=1e-2,
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


def train(encA, decA, encB, decB, optimizer):
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
        print(i, end="\r")
        # data0, data1 = paired modalA&B
        # data2, data3 = random modalA&B
        if data[0].size()[0] == args.batch_size:
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
                lamb1=args.lambda_text1,
                lamb2=args.lambda_text2,
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


def save_ckpt():
    if not os.path.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    torch.save(
        encA.state_dict(),
        "%s/%s-encA.rar"
        % (
            args.ckpt_path,
            MODEL_NAME,
        ),
    )
    torch.save(
        decA.state_dict(),
        "%s/%s-decA.rar"
        % (
            args.ckpt_path,
            MODEL_NAME,
        ),
    )
    torch.save(
        encB.state_dict(),
        "%s/%s-encB.rar"
        % (
            args.ckpt_path,
            MODEL_NAME,
        ),
    )
    torch.save(
        decB.state_dict(),
        "%s/%s-decB.rar"
        % (
            args.ckpt_path,
            MODEL_NAME,
        ),
    )


def test(encA, decA, encB, decB, epoch):
    encA.eval()
    decA.eval()
    encB.eval()
    decB.eval()
    epoch_elbo = 0.0
    N = 0
    for i, data in enumerate(test_loader):
        if data[0].size()[0] == args.batch_size:
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
                lamb1=args.lambda_text1,
                lamb2=args.lambda_text2,
                beta1=BETA1,
                beta2=BETA2,
                bias=BIAS_TEST,
            )

            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.item()

    return epoch_elbo / N


best_loss = np.inf

for e in range(args.ckpt_epochs, args.epochs):
    print("====> Epoch: %d" % e)
    train_start = time.time()
    train_elbo, rec_lossA, rec_lossB = train(encA, decA, encB, decB, optimizer)
    train_end = time.time()

    test_elbo = test(encA, decA, encB, decB, e)

    print(test_elbo, best_loss)
    if test_elbo < best_loss:
        print("saving best model")
        save_ckpt()
        best_loss = test_elbo

    print(
        "[Epoch %d] Train: ELBO %.4e RECA %.4f RECB %.4f (%ds) Test: ELBO %.4e"
        % (
            e,
            train_elbo,
            rec_lossA[1],
            rec_lossB[1],
            train_end - train_start,
            test_elbo,
        )
    )
