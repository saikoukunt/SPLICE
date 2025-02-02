import argparse
import os
import sys

import numpy as np
import torch as torch

from splice.base import ConvLayer
from splice.splice import SPLICE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conv",
        default=False,
        required=False,
        help="True if convolutional neural networks should be used",
    )
    parser.add_argument(
        "--angle",
        required=True,
        help='"shared" to train on the dataset where angle of rotation is shared between the two views, "private" to train on the dataset where angle is private.',
    )
    args = vars(parser.parse_args(sys.argv[1:]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args["angle"] == "shared":
        # train = np.load("../../data/sprites/single-pose_shared-angle_train.npz")
        # test = np.load("../../data/sprites/single-pose_shared-angle_test.npz")
        train = np.load("../../data/sprites/single_pose_train.npz")
        test = np.load("../../data/sprites/single_pose_test.npz")
        filepath = os.path.join(
            "..", "..", "results", "models", "sprites", "splice_sprites_shared-angle.pt"
        )
        n_shared = 2
        n_private_a = 500
        n_private_b = 500
        lr = 1e-4
    elif args["angle"] == "private":
        train = np.load("../../data/sprites/single-pose_private-angle_train.npz")
        test = np.load("../../data/sprites/single-pose_private-angle_test.npz")
        filepath = os.path.join(
            "..",
            "..",
            "results",
            "models",
            "sprites",
            "splice_sprites_private-angle.pt",
        )
        n_shared = 500
        n_private_a = 2
        n_private_b = 2
        lr = 1e-4
    else:
        parser.error('Unknown option for angle: must be either "private" or "shared".')

    if args["conv"]:
        A_train = torch.Tensor(train["view1"].transpose(0, 3, 1, 2)).to(device)
        B_train = torch.Tensor(train["view2"].transpose(0, 3, 1, 2)).to(device)
        A_test = torch.Tensor(test["view1"].transpose(0, 3, 1, 2)).to(device)
        B_test = torch.Tensor(test["view2"].transpose(0, 3, 1, 2)).to(device)

        enc_layers = {}
        enc_layers["conv"] = [
            ConvLayer(3, 256, 2, 2, 0),
            ConvLayer(256, 256, 2, 2, 0),
            ConvLayer(256, 256, 2, 2, 0),
            ConvLayer(256, 256, 2, 2, 0),
        ]
        enc_layers["fc"] = [256 * 4 * 4, 512]

        dec_layers = {}
        dec_layers["fc"] = [512, 256 * 4 * 4]
        dec_layers["conv"] = [
            ConvLayer(256, 256, 2, 2, 0),
            ConvLayer(256, 256, 2, 2, 0),
            ConvLayer(256, 256, 2, 2, 0),
            ConvLayer(256, 3, 2, 2, 0),
        ]
    else:
        A_train = torch.Tensor(train["view1"].reshape(-1, 64 * 64 * 3)).to(device)
        B_train = torch.Tensor(train["view2"].reshape(-1, 64 * 64 * 3)).to(device)
        A_test = torch.Tensor(test["view1"].reshape(-1, 64 * 64 * 3)).to(device)
        B_test = torch.Tensor(test["view2"].reshape(-1, 64 * 64 * 3)).to(device)
        enc_layers = [1024, 512, 512, 2048, 1024, 512]
        dec_layers = [512, 1024, 2048, 512, 512, 1024]

    model = SPLICE(
        n_a=64 * 64 * 3,
        n_b=64 * 64 * 3,
        n_private_a=n_private_a,
        n_private_b=n_private_b,
        n_shared=n_shared,
        conv=args["conv"],
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        msr_layers=dec_layers,
        size=(256, 4, 4),
    ).to(device)

    # model.load_state_dict(torch.load(filepath))

    model.fit(
        A_train,
        B_train,
        A_test,
        B_test,
        model_filepath=filepath,
        batch_size=1000,
        lr=1e-4,
        epochs=5000,
        end_factor=1,
        disent_start=250,
        msr_iter_restart=50,
        msr_iter_normal=7,
        c_disent=1,
        device=device,
        weight_decay=1e-3,
        msr_weight_decay=1e-3,
        checkpoint_freq=10,
        msr_restart=50000,
    )
