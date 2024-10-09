import os
import sys

import numpy as np
import pandas as pd
import torch

from splice.splice import SPLICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("Please provide the number of private dimensions")
    n_private = int(args[0])

    data = np.load(
        r"C:\Users\Harris_Lab\Projects\SPLICE\data\mnist\mnist_rotated_360.npz"
    )

    X = torch.Tensor(data["original"][:50000]).to(device)
    Y = torch.Tensor(data["rotated"][:50000]).to(device)

    X_val = torch.Tensor(data["original"][50000:60000]).to(device)

    Y_val = torch.Tensor(data["rotated"][50000:60000]).to(device)

    X_test = torch.Tensor(data["original"][60000:]).to(device)
    Y_test = torch.Tensor(data["rotated"][60000:]).to(device)
    angles_test = data["angles"][60000:]
    labels_test = data["labels"][60000:]

    model = SPLICE(
        n_a=784,
        n_b=784,
        n_shared=30,
        n_priv_a=0,
        n_priv_b=n_private,
        layers_enc=[256, 128, 64, 32],
        layers_dec=[32, 64, 128, 256],
        layers_msr=[32, 32, 64, 128, 256],
    ).to(device)

    filepath = os.path.join(
        "..", "..", "results", "models", "mnist", "splice_mnist_%dD.pt" % n_private
    )

    # if os.path.exists(filepath):
    #     model.load_state_dict(torch.load(filepath))
    # else:
    model.fit(
        X,
        Y,
        X_val,
        Y_val,
        model_filepath=filepath,
        batch_size=100,
        epochs=50,
        lr=1e-3,
        end_factor=1 / 100,
        rec_iter=2,
        disent_start=10,
        msr_restart=5,
        msr_iter_normal=3,
        msr_iter_restart=10,
        c_disent=0.1,
        device=device,
        weight_decay=1e-3,
        print_every=1,
    )
