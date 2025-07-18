import os
import sys

import numpy as np
import torch

from splice.splice import SPLICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        raise ValueError("Please provide the number of private dimensions")
    n_private = int(args[0])

    data = np.load(
        r"C:\Users\Harris_Lab\Projects\SPLICE\data\mnist\mnist_rotated_shared-digit.npz"
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
        n_private_a=0,
        n_private_b=n_private,
        enc_layers=[256, 128, 64, 30 + n_private],
        dec_layers=[30 + n_private, 64, 128, 256],
        msr_layers=[30 + n_private, 64, 128, 256],
    ).to(device)

    filepath = os.path.join(
        "..", "..", "results", "models", "mnist", "splice_mnist_%dD.pt" % n_private
    )

    retrain = False

    if os.path.exists(filepath) and not retrain:
        model.load_state_dict(torch.load(filepath))
    else:
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
            disent_start=5,
            msr_restart=1000,
            msr_iter_normal=7,
            msr_iter_restart=30,
            c_disent=1,
            device=device,
            weight_decay=0,
            msr_weight_decay=0,
            checkpoint_freq=1,
        )
    isomap_filepath = os.path.join(
        "..",
        "..",
        "results",
        "models",
        "mnist",
        "splice_isomap_mnist_%dD.pt" % n_private,
    )
    fix_index = np.argwhere(data["labels"] == 7).flatten()[1]

    model.fit_isomap_splice(
        X,
        Y,
        X_val,
        Y_val,
        isomap_filepath,
        fix_index=fix_index,
        epochs=10001,
        lr=1e-3,
        end_factor=1,
        disent_start=1000,
        msr_restart=10005,
        msr_iter_normal=7,
        msr_iter_restart=7,
        c_disent=1,
        device=device,
        weight_decay=0,
        msr_weight_decay=0,
        checkpoint_freq=100,
        n_landmarks=200,
        n_neighbors_shared_a=100,
        n_neighbors_shared_b=100,
        n_neighbors_private_a=100,
        n_neighbors_private_b=100,
        c_prox=0.01,
    )
