import os

import numpy as np
import torch

from splice.splice import SPLICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    data = np.load(
        r"C:\Users\Harris_Lab\Projects\SPLICE\data\mnist\mnist_rotated_shared-digit_missing.npz"
    )

    X = torch.Tensor(data["view1"][:50000]).to(device)
    Y = torch.Tensor(data["view2"][:50000]).to(device)
    X_val = torch.Tensor(data["view1"][50000:60000]).to(device)
    Y_val = torch.Tensor(data["view2"][50000:60000]).to(device)

    model = SPLICE(
        n_a=784,
        n_b=784,
        n_shared=30,
        n_private_a=0,
        n_private_b=3,
        enc_layers=[256, 128, 64, 64],
        dec_layers=[64, 64, 128, 256],
        msr_layers=[64, 64, 128, 256],
    ).to(device)

    filepath = os.path.join(
        "..", "..", "results", "models", "mnist", "splice_mnist_missing-rotations.pt"
    )
    retrain_splice = True

    if os.path.exists(filepath) and not retrain_splice:
        model.load_state_dict(torch.load(filepath))
    else:
        model.fit(
            X,
            Y,
            X_val,
            Y_val,
            model_filepath=filepath,
            batch_size=100,
            epochs=100,
            lr=1e-3,
            end_factor=1 / 500,
            disent_start=10,
            msr_iter_normal=7,
            msr_iter_restart=30,
            c_disent=1,
            device=device,
            weight_decay=1e-3,
            msr_weight_decay=1e-3,
            checkpoint_freq=1,
        )

    retrain_isomap = True
    isomap1_filepath = os.path.join(
        "..",
        "..",
        "results",
        "models",
        "mnist",
        "splice_isomap_mnist_missing-rotations-fix7.pt",
    )
    fix_index = np.argwhere(data["labels"] == 7).flatten()[1]
    if retrain_isomap or not os.path.exists(isomap1_filepath):
        model.fit_isomap_splice(
            X,
            Y,
            X_val,
            Y_val,
            model_filepath=isomap1_filepath,
            fix_index=fix_index,
            batch_size=100,
            epochs=100,
            lr=1e-5,
            end_factor=1 / 100,
            c_disent=1,
            disent_start=0,
            msr_iter_normal=7,
            msr_iter_restart=30,
            device=device,
            weight_decay=1e-3,
            msr_weight_decay=1e-3,
            checkpoint_freq=1,
            c_prox=1,
        )

    isomap2_filepath = os.path.join(
        "..",
        "..",
        "results",
        "models",
        "mnist",
        "splice_isomap_mnist_missing-rotations-fix3.pt",
    )
    fix_index = np.argwhere(data["labels"] == 3).flatten()[1]
    model.load_state_dict(torch.load(filepath))
    if retrain_isomap or not os.path.exists(isomap2_filepath):
        model.fit_isomap_splice(
            X,
            Y,
            X_val,
            Y_val,
            model_filepath=isomap2_filepath,
            fix_index=fix_index,
            batch_size=100,
            epochs=100,
            lr=1e-5,
            end_factor=1 / 100,
            c_disent=1,
            disent_start=0,
            msr_iter_normal=7,
            msr_iter_restart=30,
            device=device,
            weight_decay=1e-3,
            msr_weight_decay=1e-3,
            checkpoint_freq=1,
            c_prox=1,
        )
