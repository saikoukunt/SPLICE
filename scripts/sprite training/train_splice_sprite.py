import os

import numpy as np
import torch as torch

from splice.base import ConvLayer
from splice.splice import SPLICE

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

conv = False
retrain = True

train = np.load("../../data/sprites/single_pose_train.npz")
test = np.load("../../data/sprites/single_pose_test.npz")
filepath = os.path.join("..", "..", "results", "models", "sprites")

# train = np.load("./data/sprites/train.npz")
# test = np.load("./data/sprites/test.npz")

# if conv:
#     A_train = torch.Tensor(train["view1"][:50000].transpose(0, 3, 1, 2)).to(device)
#     B_train = torch.Tensor(train["view2"][:50000].transpose(0, 3, 1, 2)).to(device)
#     A_test = torch.Tensor(test["view1"][:1000].transpose(0, 3, 1, 2)).to(device)
#     B_test = torch.Tensor(test["view2"][:1000].transpose(0, 3, 1, 2)).to(device)

#     enc_layers = {}
#     enc_layers["conv"] = [
#         ConvLayer(3, 256, 5, 1, 2),
#         ConvLayer(256, 256, 5, 2, 2),
#         ConvLayer(256, 256, 5, 2, 2),
#         ConvLayer(256, 256, 5, 2, 2),
#     ]
#     enc_layers["fc"] = [256 * 8 * 8, 2048]

#     dec_layers = {}
#     dec_layers["fc"] = [2048, 256 * 8 * 8]
#     dec_layers["conv"] = [
#         ConvLayer(256, 256, 5, 2, 2, 1),
#         ConvLayer(256, 256, 5, 2, 2, 1),
#         ConvLayer(256, 256, 5, 2, 2, 1),
#         ConvLayer(256, 3, 5, 1, 2, 0),
#     ]

#     model = SPLICE(
#         n_a=64 * 64 * 3,
#         n_b=64 * 64 * 3,
#         n_private_a=288,
#         n_private_b=288,
#         n_shared=2,
#         conv=True,
#         enc_layers=enc_layers,
#         dec_layers=dec_layers,
#         msr_layers=dec_layers,
#         size=(256, 8, 8),
#     ).to(device)

#     filepath = os.path.join(filepath, "splice_sprites_conv.pt")

#     if os.path.exists(filepath) and not retrain:
#         model.load_state_dict(torch.load(filepath))
#     else:
#         model.fit(
#             A_train,
#             B_train,
#             A_test,
#             B_test,
#             model_filepath=filepath,
#             batch_size=25,
#             lr=1e-4,
#             epochs=200,
#             end_factor=1 / 50,
#             disent_start=0,
#             msr_iter_restart=5,
#             msr_iter_normal=5,
#             c_disent=1,
#             device=device,
#             weight_decay=1e-2,
#             msr_weight_decay=1e-1,
#             print_every=1,
#             msr_restart=20,
#         )

# else:
A_train = torch.Tensor(train["view1"].reshape(-1, 64 * 64 * 3)).to(device)
B_train = torch.Tensor(train["view2"].reshape(-1, 64 * 64 * 3)).to(device)
A_test = torch.Tensor(test["view1"].reshape(-1, 64 * 64 * 3)).to(device)
B_test = torch.Tensor(test["view2"].reshape(-1, 64 * 64 * 3)).to(device)

model = SPLICE(
    n_a=64 * 64 * 3,
    n_b=64 * 64 * 3,
    n_private_a=500,
    n_private_b=500,
    n_shared=2,
    conv=False,
    enc_layers=[1024, 512, 512, 2048, 2048, 2048, 512],
    dec_layers=[512, 2048, 2048, 2048, 512, 512, 1024],
    msr_layers=[512, 2048, 2048, 2048, 512, 512, 1024],
    size=None,
).to(device)

filepath = os.path.join(filepath, "splice_sprites.pt")

if os.path.exists(filepath) and not retrain:
    model.load_state_dict(torch.load(filepath))
else:
    model.fit(
        A_train,
        B_train,
        A_test,
        B_test,
        model_filepath=filepath,
        batch_size=1000,
        lr=1e-4,
        epochs=10000,
        end_factor=1 / 50,
        disent_start=0,
        msr_iter_restart=5,
        msr_iter_normal=7,
        c_disent=1,
        device=device,
        weight_decay=1e-3,
        msr_weight_decay=1e-1,
        checkpoint_freq=10,
        msr_restart=50000,
    )


# isomap_filepath = os.path.join(
#     "..",
#     "..",
#     "results",
#     "models",
#     "sprites",
#     "splice_sprites_isomap.pt",
# )
