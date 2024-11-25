import os

import matplotlib.pyplot as plt
import numpy as np
import torch as torch
from sklearn.decomposition import PCA

from splice.base import ConvLayer, decoder, encoder
from splice.nonalternating_baseline import DCCA
from splice.splice import SPLICE

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

train = np.load("../../data/sprites/train.npz")
test = np.load("../../data/sprites/test.npz")


A_train = torch.Tensor(train["view1"][:50000].transpose(0, 3, 1, 2)).to(device)
B_train = torch.Tensor(train["view2"][:50000].transpose(0, 3, 1, 2)).to(device)

A_test = torch.Tensor(test["view1"][:1000].transpose(0, 3, 1, 2)).to(device)
B_test = torch.Tensor(test["view2"][:1000].transpose(0, 3, 1, 2)).to(device)

# A_train = torch.Tensor(train["view1"][:50000].reshape(-1, 64 * 64 * 3)).to(device)
# B_train = torch.Tensor(train["view2"][:50000].reshape(-1, 64 * 64 * 3)).to(device)

# A_test = torch.Tensor(test["view1"][:1000].reshape(-1, 64 * 64 * 3)).to(device)
# B_test = torch.Tensor(test["view2"][:1000].reshape(-1, 64 * 64 * 3)).to(device)

enc_layers = {}
enc_layers["conv"] = [
    ConvLayer(3, 64, 4, 2, 1),
    ConvLayer(64, 64, 4, 2, 1),
    ConvLayer(64, 64, 4, 2, 1),
]
enc_layers["fc"] = [64 * 8 * 8, 256]

dec_layers = {}
dec_layers["fc"] = [256, 64 * 8 * 8]
dec_layers["conv"] = [
    ConvLayer(64, 64, 4, 2, 1),
    ConvLayer(64, 64, 4, 2, 1),
    ConvLayer(64, 3, 4, 2, 1),
]

model = SPLICE(
    n_a=64 * 64 * 3,
    n_b=64 * 64 * 3,
    n_priv_a=30,
    n_priv_b=30,
    n_shared=2,
    conv=True,
    layers_enc=enc_layers,
    layers_dec=dec_layers,
    layers_msr=dec_layers,
    size=(64, 8, 8),
).to(device)

<<<<<<< HEAD
model = SPLICE(
    n_a=64 * 64 * 3,
    n_b=64 * 64 * 3,
    n_priv_a=30,
    n_priv_b=30,
    n_shared=2,
    conv=False,
    layers_enc=[200, 200, 100, 100, 100, 100],
    layers_dec=[200, 200, 100, 100, 100, 100],
    layers_msr=[200, 200, 100, 100, 100, 100],
    size=None,
).to(device)
=======
# model = SPLICE(
#     n_a=64 * 64 * 3,
#     n_b=64 * 64 * 3,
#     n_priv_a=30,
#     n_priv_b=30,
#     n_shared=2,
#     conv=False,
#     layers_enc=[200, 200, 100, 100, 100, 100],
#     layers_dec=[200, 200, 100, 100, 100, 100],
#     layers_msr=[200, 200, 100, 100, 100, 100],
#     size=None,
# ).to(device)
>>>>>>> 6bb97c23a9aa83707f5a306cb03c3b1d37406d5a

filepath = os.path.join("..", "..", "results", "models", "sprites", "splice_sprites.pt")

retrain = True

if os.path.exists(filepath) and not retrain:
    model.load_state_dict(torch.load(filepath))
else:
    model.fit(
        A_train,
        B_train,
        B_test,
        A_test,
        model_filepath=filepath,
        batch_size=100,
        lr=1e-4,
        epochs=200,
        end_factor=1,
        disent_start=0,
        disent_iter=1,
        rec_iter=2,
        msr_iter_restart=30,
        msr_iter_normal=10,
        c_disent=0.1,
        device=device,
        weight_decay=1e-3,
        print_every=1,
        msr_restart=100,
    )

# isomap_filepath = os.path.join(
#     "..",
#     "..",
#     "results",
#     "models",
#     "sprites",
#     "splice_sprites_isomap.pt",
# )
