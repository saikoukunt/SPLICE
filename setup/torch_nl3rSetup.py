import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def carlosPlus(x, logof2):
    return 2 * (F.softplus(x) - logof2)


def makePlaceCellRepresentation(posRange, nCells, position, centers=None, sigma=None):
    """
    Same as the function in nl3rSetup.py, but can specify centers and sigma

    Args:
        posRange (np.array): the range of the environment
        nCells (int): the number of place cells
        position (np.array): the position of the agent in the environment
        centers (np.array, optional): centers of place cells
        sigma (float, optional): std of place cells
    Returns:
        (np.array): the place cell representation
        centers (np.array): the centers of the place cells
    """

    if centers is None:
        centers = np.linspace(posRange[0], posRange[1], nCells)
    if sigma is None:
        sigma = centers[1] - centers[0]

    ppos = np.dot(np.reshape(position, (len(position), 1)), np.ones((1, nCells)))
    ccen = np.dot(np.ones((len(position), 1)), np.reshape(centers, (1, nCells)))

    return np.exp(-((ppos - ccen) ** 2) / (2 * sigma**2)), centers


class nl3ize(nn.Module):
    """
    nl3ize -- a pytorch model that takes nInputs, nOutputs, a desired number of hidden layers,
    and a desired number of hidden units per layer, and returns a model with that shape.
    It is nonlinear, with multiple layers and many parameters, and it can  be trained
    to do nonlinear regression.

    Same as nl3ize in nl3rSetup.py, but removed scaling layers and used regular softplus activation

    Args:
        nInputs     (int):          the number of inputs
        nOutputs    (int):          the number of outputs
        nLayers     (int):          the number of new nonlinear hidden layers
        nUnits      (int):          the number of units per new hidden layer.
                                    nUnits must be >= min(nRows, nCols)

    Returns: a nonlinear trainable model.
    """

    def __init__(self, nInputs, nOutputs, nLayers, nUnits):
        super(nl3ize, self).__init__()
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.nLayers = nLayers
        self.nUnits = nUnits

        if self.nUnits < min(
            self.nInputs, self.nOutputs
        ):  # also try gradual taper architecture
            raise RuntimeError("nUnits must be >=min(nInputs, nOutputs)")
        if self.nLayers < 1:
            raise RuntimeError("nUnits must be >= 1")

        layers = nn.ModuleList()
        for l in range(self.nLayers):
            if l == 0:
                layers.append(nn.Linear(nInputs, nUnits))
            else:
                layers.append(nn.Linear(nUnits, nUnits))

        # for layer in layers:
        #     layer.apply(self.init_weights)

        layers.append(nn.Linear(nUnits, nOutputs))

        self.layers = layers

    def init_weights(self, m):
        torch.nn.init.normal_(m.weight, std=np.sqrt(1 / m.in_features))
        torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = carlosPlus(layer(x), np.log(2))
        return x


class encoder(nn.Module):
    """
    Triangle shaped pytorch model that gradually reduces dimensionality from
    nInputs to nOutputs using dense layers with nUnits = some power of 2

    Args:
        nInputs     (int):          the number of inputs
        nOutputs    (int):          the number of outputs

    """

    def __init__(self, nInputs, nOutputs, base, expand):
        super(encoder, self).__init__()

        nFirst = 0
        while base**nFirst < nInputs:
            nFirst += 1
        nFirst = base ** (nFirst - 1)  # nearest power of 2 < nInputs
        if expand:
            nFirst = base ** (nFirst)

        nLast = 0
        while base**nLast < nOutputs:
            nLast += 1
        nLast = base ** (nLast)  # nearest power of 2 > nOutputs

        # add layers, halving number of hidden units between layers
        layers = nn.ModuleList()
        if nFirst != nInputs:
            layers.append(nn.Linear(nInputs, nFirst))
        nHidden = nFirst
        while nHidden > nLast:
            layers.append(nn.Linear(nHidden, int(nHidden / base)))
            nHidden = int(nHidden / base)
        if nLast != nOutputs:
            layers.append(nn.Linear(nLast, nOutputs))

        for i, layer in enumerate(layers):
            layer.apply(self.init_weights)

        self.layers = layers

    def init_weights(self, m):
        torch.nn.init.normal_(m.weight, std=np.sqrt(1 / m.in_features))
        torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = carlosPlus(layer(x), np.log(2))
            # x = F.leaky_relu(layer(x))
        return x


class var_encoder(nn.Module):
    """
    Triangle shaped pytorch model that gradually reduces dimensionality from
    nInputs to nOutputs using dense layers with nUnits = some power of 2

    Args:
        nInputs     (int):          the number of inputs
        nOutputs    (int):          the number of outputs

    """

    def __init__(self, nInputs, nOutputs, true_vae):
        super(var_encoder, self).__init__()
        self.true_vae = true_vae

        nFirst = 0
        while 2**nFirst < nInputs:
            nFirst += 1
        nFirst = 2 ** (nFirst - 1)  # nearest power of 2 < nInputs
        # nFirst = 2 ** (nFirst)
        nLast = 0
        while 2**nLast < nOutputs:
            nLast += 1
        nLast = 2 ** (nLast)  # nearest power of 2 > nOutputs

        # add layers, halving number of hidden units between layers
        layers = nn.ModuleList()
        if nFirst != nInputs:
            layers.append(nn.Linear(nInputs, nFirst))
        nHidden = nFirst
        while nHidden > nLast:
            layers.append(nn.Linear(nHidden, int(nHidden / 2)))
            nHidden = int(nHidden / 2)
        if nLast != nOutputs:
            layers.append(nn.Linear(nLast, nOutputs))
            layers.append(nn.Linear(nLast, nOutputs))
        else:
            layers.append(nn.Linear(nHidden * 2, nHidden))

        for i, layer in enumerate(layers):
            layer.apply(self.init_weights)

        self.layers = layers

    def init_weights(self, m):
        torch.nn.init.normal_(m.weight, std=np.sqrt(1 / m.in_features))
        torch.nn.init.zeros_(m.bias)

    def forward(self, x, logVar):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 2:
                x = carlosPlus(layer(x), np.log(2))

        if self.true_vae:
            mu = self.layers[-2](x)
            logVar = self.layers[-1](x)
        else:
            mu = F.tanh(self.layers[-2](x))
            logVar = logVar

        return mu, logVar


class decoder(nn.Module):
    """
    Triangle shaped pytorch model that gradually increases dimensionality from
    nInputs to nOutputs using dense layers with nUnits = some power of 2

    Args:
        nInputs     (int):          the number of inputs
        nOutputs    (int):          the number of outputs

    """

    def __init__(self, nInputs, nOutputs, base, expand):
        super(decoder, self).__init__()

        nFirst = 0
        while base**nFirst < nInputs:
            nFirst += 1
        nFirst = base ** (nFirst)  # nearest power of 2 > nInputs

        nLast = 0
        while base**nLast < nOutputs:
            nLast += 1
        nLast = base ** (nLast - 1)  # nearest power of 2 < nOutputs
        if expand:
            nLast = base ** (nLast)

        layers = nn.ModuleList()

        # add layers, doubling number of hidden units between layers
        if nInputs != nFirst:
            layers.append(nn.Linear(nInputs, nFirst))
        nHidden = nFirst
        while nHidden < nLast:
            layers.append(nn.Linear(nHidden, nHidden * base))
            nHidden *= base
        if nLast != nOutputs:
            layers.append(nn.Linear(nLast, nOutputs))

        for i, layer in enumerate(layers):
            layer.apply(self.init_weights)

        self.layers = layers

    def init_weights(self, m):
        torch.nn.init.normal_(m.weight, std=np.sqrt(1 / m.in_features))
        torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer(x)
            else:
                x = carlosPlus(layer(x), np.log(2))
        return x


class bottleneck(nn.Module):
    """
    bottleneck -- a pytorch model that puts inputs through two successive nl3ize models, the
    first one lowering dimensionality, the second one raising it back up to the dimensionality
    of the targets.

    nInputs     (int):          the number of inputs
    nOutputs    (int):          the number of outputs
    bottleDim   (int):          the dimensionality of the bottleneck
    nLayers     (int):          the number of nonlinear hidden layers in each of the
                                encoder and the decoder (i.e., the model goes through nLayers
                                before reaching a layer with bottleDim units, and then goes through
                                nLayers more before reaching the output layer)
    nUnits      (int):          the number of units in each of the hidden layers.
                                nUnits must be >= min(nInputs, nOutputs, bottleDim)
    """

    def __init__(self, nInputs, nOutputs, bottleDim, nLayers, nHidden):
        super(bottleneck, self).__init__()
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.bottleDim = bottleDim
        self.nLayers = nLayers
        self.nHidden = nHidden

        self.encoder = nl3ize(self.nInputs, self.bottleDim, self.nLayers, self.nHidden)
        self.decoder = nl3ize(self.bottleDim, self.nOutputs, self.nLayers, self.nHidden)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class bottleneck_taper(nn.Module):
    """
    bottleneck_taper -- a pytorch model that puts inputs through an hourglass
    shaped autoencoder that gradually lowers and raises dimensionality through
    a bottleneck.

    nInputs     (int):          the number of inputs
    nOutputs    (int):          the number of outputs
    bottleDim   (int):          the dimensionality of the bottleneck
    """

    def __init__(self, nInputs, nOutputs, bottleDim, base=2, expand=False):
        super(bottleneck_taper, self).__init__()
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.bottleDim = bottleDim

        self.encoder = encoder(self.nInputs, self.bottleDim, base, expand)
        self.decoder = decoder(self.bottleDim, self.nOutputs, base, expand)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class vae_taper(nn.Module):
    """
    bottleneck_taper -- a pytorch model that puts inputs through an hourglass
    shaped autoencoder that gradually lowers and raises dimensionality through
    a bottleneck.

    nInputs     (int):          the number of inputs
    nOutputs    (int):          the number of outputs
    bottleDim   (int):          the dimensionality of the bottleneck
    """

    def __init__(self, nInputs, nOutputs, bottleDim, true_vae=False):
        super(vae_taper, self).__init__()
        self.true_vae = true_vae
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.bottleDim = bottleDim

        self.encoder = var_encoder(self.nInputs, self.bottleDim, true_vae)
        self.decoder = decoder(self.bottleDim, self.nOutputs)

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, logVar):
        mu, logVar = self.encoder(x, logVar)
        z = self.reparameterize(mu, logVar)
        return self.decoder(z), mu, logVar
