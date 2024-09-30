"""
This module defines the encoder, decoder, and activation function used in the SPLICE model.

Functions:
    carlosPlus: Variant of the softplus activation function.

Classes:
    encoder: Feedforward encoder network.
    decoder: Feedforward decoder network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def carlosPlus(x):
    """
    Variant of the softplus activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor.
    """
    return 2 * (F.softplus(x) - np.log(2))


class encoder(nn.Module):
    """
    Feedforward encoder network.

    Attributes:
        nInputs (int): Number of input features.
        nOutputs (int): Number of output features.
        layers (ModuleList): List of linear layers.
    """

    def __init__(self, nInputs, nOutputs, layers, nl=carlosPlus, conv=False):
        """
        Constructor for the encoder class.

        Args:
            nInputs (int): Number of input features.
            nOutputs (int): Number of output features.
            layers (list): Hidden layer sizes.
        """
        if nInputs == 0 or nOutputs == 0:
            return None

        super(encoder, self).__init__()
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.nl = nl
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.nInputs, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers.append(nn.Linear(layers[-1], self.nOutputs))

    def forward(self, x):
        """
        Forward pass through the encoder network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        for layer in self.layers:
            x = self.nl(layer(x))
        return x


class decoder(nn.Module):
    """
    Feedforward decoder network.

    Attributes:
        nInputs (int): Number of input features.
        nOutputs (int): Number of output features.
        layers (ModuleList): List of linear layers.
    """

    def __init__(self, nInputs, nOutputs, layers, nl=carlosPlus, conv=False):
        """
        Constructor for the decoder class.

        Args:
            nInputs (int): Number of input features.
            nOutputs (int): Number of output features.
            layers (_type_): Hidden layer sizes.
        """
        if nInputs == 0 or nOutputs == 0:
            return None

        super(decoder, self).__init__()

        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.nl = nl
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(nInputs, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers.append(nn.Linear(layers[-1], nOutputs))

    def forward(self, x):
        """
        Forward pass through the decoder network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """

        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.nl(layer(x))
            else:
                x = layer(x)

        return x


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Unflatten3D(nn.Module):
    def __init__(self, size):
        super(Unflatten3D, self).__init__()
        self.size = size

    def forward(self, x):
        x = x.view(x.size()[0], self.size[0], self.size[1], self.size[2])
        return x


class conv_encoder(nn.Module):
    def __init__(self, z_dim, convLayers, fcLayers, nl=nn.ReLU):
        if z_dim == 0:
            return None

        super().__init__()
        self.layers = nn.Sequential()

        for layer in convLayers:
            self.layers.append(
                nn.Conv2d(
                    layer["in_channels"],
                    layer["out_channels"],
                    layer["kernel_size"],
                    layer["stride"],
                    layer["padding"],
                )
            )
            self.layers.append(nl(True))

        self.layers.append(Flatten3D())

        for i in range(len(fcLayers) - 1):
            self.layers.append(nn.Linear(fcLayers[i], fcLayers[i + 1]))
            self.layers.append(nl(True))

        self.layers.append(nn.Linear(fcLayers[-1], z_dim))

    def forward(self, x):
        return self.layers(x)


class conv_decoder(nn.Module):
    def __init__(self, z_dim, fcLayers, convLayers, nl=nn.ReLU):
        if z_dim == 0:
            return None

        super().__init__()
        self.layers = nn.Sequential()

        self.layers.append(nn.Linear(z_dim, fcLayers[0]))
        self.layers.append(nl(True))
        for i in range(len(fcLayers) - 1):
            self.layers.append(nn.Linear(fcLayers[i], fcLayers[i + 1]))
            self.layers.append(nl(True))

        self.layers.append(Unflatten3D(convLayers[0]["size"]))

        for layer in reversed(convLayers):
            self.layers.append(nl(True))
            self.layers.append(
                nn.ConvTranspose2d(
                    layer["in_channels"],
                    layer["out_channels"],
                    layer["kernel_size"],
                    layer["stride"],
                    layer["padding"],
                )
            )
        self.layers.pop(0)

    def forward(self, x):
        return self.layers(x)
