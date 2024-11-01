"""
This module defines the encoder, decoder, and activation function used in the SPLICE model.

Functions:
    carlosPlus: Variant of the softplus activation function.

Classes:
    encoder: Feedforward encoder network.
    decoder: Feedforward decoder network.
"""

from typing import List

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class carlosPlus(nn.Module):
    """
    Variant of the softplus activation function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor.
    """

    def __init__(self, _=None):
        super(carlosPlus, self).__init__()

    def forward(self, x):
        return 2 * (F.softplus(x) - np.log(2))


class encoder(nn.Module):
    """
    Feedforward encoder network.

    Attributes:
        nInputs (int): Number of input features.
        nOutputs (int): Number of output features.
        layers (ModuleList): List of linear layers.
    """

    def __init__(self, nInputs, z_dim, layers, nl=carlosPlus, conv=False):
        """
        Constructor for the encoder class.

        Args:
            nInputs (int): Number of input features.
            nOutputs (int): Number of output features.
            layers (list): Hidden layer sizes.
        """
        super().__init__()

        if nInputs == 0 or z_dim == 0:
            self.layers = None
        else:
            self.nInputs = nInputs
            self.z_dim = z_dim
            self.layers = nn.Sequential()

            if not conv:
                self.layers.append(nn.Linear(self.nInputs, layers[0]))
                self.layers.append(nl())
                for i in range(len(layers) - 1):
                    self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                    self.layers.append(nl())
                self.layers.append(nn.Linear(layers[-1], self.z_dim))
            else:
                fcLayers = layers["fc"]
                convLayers = layers["conv"]

                self.layers = nn.Sequential()

                for layer in convLayers:
                    self.layers.append(
                        nn.Conv2d(
                            layer.in_channels,
                            layer.out_channels,
                            layer.kernel_size,
                            layer.stride,
                            layer.padding,
                        )
                    )
                    self.layers.append(nl(True))

                if len(fcLayers) > 0:
                    self.layers.append(Flatten3D())

                    for i in range(len(fcLayers) - 1):
                        self.layers.append(nn.Linear(fcLayers[i], fcLayers[i + 1]))
                        self.layers.append(nl(True))

                    self.layers.append(nn.Linear(fcLayers[-1], z_dim))

    def forward(self, x):
        """
        Forward pass through the encoder network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.layers is None:
            return None

        return self.layers(x)


class decoder(nn.Module):
    """
    Feedforward decoder network.

    Attributes:
        nInputs (int): Number of input features.
        nOutputs (int): Number of output features.
        layers (ModuleList): List of linear layers.
    """

    def __init__(self, z_dim, nOutputs, layers, nl=carlosPlus, conv=False, size=None):
        """
        Constructor for the decoder class.

        Args:
            nInputs (int): Number of input features.
            nOutputs (int): Number of output features.
            layers (_type_): Hidden layer sizes.
        """
        super().__init__()

        if z_dim == 0 or nOutputs == 0:
            self.layers = None

        else:
            self.z_dim = z_dim
            self.nOutputs = nOutputs
            self.layers = nn.Sequential()

            if not conv:
                self.layers.append(nn.Linear(z_dim, layers[0]))
                self.layers.append(nl())
                for i in range(len(layers) - 1):
                    self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                    self.layers.append(nl())
                self.layers.append(nn.Linear(layers[-1], nOutputs))

            else:
                fcLayers = layers["fc"]
                convLayers = layers["conv"]

                if len(fcLayers) > 0:
                    self.layers.append(nn.Linear(z_dim, fcLayers[0]))
                    self.layers.append(nl(True))
                    for i in range(len(fcLayers) - 1):
                        self.layers.append(nn.Linear(fcLayers[i], fcLayers[i + 1]))
                        self.layers.append(nl(True))

                    self.layers.append(Unflatten3D(size))

                for layer in convLayers:
                    self.layers.append(nl(True))
                    self.layers.append(
                        nn.ConvTranspose2d(
                            layer.in_channels,
                            layer.out_channels,
                            layer.kernel_size,
                            layer.stride,
                            layer.padding,
                        )
                    )

    def forward(self, x):
        """
        Forward pass through the decoder network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        if self.layers is None:
            return None

        return self.layers(x)


class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.reshape(x.size()[0], -1)
        return x


class Unflatten3D(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        x = x.reshape(x.size()[0], self.size[0], self.size[1], self.size[2])
        return x
