"""
This module defines the encoder, decoder, and activation function used in the SPLICE model.

Functions:
    carlosPlus: Variant of the softplus activation function.

Classes:
    encoder: Feedforward encoder network.
    decoder: Feedforward decoder network.
"""

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

    def __init__(self, nInputs, nOutputs, layers):
        """
        Constructor for the encoder class.

        Args:
            nInputs (int): Number of input features.
            nOutputs (int): Number of output features.
            layers (list): Hidden layer sizes.
        """
        super(encoder, self).__init__()
        self.nInputs = nInputs
        self.nOutputs = nOutputs
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
            x = carlosPlus(layer(x))
        return x


class decoder(nn.Module):
    """
    Feedforward decoder network.

    Attributes:
        nInputs (int): Number of input features.
        nOutputs (int): Number of output features.
        layers (ModuleList): List of linear layers.
    """

    def __init__(self, nInputs, nOutputs, layers):
        """
        Constructor for the decoder class.

        Args:
            nInputs (int): Number of input features.
            nOutputs (int): Number of output features.
            layers (_type_): Hidden layer sizes.
        """
        super(decoder, self).__init__()

        self.nInputs = nInputs
        self.nOutputs = nOutputs
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
            if i == len(self.layers) - 1:  # don't apply nonlinearity to last layer
                x = layer(x)
            else:
                x = carlosPlus(layer(x))
        return x
