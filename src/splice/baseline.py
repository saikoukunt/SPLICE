import numpy as np
import torch
import torch.nn as nn

from splice.base import decoder, encoder


class DCCA(nn.Module):
    def __init__(self, n_a, n_b, z_dim, device, conv=True, layers=[]):
        super().__init__()
        self.z_dim = z_dim

        if conv:
            self.encoder_a = conv_encoder(z_dim).to(device)
            self.encoder_b = conv_encoder(z_dim).to(device)
        else:
            self.encoder_a = encoder(n_a, z_dim, layers, nl=nn.Sigmoid).to(device)
            self.encoder_b = encoder(n_b, z_dim, layers, nl=nn.Sigmoid).to(device)

    def loss(self, z_a, z_b, g):
        mse = nn.MSELoss()
        cca_loss = (mse(z_a, g) + mse(z_b, g)) / 2

        return cca_loss

    def forward(self, x_a, x_b):
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)

        return z_a, z_b

    def update_G(self, x_a, x_b, batch_size):
        with torch.no_grad():
            z_a, z_b = self(x_a, x_b)
            Y = z_a + z_b

            Y = Y - torch.mean(Y, axis=0)  # type: ignore
            G, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            G = G @ Vh
            G = np.sqrt(batch_size) * G

        return G


class Karakasis(nn.Module):
    def __init__(self, n_a, n_b, z_dim, device, _lambda, conv=True, layers=[]):
        super().__init__()
        self.z_dim = z_dim
        if conv:
            self.encoder_a = conv_encoder(z_dim).to(device)
            self.encoder_b = conv_encoder(z_dim).to(device)
            self.decoder_a = conv_decoder(z_dim).to(device)
            self.decoder_b = conv_decoder(z_dim).to(device)
        else:
            self.encoder_a = encoder(n_a, z_dim, layers, nl=nn.functional.sigmoid).to(
                device
            )
            self.encoder_b = encoder(n_b, z_dim, layers, nl=nn.functional.sigmoid).to(
                device
            )
            self.decoder_a = decoder(
                z_dim, n_a, layers[::-1], nl=nn.functional.sigmoid
            ).to(device)
            self.decoder_b = decoder(
                z_dim, n_b, layers[::-1], nl=nn.functional.sigmoid
            ).to(device)

        self._lambda = _lambda

    def forward(self, x_a, x_b):
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)
        x_a_hat = self.decoder_a(z_b)
        x_b_hat = self.decoder_b(z_a)

        return x_a_hat, x_b_hat, z_a, z_b

    def loss(self, x_a, x_b, a_hat, b_hat, z_a, z_b, g):
        mse = nn.MSELoss()
        cca_loss = (mse(z_a, g) + mse(z_b, g)) / 2
        recon_loss_a = mse(x_a, a_hat)
        recon_loss_b = mse(x_b, b_hat)

        return (
            (1 - self._lambda) * cca_loss
            + self._lambda * (recon_loss_a + recon_loss_b) / 2,
            cca_loss,
            recon_loss_a,
            recon_loss_b,
        )

    def update_G(self, x_a, x_b, batch_size):
        with torch.no_grad():
            a_hat, b_hat, z_a, z_b = self(x_a, x_b)
            Y = z_a + z_b

            Y = Y - torch.mean(Y, axis=0)  # type: ignore
            G, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            G = G @ Vh
            G = np.sqrt(batch_size) * G

        return G


class DCCAE(nn.Module):
    def __init__(self, n_a, n_b, z_dim, device, _lambda, conv=True, layers=[]):
        super().__init__()
        self.z_dim = z_dim
        if conv:
            self.encoder_a = conv_encoder(z_dim).to(device)
            self.encoder_b = conv_encoder(z_dim).to(device)
            self.decoder_a = conv_decoder(z_dim).to(device)
            self.decoder_b = conv_decoder(z_dim).to(device)
        else:
            self.encoder_a = encoder(n_a, z_dim, layers, nl=nn.Sigmoid).to(
                device
            )
            self.encoder_b = encoder(n_b, z_dim, layers, nl=nn.Sigmoid).to(
                device
            )
            self.decoder_a = decoder(
                z_dim, n_a, layers[::-1], nl=nn.Sigmoid
            ).to(device)
            self.decoder_b = decoder(
                z_dim, n_b, layers[::-1], nl=nn.Sigmoid
            ).to(device)

        self._lambda = _lambda

    def forward(self, x_a, x_b):
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)
        x_a_hat = self.decoder_a(z_a)
        x_b_hat = self.decoder_b(z_b)

        return x_a_hat, x_b_hat, z_a, z_b

    def loss(self, x_a, x_b, a_hat, b_hat, z_a, z_b, g):
        mse = nn.MSELoss()
        cca_loss = (mse(z_a, g) + mse(z_b, g)) / 2
        recon_loss_a = mse(x_a, a_hat)
        recon_loss_b = mse(x_b, b_hat)

        return (
            (1 - self._lambda) * cca_loss
            + self._lambda * (recon_loss_a + recon_loss_b) / 2,
            cca_loss,
            recon_loss_a,
            recon_loss_b,
        )

    def update_G(self, x_a, x_b, batch_size):
        with torch.no_grad():
            a_hat, b_hat, z_a, z_b = self(x_a, x_b)
            Y = z_a + z_b

            Y = Y - torch.mean(Y, axis=0)  # type: ignore
            G, S, Vh = torch.linalg.svd(Y, full_matrices=False)
            G = G @ Vh
            G = np.sqrt(batch_size) * G

        return G
    
class Flatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class Unflatten3D(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], 32, 7, 7)
        return x
    
class conv_encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            Flatten3D(),
            nn.Linear(7 * 7 * 32, 256),
            nn.ReLU(True),
            nn.Linear(256, z_dim),
        )

    def forward(self, x):
        return self.layers(x)


class conv_decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 7 * 7 * 32),
            nn.ReLU(True),
            Unflatten3D(),
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.layers(x)
