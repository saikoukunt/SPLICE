import numpy as np
import torch
import torch.nn as nn

from splice.base import decoder, encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_cca_loss(H1, H2, n_components_):
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9

    # Transpose matrices so each column is a sample
    H1, H2 = H1.t(), H2.t()

    o1 = o2 = H1.size(0)

    m = H1.size(1)

    H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
    H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

    # Compute covariance matrices and add diagonal so they are
    # positive definite
    SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
    SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(
        o1, device=device
    )
    SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(
        o2, device=device
    )

    # Calculate the root inverse of covariance matrices by using
    # eigen decomposition
    [D1, V1] = torch.linalg.eigh(SigmaHat11)
    [D2, V2] = torch.linalg.eigh(SigmaHat22)

    # Additional code to increase numerical stability
    posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    # Compute sigma hat matrices using the edited covariance matrices
    SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1**-0.5)), V1.t())
    SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2**-0.5)), V2.t())

    # Compute the T matrix, whose matrix trace norm is the loss
    Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    # just the top n_components_ singular values are used to
    # compute the loss
    U, V = torch.linalg.eigh(torch.matmul(Tval.t(), Tval))
    U = U.topk(n_components_)[0]
    corr = torch.sum(torch.sqrt(U))
    return 1 - corr / n_components_


class DCCA(nn.Module):
    def __init__(self, n_a, n_b, z_dim, device, layers=[]):
        super().__init__()
        self.z_dim = z_dim

        self.encoder_a = encoder(n_a, z_dim, layers, nl=nn.Sigmoid).to(device)
        self.encoder_b = encoder(n_b, z_dim, layers, nl=nn.Sigmoid).to(device)

    def loss(self, z_a, z_b):
        return calc_cca_loss(z_a, z_b, self.z_dim)

    def forward(self, x_a, x_b):
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)

        return z_a, z_b


class DCCAE(nn.Module):
    def __init__(self, n_a, n_b, z_dim, device, _lambda, layers=[]):
        super().__init__()
        self.z_dim = z_dim
        self._lambda = _lambda

        self.encoder_a = encoder(n_a, z_dim, layers, nl=nn.Sigmoid).to(device)
        self.encoder_b = encoder(n_b, z_dim, layers, nl=nn.Sigmoid).to(device)
        self.decoder_a = decoder(z_dim, n_a, layers[::-1], nl=nn.Sigmoid).to(device)
        self.decoder_b = decoder(z_dim, n_b, layers[::-1], nl=nn.Sigmoid).to(device)

    def forward(self, x_a, x_b):
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)

        a_hat = self.decoder_a(z_a)
        b_hat = self.decoder_b(z_b)

        return a_hat, b_hat, z_a, z_b

    def loss(self, x_a, x_b, a_hat, b_hat, z_a, z_b):
        mse = nn.MSELoss()

        cca_loss = calc_cca_loss(z_a, z_b, self.z_dim)
        recon_loss_a = mse(x_a, a_hat)
        recon_loss_b = mse(x_b, b_hat)

        return (
            (1 - self._lambda) * cca_loss
            + self._lambda * (recon_loss_a + recon_loss_b) / 2,
            cca_loss,
            recon_loss_a,
            recon_loss_b,
        )


class Karakasis(nn.Module):
    def __init__(self, n_a, n_b, z_dim, device, _lambda, layers=[]):
        super().__init__()
        self.z_dim = z_dim
        self._lambda = _lambda

        self.encoder_a = encoder(n_a, z_dim, layers, nl=nn.Sigmoid).to(device)
        self.encoder_b = encoder(n_b, z_dim, layers, nl=nn.Sigmoid).to(device)
        self.decoder_a = decoder(z_dim, n_a, layers[::-1], nl=nn.Sigmoid).to(device)
        self.decoder_b = decoder(z_dim, n_b, layers[::-1], nl=nn.Sigmoid).to(device)

    def forward(self, x_a, x_b):
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)

        a_hat = self.decoder_a(z_b)
        b_hat = self.decoder_b(z_a)

        return a_hat, b_hat, z_a, z_b

    def loss(self, x_a, x_b, a_hat, b_hat, z_a, z_b):
        mse = nn.MSELoss()

        cca_loss = calc_cca_loss(z_a, z_b, self.z_dim)
        recon_loss_a = mse(x_a, a_hat)
        recon_loss_b = mse(x_b, b_hat)

        return (
            (1 - self._lambda) * cca_loss
            + self._lambda * (recon_loss_a + recon_loss_b) / 2,
            cca_loss,
            recon_loss_a,
            recon_loss_b,
        )
