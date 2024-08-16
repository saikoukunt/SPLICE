import numpy as np
import torch
import torch.nn as nn

from splice.base import conv_decoder, conv_encoder, decoder, encoder


class DCCA(nn.Module):
    def __init__(self, n_a, n_b, z_dim, device, conv=True, layers=[]):
        super().__init__()
        self.z_dim = z_dim

        if conv:
            self.encoder_a = conv_encoder(z_dim).to(device)
            self.encoder_b = conv_encoder(z_dim).to(device)
        else:
            self.encoder_a = encoder(n_a, z_dim, layers, nl=nn.functional.sigmoid).to(
                device
            )
            self.encoder_b = encoder(n_b, z_dim, layers, nl=nn.functional.sigmoid).to(
                device
            )

        self.cca_loss = cca_loss(
            z_dim, use_all_singular_values=False, device=device
        ).loss

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
        self.cca_loss = cca_loss(
            z_dim, use_all_singular_values=False, device=device
        ).loss

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
        self.cca_loss = cca_loss(
            z_dim, use_all_singular_values=False, device=device
        ).loss

    def forward(self, x_a, x_b):
        z_a = self.encoder_a(x_a)
        z_b = self.encoder_b(x_b)
        x_a_hat = self.decoder_a(z_a)
        x_b_hat = self.decoder_b(z_b)

        return x_a_hat, x_b_hat, z_a, z_b

    def loss(self, x_a, x_b, a_hat, b_hat, z_a, z_b):
        cca_loss = self.cca_loss(z_a, z_b)
        recon_loss_a = torch.mean((x_a - a_hat) ** 2)
        recon_loss_b = torch.mean((x_b - b_hat) ** 2)

        return (
            cca_loss + self._lambda * (recon_loss_a + recon_loss_b),
            cca_loss,
            recon_loss_a,
            recon_loss_b,
        )


class cca_loss:
    """
    An implementation of the loss function of linear CCA as introduced
    in the original paper for ``DCCA`` [#1DCCA]_. Details of how this loss
    is computed can be found in the paper or in the documentation for
    ``DCCA``.

    Parameters
    ----------
    n_components : int (positive)
        The output dimensionality of the CCA transformation.
    use_all_singular_values : boolean
        Whether or not to use all the singular values in the loss calculation.
        If False, only use the top n_components singular values.
    device : torch.device object
        The torch device being used in DCCA.

    Attributes
    ----------
    n_components_ : int (positive)
        The output dimensionality of the CCA transformation.
    use_all_singular_values_ : boolean
        Whether or not to use all the singular values in the loss calculation.
        If False, only use the top ``n_components`` singular values.
    device_ : torch.device object
        The torch device being used in DCCA.

    """

    def __init__(self, n_components, use_all_singular_values, device):
        self.n_components_ = n_components
        self.use_all_singular_values_ = use_all_singular_values
        self.device_ = device

    def loss(self, H1, H2):
        """
        Compute the loss (negative correlation) between 2 views. Details can
        be found in [#1DCCA]_ or the documentation for ``DCCA``.

        Parameters
        ----------
        H1: torch.tensor, shape (n_samples, n_features)
            View 1 data.
        H2: torch.tensor, shape (n_samples, n_features)
            View 2 data.
        """

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
            o1, device=self.device_
        )
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(
            o2, device=self.device_
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
        Tval = torch.matmul(
            torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv
        )

        if self.use_all_singular_values_:
            # all singular values are used to calculate the correlation (and
            # thus the loss as well)
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            corr = torch.sqrt(tmp)
        else:
            # just the top self.n_components_ singular values are used to
            # compute the loss
            U, V = torch.linalg.eigh(torch.matmul(Tval.t(), Tval))
            U = U.topk(self.n_components_)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr
