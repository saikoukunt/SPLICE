"""
This module defines the SPLICE model.

Classes:
    splice_model: SPLICE model.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR

from splice.base import decoder, encoder


class splice_model(nn.Module):
    def __init__(
        self,
        n_a,
        n_b,
        n_shared,
        n_priv_a,
        n_priv_b,
        layers,
        layers_msr,
        msr_scheme="obs",
    ):
        """
        The SPLICE model.

        Args:
            n_a (int): Number of features in dataset A.
            n_b (int): Number of features in dataset B.
            n_shared (int): Number of shared latents.
            n_priv_a (int): Number of private latents for dataset A.
            n_priv_b (int): Number of private latents for dataset B.
            layers (list[int]): Hidden layer sizes for the encoder and decoder networks.
            layers_msr (list[int]): Hidden layer sizes for the measurement networks.
            msr_scheme (str, optional): "obs" if measurement networks should predict datasets,
                "shared" if they shuold predict shared latents. Defaults to "obs".

        Raises:
            ValueError: Hidden layer size must be greater than the latent size.
            ValueError: Number of shared latents must be greater than 0.
            ValueError: Measurement scheme must be 'obs' or 'shared'.
        """
        # input validation
        if (
            (max(layers) < n_shared)
            or (max(layers) < n_priv_a)
            or (max(layers) < n_priv_b)
            or (max(layers_msr) < n_priv_a)
            or (max(layers_msr) < n_priv_b)
        ):
            raise ValueError("Hidden layer size must be greater than the latent size.")

        if n_shared == 0:
            raise ValueError("Number of shared latents must be greater than 0.")

        if msr_scheme not in ["obs", "shared"]:
            raise ValueError("Measurement scheme must be 'obs' or 'shared'.")

        super().__init__()
        self.n_a = n_a
        self.n_b = n_b
        self.n_shared = n_shared
        self.n_priv_a = n_priv_a
        self.n_priv_b = n_priv_b
        self.msr_scheme = msr_scheme
        self.layers_msr = layers_msr

        # separate encoders generate private and shared latent representations
        self.F_a = encoder(self.n_a, self.n_priv_a, layers) if n_priv_a > 0 else None
        self.F_a2b = encoder(self.n_a, self.n_shared, layers)
        self.F_b2a = encoder(self.n_b, self.n_shared, layers)
        self.F_b = encoder(self.n_b, self.n_priv_b, layers) if n_priv_b > 0 else None

        # decoders reconstruct datasets from combined shared and private latents
        self.G_a = decoder(self.n_priv_a + self.n_shared, self.n_a, layers[::-1])
        self.G_b = decoder(self.n_priv_b + self.n_shared, self.n_b, layers[::-1])

        # measurement networks predict opposite datasets or shared latents
        if msr_scheme == "obs":
            self.M_a2b = decoder(self.n_priv_a, self.n_b, layers_msr)
            self.M_b2a = decoder(self.n_priv_b, self.n_a, layers_msr)
        elif msr_scheme == "shared":
            self.M_a2b = encoder(self.n_priv_a, self.n_shared, layers_msr)
            self.M_b2a = encoder(self.n_priv_b, self.n_shared, layers_msr)
        self.M_a2b = self.M_a2b if n_priv_a > 0 else None
        self.M_b2a = self.M_b2a if n_priv_b > 0 else None

    def forward(self, a, b):
        """
        Forward pass through the SPLICE model.

        Args:
            a (Tensor): Dataset A with shape (n_samples, n_a).
            b (_type_): Dataset B with shape (n_samples, n_b).

        Returns:
            z_a (Tensor): Private latents for dataset A.
            z_b2a (Tensor): Shared latents from dataset B to A.
            z_a2b (Tensor): Shared latents from dataset A to B.
            z_b (Tensor): Private latents for dataset B.
            m_b2a (Tensor): Measurements from dataset B to A.
            m_a2b (Tensor): Measurements from dataset A to B.
            a_hat (Tensor): Reconstructed dataset A.
            b_hat (Tensor): Reconstructed dataset B.
        """
        # private latents come from the same dataset, shared latents cross datasets
        z_a = self.F_a(a) if self.F_a is not None else torch.zeros(a.shape[0])
        z_b2a = self.F_b2a(b)
        z_a2b = self.F_a2b(a)
        z_b = self.F_b(b) if self.F_b is not None else torch.zeros(b.shape[0])

        # reconstruction using concatenated latents
        a_hat = self.G_a(torch.hstack((z_a, z_b2a)))
        b_hat = self.G_b(torch.hstack((z_a2b, z_b)))

        # measurement networks predict opposite datasets/shared latents
        m_a2b = self.M_a2b(z_a) if self.M_a2b is not None else torch.zeros_like(b)
        m_b2a = self.M_b2a(z_b) if self.M_b2a is not None else torch.zeros_like(a)

        return z_a, z_b2a, z_a2b, z_b, m_b2a, m_a2b, a_hat, b_hat

    def freeze_all_except(self, *args):
        """
        Freeze all model parameters except those in the specified networks.

        Args:
            args (nn.Module): Networks to unfreeze.
        """
        for param in self.parameters():
            param.requires_grad = False
        for net in args:
            for param in net.parameters():
                param.requires_grad = True

    def restart_measurement_networks(self, device):
        """
        Cold restart the measurement networks.

        Args:
            device (torch.device): Device on which the model and data are stored.

        Returns:
            list: Network parameters of the new measurement networks.
        """
        if self.msr_scheme == "obs":
            self.M_b2a = decoder(self.n_priv_b, self.n_a, self.layers_msr).to(device)
            self.M_a2b = decoder(self.n_priv_a, self.n_b, self.layers_msr).to(device)
        elif self.msr_scheme == "shared":
            self.M_b2a = encoder(self.n_priv_b, self.n_shared, self.layers_msr).to(
                device
            )
            self.M_a2b = encoder(self.n_priv_a, self.n_shared, self.layers_msr).to(
                device
            )
        self.M_b2a = self.M_b2a if self.n_priv_b > 0 else None
        self.M_a2b = self.M_a2b if self.n_priv_a > 0 else None

        msr_params = []
        if self.M_b2a is not None:
            msr_params += list(self.M_b2a.parameters())
        if self.M_a2b is not None:
            msr_params += list(self.M_a2b.parameters())

        return msr_params

    def fit(
        self,
        a_train,
        b_train,
        a_test,
        b_test,
        epochs=25000,
        lr=1e-3,
        end_factor=1 / 100,
        verbose=True,
        disent_start=1000,
        msr_restart=1000,
        msr_iter_normal=3,
        msr_iter_restart=1000,
        c_disent=0.05,
        device=torch.device("cuda"),
    ):
        """
        Fit the SPLICE model with the Adam optimizer, linear learning rate decay, and no
        batching.

        Args:
            a_train (Tensor): Training dataset A with shape (n_train, n_a).
            b_train (Tensor): Training dataset B with shape (n_train, n_b).
            a_test (Tensor): Testing dataset A with shape (n_test, n_a).
            b_test (Tensor): Testing dataset B with shape (n_test, n_b).
            epochs (int, optional): Number of training epochs. Defaults to 25000.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            end_factor (float, optional): Factor by which to reduce the learning rate. Defaults to 1/100.
            verbose (bool, optional): Whether to print training progress. Defaults to True.
            disent_start (int, optional): Epoch at which to start disentanglement loss. Defaults to 1000.
            msr_restart (int, optional): Interval at which to cold restart measurement networks. Defaults to 1000.
            msr_iter_normal (int, optional): Number of iterations to train measurement networks. Defaults to 3.
            msr_iter_restart (int, optional): Number of iterations to train measurement networks after restart. Defaults to 1000.
            c_disent (float, optional): Weight of the disentanglement loss. Defaults to 0.1.

        Raises:
            ValueError: Training dataset A has the incorrect number of features.
            ValueError: Training dataset B has the incorrect number of features.
            ValueError: Testing dataset A has the incorrect number of features.
            ValueError: Testing dataset B has the incorrect number of features.

        Yields:
            A trained SPLICE model.
        """

        # input validation
        if a_train.shape[1] != self.n_a:
            raise ValueError("Training dataset A has the incorrect number of features.")
        if b_train.shape[1] != self.n_b:
            raise ValueError("Training dataset B has the incorrect number of features.")
        if a_test.shape[1] != self.n_a:
            raise ValueError("Testing dataset A has the incorrect number of features.")
        if b_test.shape[1] != self.n_b:
            raise ValueError("Testing dataset B has the incorrect number of features.")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = LinearLR(
            optimizer, total_iters=epochs, start_factor=1, end_factor=end_factor
        )

        msr_params = []
        if self.M_a2b is not None:
            msr_params += list(self.M_a2b.parameters())
        if self.M_b2a is not None:
            msr_params += list(self.M_b2a.parameters())
        msr_optimizer = torch.optim.Adam(msr_params, lr=lr)

        best_params = copy.deepcopy(self.state_dict())
        best_loss = float("inf")

        for epoch in range(epochs):
            if verbose:
                print(f"{epoch}", end="\r")

            # 1) train encoders/decoders to minimize data reconstruction loss
            self.freeze_all_except(
                self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
            )

            _, _, _, _, _, _, a_hat, b_hat = self(a_train, b_train)
            l_rec_a = F.mse_loss(a_hat, a_train)
            l_rec_b = F.mse_loss(b_hat, b_train)
            rec_loss = l_rec_a + l_rec_b

            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()

            disent_loss = torch.tensor([1]).to(device)
            disent_rec_loss = torch.tensor([1]).to(device)
            msr_loss = torch.tensor([0]).to(device)
            if epoch >= disent_start:

                # 2) train measurement networks to minimize measurement loss
                # cold restart periodically to avoid local minima
                if epoch % msr_restart == 0:
                    msr_params = self.restart_measurement_networks(device)
                    msr_optimizer = torch.optim.Adam(msr_params, lr=lr)
                    msr_iter = msr_iter_restart
                else:
                    msr_iter = msr_iter_normal

                self.freeze_all_except(self.M_a2b, self.M_b2a)

                for i in range(msr_iter):
                    _, _, _, _, m_b2a, m_a2b, a_hat, b_hat = self(a_train, b_train)

                    msr_loss = 0
                    if self.msr_scheme == "obs":
                        l_msr_a = F.mse_loss(m_b2a, a_train)
                        l_msr_b = F.mse_loss(m_a2b, b_train)
                        # normalize by variance of target variables and # of targets to make loss scale-invariant
                        l_msr_a *= self.n_a / a_train.var(dim=0).sum()
                        l_msr_b *= self.n_b / b_train.var(dim=0).sum()
                    elif self.msr_scheme == "shared":
                        l_msr_a = F.mse_loss(m_b2a, z_a2b)
                        l_msr_b = F.mse_loss(m_a2b, z_b2a)
                        # normalize by variance of target variables and # of targets to make loss scale-invariant
                        l_msr_a *= self.n_shared / z_a2b.var(dim=0).sum()
                        l_msr_b *= self.n_shared / z_b2a.var(dim=0).sum()
                    msr_loss = l_msr_a + l_msr_b  # type: ignore

                    msr_optimizer.zero_grad()
                    msr_loss.backward()
                    msr_optimizer.step()

                # 3) train private encoders to minimize disentanglement loss
                self.freeze_all_except(self.F_a, self.F_b)

                _, z_b2a, z_a2b, _, m_b2a, m_a2b, a_hat, b_hat = self(a_train, b_train)

                l_rec_a = F.mse_loss(a_hat, a_train)
                l_rec_b = F.mse_loss(b_hat, b_train)

                l_disent_a = 0
                l_disent_b = 0

                if self.msr_scheme == "obs":
                    l_disent_a = m_b2a.var(dim=0).sum() / a_train.var(dim=0).sum()
                    l_disent_b = m_a2b.var(dim=0).sum() / b_train.var(dim=0).sum()
                elif self.msr_scheme == "shared":
                    l_disent_a = m_b2a.var(dim=0).sum() / z_a2b.var(dim=0).sum()
                    l_disent_b = m_a2b.var(dim=0).sum() / z_b2a.var(dim=0).sum()
                disent_loss = c_disent * (l_disent_a + l_disent_b)
                #     disent_rec_loss = l_rec_a + l_rec_b + disent_loss

                optimizer.zero_grad()
                disent_loss.backward()  # type: ignore
                optimizer.step()

            scheduler.step()

            # 4) save best model + print progress
            if epoch % 500 == 0:
                if rec_loss.item() < best_loss:
                    best_loss = rec_loss.item()
                    best_params = copy.deepcopy(self.state_dict())

                if verbose:
                    z_a, z_b2a, z_a2b, z_b, m_b2a, m_a2b, a_hat, b_hat = self(
                        a_test, b_test
                    )

                    l_rec_a_test = F.mse_loss(a_hat, a_test)
                    l_rec_b_test = F.mse_loss(b_hat, b_test)

                    if self.msr_scheme == "obs":
                        l_disent_a = m_b2a.var(dim=0).sum() / a_test.var(dim=0).sum()
                        l_disent_b = m_a2b.var(dim=0).sum() / b_test.var(dim=0).sum()

                        l_msr_a = F.mse_loss(m_b2a, a_test)
                        l_msr_b = F.mse_loss(m_a2b, b_test)
                        # normalize by variance of target variables and # of targets to make loss scale-invariant
                        l_msr_a *= self.n_a / a_test.var(dim=0).sum()
                        l_msr_b *= self.n_b / b_test.var(dim=0).sum()
                    elif self.msr_scheme == "shared":
                        l_disent_a = m_b2a.var(dim=0).sum() / z_a2b.var(dim=0).sum()
                        l_disent_b = m_a2b.var(dim=0).sum() / z_b2a.var(dim=0).sum()

                        l_msr_a = F.mse_loss(m_b2a, z_a2b)
                        l_msr_b = F.mse_loss(m_a2b, z_b2a)
                        # normalize by variance of target variables and # of targets to make loss scale-invariant
                        l_msr_a *= self.n_shared / z_a2b.var(dim=0).sum()
                        l_msr_b *= self.n_shared / z_b2a.var(dim=0).sum()

                    disent_loss_test = l_disent_a + l_disent_b  # type: ignore
                    msr_loss_test = l_msr_a + l_msr_b  # type: ignore

                    print(
                        "Epoch %d \t source loss: %.4f | %.4f \t target loss: %.4f | %.4f \t disent loss: %.4f | %.4f \t msr loss: %.4f | %.4f"
                        % (
                            epoch,
                            l_rec_a.item(),
                            l_rec_a_test.item(),
                            l_rec_b.item(),
                            l_rec_b_test.item(),
                            disent_loss.item(),  # type: ignore
                            disent_loss_test.item(),
                            msr_loss.item(),
                            msr_loss_test.item(),
                        )
                    )

        self.load_state_dict(best_params)
