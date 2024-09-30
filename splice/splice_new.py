import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import lil_array
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import LinearLR

from splice.base import carlosPlus, decoder, encoder, conv_decoder, conv_encoder
from splice.utils import calculate_isomap_dists, iso_loss_func


class SPLICECore(nn.Module):
    def __init__(
        self,
        n_a,
        n_b,
        n_shared,
        n_priv_a,
        n_priv_b,
        layers_enc,
        layers_dec,
        nl=carlosPlus,
    ):
        super().__init__()

        self.n_a = n_a
        self.n_b = n_b
        self.n_shared = n_shared
        self.n_priv_a = n_priv_a
        self.n_priv_b = n_priv_b
        self.nl = nl

        self.F_a = encoder(self.n_a, self.n_priv_a, layers_enc)
        self.F_a2b = encoder(self.n_a, self.n_shared, layers_enc)
        self.F_b2a = encoder(self.n_b, self.n_shared, layers_enc)
        self.F_b = encoder(self.n_b, self.n_priv_b, layers_enc)

        self.G_a = decoder(self.n_priv_a + self.n_shared, self.n_a, layers_dec)
        self.G_b = decoder(self.n_priv_b + self.n_shared, self.n_b, layers_dec)

    def forward(self, x_a, x_b):
        z_a, z_b2a, z_a2b, z_b = self.encode(x_a, x_b)
        a_hat, b_hat = self.decode(z_a, z_b2a, z_a2b, z_b)

        return z_a, z_b2a, z_a2b, z_b, a_hat, b_hat

    def encode(self, x_a, x_b):
        z_a = self.F_a(x_a) if self.F_a is not None else None
        z_a2b = self.F_a2b(x_a)
        z_b2a = self.F_b2a(x_b)
        z_b = self.F_b(x_b) if self.F_b is not None else None

        return z_a, z_b2a, z_a2b, z_b

    def decode(self, z_a, z_b2a, z_a2b, z_b):
        a_latents = torch.cat([z_a, z_b2a], dim=1) if z_a is not None else z_b2a
        b_latents = torch.cat([z_b, z_a2b], dim=1) if z_b is not None else z_a2b

        a_hat = self.G_a(a_latents)
        b_hat = self.G_b(b_latents)

        return a_hat, b_hat

    def project_to_submanifolds(self, a, b, fix_index=None):
        pass


class SPLICE(SPLICECore):
    def __init__(
        self,
        n_a,
        n_b,
        n_shared,
        n_priv_a,
        n_priv_b,
        layers_enc,
        layers_dec,
        layers_msr,
        nl=carlosPlus,
    ):
        super().__init__(
            n_a, n_b, n_shared, n_priv_a, n_priv_b, layers_enc, layers_dec, nl
        )
        self.layers_msr = layers_msr
        if n_priv_a > 0:
            self.M_a2b = decoder(n_priv_a, n_b, layers_msr, nl)
        if n_priv_b > 0:
            self.M_b2a = decoder(n_priv_b, n_a, layers_msr, nl)

    def forward(self, x_a, x_b):
        z_a, z_b2a, z_a2b, z_b, a_hat, b_hat = super()(x_a, x_b)
        m_a2b = self.M_a2b(z_a) if self.n_priv_a > 0 else None
        m_b2a = self.M_b2a(z_b) if self.n_priv_b > 0 else None

        return z_a, z_b2a, z_a2b, z_b, m_a2b, m_b2a, a_hat, b_hat

    def measure(self, x_a, x_b):
        z_a, z_b2a, z_a2b, z_b = self.encode(x_a, x_b)
        m_a2b = self.M_a2b(z_a) if self.M_a2b is not None else None
        m_b2a = self.M_b2a(z_b) if self.M_b2a is not None else None

        return z_a, z_b2a, z_a2b, z_b, m_a2b, m_b2a

    def fit(
        self,
        a_train,
        b_train,
        a_test,
        b_test,
        model_filepath,
        batch_size,
        epochs=25000,
        lr=1e-3,
        end_factor=1 / 100,
        disent_start=1000,
        msr_restart=1000,
        msr_iter_normal=3,
        msr_iter_restart=1000,
        c_disent=0.1,
        device=torch.device("cuda"),
        weight_decay=0,
        print_every=500,
    ):
        # input validation
        if a_train.shape[1] != self.n_a:
            raise ValueError("Training dataset A has the incorrect number of features.")
        if b_train.shape[1] != self.n_b:
            raise ValueError("Training dataset B has the incorrect number of features.")
        if a_test.shape[1] != self.n_a:
            raise ValueError("Testing dataset A has the incorrect number of features.")
        if b_test.shape[1] != self.n_b:
            raise ValueError("Testing dataset B has the incorrect number of features.")

        n_batches = math.ceil(a_train.shape[0] / batch_size)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = LinearLR(
            optimizer, total_iters=epochs, start_factor=1, end_factor=end_factor
        )

        msr_params = []
        if self.M_a2b is not None:
            msr_params += list(self.M_a2b.parameters())
        if self.M_b2a is not None:
            msr_params += list(self.M_b2a.parameters())
        msr_optimizer = torch.optim.Adam(msr_params, lr=lr)

        best_loss = float("inf")

        for epoch in range(epochs):
            print(f"{epoch}", end="\r")

            # 1) train encoders/decoders to minimize data reconstruction loss
            self.freeze_all_except(
                self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
            )
            cumulative_l_rec_a = 0
            cumulative_l_rec_b = 0

            for batch_start in range(0, a_train.shape[0], batch_size):
                a_batch = a_train[batch_start : batch_start + batch_size]
                b_batch = b_train[batch_start : batch_start + batch_size]

                _, _, _, _, _, _, a_hat, b_hat = self.forward(a_batch, b_batch)
                l_rec_a = F.mse_loss(a_hat, a_batch)
                l_rec_b = F.mse_loss(b_hat, b_batch)
                cumulative_l_rec_a += l_rec_a.item()
                cumulative_l_rec_b += l_rec_b.item()

                reconstruction_loss = l_rec_a + l_rec_b

                optimizer.zero_grad()
                reconstruction_loss.backward()
                optimizer.step()

            disentangle_loss = torch.tensor([1]).to(device)
            measurement_loss = torch.tensor([0]).to(device)
            if epoch >= disent_start:
                # 2) train measurement networks to minimize measurement loss
                # cold restart periodically to avoid local minima
                if epoch % msr_restart == 0:
                    msr_params = self.restart_measurement_networks(device)
                    msr_optimizer = torch.optim.Adam(msr_params, lr=lr)  # type: ignore
                    msr_iter = msr_iter_restart
                else:
                    msr_iter = msr_iter_normal

                self.freeze_all_except(self.M_a2b, self.M_b2a)

                for _ in range(msr_iter):
                    cumulative_measurement_loss = 0
                    for batch_start in range(0, a_train.shape[0], batch_size):
                        a_batch = a_train[batch_start : batch_start + batch_size]
                        b_batch = b_train[batch_start : batch_start + batch_size]

                        _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)

                        l_msr_a = F.mse_loss(m_a2b, b_batch) if self.M_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                        l_msr_b = F.mse_loss(m_b2a, a_batch) if self.M_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore
                        # normalize by variance of target variables and # of targets to make loss scale-invariant
                        l_msr_a *= self.n_b / b_batch.var(dim=0).sum()
                        l_msr_b *= self.n_a / a_batch.var(dim=0).sum()

                        measurement_loss = l_msr_a + l_msr_b
                        cumulative_measurement_loss += measurement_loss.item()

                        msr_optimizer.zero_grad()
                        measurement_loss.backward()
                        msr_optimizer.step()

                # 3) train private encoders to minimize disentanglement loss
                self.freeze_all_except(self.F_a, self.F_b)

                cumulative_disentangle_loss = 0
                for batch_start in range(0, a_train.shape[0], batch_size):
                    a_batch = a_train[batch_start : batch_start + batch_size]
                    b_batch = b_train[batch_start : batch_start + batch_size]

                    _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)

                    l_disent_a = m_a2b.var(dim=0).sum() / b_batch.var(dim=0).sum() if self.M_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                    l_disent_b = m_b2a.var(dim=0).sum() / a_batch.var(dim=0).sum() if self.M_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore

                    disentangle_loss = c_disent * (l_disent_a + l_disent_b)
                    cumulative_disentangle_loss += disentangle_loss.item()

                    optimizer.zero_grad()
                    disentangle_loss.backward()
                    optimizer.step()

                # 4) save best model + print progress
                if epoch % print_every == 0:
                    loss = (
                        cumulative_l_rec_a / n_batches
                        + cumulative_l_rec_b / n_batches
                        + c_disent * cumulative_disentangle_loss / n_batches
                        - cumulative_measurement_loss / n_batches
                    )
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(self.state_dict(), model_filepath)

                    _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = self(a_test, b_test)

                    test_reconstruction_loss_a = F.mse_loss(a_hat, a_test)
                    test_reconstruction_loss_b = F.mse_loss(b_hat, b_test)
                    test_disentangle_loss = (
                        m_a2b.var(dim=0).sum() + m_b2a.var(dim=0).sum()
                    )
                    test_measurement_loss = (
                        self.n_b * F.mse_loss(m_a2b, b_test) / b_test.var(dim=0).sum()
                        + self.n_a * F.mse_loss(m_b2a, a_test) / a_test.var(dim=0).sum()
                    )

                    print(
                        "Epoch: %d \t A reconstruction: %.4f | %.4f \t B reconstruction: %.4f | %.4f \t Disentangling: %.4f | %.4f \t Measurement: %.4f | %.4f"
                        % (
                            epoch,
                            cumulative_l_rec_a / n_batches,
                            test_reconstruction_loss_a.item(),
                            cumulative_l_rec_b / n_batches,
                            test_reconstruction_loss_b.item(),
                            c_disent * cumulative_disentangle_loss / n_batches,
                            test_disentangle_loss.item(),
                            cumulative_measurement_loss / n_batches,
                            test_measurement_loss.item(),
                        )
                    )

            scheduler.step()

        self.load_state_dict(torch.load(model_filepath))

    def fit_isomap_splice(
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
        c_disent=0.1,
        device=torch.device("cuda"),
        weight_decay=0,
        n_neighbors=100,
        n_landmarks=100,
        c_prox=50,
    ):
        pass

    def freeze_all_except(self, *args):
        for param in self.parameters():
            param.requires_grad = False
        for net in args:
            for param in net.parameters():
                param.requires_grad = True

    def restart_measurement_networks(self, device):
        if self.n_priv_a > 0:
            self.M_a2b = decoder(self.n_priv_a, self.n_b, self.layers_msr, self.nl)
        if self.n_priv_b > 0:
            self.M_b2a = decoder(self.n_priv_b, self.n_a, self.layers_msr, self.nl)

        msr_params = []
        if self.M_a2b is not None:
            msr_params += list(self.M_a2b.parameters())
        if self.M_b2a is not None:
            msr_params += list(self.M_b2a.parameters())

        return msr_params
