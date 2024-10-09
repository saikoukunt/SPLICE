import copy
import math
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import lil_array
from scipy.sparse.csgraph import dijkstra
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import LinearLR

from splice.base import carlosPlus, conv_decoder, conv_encoder, decoder, encoder
from splice.utils import calculate_isomap_dists, iso_loss_func

# TODO: add convolutional option


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

        self.F_a = encoder(self.n_a, self.n_priv_a, layers_enc, nl)
        self.F_a2b = encoder(self.n_a, self.n_shared, layers_enc, nl)
        self.F_b2a = encoder(self.n_b, self.n_shared, layers_enc, nl)
        self.F_b = encoder(self.n_b, self.n_priv_b, layers_enc, nl)

        self.G_a = decoder(self.n_priv_a + self.n_shared, self.n_a, layers_dec, nl)
        self.G_b = decoder(self.n_priv_b + self.n_shared, self.n_b, layers_dec, nl)

    def forward(self, x_a, x_b):
        z_a, z_b2a, z_a2b, z_b = self.encode(x_a, x_b)
        a_hat, b_hat = self.decode(z_a, z_b2a, z_a2b, z_b)

        return z_a, z_b2a, z_a2b, z_b, a_hat, b_hat

    def encode(self, x_a, x_b):
        z_a = self.F_a(x_a)
        z_a2b = self.F_a2b(x_a)
        z_b2a = self.F_b2a(x_b)
        z_b = self.F_b(x_b)

        return z_a, z_b2a, z_a2b, z_b

    def decode(self, z_a, z_b2a, z_a2b, z_b):
        a_latents = torch.cat([z_a, z_b2a], dim=1) if z_a is not None else z_b2a
        b_latents = torch.cat([z_b, z_a2b], dim=1) if z_b is not None else z_a2b

        a_hat = self.G_a(a_latents)
        b_hat = self.G_b(b_latents)

        return a_hat, b_hat

    def project_to_submanifolds(self, a, b, fix_index=None):
        z_a, z_b2a, z_a2b, z_b, a_hat, b_hat = SPLICECore.forward(self, a, b)

        if fix_index is None:
            fix_index = np.random.randint(0, a.shape[0])

        a_in = (
            torch.hstack((torch.ones_like(z_a) * z_a[fix_index], z_b2a))
            if z_a is not None
            else z_b2a
        )
        b_in = (
            torch.hstack((torch.ones_like(z_b) * z_b[fix_index], z_a2b))
            if z_b is not None
            else z_a2b
        )
        a_shared_subm = self.G_a(a_in)
        b_shared_subm = self.G_b(b_in)

        if z_a is not None:
            a_in = torch.cat([z_a, torch.ones_like(z_b2a) * z_b2a[fix_index]], dim=1)
            a_private_subm = self.G_a(a_in)
        else:
            a_private_subm = None

        if z_b is not None:
            b_in = torch.cat([z_b, torch.ones_like(z_a2b) * z_a2b[fix_index]], dim=1)
            b_private_subm = self.G_b(b_in)
        else:
            b_private_subm = None

        return a_shared_subm, b_shared_subm, a_private_subm, b_private_subm


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
        if n_shared == 0:
            raise ValueError("Shared dimensionality cannot be 0.")

        self.layers_msr = layers_msr
        self.M_a2b = decoder(n_priv_a, n_b, layers_msr, nl)
        self.M_b2a = decoder(n_priv_b, n_a, layers_msr, nl)

    def forward(self, x_a, x_b):
        z_a, z_b2a, z_a2b, z_b, a_hat, b_hat = super().forward(x_a, x_b)
        m_a2b = self.M_a2b(z_a)
        m_b2a = self.M_b2a(z_b)

        return z_a, z_b2a, z_a2b, z_b, m_a2b, m_b2a, a_hat, b_hat

    def measure(self, x_a, x_b):
        z_a, z_b2a, z_a2b, z_b = self.encode(x_a, x_b)
        m_a2b = self.M_a2b(z_a)
        m_b2a = self.M_b2a(z_b)

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
        rec_iter=1,
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
        # n_msr_batches = math.ceil(a_train.shape[0] / msr_batch_size)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = LinearLR(
            optimizer, total_iters=epochs, start_factor=1, end_factor=end_factor
        )

        msr_params = list(self.M_a2b.parameters()) + list(self.M_b2a.parameters())
        msr_optimizer = torch.optim.AdamW(msr_params, lr=lr)

        best_loss = float("inf")

        for epoch in range(epochs):
            shuffle_inds = np.random.permutation(a_train.shape[0])

            cumulative_l_rec_a = 0
            cumulative_l_rec_b = 0
            cumulative_measurement_loss = 0
            cumulative_disentangle_loss = 0

            for batch_start in range(0, a_train.shape[0], batch_size):
                print(f"Epoch {epoch}, Batch {batch_start}", end="\r")
                a_batch = a_train[shuffle_inds][batch_start : batch_start + batch_size]
                b_batch = b_train[shuffle_inds][batch_start : batch_start + batch_size]

                # 1) train encoders/decoders to minimize data reconstruction loss
                self.freeze_all_except(
                    self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
                )

                for __ in range(rec_iter):
                    _, _, _, _, a_hat, b_hat = super().forward(a_batch, b_batch)
                    l_rec_a = F.mse_loss(a_hat, a_batch)
                    l_rec_b = F.mse_loss(b_hat, b_batch)

                    if __ == rec_iter - 1:
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
                        msr_optimizer = torch.optim.AdamW(msr_params, lr=lr)  # type: ignore
                        msr_iter = msr_iter_restart
                    else:
                        msr_iter = msr_iter_normal

                    self.freeze_all_except(self.M_a2b, self.M_b2a)

                    for __ in range(msr_iter):

                        _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)

                        l_msr_a = F.mse_loss(m_a2b, b_batch) if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                        l_msr_b = F.mse_loss(m_b2a, a_batch) if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore
                        # normalize by variance of target variables and # of targets to make loss scale-invariant
                        l_msr_a *= self.n_b / b_batch.var(dim=0).sum()
                        l_msr_b *= self.n_a / a_batch.var(dim=0).sum()

                        measurement_loss = l_msr_a + l_msr_b
                        if __ == msr_iter - 1:
                            cumulative_measurement_loss += measurement_loss.item()

                        msr_optimizer.zero_grad()
                        measurement_loss.backward()
                        msr_optimizer.step()

                    print(
                        "                                                   ", end="\r"
                    )

                    # 3) train private encoders to minimize disentanglement loss
                    self.freeze_all_except(self.F_a, self.F_b)

                    _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)

                    l_disent_a = m_a2b.var(dim=0).sum() / b_batch.var(dim=0).sum() if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                    l_disent_b = m_b2a.var(dim=0).sum() / a_batch.var(dim=0).sum() if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore

                    disentangle_loss = c_disent * (l_disent_a + l_disent_b)
                    cumulative_disentangle_loss += disentangle_loss.item()

                    optimizer.zero_grad()
                    disentangle_loss.backward()
                    optimizer.step()

                    print(
                        "                                                   ", end="\r"
                    )

            # 4) save best model + print progress
            if epoch % print_every == 0:
                _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = self(a_test, b_test)

                test_reconstruction_loss_a = F.mse_loss(a_hat, a_test)
                test_reconstruction_loss_b = F.mse_loss(b_hat, b_test)

                test_disent_a = m_a2b.var(dim=0).sum() / b_test.var(dim=0).sum() if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_disent_b = m_b2a.var(dim=0).sum() / a_test.var(dim=0).sum() if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_disentangle_loss = test_disent_a + test_disent_b

                test_msr_a = F.mse_loss(m_a2b, b_test) if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_msr_b = F.mse_loss(m_b2a, a_test) if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_msr_a *= self.n_b / b_test.var(dim=0).sum()
                test_msr_b *= self.n_a / a_test.var(dim=0).sum()

                test_measurement_loss = (
                    test_msr_a + test_msr_b
                    if epoch >= disent_start
                    else torch.Tensor([0]).to(device)
                )

                # logic so small fluctuations in measurement loss don't prevent the model from saving
                max_measurement_loss = (
                    2 if (self.n_priv_a > 0) and (self.n_priv_b > 0) else 1
                )
                capped_measurement_loss = (
                    max_measurement_loss
                    if test_measurement_loss > max_measurement_loss
                    else test_measurement_loss
                )

                test_loss = (
                    test_reconstruction_loss_a
                    + test_reconstruction_loss_b
                    + c_disent * test_disentangle_loss
                    - capped_measurement_loss
                )

                if (test_loss < best_loss) and (epoch >= disent_start):
                    best_loss = test_loss
                    torch.save(self.state_dict(), model_filepath)
                    print("saving new best model")

                print(
                    "Epoch %d:        A reconstruction: %.4f | %.4f \t B reconstruction: %.4f | %.4f \t Disentangling: %.4f | %.4f \t Measurement: %.4f | %.4f"
                    % (
                        epoch,
                        cumulative_l_rec_a / n_batches,
                        test_reconstruction_loss_a.item(),
                        cumulative_l_rec_b / n_batches,
                        test_reconstruction_loss_b.item(),
                        cumulative_disentangle_loss / n_batches / c_disent,
                        test_disentangle_loss.item(),
                        cumulative_measurement_loss / n_batches,
                        test_measurement_loss.item(),
                    )
                )

            scheduler.step()

        # self.load_state_dict(torch.load(model_filepath))

    def fit_isomap_splice(
        self,
        a_train,
        b_train,
        a_test,
        b_test,
        model_filepath,
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
        n_neighbors=100,
        n_landmarks=100,
        c_prox=50,
    ):
        a_shared_subm, b_shared_subm, a_private_subm, b_private_subm = (
            self.project_to_submanifolds(a_train, b_train)
        )

        landmark_inds = np.random.choice(a_train.shape[0], n_landmarks, replace=False)

        a_private_dists = calculate_isomap_dists(
            a_private_subm, n_neighbors, landmark_inds
        ).to(device)
        b_private_dists = calculate_isomap_dists(
            b_private_subm, n_neighbors, landmark_inds
        ).to(device)
        a_shared_dists = calculate_isomap_dists(
            a_shared_subm, n_neighbors, landmark_inds
        ).to(device)
        b_shared_dists = calculate_isomap_dists(
            b_shared_subm, n_neighbors, landmark_inds
        ).to(device)

        # input validation
        if a_train.shape[1] != self.n_a:
            raise ValueError("Training dataset A has the incorrect number of features.")
        if b_train.shape[1] != self.n_b:
            raise ValueError("Training dataset B has the incorrect number of features.")
        if a_test.shape[1] != self.n_a:
            raise ValueError("Testing dataset A has the incorrect number of features.")
        if b_test.shape[1] != self.n_b:
            raise ValueError("Testing dataset B has the incorrect number of features.")

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = LinearLR(
            optimizer, total_iters=epochs, start_factor=1, end_factor=end_factor
        )

        msr_params = list(self.M_a2b.parameters()) + list(self.M_b2a.parameters())
        msr_optimizer = torch.optim.Adam(msr_params, lr=lr, weight_decay=weight_decay)

        best_loss = float("inf")

        for epoch in range(epochs):
            print(f"{epoch}", end="\r")

            # 1) train encoders/decoders to minimize data reconstruction loss and isomap loss
            self.freeze_all_except(
                self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
            )

            z_a, z_b2a, z_a2b, z_b, a_hat, b_hat = super().forward(a_train, b_train)

            mse_rec_a, prox_shared_a = iso_loss_func(
                a_train, a_hat, z_b2a, a_shared_dists, landmark_inds
            )
            mse_rec_b, prox_shared_b = iso_loss_func(
                b_train, b_hat, z_a2b, b_shared_dists, landmark_inds
            )
            mse_rec_a, prox_private_a = iso_loss_func(
                a_train, a_hat, z_a, a_private_dists, landmark_inds
            )
            mse_rec_b, prox_private_b = iso_loss_func(
                b_train, b_hat, z_b, b_private_dists, landmark_inds
            )

            rec_loss = (
                mse_rec_a
                + mse_rec_b
                + c_prox
                * (prox_shared_a + prox_shared_b + prox_private_a + prox_private_b)
            )

            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()

            disentangle_loss = torch.tensor([1]).to(device)
            measurement_loss = torch.tensor([0]).to(device)
            if epoch >= disent_start:

                # 2) train measurement networks to minimize measurement loss
                # cold restart periodically to avoid local minima
                if epoch % msr_restart == 0:
                    msr_params = self.restart_measurement_networks(device)
                    msr_optimizer = torch.optim.AdamW(
                        msr_params, lr=lr, weight_decay=weight_decay
                    )
                    msr_iter = msr_iter_restart
                else:
                    msr_iter = msr_iter_normal

                self.freeze_all_except(self.M_a2b, self.M_b2a)

                for i in range(msr_iter):
                    measurement_loss = torch.tensor([0]).to(device)
                    _, _, _, _, m_a2b, m_b2a = self.measure(a_train, b_train)

                    l_msr_a = (
                        F.mse_loss(m_a2b, b_train)
                        if m_a2b is not None
                        else torch.Tensor([0]).to(device)
                    )
                    l_msr_b = (
                        F.mse_loss(m_b2a, a_train)
                        if m_b2a is not None
                        else torch.Tensor([0]).to(device)
                    )
                    # normalize by variance of target variables and # of targets to make loss scale-invariant
                    l_msr_a *= self.n_b / b_train.var(dim=0).sum()
                    l_msr_b *= self.n_a / a_train.var(dim=0).sum()

                    measurement_loss = l_msr_a + l_msr_b

                    msr_optimizer.zero_grad()
                    measurement_loss.backward()
                    msr_optimizer.step()

                # 3) train private encoders to minimize disentanglement loss
                self.freeze_all_except(self.F_a, self.F_b)

                _, _, _, _, m_a2b, m_b2a = self.measure(a_train, b_train)

                l_disent_a = (
                    m_a2b.var(dim=0).sum() / b_train.var(dim=0).sum()
                    if m_a2b is not None
                    else torch.Tensor([0]).to(device)
                )
                l_disent_b = (
                    m_b2a.var(dim=0).sum() / a_train.var(dim=0).sum()
                    if m_b2a is not None
                    else torch.Tensor([0]).to(device)
                )

                disentangle_loss = c_disent * (l_disent_a + l_disent_b)

                optimizer.zero_grad()
                disentangle_loss.backward()
                optimizer.step()

            # 4) save best model + print progress
            if epoch % print_every == 0:
                _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = self.forward(a_test, b_test)

                test_reconstruction_loss_a = F.mse_loss(a_hat, a_test)
                test_reconstruction_loss_b = F.mse_loss(b_hat, b_test)

                test_disent_a = (
                    m_a2b.var(dim=0).sum() / b_test.var(dim=0).sum()
                    if m_a2b is not None
                    else torch.Tensor([0]).to(device)
                )
                test_disent_b = (
                    m_b2a.var(dim=0).sum() / a_test.var(dim=0).sum()
                    if m_b2a is not None
                    else torch.Tensor([0]).to(device)
                )
                test_disentangle_loss = test_disent_a + test_disent_b

                test_msr_a = (
                    F.mse_loss(m_a2b, b_test)
                    if m_a2b is not None
                    else torch.Tensor([0]).to(device)
                )
                test_msr_b = (
                    F.mse_loss(m_b2a, a_test)
                    if m_b2a is not None
                    else torch.Tensor([0]).to(device)
                )
                test_msr_a *= self.n_b / b_test.var(dim=0).sum()
                test_msr_b *= self.n_a / a_test.var(dim=0).sum()

                test_measurement_loss = (
                    test_msr_a + test_msr_b
                    if epoch >= disent_start
                    else torch.Tensor([0]).to(device)
                )

                test_loss = (
                    test_reconstruction_loss_a
                    + test_reconstruction_loss_b
                    + c_disent * test_disentangle_loss
                    + c_prox
                    * (prox_private_a + prox_private_b + prox_shared_a + prox_shared_b)
                    - test_measurement_loss
                )

                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(self.state_dict(), model_filepath)

                print(
                    "Epoch: %d \t A reconstruction: %.4f | %.4f \t B reconstruction: %.4f | %.4f \t Disentangling: %.4f | %.4f \t Measurement: %.4f | %.4f \t Isomap A: %.4f | %.4f \t Isomap B: %.4f | %.4f"
                    % (
                        epoch,
                        mse_rec_a.item(),
                        test_reconstruction_loss_a.item(),
                        mse_rec_b.item(),
                        test_reconstruction_loss_b.item(),
                        disentangle_loss.item(),
                        test_disentangle_loss.item(),
                        measurement_loss.item(),
                        test_measurement_loss.item(),
                        prox_shared_a.item(),
                        prox_private_a.item(),
                        prox_shared_b.item(),
                        prox_private_b.item(),
                    )
                )

            scheduler.step()

        # self.load_state_dict(torch.load(model_filepath))

    def freeze_all_except(self, *args):
        for param in self.parameters():
            param.requires_grad = False
        for net in args:
            for param in net.parameters():
                param.requires_grad = True

    def restart_measurement_networks(self, device):
        self.M_a2b = decoder(self.n_priv_a, self.n_b, self.layers_msr, self.nl).to(
            device
        )
        self.M_b2a = decoder(self.n_priv_b, self.n_a, self.layers_msr, self.nl).to(
            device
        )

        msr_params = list(self.M_a2b.parameters()) + list(self.M_b2a.parameters())

        return msr_params
