import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.optim.lr_scheduler import LinearLR

from splice.base import carlosPlus, decoder, encoder
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
        conv=False,
        nl=carlosPlus,
        size=None,
    ):
        super().__init__()

        self.n_a = n_a
        self.n_b = n_b
        self.n_shared = n_shared
        self.n_priv_a = n_priv_a
        self.n_priv_b = n_priv_b
        self.nl = nl

        self.F_a = encoder(self.n_a, self.n_priv_a, layers_enc, nl, conv)
        self.F_a2b = encoder(self.n_a, self.n_shared, layers_enc, nl, conv)
        self.F_b2a = encoder(self.n_b, self.n_shared, layers_enc, nl, conv)
        self.F_b = encoder(self.n_b, self.n_priv_b, layers_enc, nl, conv)

        self.G_a = decoder(
            self.n_priv_a + self.n_shared, self.n_a, layers_dec, nl, conv, size=size
        )
        self.G_b = decoder(
            self.n_priv_b + self.n_shared, self.n_b, layers_dec, nl, conv, size=size
        )

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
        z_a, z_b2a, z_a2b, z_b, _, _ = SPLICECore.forward(self, a, b)

        if fix_index is None:
            fix_index = np.random.randint(0, a.shape[0], fix_index)

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
            z_a_fix = torch.ones_like(z_a) * z_a[fix_index]
            a_in = torch.cat([z_a_fix, z_b2a], dim=1)
            a_shared_subm = self.G_a(a_in)

            z_b2a_fix = torch.ones_like(z_b2a) * z_b2a[fix_index]
            a_in = torch.cat([z_a, z_b2a_fix], dim=1)
            a_private_subm = self.G_a(a_in)
        else:
            a_private_subm = None

        if z_b is not None:
            z_b_fix = torch.ones_like(z_b) * z_b[fix_index]
            b_in = torch.cat([z_b_fix, z_a2b], dim=1)
            b_shared_subm = self.G_b(b_in)

            z_a2b_fix = torch.ones_like(z_a2b) * z_a2b[fix_index]
            b_in = torch.cat([z_b, z_a2b_fix], dim=1)
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
        conv=False,
        nl=carlosPlus,
        size=None,
    ):
        super().__init__(
            n_a,
            n_b,
            n_shared,
            n_priv_a,
            n_priv_b,
            layers_enc,
            layers_dec,
            conv=conv,
            nl=nl,
            size=size,
        )
        if n_shared == 0:
            raise ValueError("Shared dimensionality cannot be 0.")

        self.conv = conv
        self.size = size
        self.layers_msr = layers_msr
        self.M_a2b = decoder(n_priv_a, n_b, layers_msr, nl, conv=conv, size=size)
        self.M_b2a = decoder(n_priv_b, n_a, layers_msr, nl, conv=conv, size=size)

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
        disent_iter=1,
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
        if not self.conv:
            if a_train.shape[1] != self.n_a:
                raise ValueError(
                    "Training dataset A has the incorrect number of features."
                )
            if b_train.shape[1] != self.n_b:
                raise ValueError(
                    "Training dataset B has the incorrect number of features."
                )
            if a_test.shape[1] != self.n_a:
                raise ValueError(
                    "Testing dataset A has the incorrect number of features."
                )
            if b_test.shape[1] != self.n_b:
                raise ValueError(
                    "Testing dataset B has the incorrect number of features."
                )

        n_batches = math.ceil(a_train.shape[0] / batch_size)
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = LinearLR(
            optimizer, total_iters=epochs, start_factor=1, end_factor=end_factor
        )

        if not (self.n_priv_a == 0 and self.n_priv_b == 0):
            msr_params = list(self.M_a2b.parameters()) + list(self.M_b2a.parameters())
            msr_optimizer = torch.optim.AdamW(
                msr_params, lr=lr, weight_decay=weight_decay
            )

        a_train_var = a_train.reshape(a_train.shape[0], -1).var(dim=0).sum()
        b_train_var = b_train.reshape(a_train.shape[0], -1).var(dim=0).sum()
        best_loss = float("inf")

        for epoch in range(epochs):
            cumulative_l_rec_a = 0
            cumulative_l_rec_b = 0
            cumulative_measurement_loss = 0
            cumulative_disentangle_loss = 0

            for batch_start in range(0, a_train.shape[0], batch_size):
                # 1) train networks to minimize reconstruction loss
                self.freeze_all_except(
                    self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
                )

                for i in range(rec_iter):
                    print(
                        f"Epoch {epoch}, Rec Sample {batch_start}, Iter {i+1}", end="\r"
                    )
                    a_batch = a_train[batch_start : batch_start + batch_size]
                    b_batch = b_train[batch_start : batch_start + batch_size]

                    _, _, _, _, a_hat, b_hat = super().forward(a_batch, b_batch)
                    l_rec_a = F.mse_loss(a_hat, a_batch)
                    l_rec_b = F.mse_loss(b_hat, b_batch)

                    if i == rec_iter - 1:
                        cumulative_l_rec_a += l_rec_a.item()
                        cumulative_l_rec_b += l_rec_b.item()

                    reconstruction_loss = l_rec_a + l_rec_b

                    optimizer.zero_grad()
                    reconstruction_loss.backward()
                    optimizer.step()

                print("                                                   ", end="\r")

                if self.n_priv_a == 0 and self.n_priv_b == 0:
                    continue

                # 2) train private encoders to minimize disentanglement loss
                self.freeze_all_except(self.F_a, self.F_b)
                if epoch > disent_start:
                    for i in range(disent_iter):
                        print(f"Epoch {epoch}, Disent Sample {batch_start}", end="\r")
                        _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)

                        l_disent_a = m_a2b.var(dim=0).sum() / b_batch.var(dim=0).sum() if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                        l_disent_b = m_b2a.var(dim=0).sum() / a_batch.var(dim=0).sum() if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore

                        disentangle_loss = c_disent * (l_disent_a + l_disent_b)
                        if i == disent_iter - 1:
                            cumulative_disentangle_loss += disentangle_loss.item()

                        optimizer.zero_grad()
                        disentangle_loss.backward()
                        optimizer.step()

                        print(
                            "                                                   ",
                            end="\r",
                        )

            if (epoch >= disent_start) and not (
                self.n_priv_a == 0 and self.n_priv_b == 0
            ):
                # 3) train measurement networks to minimize measurement loss
                # cold restart periodically to avoid local minima
                if (epoch % msr_restart == 0) or (epoch == disent_start):
                    msr_params = self.restart_measurement_networks(device)
                    msr_optimizer = torch.optim.AdamW(msr_params, lr=lr, weight_decay=weight_decay)  # type: ignore
                    msr_iter = msr_iter_restart
                else:
                    msr_iter = msr_iter_normal

                self.freeze_all_except(self.M_a2b, self.M_b2a)

                for i in range(msr_iter):
                    for batch_start in range(0, a_train.shape[0], batch_size):
                        print(
                            f"Epoch {epoch}, Iter {i+1}, Msr Sample {batch_start}",
                            end="\r",
                        )
                        a_batch = a_train[batch_start : batch_start + batch_size]
                        b_batch = b_train[batch_start : batch_start + batch_size]

                        _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)

                        l_msr_a = F.mse_loss(m_a2b.reshape(a_batch.shape[0], -1), b_batch.reshape(a_batch.shape[0], -1)) if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                        l_msr_b = F.mse_loss(m_b2a.reshape(a_batch.shape[0], -1), a_batch.reshape(a_batch.shape[0], -1)) if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore
                        # normalize by variance of target variables and # of targets to make loss scale-invariant
                        l_msr_a *= self.n_b / b_train_var
                        l_msr_b *= self.n_a / a_train_var

                        measurement_loss = l_msr_a + l_msr_b
                        if i == msr_iter - 1:
                            cumulative_measurement_loss += measurement_loss.item()

                        msr_optimizer.zero_grad()
                        measurement_loss.backward()
                        msr_optimizer.step()

                        print(
                            "                                                   ",
                            end="\r",
                        )

            # 4) save best model + print progress
            if epoch % print_every == 0:
                _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = self(a_test, b_test)

                test_reconstruction_loss_a = F.mse_loss(a_hat, a_test)
                test_reconstruction_loss_b = F.mse_loss(b_hat, b_test)

                test_disent_a = m_a2b.reshape(a_test.shape[0], -1).var(dim=0).sum() / b_test.reshape(a_test.shape[0], -1).var(dim=0).sum() if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_disent_b = m_b2a.reshape(a_test.shape[0], -1).var(dim=0).sum() / a_test.reshape(a_test.shape[0], -1).var(dim=0).sum() if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_disentangle_loss = test_disent_a + test_disent_b

                test_msr_a = F.mse_loss(m_a2b.reshape(a_test.shape[0], -1), b_test.reshape(a_test.shape[0], -1)) if m_a2b is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_msr_b = F.mse_loss(m_b2a.reshape(a_test.shape[0], -1), a_test.reshape(a_test.shape[0], -1)) if m_b2a is not None else torch.Tensor([0]).to(device)  # type: ignore
                test_msr_a *= (
                    self.n_b / b_test.reshape(a_test.shape[0], -1).var(dim=0).sum()
                )
                test_msr_b *= (
                    self.n_a / a_test.reshape(a_test.shape[0], -1).var(dim=0).sum()
                )

                test_measurement_loss = (
                    test_msr_a + test_msr_b
                    if epoch >= disent_start
                    else torch.Tensor([0]).to(device)
                )
                max_measurement_loss = (
                    2 if ((self.n_priv_a > 0) and (self.n_priv_b > 0)) else 1
                )
                capped_measurement_loss = (  # so small fluctuations above max measurement loss don't cause the model to be saved
                    test_measurement_loss
                    if test_measurement_loss < max_measurement_loss
                    else torch.Tensor([max_measurement_loss]).to(device)
                )

                test_loss = (
                    test_reconstruction_loss_a
                    + test_reconstruction_loss_b
                    + c_disent * test_disentangle_loss
                    - 0.1 * capped_measurement_loss
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
        fix_index=None,
        epochs=25000,
        lr=1e-3,
        end_factor=1 / 100,
        disent_start=1000,
        msr_restart=1000,
        msr_iter_normal=3,
        msr_iter_restart=1000,
        c_disent=0.1,
        disent_iter=1,
        device=torch.device("cuda"),
        weight_decay=0,
        print_every=500,
        n_neighbors=100,
        n_landmarks=100,
        c_prox=50,
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

        print("Projecting onto submanifolds")
        a_shared_subm, b_shared_subm, a_private_subm, b_private_subm = (
            self.project_to_submanifolds(a_train, b_train)
        )

        landmark_inds = np.random.choice(a_train.shape[0], n_landmarks, replace=False)

        print("Calculating isomap distances")

        a_private_dists = (
            calculate_isomap_dists(a_private_subm, n_neighbors, landmark_inds).to(
                device
            )
            if self.n_priv_a > 0
            else None
        )

        b_private_dists = (
            calculate_isomap_dists(b_private_subm, n_neighbors, landmark_inds).to(
                device
            )
            if self.n_priv_b > 0
            else None
        )

        a_shared_dists = calculate_isomap_dists(
            a_shared_subm, n_neighbors, landmark_inds
        ).to(device)
        b_shared_dists = calculate_isomap_dists(
            b_shared_subm, n_neighbors, landmark_inds
        ).to(device)

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

            # 1) finetune shared encoders and decoders
            self.freeze_all_except(self.F_a2b, self.F_b2a, self.G_a, self.G_b)

            _, z_b2a, z_a2b, _, a_hat, b_hat = super().forward(a_train, b_train)

            mse_rec_a, prox_shared_a = iso_loss_func(
                a_train, a_hat, z_b2a, a_shared_dists, landmark_inds
            )
            mse_rec_b, prox_shared_b = iso_loss_func(
                b_train, b_hat, z_a2b, b_shared_dists, landmark_inds
            )

            shared_loss = (
                mse_rec_a + mse_rec_b + c_prox * (prox_shared_a + prox_shared_b)
            )

            optimizer.zero_grad()
            shared_loss.backward()
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
                self.freeze_all_except(self.F_a, self.F_b, self.G_a, self.G_b)

                for i in range(disent_iter):

                    z_a, _, _, z_b, m_a2b, m_b2a, a_hat, b_hat = self.forward(
                        a_train, b_train
                    )

                    mse_rec_a, prox_private_a = iso_loss_func(
                        a_train, a_hat, z_a, a_private_dists, landmark_inds
                    )
                    mse_rec_b, prox_private_b = iso_loss_func(
                        b_train, b_hat, z_b, b_private_dists, landmark_inds
                    )

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

                    disentangle_loss = l_disent_a + l_disent_b
                    private_loss = (
                        c_prox * (prox_private_a + prox_private_b)
                        + c_disent * (l_disent_a + l_disent_b)
                        + mse_rec_a
                        + mse_rec_b
                    )

                    optimizer.zero_grad()
                    private_loss.backward()
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
                    - 0.1 * test_measurement_loss
                )

                if (test_loss < best_loss) and (epoch >= disent_start):
                    best_loss = test_loss
                    torch.save(self.state_dict(), model_filepath)
                    print("saving new best model")

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
        self.M_a2b = decoder(
            self.n_priv_a,
            self.n_b,
            self.layers_msr,
            self.nl,
            conv=self.conv,
            size=self.size,
        ).to(device)
        self.M_b2a = decoder(
            self.n_priv_b,
            self.n_a,
            self.layers_msr,
            self.nl,
            conv=self.conv,
            size=self.size,
        ).to(device)

        msr_params = list(self.M_a2b.parameters()) + list(self.M_b2a.parameters())

        return msr_params

    def msr_loss(self, a_batch, b_batch, m_a2b, m_b2a):
        """
        Calculate the normalized loss for measurement networks.

        Normalized loss is obtained by dividing the mean squared error by the sum of the
        variance of the target data and multiplying by the number of features. The
        resulting loss will be between 0 and 1 for each view.

        Args:
            a_batch (torch.Tensor): Data for view A.
            b_batch (torch.Tensor): Data for view B.
            m_a2b (torch.Tensor): Measurement network prediction of view B.
            m_b2a (torch.Tensor): Measurement network prediction of view A.

        Returns:
            measurement_loss (torch.Tensor): Raw measurement loss.
            norm_msr_loss (torch.Tensor): Normalized measurement loss. This will be
                between 0 and 2 is both views have private dimensionality > 0, and
                between 0 and 1 if only one view does.
        """
        # reshape to 2D if necessary
        if len(a_batch.shape) != 2:
            if m_a2b is not None:
                m_a2b = m_a2b.reshape(-1, self.n_b)
            if m_b2a is not None:
                m_b2a = m_b2a.reshape(-1, self.n_a)
            a_batch = a_batch.reshape(-1, self.n_a)
            b_batch = b_batch.reshape(-1, self.n_b)

        l_msr_a = (
            F.mse_loss(m_a2b, b_batch)
            if m_a2b is not None
            else torch.Tensor([0]).to(self.device)
        )
        l_msr_b = (
            F.mse_loss(m_b2a, a_batch)
            if m_b2a is not None
            else torch.Tensor([0]).to(self.device)
        )
        norm_msr_a = l_msr_a * self.n_b / b_batch.var(dim=0).sum()
        norm_msr_b = l_msr_b * self.n_a / a_batch.var(dim=0).sum()
        return l_msr_a + l_msr_b, norm_msr_a + norm_msr_b

    def disent_loss(
        self, disent_start, c_disent, epoch, a_batch, b_batch, m_a2b, m_b2a
    ):
        """_summary_

        Args:
            disent_start (_type_): _description_
            c_disent (_type_): _description_
            epoch (_type_): _description_
            a_batch (_type_): _description_
            b_batch (_type_): _description_
            m_a2b (_type_): _description_
            m_b2a (_type_): _description_

        Returns:
            _type_: _description_
        """
        if epoch >= disent_start and not (
            self.n_private_a == 0 and self.n_private_b == 0
        ):
            if m_a2b is not None:
                m_a2b = m_a2b.reshape(-1, self.n_b)
            if m_b2a is not None:
                m_b2a = m_b2a.reshape(-1, self.n_a)
            a_batch = a_batch.reshape(-1, self.n_a)
            b_batch = b_batch.reshape(-1, self.n_b)

            l_disent_a = (
                ((m_a2b - b_batch.mean(dim=0)) ** 2).mean()
                if m_a2b is not None
                else torch.Tensor([0]).to(self.device)
            )
            l_disent_b = (
                ((m_b2a - a_batch.mean(dim=0)) ** 2).mean()
                if m_b2a is not None
                else torch.Tensor([0]).to(self.device)
            )
            norm_disent_a = l_disent_a / b_batch.var(dim=0).mean()
            norm_disent_b = l_disent_b / a_batch.var(dim=0).mean()
        else:
            l_disent_a = torch.Tensor([0]).to(self.device)
            l_disent_b = torch.Tensor([0]).to(self.device)
            norm_disent_a = torch.Tensor([0]).to(self.device)
            norm_disent_b = torch.Tensor([0]).to(self.device)

        disentangle_loss = c_disent * (l_disent_a + l_disent_b)
        return disentangle_loss, norm_disent_a + norm_disent_b

    def calc_isomap_dists(
        self, a_train, b_train, fix_index, n_neighbors, n_landmarks, device
    ):
        tqdm.write("Projecting onto submanifolds")
        a_shared_subm, b_shared_subm, a_private_subm, b_private_subm = (
            self.project_to_submanifolds(a_train, b_train, fix_index)
        )

        tqdm.write("Calculating isomap distances")
        landmark_inds = np.random.choice(a_train.shape[0], n_landmarks, replace=False)
        a_landmarks = a_train[landmark_inds]
        b_landmarks = b_train[landmark_inds]

        a_private_dists = (
            calculate_isomap_dists(a_private_subm, n_neighbors, landmark_inds).to(
                device
            )
            if self.n_private_a > 0
            else None
        )
        b_private_dists = (
            calculate_isomap_dists(b_private_subm, n_neighbors, landmark_inds).to(
                device
            )
            if self.n_private_b > 0
            else None
        )
        a_shared_dists = calculate_isomap_dists(
            a_shared_subm, n_neighbors, landmark_inds
        ).to(device)
        b_shared_dists = calculate_isomap_dists(
            b_shared_subm, n_neighbors, landmark_inds
        ).to(device)

        return (
            a_landmarks,
            b_landmarks,
            a_private_dists,
            b_private_dists,
            a_shared_dists,
            b_shared_dists,
        )

    def iso_loss_func(
        self,
        target,
        out,
        dists,
        batch_inds,
        z_landmarks,
        z_batch,
        calc_mse=True,
    ):
        """_summary_

        Args:
            target (_type_): _description_
            out (_type_): _description_
            z (_type_): _description_
            dists (_type_): _description_
            inds (_type_): _description_
            calc_mse (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if target is None or dists is None:
            prox = torch.Tensor([0]).to(self.device)
        else:
            dists = dists[:, batch_inds]
            prox = torch.linalg.norm(
                dists - torch.cdist(z_landmarks, z_batch), "fro"
            ) / np.sqrt(dists.shape[0] * dists.shape[1])

        if calc_mse:
            mse = torch.nn.functional.mse_loss(target, out, reduction="mean")
        else:
            mse = torch.Tensor([0]).to(self.device)

        return mse, prox

    def calculate_validation_losses(
        self, a_validation, b_validation, batch_size, c_disent, epoch
    ):
        val_reconstruction_loss_a = torch.tensor([0.0]).to(self.device)
        val_reconstruction_loss_b = torch.tensor([0.0]).to(self.device)
        val_disentangle_loss = torch.tensor([0.0]).to(self.device)
        val_measurement_loss = torch.tensor([0.0]).to(self.device)

        with torch.no_grad():
            n_batches = math.ceil(a_validation.shape[0] / batch_size)
            for batch_start in range(0, a_validation.shape[0], batch_size):
                a_batch = a_validation[batch_start : batch_start + batch_size]
                b_batch = b_validation[batch_start : batch_start + batch_size]

                _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = self(a_batch, b_batch)
                val_reconstruction_loss_a += F.mse_loss(a_hat, a_batch)
                val_reconstruction_loss_b += F.mse_loss(b_hat, b_batch)
                val_disentangle_loss += self.disent_loss(
                    0,
                    c_disent,
                    epoch,
                    a_batch,
                    b_batch,
                    m_a2b,
                    m_b2a,
                )[1]
                val_measurement_loss += self.msr_loss(a_batch, b_batch, m_a2b, m_b2a)[1]
                del _, m_a2b, m_b2a, a_hat, b_hat
                torch.cuda.empty_cache()

        val_reconstruction_loss_a = val_reconstruction_loss_a.item() / n_batches
        val_reconstruction_loss_b = val_reconstruction_loss_b.item() / n_batches
        val_disentangle_loss = val_disentangle_loss.item() / n_batches
        val_measurement_loss = val_measurement_loss.item() / n_batches
        return (
            val_reconstruction_loss_a,
            val_reconstruction_loss_b,
            val_disentangle_loss,
            val_measurement_loss,
        )

    def validate_input_data(self, a_train, b_train, a_validation, b_validation):
        if not self.conv:
            if a_train.shape[1] != self.n_a:
                raise ValueError("Incorrect number of features in training dataset A.")
            if b_train.shape[1] != self.n_b:
                raise ValueError("Incorrect number of features in training dataset B.")
            if a_validation.shape[1] != self.n_a:
                raise ValueError("Incorrect number of features in testing dataset A.")
            if b_validation.shape[1] != self.n_b:
                raise ValueError("Incorrect number of features in testing dataset B.")
        if (a_train.shape[0] != b_train.shape[0]) or (
            a_validation.shape[0] != b_validation.shape[0]
        ):
            raise ValueError("View A and B must have the same number of samples.")
