import math
import sched

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from splice.base import carlosPlus, decoder, encoder
from splice.utils import MultiViewDataset, calculate_isomap_dists


class _ARDgate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        return (weights >= 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class ARDLayer(nn.Module):
    def __init__(self, m, latent_dim):
        super().__init__()
        self.m = m
        self.latent_dim = latent_dim
        self.weights = nn.Parameter(0.5 * torch.ones(m, latent_dim))

    def forward(self, x):
        gates = []
        lats = []
        for i in range(self.m):
            gates.append(_ARDgate.apply(F.sigmoid(self.weights[i])))
            lats.append(x * gates[-1])

        return lats, gates


class SPLICE(nn.Module):
    def __init__(
        self,
        m,
        input_dims,
        latent_dim,
        enc_layers,
        dec_layers,
        act_fn=carlosPlus,
        device=torch.device("cuda"),
    ):
        # TODO: show conv usage
        super().__init__()
        if m != len(input_dims):
            raise ValueError(
                "Number of input views (m) must match the length of input_dims."
            )

        total_inputs = sum(input_dims)
        self.encoder = encoder(total_inputs, latent_dim, enc_layers, act_fn).to(device)
        self.ard = ARDLayer(m, latent_dim).to(device)
        self.decoder = nn.ModuleList()
        for i in range(m):
            self.decoder.append(
                decoder(latent_dim, input_dims[i], dec_layers, act_fn)
            ).to(device)
        self.m = m

    def forward(self, x):
        z = self.encoder(x)
        recons = []
        for i in range(self.m):
            lats, gates = self.ard(z)
            recons.append(self.decoder[i](lats[i]))

        return recons, z, lats, gates

    def project_to_submanifolds(self, x_a, x_b, fix_index=None):
        """
        Project data onto the shared and private submanifolds.

        Args:
            a (torch.Tensor): View A data.
            b (torch.Tensor): View B data.
            fix_index (torch.Tensor, optional): Index of sample that should be used to
                generate the non-varying set of latents. If None, a random index is used.
                Defaults to None.

        Returns:
            a_shared_subm (torch.Tensor): Shared submanifold for view A.
            b_shared_subm (torch.Tensor): Shared submanifold for view B.
            a_private_subm (torch.Tensor): Private submanifold for view A.
            b_private_subm (torch.Tensor): Private submanifold for view B.
        """
        z_a, z_b2a, z_a2b, z_b, _, _ = SPLICECore.forward(self, x_a, x_b)

        if fix_index is None:
            fix_index = np.random.randint(0, x_a.shape[0])

        if z_a is not None:
            z_a_fix = torch.ones_like(z_a) * z_a[fix_index]
            a_in = torch.cat([z_a_fix, z_b2a], dim=1)
            a_shared_subm = self.G_a(a_in)

            z_b2a_fix = torch.ones_like(z_b2a) * z_b2a[fix_index]
            a_in = torch.cat([z_a, z_b2a_fix], dim=1)
            a_private_subm = self.G_a(a_in)
        else:
            a_in = z_b2a
            a_shared_subm = self.G_a(a_in)
            a_private_subm = None

        if z_b is not None:
            z_b_fix = torch.ones_like(z_b) * z_b[fix_index]
            b_in = torch.cat([z_b_fix, z_a2b], dim=1)
            b_shared_subm = self.G_b(b_in)

            z_a2b_fix = torch.ones_like(z_a2b) * z_a2b[fix_index]
            b_in = torch.cat([z_b, z_a2b_fix], dim=1)
            b_private_subm = self.G_b(b_in)
        else:
            b_in = z_a2b
            b_shared_subm = self.G_b(b_in)
            b_private_subm = None

        return a_shared_subm, b_shared_subm, a_private_subm, b_private_subm

    def fit(
        self,
        x_train,
        x_validation,
        model_filepath,
        batch_size=None,
        epochs=25000,
        lr=1e-3,
        c_ard=0.1,
        end_factor=1 / 100,
        device=torch.device("cuda"),
        weight_decay=0,
        checkpoint_freq=500,
    ):
        c_ard = np.linspace(0, c_ard, epochs + 1)  # linear decay

        # self.validate_input_data(a_train, b_train, a_validation, b_validation)
        if batch_size is None:
            batch_size = x_train[0].shape[0]

        dataset = MultiViewDataset(x_train)  # Using x_train for both views

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(batch_size < x_train[0].shape[0])
        )
        n_batches = math.ceil(x_train[0].shape[0] / batch_size)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = LinearLR(
            optimizer, total_iters=epochs, start_factor=1, end_factor=end_factor
        )

        bar_format = "{n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        for epoch in tqdm(range(epochs), bar_format=bar_format, ncols=80):
            cumul_l_rec = torch.tensor([0.0]).to(device)
            cumul_ard_l1 = torch.tensor([0.0]).to(device)

            for step, (batch, idx) in enumerate(dataloader):
                recons, z, lats, gates = self(torch.cat(batch, dim=1))
                l_rec = torch.tensor([0.0]).to(device)
                ard_l1 = torch.tensor([0.0]).to(device)
                for i in range(self.m):
                    l_rec += F.mse_loss(recons[i], batch[i])
                    ard_l1 += torch.sum(gates[i])
                cumul_l_rec += 1 / n_batches * l_rec
                cumul_ard_l1 += 1 / n_batches * ard_l1

                optimizer.zero_grad()
                (l_rec + c_ard[epoch] * ard_l1).backward()
                optimizer.step()

            scheduler.step()

            # 3) save best model + print progress
            if epoch % checkpoint_freq == 0:
                tqdm.write(
                    "Epoch %d: Reconstruction loss: %.4f | ARD L1: %.4f"
                    % (epoch, cumul_l_rec.item(), cumul_ard_l1.item())
                )

    def fit_isomap_splice(
        self,
        a_train,
        b_train,
        a_validation,
        b_validation,
        model_filepath,
        fix_index=None,
        n_neighbors_shared_a=100,
        n_neighbors_private_a=100,
        n_neighbors_shared_b=100,
        n_neighbors_private_b=100,
        n_landmarks=100,
        c_prox=0.1,
        batch_size=None,
        epochs=25000,
        lr=1e-3,
        end_factor=1 / 100,
        c_disent=0.1,
        disent_start=0,
        msr_restart=1000,
        msr_iter_normal=5,
        msr_iter_restart=1000,
        device=torch.device("cuda"),
        weight_decay=0.0,
        msr_weight_decay=0.0,
        checkpoint_freq=500,
        pass2_coeff=1,
    ):
        self.validate_input_data(a_train, b_train, a_validation, b_validation)
        if batch_size is None:
            batch_size = a_train.shape[0]

        (
            a_landmarks,
            b_landmarks,
            a_private_dists,
            b_private_dists,
            a_shared_dists,
            b_shared_dists,
        ) = self.calc_isomap_dists(
            a_train,
            b_train,
            fix_index,
            n_neighbors_shared_a,
            n_neighbors_private_a,
            n_neighbors_shared_b,
            n_neighbors_private_b,
            n_landmarks,
            device,
        )

        # train model
        dataset = PairedViewDataset(a_train, b_train)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(batch_size < a_train.shape[0])
        )
        n_batches = math.ceil(a_train.shape[0] / batch_size)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = LinearLR(
            optimizer, total_iters=epochs, start_factor=1, end_factor=end_factor
        )

        best_loss = float("inf")
        bar_format = "{n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        for epoch in tqdm(range(epochs), bar_format=bar_format, ncols=80):
            (
                cumul_msr_loss,
                cumul_l_rec_a,
                cumul_l_rec_b,
                cumul_disent_loss,
                cumul_prox_shared_a,
                cumul_prox_shared_b,
                cumul_prox_private_a,
                cumul_prox_private_b,
            ) = self.train_one_isomap_epoch(
                dataloader,
                n_batches,
                c_prox,
                lr,
                c_disent,
                disent_start,
                msr_restart,
                msr_iter_normal,
                msr_iter_restart,
                device,
                msr_weight_decay,
                a_landmarks,
                b_landmarks,
                a_private_dists,
                b_private_dists,
                a_shared_dists,
                b_shared_dists,
                optimizer,
                epoch,
                pass2_coeff,
            )
            scheduler.step()

            # 3) save best model + print progress
            if epoch % checkpoint_freq == 0:
                (
                    val_reconstruction_loss_a,
                    val_reconstruction_loss_b,
                    val_disentangle_loss,
                    val_measurement_loss,
                ) = self.calculate_validation_losses(
                    a_validation, b_validation, batch_size, c_disent, epoch
                )

                tqdm.write(
                    "Epoch %d:        A reconstruction: %.4f | %.4f \t B reconstruction: %.4f | %.4f \t Disentangling: %.4f | %.4f \t Measurement: %.4f | %.4f \t Private Isomap: %.4f | %.4f \t Shared Isomap: %.4f | %.4f"
                    % (
                        epoch,
                        cumul_l_rec_a,
                        val_reconstruction_loss_a,
                        cumul_l_rec_b,
                        val_reconstruction_loss_b,
                        cumul_disent_loss,
                        val_disentangle_loss,
                        cumul_msr_loss,
                        val_measurement_loss,
                        cumul_prox_private_a,
                        cumul_prox_private_b,
                        cumul_prox_shared_a,
                        cumul_prox_shared_b,
                    ),
                )

                validation_loss = (
                    val_reconstruction_loss_a
                    + val_reconstruction_loss_b
                    + cumul_prox_private_a
                    + cumul_prox_private_b
                    + cumul_prox_shared_a
                    + cumul_prox_shared_b
                )
                if (
                    (validation_loss < best_loss)
                    and (epoch >= disent_start)
                    and (model_filepath is not None)
                ):
                    best_loss = validation_loss
                    torch.save(self.state_dict(), model_filepath)
                    tqdm.write("saving new best model")

    def train_one_epoch(
        self,
        dataloader,
        n_batches,
        lr,
        disent_start,
        msr_restart,
        msr_iter_normal,
        msr_iter_restart,
        c_disent,
        device,
        msr_weight_decay,
        optimizer,
        epoch,
    ):
        self.train()
        cumul_msr_loss = torch.Tensor([0]).to(device)
        cumul_l_rec_a = torch.Tensor([0]).to(device)
        cumul_l_rec_b = torch.Tensor([0]).to(device)
        cumul_disent_loss = torch.Tensor([0]).to(device)

        # cold restart measurement networks periodically to avoid local minima
        if (epoch >= disent_start) and not (
            self.n_private_a == 0 and self.n_private_b == 0
        ):
            if (epoch % msr_restart == 0) or (epoch == disent_start):
                msr_params = self.restart_measurement_networks(device)
                self.msr_optimizer = torch.optim.AdamW(
                    msr_params, lr=lr, weight_decay=msr_weight_decay
                )
                msr_iter = msr_iter_restart
            else:
                msr_iter = msr_iter_normal

            self.freeze_all_except(self.M_a2b, self.M_b2a)
            for i in range(msr_iter):
                for step, (a_batch, b_batch, idx) in enumerate(dataloader):
                    # Step 1) minimize measurement loss
                    _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)
                    measurement_loss, norm_msr_loss = self.msr_loss(
                        a_batch, b_batch, m_a2b, m_b2a
                    )
                    self.msr_optimizer.zero_grad()
                    norm_msr_loss.backward()
                    self.msr_optimizer.step()

                    if i == msr_iter - 1:
                        cumul_msr_loss += 1 / n_batches * norm_msr_loss
            torch.cuda.empty_cache()

        self.freeze_all_except(
            self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
        )
        for step, (a_batch, b_batch, idx) in enumerate(dataloader):
            # Step 2) minimize reconstruction loss
            _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = self(a_batch, b_batch)
            l_rec_a = F.mse_loss(a_hat, a_batch)
            l_rec_b = F.mse_loss(b_hat, b_batch)
            reconstruction_loss = l_rec_a + l_rec_b

            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()

            cumul_l_rec_a += 1 / n_batches * l_rec_a
            cumul_l_rec_b += 1 / n_batches * l_rec_b
            torch.cuda.empty_cache()

            # Step 3) minimize disentangling loss
            if epoch >= disent_start:
                _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)
                disentangle_loss, norm_disent_loss = self.disent_loss(
                    disent_start,
                    c_disent,
                    epoch,
                    a_batch,
                    b_batch,
                    m_a2b,
                    m_b2a,
                )

                optimizer.zero_grad()
                norm_disent_loss.backward()
                optimizer.step()

                cumul_disent_loss += 1 / n_batches * norm_disent_loss
            torch.cuda.empty_cache()
        return cumul_msr_loss, cumul_l_rec_a, cumul_l_rec_b, cumul_disent_loss

    def train_one_isomap_epoch(
        self,
        dataloader,
        n_batches,
        c_prox,
        lr,
        c_disent,
        disent_start,
        msr_restart,
        msr_iter_normal,
        msr_iter_restart,
        device,
        msr_weight_decay,
        a_landmarks,
        b_landmarks,
        a_private_dists,
        b_private_dists,
        a_shared_dists,
        b_shared_dists,
        optimizer,
        epoch,
        pass2_coeff,
    ):
        cumul_msr_loss = torch.Tensor([0]).to(device)
        cumul_l_rec_a = torch.Tensor([0]).to(device)
        cumul_l_rec_b = torch.Tensor([0]).to(device)
        cumul_disent_loss = torch.Tensor([0]).to(device)
        cumul_prox_shared_a = torch.Tensor([0]).to(device)
        cumul_prox_shared_b = torch.Tensor([0]).to(device)
        cumul_prox_private_a = torch.Tensor([0]).to(device)
        cumul_prox_private_b = torch.Tensor([0]).to(device)

        # Pass 1) train measurement networks
        norm_msr_loss = torch.tensor([0]).to(device)
        if (epoch >= disent_start) and not (
            self.n_private_a == 0 and self.n_private_b == 0
        ):
            # cold restart periodically to avoid local minima
            if (epoch % msr_restart == 0) or (epoch == disent_start):
                msr_params = self.restart_measurement_networks(device)
                self.msr_optimizer = torch.optim.AdamW(
                    msr_params, lr=lr, weight_decay=msr_weight_decay
                )
                msr_iter = msr_iter_restart
            else:
                msr_iter = msr_iter_normal

            self.freeze_all_except(self.M_a2b, self.M_b2a)
            for i in range(msr_iter):
                for step, (a_batch, b_batch, idx) in enumerate(dataloader):
                    _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)
                    measurement_loss, norm_msr_loss = self.msr_loss(
                        a_batch, b_batch, m_a2b, m_b2a
                    )
                    self.msr_optimizer.zero_grad()
                    norm_msr_loss.backward()
                    self.msr_optimizer.step()

                    if i == msr_iter - 1:
                        cumul_msr_loss += 1 / n_batches * norm_msr_loss
                    del a_batch, b_batch, m_a2b, m_b2a
            torch.cuda.empty_cache()

        self.freeze_all_except(
            self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
        )

        for step, (a_batch, b_batch, idx) in enumerate(dataloader):
            # Step 2) minimize reconstruction loss
            z_a, z_b2a, z_a2b, z_b, m_a2b, m_b2a, a_hat, b_hat = self(a_batch, b_batch)
            zl_a, zl_b2a, zl_a2b, zl_b = self.encode(a_landmarks, b_landmarks)

            _, prox_private_a = self.iso_loss_func(
                a_batch,
                a_hat,
                a_private_dists,
                idx,
                zl_a,
                z_a,
                calc_mse=False,
            )
            _, prox_private_b = self.iso_loss_func(
                b_batch,
                b_hat,
                b_private_dists,
                idx,
                zl_b,
                z_b,
                calc_mse=False,
            )
            mse_rec_a, prox_shared_a = self.iso_loss_func(
                a_batch,
                a_hat,
                a_shared_dists,
                idx,
                zl_b2a,
                z_b2a,
            )
            mse_rec_b, prox_shared_b = self.iso_loss_func(
                b_batch,
                b_hat,
                b_shared_dists,
                idx,
                zl_a2b,
                z_a2b,
            )
            reconstruction_loss = mse_rec_a + mse_rec_b
            prox_loss = c_prox * (
                prox_private_a + prox_private_b + prox_shared_a + prox_shared_b
            )
            pass2_loss = pass2_coeff * reconstruction_loss + pass2_coeff * prox_loss

            optimizer.zero_grad()
            pass2_loss.backward()
            optimizer.step()

            cumul_l_rec_a += 1 / n_batches * mse_rec_a
            cumul_l_rec_b += 1 / n_batches * mse_rec_b
            cumul_prox_private_a += 1 / n_batches * prox_private_a
            cumul_prox_private_b += 1 / n_batches * prox_private_b
            cumul_prox_shared_a += 1 / n_batches * prox_shared_a
            cumul_prox_shared_b += 1 / n_batches * prox_shared_b
            torch.cuda.empty_cache()

            # Step 3) minimize disentangling loss
            if epoch >= disent_start:
                _, _, _, _, m_a2b, m_b2a = self.measure(a_batch, b_batch)
                disentangle_loss, norm_disent_loss = self.disent_loss(
                    disent_start,
                    c_disent,
                    epoch,
                    a_batch,
                    b_batch,
                    m_a2b,
                    m_b2a,
                )

                optimizer.zero_grad()
                norm_disent_loss.backward()
                optimizer.step()

                cumul_disent_loss += 1 / n_batches * norm_disent_loss
                torch.cuda.empty_cache()

        return (
            cumul_msr_loss,
            cumul_l_rec_a,
            cumul_l_rec_b,
            cumul_disent_loss,
            cumul_prox_shared_a,
            cumul_prox_shared_b,
            cumul_prox_private_a,
            cumul_prox_private_b,
        )

    def freeze_all_except(self, *args):
        for param in self.parameters():
            param.requires_grad = False
        for net in args:
            for param in net.parameters():
                param.requires_grad = True

    def restart_measurement_networks(self, device):
        self.M_a2b = decoder(
            self.n_private_a,
            self.n_b,
            self.msr_layers,
            self.act_fn,
            conv=self.conv,
            size=self.size,
        ).to(device)
        self.M_b2a = decoder(
            self.n_private_b,
            self.n_a,
            self.msr_layers,
            self.act_fn,
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
                m_a2b.var(dim=0).sum()
                # (m_a2b - b_batch.mean(dim=0)).var(dim=0).sum()
                if m_a2b is not None
                else torch.Tensor([0]).to(self.device)
            )
            l_disent_b = (
                m_b2a.var(dim=0).sum()
                # (m_b2a - a_batch.mean(dim=0)).var(dim=0).sum()
                if m_b2a is not None
                else torch.Tensor([0]).to(self.device)
            )
            norm_disent_a = l_disent_a / b_batch.var(dim=0).sum()
            norm_disent_b = l_disent_b / a_batch.var(dim=0).sum()
        else:
            l_disent_a = torch.Tensor([0]).to(self.device)
            l_disent_b = torch.Tensor([0]).to(self.device)
            norm_disent_a = torch.Tensor([0]).to(self.device)
            norm_disent_b = torch.Tensor([0]).to(self.device)

        disentangle_loss = c_disent * (l_disent_a + l_disent_b)
        return disentangle_loss, norm_disent_a + norm_disent_b

    def calc_isomap_dists(
        self,
        a_train,
        b_train,
        fix_index,
        n_neighbors_shared_a,
        n_neighbors_private_a,
        n_neighbors_shared_b,
        n_neighbors_private_b,
        n_landmarks,
        device,
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
            calculate_isomap_dists(
                a_private_subm, n_neighbors_private_a, landmark_inds
            ).to(device)
            if self.n_private_a > 0
            else None
        )
        b_private_dists = (
            calculate_isomap_dists(
                b_private_subm, n_neighbors_private_b, landmark_inds
            ).to(device)
            if self.n_private_b > 0
            else None
        )
        a_shared_dists = calculate_isomap_dists(
            a_shared_subm, n_neighbors_shared_a, landmark_inds
        ).to(device)
        b_shared_dists = calculate_isomap_dists(
            b_shared_subm, n_neighbors_shared_b, landmark_inds
        ).to(device)

        if (a_private_dists is not None and a_private_dists.isnan().any()) or (
            b_private_dists is not None and b_private_dists.isnan().any()
        ):
            raise ValueError(
                "NaN values in private distances, try increasing n_neighbors_private"
            )
        elif a_shared_dists.isnan().any() or b_shared_dists.isnan().any():
            raise ValueError(
                "NaN values in shared distances, try increasing n_neighbors_shared"
            )

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
