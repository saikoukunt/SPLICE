import math

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from tqdm import tqdm

from splice.base import carlosPlus, decoder, encoder
from splice.utils import calculate_isomap_dists, PairedViewDataset


class SPLICECore(nn.Module):
    def __init__(
        self,
        n_a,
        n_b,
        n_shared,
        n_private_a,
        n_private_b,
        enc_layers,
        dec_layers,
        conv=False,
        act_fn=carlosPlus,
        size=None,
        device=torch.device("cuda"),
    ):
        """
        Initialize the SPLICE model.

        Args:
            n_a (int): Number of features in view A.
            n_b (int): Number of features in view B.
            n_shared (int): Dimensionality of shared latents. Must be greater than 0.
            n_private_a (int): Dimensionality of view A private latents.
            n_private_b (int): Dimensionality of view B private latents.
            enc_layers (list): Hidden layer sizes for the encoder.
            dec_layers (list): Hidden layer sizes for the decoder.
            conv (bool, optional): Whether to use convolutional layers. Defaults to False.
            act_fn (nn.Module, optional): Activation function. Defaults to carlosPlus.
            size (tuple, optional): If conv is True, the output size of the final
                convolutional layer in the encoder. Defaults to None.
            device (torch.device, optional): Device containing model.
        """
        # TODO: show conv usage
        super().__init__()
        if n_shared <= 0:
            raise ValueError("Shared dimensionality cannot be 0.")

        self.n_a = n_a
        self.n_b = n_b
        self.n_shared = n_shared
        self.n_private_a = n_private_a
        self.n_private_b = n_private_b
        self.act_fn = act_fn
        self.device = device

        self.F_a = encoder(self.n_a, self.n_private_a, enc_layers, act_fn, conv)
        self.F_a2b = encoder(self.n_a, self.n_shared, enc_layers, act_fn, conv)
        self.F_b2a = encoder(self.n_b, self.n_shared, enc_layers, act_fn, conv)
        self.F_b = encoder(self.n_b, self.n_private_b, enc_layers, act_fn, conv)

        self.G_a = decoder(
            self.n_private_a + self.n_shared,
            self.n_a,
            dec_layers,
            act_fn,
            conv,
            size=size,
        )
        self.G_b = decoder(
            self.n_private_b + self.n_shared,
            self.n_b,
            dec_layers,
            act_fn,
            conv,
            size=size,
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
            a_in = torch.hstack((z_a_fix, z_b2a))
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
            b_in = torch.hstack((z_b_fix, z_a2b))
            b_shared_subm = self.G_b(b_in)

            z_a2b_fix = torch.ones_like(z_a2b) * z_a2b[fix_index]
            b_in = torch.cat([z_b, z_a2b_fix], dim=1)
            b_private_subm = self.G_b(b_in)
        else:
            b_in = z_a2b
            b_shared_subm = self.G_b(b_in)
            b_private_subm = None

        return a_shared_subm, b_shared_subm, a_private_subm, b_private_subm


class SPLICE(SPLICECore):
    def __init__(
        self,
        n_a,
        n_b,
        n_shared,
        n_private_a,
        n_private_b,
        enc_layers,
        dec_layers,
        msr_layers,
        conv=False,
        act_fn=carlosPlus,
        size=None,
        device=torch.device("cuda"),
    ):
        super().__init__(
            n_a,
            n_b,
            n_shared,
            n_private_a,
            n_private_b,
            enc_layers,
            dec_layers,
            conv=conv,
            act_fn=act_fn,
            size=size,
            device=device,
        )
        self.conv = conv
        self.size = size
        self.msr_layers = msr_layers
        self.M_a2b = decoder(n_private_a, n_b, msr_layers, act_fn, conv=conv, size=size)
        self.M_b2a = decoder(n_private_b, n_a, msr_layers, act_fn, conv=conv, size=size)

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
        a_validation,
        b_validation,
        model_filepath,
        batch_size=None,
        epochs=25000,
        lr=1e-3,
        end_factor=1 / 100,
        disent_start=0,
        msr_restart=1000,
        msr_iter_normal=5,
        msr_iter_restart=1000,
        c_disent=1,
        device=torch.device("cuda"),
        weight_decay=0,
        msr_weight_decay=0,
        checkpoint_freq=500,
    ):
        """

        Train the SPLICE model.

        Training consists of two steps:
        1) Training measurement networks to predict one view from the opposite view's
            private latents -- a well-trained measurement network gives us a
            measurement of the predictability of one view from the other's private
            latents.
        2) Training the encoders and decoders to minimize reconstruction loss and
            enforce disentanglement. Disentangling is accomplished via an adversarial
            scheme that encourages the private encoders to minimize the accuracy of the
            measurement networks.
        The best model weights (as measured by validation reconstruction loss) are saved
        periodically.

        Args:
            a_train (torch.Tensor): Training data for view A.
            b_train (torch.Tensor): Training data for view B.
            a_validation (torch.Tensor): Validation data for view A.
            b_validation (torch.Tensor): Validation data for view B.
            model_filepath (string): Path to save the model.
            batch_size (int): Batch size for training.
            epochs (int, optional): Number of training epochs. Defaults to 25000.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            end_factor (float_, optional): Learning rate decay over all epochs. Defaults
                to 1/100.
            disent_start (int, optional): Epoch when disentangling begins. Defaults to 0.
            msr_restart (int, optional): Frequency (in number of epochs) of restarting
                measurement networks. Defaults to 1000.
            msr_iter_normal (int, optional): Number of iterations to train measurement
                networks during non-restart epochs. Defaults to 5.
            msr_iter_restart (int, optional): Number of iterations to train measurement
                networks during restart epochs. Defaults to 1000.
            c_disent (float, optional): Weight of disentangling loss. Defaults to 1.
            device (torch.Device, optional): Device containing model and input tensors.
                Defaults to torch.device("cuda").
            weight_decay (float, optional): Weight decay for encoders and decoders.
                Defaults to 0.
            msr_weight_decay (float, optional): Weight decay for measurement networks.
                Defaults to 0.
            checkpoint_freq (int, optional): Frequency of saving best model and
                printing progress. Defaults to 500.

        Raises:
            ValueError: Incorrect number of features in input datasets.

        Yields:
            self: Trained SPLICE model.
        """
        self.validate_input_data(a_train, b_train, a_validation, b_validation)
        if batch_size is None:
            batch_size = a_train.shape[0]

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
            cumul_msr_loss, cumul_l_rec_a, cumul_l_rec_b, cumul_disent_loss = (
                self.train_one_epoch(
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
                )
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
                    "Epoch %d:        A reconstruction: %.4f | %.4f \t B reconstruction: %.4f | %.4f \t Disentangling: %.4f | %.4f \t Measurement: %.4f | %.4f"
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
                    ),
                )

                validation_loss = val_reconstruction_loss_a + val_reconstruction_loss_b
                if (validation_loss < best_loss) and (epoch >= disent_start):
                    best_loss = validation_loss
                    torch.save(self.state_dict(), model_filepath)
                    tqdm.write("saving new best model")

    def fit_isomap_splice(
        self,
        a_train,
        b_train,
        a_validation,
        b_validation,
        model_filepath,
        fix_index=None,
        n_neighbors=100,
        n_landmarks=100,
        c_prox=50.0,
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
            a_train, b_train, fix_index, n_neighbors, n_landmarks, device
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
                if (validation_loss < best_loss) and (epoch >= disent_start):
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
        cumul_msr_loss = torch.Tensor([0]).to(device)
        cumul_l_rec_a = torch.Tensor([0]).to(device)
        cumul_l_rec_b = torch.Tensor([0]).to(device)
        cumul_disent_loss = torch.Tensor([0]).to(device)

        # Pass 1) train measurement networks
        if (epoch >= disent_start) and not (
            self.n_private_a == 0 and self.n_private_b == 0
        ):
            # cold restart measurement networks periodically to avoid local minima
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
                    measurement_loss.backward()
                    self.msr_optimizer.step()

                    if i == msr_iter - 1:
                        cumul_msr_loss += 1 / n_batches * norm_msr_loss
                    del a_batch, b_batch, m_a2b, m_b2a
                    torch.cuda.empty_cache()

            # Pass 2) train encoders and decoders
        torch.cuda.empty_cache()
        self.freeze_all_except(
            self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
        )

        for step, (a_batch, b_batch, idx) in enumerate(dataloader):
            _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = self(a_batch, b_batch)
            l_rec_a = F.mse_loss(a_hat, a_batch)
            l_rec_b = F.mse_loss(b_hat, b_batch)
            reconstruction_loss = l_rec_a + l_rec_b
            disentangle_loss, norm_disent_loss = self.disent_loss(
                disent_start,
                c_disent,
                epoch,
                a_batch,
                b_batch,
                m_a2b,
                m_b2a,
            )
            pass2_loss = reconstruction_loss + disentangle_loss
            optimizer.zero_grad()
            pass2_loss.backward()
            optimizer.step()

            cumul_l_rec_a += 1 / n_batches * l_rec_a
            cumul_l_rec_b += 1 / n_batches * l_rec_b
            cumul_disent_loss += 1 / n_batches * norm_disent_loss
            del _, a_batch, b_batch, m_a2b, m_b2a, a_hat, b_hat
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
                    measurement_loss.backward()
                    self.msr_optimizer.step()

                    if i == msr_iter - 1:
                        cumul_msr_loss += 1 / n_batches * norm_msr_loss
                    del a_batch, b_batch, m_a2b, m_b2a
                    torch.cuda.empty_cache()

            # Pass 2) train encoders and decoders
        torch.cuda.empty_cache()
        self.freeze_all_except(
            self.F_a, self.F_b, self.F_a2b, self.F_b2a, self.G_a, self.G_b
        )

        for step, (a_batch, b_batch, idx) in enumerate(dataloader):
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
            disentangle_loss, norm_disent_loss = self.disent_loss(
                disent_start,
                c_disent,
                epoch,
                a_batch,
                b_batch,
                m_a2b,
                m_b2a,
            )
            pass2_loss = reconstruction_loss + disentangle_loss + prox_loss
            optimizer.zero_grad()
            pass2_loss.backward()
            optimizer.step()

            cumul_l_rec_a += 1 / n_batches * mse_rec_a
            cumul_l_rec_b += 1 / n_batches * mse_rec_b
            cumul_disent_loss += 1 / n_batches * norm_disent_loss
            cumul_prox_private_a += 1 / n_batches * prox_private_a**2
            cumul_prox_private_b += 1 / n_batches * prox_private_b**2
            cumul_prox_shared_a += 1 / n_batches * prox_shared_a**2
            cumul_prox_shared_b += 1 / n_batches * prox_shared_b**2
            del a_batch, b_batch, z_a, z_b2a, z_a2b, z_b, m_a2b, m_b2a, a_hat, b_hat
            del zl_a, zl_b2a, zl_a2b, zl_b
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
            measurement_loss (torch.Tensor): Normalized measurement loss.
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

        return mse, torch.sqrt(prox)

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
