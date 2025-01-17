import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from splice.utils import PairedViewDataset, calculate_isomap_dists
from splice.splice import SPLICE


def setup_DDP(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def prepare_DDP(
    dataset,
    batch_size,
    rank,
    world_size,
    pin_memory=False,
    num_workers=0,
    shuffle=False,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(batch_size < dataset.shape[0]),
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )

    return dataloader


def cleanup_DDP():
    dist.destroy_process_group()


def train_ddp(
    n_a,
    n_b,
    n_shared,
    n_private_a,
    n_private_b,
    enc_layers,
    dec_layers,
    msr_layers,
    world_size,
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
    weight_decay=0,
    msr_weight_decay=0,
    checkpoint_freq=500,
):
    mp.spawn(
        _train_ddp,
        args=[
            n_a,
            n_b,
            n_shared,
            n_private_a,
            n_private_b,
            enc_layers,
            dec_layers,
            msr_layers,
            world_size,
            a_train,
            b_train,
            a_validation,
            b_validation,
            model_filepath,
            batch_size,
            epochs,
            lr,
            end_factor,
            disent_start,
            msr_restart,
            msr_iter_normal,
            msr_iter_restart,
            c_disent,
            weight_decay,
            msr_weight_decay,
            checkpoint_freq,
        ],
        nproc=world_size,
    )


def _train_ddp(
    rank,
    n_a,
    n_b,
    n_shared,
    n_private_a,
    n_private_b,
    enc_layers,
    dec_layers,
    msr_layers,
    world_size,
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
    weight_decay=0,
    msr_weight_decay=0,
    checkpoint_freq=500,
):
    setup_DDP(rank, world_size)
    model = SPLICE(
        n_a,
        n_b,
        n_shared,
        n_private_a,
        n_private_b,
        enc_layers,
        dec_layers,
        msr_layers,
        device=rank,
    )

    model.validate_input_data(a_train, b_train, a_validation, b_validation)
    if batch_size is None:
        batch_size = a_train.shape[0]
    dataloader = prepare_DDP(
        PairedViewDataset(a_train, b_train).to(rank), batch_size, rank, world_size
    )
    a_validation = a_validation.to(rank)
    b_validation = b_validation.to(rank)
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )

    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = LinearLR(
        optimizer, start_factor=1, end_factor=end_factor, total_iters=epochs
    )

    msr_params = list(model.module.M_a2b.parameters() + model.module.M_b2a.parameters())
    msr_optimizer = torch.optim.Adam(msr_params, lr=lr)

    best_loss = float("inf")
    bar_format = "{n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    for epoch in tqdm(range(epochs), bar_format=bar_format, ncols=80):
        # Pass 1) train measurement networks
        if (epoch >= disent_start) and not (
            model.n_private_a == 0 and model.n_private_b == 0
        ):
            # cold restart measurement networks periodically to avoid local minima
            if (epoch % msr_restart == 0) or (epoch == disent_start):
                msr_params = model.module.restart_measurement_networks(rank)
                msr_optimizer = torch.optim.AdamW(
                    msr_params, lr=lr, weight_decay=msr_weight_decay
                )
                msr_iter = msr_iter_restart
            else:
                msr_iter = msr_iter_normal

            model.module.freeze_all_except(model.module.M_a2b, model.module.M_b2a)

            for i in range(msr_iter):
                for step, (a_batch, b_batch, idx) in enumerate(dataloader):
                    _, _, _, _, m_a2b, m_b2a, _, _ = model(a_batch, b_batch)
                    measurement_loss, norm_msr_loss = model.module.msr_loss(
                        a_batch, b_batch, m_a2b, m_b2a
                    )
                    msr_optimizer.zero_grad()
                    measurement_loss.backward()
                    msr_optimizer.step()

                    del a_batch, b_batch, m_a2b, m_b2a
                    torch.cuda.empty_cache()

        # Pass 2) train encoders and decoders
        torch.cuda.empty_cache()
        model.module.freeze_all_except(
            model.module.F_a,
            model.module.F_b,
            model.module.F_a2b,
            model.module.F_b2a,
            model.module.G_a,
            model.module.G_b,
        )

        for step, (a_batch, b_batch, idx) in enumerate(dataloader):
            _, _, _, _, m_a2b, m_b2a, a_hat, b_hat = model(a_batch, b_batch)
            l_rec_a = F.mse_loss(a_hat, a_batch)
            l_rec_b = F.mse_loss(b_hat, b_batch)
            reconstruction_loss = l_rec_a + l_rec_b
            disentangle_loss, norm_disent_loss = model.disent_loss(
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

            del _, a_batch, b_batch, m_a2b, m_b2a, a_hat, b_hat
            torch.cuda.empty_cache()

        scheduler.step()

        if epoch % checkpoint_freq == 0:
            (
                val_reconstruction_loss_a,
                val_reconstruction_loss_b,
                val_disentangle_loss,
                val_measurement_loss,
            ) = model.module.calculate_validation_losses(
                a_validation, b_validation, batch_size, c_disent, epoch
            )

        tqdm.write(
            "Epoch %d:        A reconstruction: %.4f | %.4f \t B reconstruction: %.4f | %.4f \t Disentangling: %.4f | %.4f \t Measurement: %.4f | %.4f"
            % (
                epoch,
                l_rec_a,
                val_reconstruction_loss_a,
                l_rec_b,
                val_reconstruction_loss_b,
                norm_disent_loss,
                val_disentangle_loss,
                norm_msr_loss,
                val_measurement_loss,
            ),
        )

        validation_loss = val_reconstruction_loss_a + val_reconstruction_loss_b
        if (
            (validation_loss < best_loss)
            and (epoch >= disent_start)
            and (dist.get_rank() == 0)
        ):
            best_loss = validation_loss
            torch.save(model.module.state_dict(), model_filepath)
            tqdm.write("saving new best model")


def train_isomap_ddp(
    n_a,
    n_b,
    n_shared,
    n_private_a,
    n_private_b,
    enc_layers,
    dec_layers,
    msr_layers,
    world_size,
    a_train,
    b_train,
    a_validation,
    b_validation,
    load_filepath,
    model_filepath,
    fix_index=None,
    n_neighbors=100,
    n_landmarks=100,
    c_prox=50.0,
    batch_size=None,
    epochs=25000,
    lr=1e-3,
    end_factor=1 / 100,
    disent_start=0,
    msr_restart=1000,
    msr_iter_normal=5,
    msr_iter_restart=1000,
    c_disent=1,
    weight_decay=0,
    msr_weight_decay=0,
    checkpoint_freq=500,
):
    model = SPLICE(
        n_a,
        n_b,
        n_shared,
        n_private_a,
        n_private_b,
        enc_layers,
        dec_layers,
        msr_layers,
        device=torch.device(0),
    )

    model.load_state_dict(torch.load(load_filepath))

    model.validate_input_data(a_train, b_train, a_validation, b_validation)
    if batch_size is None:
        batch_size = a_train.shape[0]

    (
        a_landmarks,
        b_landmarks,
        a_private_dists,
        b_private_dists,
        a_shared_dists,
        b_shared_dists,
    ) = model.calc_isomap_dists(
        a_train, b_train, fix_index, n_neighbors, n_landmarks, torch.device(0)
    )

    mp.spawn(
        _train_isomap_ddp,
        args=[
            n_a,
            n_b,
            n_shared,
            n_private_a,
            n_private_b,
            enc_layers,
            dec_layers,
            msr_layers,
            world_size,
            a_train,
            b_train,
            a_validation,
            b_validation,
            load_filepath,
            model_filepath,
            a_landmarks,
            b_landmarks,
            a_private_dists,
            b_private_dists,
            a_shared_dists,
            b_shared_dists,
            c_prox,
            batch_size,
            epochs,
            lr,
            end_factor,
            disent_start,
            msr_restart,
            msr_iter_normal,
            msr_iter_restart,
            c_disent,
            weight_decay,
            msr_weight_decay,
            checkpoint_freq,
        ],
        nprocs=world_size,
    )


def _train_isomap_ddp(
    rank,
    n_a,
    n_b,
    n_shared,
    n_private_a,
    n_private_b,
    enc_layers,
    dec_layers,
    msr_layers,
    world_size,
    a_train,
    b_train,
    a_validation,
    b_validation,
    load_filepath,
    model_filepath,
    a_landmarks,
    b_landmarks,
    a_private_dists,
    b_private_dists,
    a_shared_dists,
    b_shared_dists,
    c_prox=50.0,
    batch_size=None,
    epochs=25000,
    lr=1e-3,
    end_factor=1 / 100,
    disent_start=0,
    msr_restart=1000,
    msr_iter_normal=5,
    msr_iter_restart=1000,
    c_disent=1,
    weight_decay=0,
    msr_weight_decay=0,
    checkpoint_freq=500,
):
    setup_DDP(rank, world_size)
    model = SPLICE(
        n_a,
        n_b,
        n_shared,
        n_private_a,
        n_private_b,
        enc_layers,
        dec_layers,
        msr_layers,
        device=rank,
    )
    model.load_state_dict(torch.load(load_filepath))

    model.validate_input_data(a_train, b_train, a_validation, b_validation)
    if batch_size is None:
        batch_size = a_train.shape[0]
    dataloader = prepare_DDP(
        PairedViewDataset(a_train, b_train).to(rank), batch_size, rank, world_size
    )
    a_validation = a_validation.to(rank)
    b_validation = b_validation.to(rank)
    model = DDP(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
    )

    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = LinearLR(
        optimizer, start_factor=1, end_factor=end_factor, total_iters=epochs
    )

    msr_params = list(model.module.M_a2b.parameters() + model.module.M_b2a.parameters())
    msr_optimizer = torch.optim.Adam(msr_params, lr=lr)

    best_loss = float("inf")
    bar_format = "{n_fmt}/{total_fmt} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    for epoch in tqdm(range(epochs), bar_format=bar_format, ncols=80):
        # Pass 1) train measurement networks
        if (epoch >= disent_start) and not (
            model.module.n_private_a == 0 and model.module.n_private_b == 0
        ):
            # cold restart measurement networks periodically to avoid local minima
            if (epoch % msr_restart == 0) or (epoch == disent_start):
                msr_params = model.module.restart_measurement_networks(rank)
                msr_optimizer = torch.optim.AdamW(
                    msr_params, lr=lr, weight_decay=msr_weight_decay
                )
                msr_iter = msr_iter_restart
            else:
                msr_iter = msr_iter_normal

            model.module.freeze_all_except(model.module.M_a2b, model.module.M_b2a)

            for i in range(msr_iter):
                for step, (a_batch, b_batch, idx) in enumerate(dataloader):
                    _, _, _, _, m_a2b, m_b2a, _, _ = model(a_batch, b_batch)
                    measurement_loss, norm_msr_loss = model.module.msr_loss(
                        a_batch, b_batch, m_a2b, m_b2a
                    )
                    msr_optimizer.zero_grad()
                    measurement_loss.backward()
                    msr_optimizer.step()

                    del a_batch, b_batch, m_a2b, m_b2a
                    torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        model.module.freeze_all_except(
            model.module.F_a,
            model.module.F_b,
            model.module.F_a2b,
            model.module.F_b2a,
            model.module.G_a,
            model.module.G_b,
        )

        torch.cuda.empty_cache()
        model.module.freeze_all_except(
            model.module.F_a,
            model.module.F_b,
            model.module.F_a2b,
            model.module.F_b2a,
            model.module.G_a,
            model.module.G_b,
        )

        zl_a, zl_b2a, zl_a2b, zl_b = model.module.encode(a_landmarks, b_landmarks)
        for step, (a_batch, b_batch, idx) in enumerate(dataloader):
            z_a, z_b2a, z_a2b, z_b, m_a2b, m_b2a, a_hat, b_hat = model(a_batch, b_batch)

            _, prox_private_a = model.module.iso_loss_func(
                a_batch,
                a_hat,
                a_private_dists,
                idx,
                zl_a,
                z_a,
                calc_mse=False,
            )
            _, prox_private_b = model.module.iso_loss_func(
                b_batch,
                b_hat,
                b_private_dists,
                idx,
                zl_b,
                z_b,
                calc_mse=False,
            )
            mse_rec_a, prox_shared_a = model.module.iso_loss_func(
                a_batch,
                a_hat,
                a_shared_dists,
                idx,
                zl_b2a,
                z_b2a,
            )
            mse_rec_b, prox_shared_b = model.module.iso_loss_func(
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
            disentangle_loss, norm_disent_loss = model.module.disent_loss(
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

            del a_batch, b_batch, z_a, z_b2a, z_a2b, z_b, m_a2b, m_b2a, a_hat, b_hat
            torch.cuda.empty_cache()
        del zl_a, zl_b2a, zl_a2b, zl_b
        torch.cuda.empty_cache()
        scheduler.step()

        if epoch % checkpoint_freq == 0:
            (
                val_reconstruction_loss_a,
                val_reconstruction_loss_b,
                val_disentangle_loss,
                val_measurement_loss,
            ) = model.module.calculate_validation_losses(
                a_validation, b_validation, batch_size, c_disent, epoch
            )

            tqdm.write(
                "Epoch %d:        A reconstruction: %.4f | %.4f \t B reconstruction: %.4f | %.4f \t Disentangling: %.4f | %.4f \t Measurement: %.4f | %.4f \t Private Isomap: %.4f | %.4f \t Shared Isomap: %.4f | %.4f"
                % (
                    epoch,
                    mse_rec_a,
                    val_reconstruction_loss_a,
                    mse_rec_b,
                    val_reconstruction_loss_b,
                    norm_disent_loss,
                    val_disentangle_loss,
                    norm_msr_loss,
                    val_measurement_loss,
                    prox_private_a,
                    prox_private_b,
                    prox_shared_a,
                    prox_shared_b,
                ),
            )

            validation_loss = (
                val_reconstruction_loss_a
                + val_reconstruction_loss_b
                + prox_private_a
                + prox_private_b
                + prox_shared_a
                + prox_shared_b
            )
            if (
                (validation_loss < best_loss)
                and (epoch >= disent_start)
                and (dist.get_rank() == 0)
            ):
                best_loss = validation_loss
                torch.save(model.module.state_dict(), model_filepath)
                tqdm.write("saving new best model")
