from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import numpy as np
from copy import deepcopy
from torch.nn import functional as F

from splice.splice import ConvSplice
from splice.baseline import DCCA
from splice.utils import calculate_mnist_accuracy

import torch


def train_splice(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(
        "/cis/home/skoukun1/projects/SPLICE/data/mnist/mnist_rotated_360.npz"
    )

    a_train = torch.Tensor(data["original"][:50000]).to(device).reshape(-1, 1, 28, 28)
    b_train = torch.Tensor(data["rotated"][:50000]).to(device).reshape(-1, 1, 28, 28)

    a_validation = (
        torch.Tensor(data["original"][50000:60000]).to(device).reshape(-1, 1, 28, 28)
    )
    b_validation = (
        torch.Tensor(data["rotated"][50000:60000]).to(device).reshape(-1, 1, 28, 28)
    )

    # a_test = torch.Tensor(data["view1"][60000:]).to(device).reshape(-1, 1, 28, 28)
    # b_test = torch.Tensor(data["view2"][60000:]).to(device).reshape(-1, 1, 28, 28)

    labels = data["labels"]

    accuracy = np.zeros(5)

    for rep in range(5):
        model = ConvSplice(30, 2, device).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        msr_optimizer = torch.optim.AdamW(
            model.M_b2a.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        best_params = None
        best_loss = float("inf")

        num_epochs = 100
        batch_size = config["batch_size"]
        best_loss = np.inf
        best_params = None
        msr_restart = 3
        msr_iter_normal = 5
        msr_iter_restart = 25
        c_disent = 0.1

        for epoch in range(num_epochs):
            print(f"Rep {rep} Epoch {epoch}", end="\r")

            model.freeze_all_except(
                model.F_b, model.F_a2b, model.F_b2a, model.G_a, model.G_b
            )

            # 1) train encoders/decoders to minimize data reconstruction loss
            for i in range(0, len(a_train), batch_size):
                a_batch = a_train[i : i + batch_size]
                b_batch = b_train[i : i + batch_size]

                optimizer.zero_grad()

                _, _, _, _, a_hat, b_hat = model(a_batch, b_batch)
                l_rec_a = F.mse_loss(a_hat, a_batch)
                l_rec_b = F.mse_loss(b_hat, b_batch)
                rec_loss = l_rec_a + l_rec_b

                rec_loss.backward()
                optimizer.step()

                # if epoch > 3:
                #     # 2) train measurement networks to minimize measurement loss
                #     # cold restart periodically to avoid local minima

                #     if epoch % msr_restart == 0:
                #         msr_params = model.restart_measurement_networks(device)
                #         msr_optimizer = torch.optim.AdamW(msr_params, lr=config["lr"])
                #         msr_iter = msr_iter_restart
                #     else:
                #         msr_iter = msr_iter_normal

                #     model.freeze_all_except(model.M_b2a)

                #     for i in range(msr_iter):
                #         for j in range(0, len(a_train), batch_size):
                #             a_batch = a_train[j : j + batch_size]
                #             b_batch = b_train[j : j + batch_size]

                #             _, m_b2a = model.measure(b_batch)
                #             l_msr = (
                #                 784
                #                 * F.mse_loss(a_batch, m_b2a)
                #                 / a_batch.reshape(-1, 784).var(dim=0).sum()
                #             )

                #             msr_optimizer.zero_grad()
                #             l_msr.backward()
                #             msr_optimizer.step()

                #     # 3) train private encoders to minimize disentanglement loss
                #     model.freeze_all_except(model.F_b)

                #     for i in range(0, len(a_train), batch_size):
                #         a_batch = a_train[i : i + batch_size]
                #         b_batch = b_train[i : i + batch_size]

                #         _, m_b2a = model.measure(b_batch)

                #         l_disent = m_b2a.reshape(-1, 784).var(dim=0).sum()
                #         l_disent *= c_disent / a_batch.reshape(-1, 784).var(dim=0).sum()

                #         optimizer.zero_grad()
                #         l_disent.backward()
                #         optimizer.step()

                z_b2a, z_a2b, z_b, m_b2a, a_hat, b_hat = model(
                    a_validation, b_validation
                )
                l_rec_a = F.mse_loss(a_hat, a_validation)
                l_rec_b = F.mse_loss(b_hat, b_validation)
                # disent_loss = (
                #     c_disent
                #     * m_b2a.reshape(-1, 784).var(dim=0).sum()
                #     / a_validation.reshape(-1, 784).var(dim=0).sum()
                # )
                val_loss = l_rec_a + l_rec_b  # + disent_loss

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = deepcopy(model.state_dict())

        model.load_state_dict(best_params)

        z_train_a = []
        z_train_b = []

        for i in range(0, a_train.shape[0], batch_size):
            a_batch = a_train[i : i + batch_size]
            b_batch = b_train[i : i + batch_size]

            z_b2a, z_a2b, z_b, m_b2a, a_hat, b_hat = model(a_batch, b_batch)
            z_train_a.append(z_a2b)
            z_train_b.append(z_b2a)

        z_train_a = torch.cat(z_train_a, dim=0).detach().cpu().numpy()
        z_train_b = torch.cat(z_train_b, dim=0).detach().cpu().numpy()

        z_val_b2a, z_val_2b, z_b, m_b2a, a_hat, b_hat = model(
            a_validation, b_validation
        )

        a_acc = calculate_mnist_accuracy(
            labels[50000:60000], z_train_a, z_val_b2a.detach().cpu().numpy()
        )
        b_acc = calculate_mnist_accuracy(
            labels[50000:60000], z_train_b, z_val_2b.detach().cpu().numpy()
        )

        accuracy[rep] = max(a_acc, b_acc)
        print(accuracy[rep])

    return {"accuracy": np.mean(accuracy)}


if __name__ == "__main__":
    search_space = {
        "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.choice([0, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([20, 50, 100, 200, 500, 800, 1000]),
    }

    results = tune.run(
        train_splice,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=10,
        search_alg=HyperOptSearch(search_space, metric="accuracy", mode="max"),
    )

    print(
        "Best hyperparameters found were: ",
        results.get_best_trial("accuracy", "max", "last").last_result,
    )
