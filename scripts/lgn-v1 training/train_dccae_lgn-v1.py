import pickle
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from splice.baseline import DCCAE
from splice.utils import calculate_mnist_accuracy, update_G


def train_dccae(config):
    import warnings

    warnings.filterwarnings("error")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(
        "/cis/home/skoukun1/projects/SPLICE/data/random_18.9k/random_18.9k_train.pkl",
        "rb",
    ) as f:
        train_data = pickle.load(f)

    a_train = torch.Tensor(train_data["responses"]["lgn"][:12096]).to(device)
    b_train = torch.Tensor(train_data["responses"]["v1"][:12096]).to(device)

    a_validation = torch.Tensor(train_data["responses"]["lgn"][12096:]).to(device)
    b_validation = torch.Tensor(train_data["responses"]["v1"][12096:]).to(device)

    n_reps = 3
    cost = np.zeros(n_reps)
    recon_loss = np.zeros(n_reps)

    for rep in range(n_reps):
        model = DCCAE(
            n_a=400,
            n_b=800,
            z_dim=2,
            device=device,
            layers=[200] * config["n_layers"],
            conv=False,
            _lambda=0.5,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        num_epochs = 10000
        batch_size = config["batch_size"]
        fail_count = 0
        success = False
        best_loss = np.inf
        best_params = None

        while not success and fail_count < 5:
            try:
                G = model.update_G(a_train, b_train, batch_size)

                for epoch in range(num_epochs):
                    if epoch % 500 == 0:
                        print("REP %d EPOCH %d" % (rep, epoch))
                    for i in range(0, a_train.shape[0], batch_size):
                        a_batch = a_train[i : i + batch_size]
                        b_batch = b_train[i : i + batch_size]
                        g = G[i : i + batch_size]

                        a_hat, b_hat, z_a, z_b = model(a_batch, b_batch)
                        loss, cca_loss, recon_loss_a, recon_loss_b = model.loss(
                            a_batch, b_batch, a_hat, b_hat, z_a, z_b, g
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        del a_batch, b_batch, a_hat, b_hat, z_a, z_b

                    G = model.update_G(a_train, b_train, batch_size)

                    x_a_hat, x_b_hat, z_val_a, z_val_b = model(
                        a_validation, b_validation
                    )

                    loss, cca_loss, recon_loss_a, recon_loss_b = model.loss(
                        a_validation,
                        b_validation,
                        x_a_hat,
                        x_b_hat,
                        z_val_a,
                        z_val_b,
                        (z_val_a + z_val_b) / 2,
                    )
                    if loss < best_loss:
                        best_loss = loss
                        best_params = model.state_dict()

                model.load_state_dict(best_params)

                z_train_a = []
                z_train_b = []

                for i in range(0, a_train.shape[0], batch_size):
                    a_batch = a_train[i : i + batch_size]
                    b_batch = b_train[i : i + batch_size]

                    x_a_hat, x_b_hat, z_a, z_b = model(a_batch, b_batch)
                    z_train_a.append(z_a)
                    z_train_b.append(z_b)

                z_train_a = torch.cat(z_train_a, dim=0).detach().cpu().numpy()
                z_train_b = torch.cat(z_train_b, dim=0).detach().cpu().numpy()

                x_a_hat, x_b_hat, z_val_a, z_val_b = model(a_validation, b_validation)

                cost[rep] = best_loss
                recon_loss[rep] = (
                    F.mse_loss(x_a_hat, a_validation).item()
                    + F.mse_loss(x_b_hat, b_validation).item()
                )

                success = True

                train.report(
                    {
                        "cost": cost[rep],
                        "recon_loss": recon_loss[rep],
                    }
                )

            except Exception as e:
                print(e)
                fail_count += 1
                if fail_count == 5:
                    return {"cost": np.inf}

                continue

    return {
        "cost": np.mean(cost),
        "recon_loss": np.mean(recon_loss),
    }


if __name__ == "__main__":
    search_space = {
        "n_layers": tune.choice([1, 2, 3, 4, 5, 6]),
        "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.choice([0, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([2000, 5000, 12096]),
    }

    results = tune.run(
        train_dccae,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=100,
        search_alg=HyperOptSearch(search_space, metric="cost", mode="min"),
        scheduler=ASHAScheduler(
            metric="cost",
            mode="min",
        ),
    )

    print(
        "Best hyperparameters found were: ",
        results.get_best_trial("cost", "min", "last").last_result,
    )
