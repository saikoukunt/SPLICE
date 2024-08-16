from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import numpy as np
from copy import deepcopy

from splice.baseline import DCCA
from splice.utils import calculate_mnist_accuracy

import torch


def train_dcca(config):

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
        model = DCCA(
            n_a=784,
            n_b=784,
            z_dim=30,
            device=device,
            conv=True,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        num_epochs = 100
        batch_size = config["batch_size"]
        fail_count = 0
        success = False
        best_loss = np.inf
        best_params = None

        while not success and fail_count < 5:
            try:
                G = model.update_G(a_train, b_train, batch_size)

                for epoch in range(num_epochs):
                    print("REP %d EPOCH %d" % (rep, epoch))
                    for i in range(0, a_train.shape[0], batch_size):
                        a_batch = a_train[i : i + batch_size]
                        b_batch = b_train[i : i + batch_size]
                        g = G[i : i + batch_size]

                        z_a, z_b = model(a_batch, b_batch)
                        loss = model.loss(z_a, z_b, g)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        del a_batch, b_batch, z_a, z_b

                    G = model.update_G(a_train, b_train, batch_size)

                    z_val_a, z_val_b = model(a_validation, b_validation)

                    loss = model.loss(
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

                    z_a, z_b = model(a_batch, b_batch)
                    z_train_a.append(z_a)
                    z_train_b.append(z_b)

                z_train_a = torch.cat(z_train_a, dim=0).detach().cpu().numpy()
                z_train_b = torch.cat(z_train_b, dim=0).detach().cpu().numpy()

                z_val_a, z_val_b = model(a_validation, b_validation)

                a_acc = calculate_mnist_accuracy(
                    labels[50000:60000], z_train_a, z_val_a.detach().cpu().numpy()
                )
                b_acc = calculate_mnist_accuracy(
                    labels[50000:60000], z_train_b, z_val_b.detach().cpu().numpy()
                )

                # print("FINISHED")
                accuracy[rep] = max(a_acc, b_acc)
                print(accuracy[rep])
                success = True

            except Exception as e:
                print(e)
                fail_count += 1
                if fail_count == 5:
                    return {"accuracy": 0}

                continue

    return {"accuracy": np.mean(accuracy)}


if __name__ == "__main__":
    search_space = {
        "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
        "weight_decay": tune.choice([0, 1e-4, 1e-3, 1e-2, 1e-1]),
        "batch_size": tune.choice([100, 200, 500, 800, 1000]),
    }

    results = tune.run(
        train_dcca,
        resources_per_trial={"cpu": 1, "gpu": 1},
        num_samples=30,
        search_alg=HyperOptSearch(search_space, metric="accuracy", mode="max"),
    )

    print(
        "Best hyperparameters found were: ",
        results.get_best_trial("accuracy", "max", "last").last_result,
    )
