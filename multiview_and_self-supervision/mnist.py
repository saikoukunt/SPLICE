import sys
import numpy as np
import torch.utils

sys.path.append(".")
sys.path.append("..")

import torch
import model
import train
import utils

# import evaluate
from sklearn.datasets import fetch_openml
from scipy.ndimage import rotate


def main(args):
    torch.manual_seed(0)
    np.random.seed(12)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_dim = 30
    c_dim = [0, 2]
    phi_hidden_size = 64
    tau_hidden_size = 64
    phi_num_layers = 2
    tau_num_layers = 2
    lr_max = 1
    lr_min = 1e-3
    mmcca_decay = 1e-1
    weight_decay = 1e-4
    batch_size1 = 100
    batch_size2 = 1000
    num_iters = 50
    inner_epochs = 3
    beta = 1
    _lambda = 100

    # Encoder and decoder network
    ae_model = model.CNNDAE(z_dim, c_dim, channels=1).to(device)
    # View1 independence regularization network
    # mmcca1 = model.MMDCCA(args.z_dim, args.c_dim,
    #         [args.phi_hidden_size]*args.phi_num_layers,
    #         [args.tau_hidden_size]*args.tau_num_layers).to(device)
    # View2 independence regularization network
    mmcca2 = model.MMDCCA(
        z_dim,
        c_dim[1],
        [phi_hidden_size] * phi_num_layers,
        [tau_hidden_size] * tau_num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {
                "params": mmcca2.parameters(),
                "lr": lr_max,
                "weight_decay": mmcca_decay,
            },
            {
                "params": ae_model.parameters(),
                "lr": lr_min,
                "weight_decay": weight_decay,
            },
        ],
        lr=lr_min,
    )

    # Load data
    # print("Generating data ...")
    # mnist, mnist_labels = fetch_openml(
    #     "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    # )

    # rotated_mnist = np.zeros_like(mnist)
    # angles = np.zeros(mnist.shape[0], dtype="float32")
    # for i in range(mnist.shape[0]):
    #     rotated_mnist[i], angles[i] = rot_digit(mnist[i], restricted_rotations=False)

    # X = mnist.astype("int16").astype("float32")
    # Y = rotated_mnist.astype("int16").astype("float32")
    # X = torch.Tensor(X).reshape(-1, 1, 28, 28) / 255
    # Y = torch.Tensor(Y).reshape(-1, 1, 28, 28) / 255

    data = np.load(
        "/cis/home/skoukun1/projects/SPLICE/data/mnist/mnist_rotated_360.npz"
    )

    X = torch.Tensor(data["original"][:50000]).to(device).reshape(-1, 1, 28, 28)
    Y = torch.Tensor(data["rotated"][:50000]).to(device).reshape(-1, 1, 28, 28)

    dataset = utils.ViewDataset(X[:50000], Y[:50000])
    train_loader_b1 = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size1, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size2, shuffle=False
    )
    train_loader_b2 = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size2, shuffle=True
    )
    corr_iter = iter(train_loader_b2)

    # Start training
    best_obj = float("inf")
    model_file_name = "lyu_mnist_model.pth"

    print("Start training ...")
    for itr in range(1, num_iters + 1):

        # Solve the U subproblem
        U = train.update_U(ae_model, eval_loader, z_dim, device)

        # Update network theta and eta for multiple epochs
        for _ in range(inner_epochs):

            # Backprop to update
            corr_iter = train.train(
                ae_model,
                mmcca2,
                U,
                train_loader_b1,
                train_loader_b2,
                corr_iter,
                beta,
                _lambda,
                optimizer,
                device,
            )

            # Evaluate on the whole set
            match_err, recons_err, corr = train.eval_train(
                ae_model, mmcca2, itr, U, eval_loader, beta, _lambda, device
            )

            # Save the model
            if match_err + beta * recons_err + _lambda * corr < best_obj:
                print("Saving Model")
                torch.save(ae_model.state_dict(), model_file_name)
                best_obj = match_err + beta * recons_err + _lambda * corr


def rot_digit(m, restricted_rotations=True):
    """
    Returns the digit/image "m" by a random angle [-45,45]deg
    clips it to MNIST size
    and returns it flattened into (28*28,) shape
    """
    if restricted_rotations:
        angle = np.random.rand() * 45 - 45
    else:
        angle = np.random.rand() * 90 - 45  # will lead to ambiguities because "6" = "9"

    m = m.reshape((28, 28))
    tmp = rotate(m, angle=angle)
    xs, ys = tmp.shape
    xs = int(xs / 2)
    ys = int(ys / 2)
    rot_m = tmp[xs - 14 : xs + 14, ys - 14 : ys + 14]
    return rot_m.reshape((28 * 28,)), angle


if __name__ == "__main__":
    main(sys.argv[1:])
