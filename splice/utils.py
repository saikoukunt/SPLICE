import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import lil_array
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairedViewDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

    def __len__(self):
        return self.dataset_a.shape[0]

    def __getitem__(self, idx):
        return self.dataset_a[idx], self.dataset_b[idx], idx


class MultiViewDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return self.datasets[0].shape[0]

    def __getitem__(self, idx):
        sample = [self.datasets[i][idx] for i in range(len(self.datasets))]
        return sample, idx


def update_G(x_a, x_b, model, batch_size):
    with torch.no_grad():
        a_hat, b_hat, z_a, z_b = model(x_a, x_b)
        Y = z_a + z_b

        # Ensure Y is zero mean
        Y = Y - torch.mean(Y, axis=0)  # type: ignore
        G, S, Vh = torch.linalg.svd(Y, full_matrices=False)
        G = G @ Vh
        G = np.sqrt(batch_size) * G
    return G


def calculate_mnist_accuracy(true_labels, z_train, z_validation):

    # do spectral clustering
    clust = KMeans(n_clusters=10)
    clust.fit(z_train)
    validation_pred = clust.predict(z_validation)

    # do linear sum assignment
    confusion = confusion_matrix(true_labels, validation_pred)
    row_ind, col_ind = linear_sum_assignment(-confusion)
    label_mapping = {
        old_label: new_label for old_label, new_label in zip(col_ind, row_ind)
    }
    validation_pred = [label_mapping[label] for label in validation_pred]

    return np.mean(true_labels == validation_pred)


def calculate_isomap_dists(x, n_neighbors, landmark_inds):
    x = x.detach().cpu().numpy()

    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(x)
    pair_dists, neighbors = neigh.kneighbors(x, return_distance=True)  # type: ignore
    neighbors = neighbors[:, 1:]
    pair_dists = pair_dists[:, 1:]

    graph = lil_array((pair_dists.shape[0], pair_dists.shape[0]))
    for i in range(pair_dists.shape[0]):
        graph[i, neighbors[i]] = pair_dists[i, :]

    dists = dijkstra(graph, indices=landmark_inds, directed=False, unweighted=False)

    return torch.Tensor(dists)


def compute_corr(x1, x2):
    if x1 is None:
        return torch.Tensor(0)

    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2))) + 0.001
    sigma2 = torch.sqrt(torch.mean(x2.pow(2))) + 0.001
    corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)

    if corr.isnan():
        print(x1_mean, x2_mean, sigma1, sigma2, corr)
        np.savez_compressed(
            "nan_corr.npz", x1=x1.detach().cpu().numpy(), x2=x2.detach().cpu().numpy()
        )
        raise ValueError("NaN in correlation computation")

    return corr
