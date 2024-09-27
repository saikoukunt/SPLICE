import math
import os
import pickle
import sys

import numpy as np
import torch
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset


def make_stim(x, y, theta, width=30, height=10, grid_width=100, grid_height=100):
    """
    Generate a rotated bar stimulus on a pixel grid.

    Args:
        x (int): x-coordinate of the bar.
        y (int): y-coordinate of the bar.
        theta (float): angle of rotation in degrees.
        width (int, optional): width of the bar in pixels. Defaults to 30.
        height (int, optional): height of the bar in pixels. Defaults to 10.
        grid_width (int, optional): width of the grid in pixels. Defaults to 100.
        grid_height (int, optional): width of the bar in pixels. Defaults to 100.

    Raises:
        RuntimeError: bar out of bounds, if the at least half the bar is out of the grid.

    Returns:
        np.array: an array containing the stimulus on the grid.
    """
    grid = np.zeros((grid_width, grid_height))
    stim = np.ones((width, height))
    stim = rotate(stim, theta)

    xrad = math.floor(stim.shape[0] / 2)
    yrad = math.floor(stim.shape[1] / 2)

    # valid stimuli have at least half the bar in the grid
    if (
        (x - xrad < 0)
        or (y - yrad < 0)
        or (x + xrad >= grid.shape[0])
        or (y + yrad >= grid.shape[1])
    ):
        raise RuntimeError("bar out of bounds")

    grid[x - xrad : x - xrad + stim.shape[0], y - yrad : y - yrad + stim.shape[1]] = (
        stim
    )

    return grid


def make_lgn(x, y):
    """
    Generate an LGN receptive field.

    The receptive field to the shared "visual" stimulus is a center-surround, centered at
    the input x, y coordinates. The receptive field to the private "position" stimulus is
    centered at a random location, with a response scaling coefficient c.

    Args:
        x (int): downsampled x-coordinate of the RF to the shared response.
        y (int): downsampled y-coordinate of the RF to the shared response.

    Returns:
        visual_rf (np.ndarray): an array containing the visual receptive field on the downsampled grid.
        mu (float): the center of the private receptive field.
        c (float): the scaling coefficient for the private response.
    """
    visual_rf = np.zeros((100, 100))

    # calculate bounds
    xmin = max(0, x - 7)
    xmax = min(x + 8, visual_rf.shape[0])
    ymin = max(0, y - 7)
    ymax = min(y + 8, visual_rf.shape[1])

    # make center surround
    visual_rf[xmin:xmax, ymin:ymax] = -1 / (8 * 25)
    visual_rf[x - 2 : x + 3, y - 2 : y + 3] = 1 / 25

    # private response parameters
    c = 1.73 / 2
    mu = 20 * np.random.rand(1)[0]

    return visual_rf, mu, c


def make_v1(x, y, theta, grid_width=100, grid_height=100):
    """
    Generate an V1 receptive field.

    The receptive field to the shared "visual" stimulus is a Gabor filter, centered at
    the input x, y coordinates. The receptive field to the private "position" stimulus is
    centered at a random location, with a response scaling coefficient c.

    Args:
        x (int): downsampled x-coordinate of the RF to the shared response.
        y (int): downsampled y-coordinate of the RF to the shared response.
        theta (int): orientation of the Gabor filter. Must be 0 or 90.
        grid_width (int, optional): width of the grid in pixels. Defaults to 100.
        grid_height (int, optional): width of the bar in pixels. Defaults to 100.

    Returns:
        visual_rf (np.ndarray): an array containing the visual receptive field on the downsampled grid.
        mu (float): the center of the private receptive field.
        c (float): the scaling coefficient for the private response.
    """
    visual_rf = np.zeros((grid_width, grid_height))

    # calculate bounds
    width = 15
    height = 15

    xrad = math.floor(width / 2)
    yrad = math.floor(height / 2)

    xmin = int(max(0, x - xrad))
    xmax = int(min(x + xrad + 1, visual_rf.shape[0]))
    ymin = max(0, y - yrad)
    ymax = int(min(y + yrad + 1, visual_rf.shape[1]))

    # set weights
    visual_rf[xmin:xmax, ymin:ymax] = -1 / (14 * 25)
    if theta == 0:
        visual_rf[xmin:xmax, y - 2 : y + 3] = 1 / (3 * 25)
    elif theta == 90:
        visual_rf[x - 2 : x + 3, ymin:ymax] = 1 / (3 * 25)

    # private response parameters
    c = 1.73 / 2
    mu = 20 * np.random.rand(1)[0]

    return visual_rf, mu, c


def generate_sim(sampling="grid", mixing="linear", num_samples=18900, train_size=0.8):
    """
    Generates pseudo-LGN and pseudo-V1 simulated responses.

    ### Args:
        - `sampling` (str): `"grid"` generates 18900 samples uniformly distributed on a 20x20x grid.
                       `"random"` generates `num_samples` samples randomly sampled from a 20x20 grid.
        - `mixing` (str): `"linear"` mixes shared and private responses linearly.
        - 'num_samples' (int): If 'sampling' == 'random', number of samples to generate.
        - 'train_size' (int): Percentage of samples to use for training.

    ### Returns:
        - `train` (dict): List of dictionaries containing training samples.
            - `responses` (dict): Dictionary containing LGN and V1 responses.
                - `lgn` (np.ndarray): LGN responses.
                - `v1` (np.ndarray): V1 responses.
            - `stim_params` (dict): Dictionary containing stimulus parameters.
                - `x` (list): x-coordinates of stimuli.
                - `y` (list): y-coordinates of stimuli.
                - `lgn_p` (list): LGN private variables.
                - `v1_p` (list): V1 private variables.
                - `stim_arr` (np.ndarray): stimuli.
        - `test` (dict): List of dictionaries containing testing samples. Same structure as `train`.
        - `field_params` (dict): Dictionary containing receptive field parameters.
                - `lgn_field` (np.ndarray): LGN visual receptive fields.
                - `lgn_mu` (np.ndarray): LGN place field centers.
                - `lgn_c` (np.ndarray): LGN private response coefficients.
                - `v1_field` (np.ndarray): V1 visual receptive fields.
                - `v1_mu` (np.ndarray): V1 place field centers.
                - `v1_c` (np.ndarray): V1 private response coefficients.
    """

    # generate stimuli
    match sampling:
        case "grid":
            stim_arr = []
            x = []
            y = []
            lgn_p = []
            v1_p = []

            for i in np.arange(0, 100, 1):
                for j in np.arange(0, 100, 1):
                    try:
                        for k in range(3):
                            stim_arr.append(
                                make_stim(i, j, 0, grid_width=100, grid_height=100)
                            )
                            x.append(i)
                            y.append(j)
                            lgn_p.append(20 * np.random.rand(1)[0])
                            v1_p.append(20 * np.random.rand(1)[0])
                    except RuntimeError:
                        continue

            stim_arr = np.array(stim_arr).reshape(-1, 100 * 100)
            x = np.array(x)
            y = np.array(y)
            lgn_p = np.array(lgn_p)
            v1_p = np.array(v1_p)
        case "random":
            min_x = 15  # valid stimuli have at least half the bar (30x10) in the grid
            max_x = 84
            min_y = 5
            max_y = 94

            x = np.random.uniform(min_x, max_x, num_samples).astype(int)
            y = np.random.uniform(min_y, max_y, num_samples).astype(int)

            lgn_p = 20 * np.random.rand(num_samples)
            v1_p = 20 * np.random.rand(num_samples)
            stim_arr = np.zeros((num_samples, 100, 100))

            for i in range(num_samples):
                stim_arr[i] = make_stim(x[i], y[i], 0, grid_width=100, grid_height=100)

            stim_arr = stim_arr.reshape(-1, 100 * 100)
        case _:
            raise ValueError(
                "Invalid sampling method. Usage: python simulate_lgn_v1.py <file_desc> <sampling>"
            )

    # generate shared and private receptive field params
    lgn_field = np.zeros((20, 20, 100, 100))
    lgn_mu = np.zeros((20, 20))
    lgn_c = np.zeros((20, 20))

    v1_field = np.zeros((20, 20, 2, 100, 100))
    v1_mu = np.zeros((20, 20, 2))
    v1_c = np.zeros((20, 20, 2))
    rots = [0, 90]

    for i in range(20):
        for j in range(20):
            lgn_field[i, j, :, :], lgn_mu[i, j], lgn_c[i, j] = make_lgn(i * 5, j * 5)

            for k in range(len(rots)):
                v1_field[i, j, k], v1_mu[i, j, k], v1_c[i, j, k] = make_v1(
                    i * 5,
                    j * 5,
                    rots[k],
                )
    lgn_field = lgn_field.reshape(-1, 100 * 100)
    lgn_mu = lgn_mu.reshape(20 * 20)
    lgn_c = lgn_c.reshape(20 * 20)

    v1_field = v1_field.reshape(-1, 2, 100 * 100)
    v1_field = np.vstack(  # first 400 are 0 degree, next 400 are 90 degree
        (v1_field[:, 0, :], v1_field[:, 1, :])
    )
    v1_mu = v1_mu.reshape(20 * 20 * 2)
    v1_c = v1_c.reshape(20 * 20 * 2)

    # generate shared and private responses
    lgn_shared_resp = stim_arr @ lgn_field.T
    v1_shared_resp = stim_arr @ v1_field.T

    lgn_priv_resp = lgn_c * np.exp(
        -((lgn_mu * np.ones((len(stim_arr), 400)) - lgn_p.reshape(-1, 1)) ** 2)
    )
    v1_priv_resp = v1_c * np.exp(
        -((v1_mu * np.ones((len(stim_arr), 800)) - v1_p.reshape(-1, 1)) ** 2)
    )

    match mixing:
        case "linear":
            lgn_resp = lgn_shared_resp + lgn_priv_resp
            v1_resp = v1_shared_resp + v1_priv_resp
        case _:
            raise ValueError(
                "Invalid mixing method. Usage: python generate_lgn_v1.py <file_desc> <sampling>"
            )

    # split into training and testing sets
    train_inds, test_inds = train_test_split(
        np.arange(len(stim_arr)), train_size=train_size
    )

    lgn_resp_train, lgn_resp_test = lgn_resp[train_inds], lgn_resp[test_inds]
    v1_resp_train, v1_resp_test = v1_resp[train_inds], v1_resp[test_inds]
    x_train, x_test = x[train_inds], x[test_inds]
    y_train, y_test = y[train_inds], y[test_inds]
    lgn_p_train, lgn_p_test = lgn_p[train_inds], lgn_p[test_inds]
    v1_p_train, v1_p_test = v1_p[train_inds], v1_p[test_inds]
    stim_arr_train, stim_arr_test = stim_arr[train_inds], stim_arr[test_inds]

    return (
        {
            "responses": {
                "lgn": lgn_resp_train,
                "v1": v1_resp_train,
            },
            "stim_params": {
                "x": x_train,
                "y": y_train,
                "lgn_p": lgn_p_train,
                "v1_p": v1_p_train,
                "stim_arr": stim_arr_train,
            },
        },
        {
            "responses": {
                "lgn": lgn_resp_test,
                "v1": v1_resp_test,
            },
            "stim_params": {
                "x": x_test,
                "y": y_test,
                "lgn_p": lgn_p_test,
                "v1_p": v1_p_test,
                "stim_arr": stim_arr_test,
            },
        },
        {
            "lgn_field": lgn_field,
            "lgn_mu": lgn_mu,
            "lgn_c": lgn_c,
            "v1_field": v1_field,
            "v1_mu": v1_mu,
            "v1_c": v1_c,
        },
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simulate_lgn_v1.py <file_desc> <sampling> <num_samples")
        sys.exit(1)

    file_desc = sys.argv[1]
    sampling = sys.argv[2] if len(sys.argv) >= 3 else "grid"
    num_samples = int(sys.argv[3]) if len(sys.argv) >= 4 else 18900

    train, test, field_params = generate_sim(sampling=sampling, num_samples=num_samples)

    os.makedirs(os.path.join("..", "..", "data", file_desc), exist_ok=True)
    pickle.dump(
        train,
        open(
            os.path.join("..", "..", "data", file_desc, f"{file_desc}_train.pkl"), "wb"
        ),
    )
    pickle.dump(
        test,
        open(
            os.path.join("..", "..", "data", file_desc, f"{file_desc}_test.pkl"), "wb"
        ),
    )
    pickle.dump(
        field_params,
        open(
            os.path.join(
                "..", "..", "data", file_desc, f"{file_desc}_field_params.pkl"
            ),
            "wb",
        ),
    )
