import sys

import numpy as np
from scipy.ndimage import rotate
from sympy import N
from tqdm import trange

if __name__ == "__main__":
    data = np.load(
        "../../data/sprites/npy/single_pose.npy",
    )
    train_data = data[: int(data.shape[0] * 0.8)]
    test_data = data[int(data.shape[0] * 0.8) :]

    n_train = 20000
    n_test = 5000

    if sys.argv[1] == "--angle" and sys.argv[2] == "shared":
        # randomly pair sprites and rotate by a random angle
        train_angles = np.random.rand(n_train) * 360
        test_angles = np.random.rand(n_test) * 360

        train_view1_inds = np.random.choice(np.arange(train_data.shape[0]), n_train)
        test_view1_inds = np.random.choice(np.arange(test_data.shape[0]), n_test)
        train_view2_inds = np.random.choice(np.arange(train_data.shape[0]), n_train)
        test_view2_inds = np.random.choice(np.arange(test_data.shape[0]), n_test)

        train_view1 = np.zeros((n_train, 64, 64, 3))
        train_view2 = np.zeros((n_train, 64, 64, 3))
        test_view1 = np.zeros((n_test, 64, 64, 3))
        test_view2 = np.zeros((n_test, 64, 64, 3))

        for i in trange(n_train, ncols=80):
            train_view1[i] = rotate(
                train_data[train_view1_inds[i]], angle=train_angles[i], reshape=False
            )
            train_view2[i] = rotate(
                train_data[train_view2_inds[i]], angle=train_angles[i], reshape=False
            )

        for i in trange(n_test, ncols=80):
            test_view1[i] = rotate(
                test_data[test_view1_inds[i]], angle=test_angles[i], reshape=False
            )
            test_view2[i] = rotate(
                test_data[test_view2_inds[i]], angle=test_angles[i], reshape=False
            )

        np.savez_compressed(
            f"../../data/sprites/single-pose_shared-angle_train.npz",
            view1=train_view1,
            view2=train_view2,
            angles=train_angles,
            view1_inds=train_view1_inds,
            view2_inds=train_view2_inds,
        )

        np.savez_compressed(
            f"../../data/sprites/single-pose_shared-angle_test.npz",
            view1=test_view1,
            view2=test_view2,
            angles=test_angles,
            view1_inds=test_view1_inds,
            view2_inds=test_view2_inds,
        )

    elif sys.argv[1] == "--angle" and sys.argv[2] == "private":
        train_angles = np.random.rand(n_train, 2) * 360
        test_angles = np.random.rand(n_test, 2) * 360

        train_inds = np.random.choice(np.arange(train_data.shape[0]), n_train)
        test_inds = np.random.choice(np.arange(test_data.shape[0]), n_test)
        train_view1 = np.zeros((n_train, 64, 64, 3))
        train_view2 = np.zeros((n_train, 64, 64, 3))
        test_view1 = np.zeros((n_test, 64, 64, 3))
        test_view2 = np.zeros((n_test, 64, 64, 3))

        for i in trange(n_train, ncols=80):
            train_view1[i] = rotate(
                train_data[train_inds[i]], angle=train_angles[i, 0], reshape=False
            )
            train_view2[i] = rotate(
                train_data[train_inds[i]], angle=train_angles[i, 1], reshape=False
            )

        for i in trange(n_test, ncols=80):
            test_view1[i] = rotate(
                test_data[test_inds[i]], angle=test_angles[i, 0], reshape=False
            )
            test_view2[i] = rotate(
                test_data[test_inds[i]], angle=test_angles[i, 1], reshape=False
            )

        np.savez_compressed(
            f"../../data/sprites/single-pose_private-angle_train.npz",
            view1=train_view1,
            view2=train_view2,
            angles=train_angles,
            inds=train_inds,
        )

        np.savez_compressed(
            f"../../data/sprites/single-pose_private-angle_test.npz",
            view1=test_view1,
            view2=test_view2,
            angles=test_angles,
            inds=test_inds,
        )

        pass
    else:
        raise ValueError(
            "Incorrect usage -- specify a value for the --angle option. Valid values are private or shared)"
        )
