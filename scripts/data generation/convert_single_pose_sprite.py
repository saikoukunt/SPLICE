import numpy as np
from scipy.ndimage import rotate
from sympy import N


def rot_image(m):
    """
    Returns the image "m" by a random angle [-45,45]deg
    clips it to MNIST size
    and returns it flattened into (28*28,) shape
    """
    angle = np.random.rand() * 360  # will lead to ambiguities because "6" = "9"
    rot_m = rotate(m, angle=angle, reshape=False)

    return rot_m, angle


data = np.load(
    "../../data/sprites/npy/single_pose.npy",
)
train_data = data[: int(data.shape[0] * 0.8)]
test_data = data[int(data.shape[0] * 0.8) :]

# randomly pair sprites and rotate by a random angle
n_pairs_train = 20000
n_pairs_test = 5000

train_angles = np.random.rand(n_pairs_train) * 360
test_angles = np.random.rand(n_pairs_test) * 360

train_view1_inds = np.random.choice(np.arange(train_data.shape[0]), n_pairs_train)
test_view1_inds = np.random.choice(np.arange(test_data.shape[0]), n_pairs_test)
train_view2_inds = np.random.choice(np.arange(train_data.shape[0]), n_pairs_train)
test_view2_inds = np.random.choice(np.arange(test_data.shape[0]), n_pairs_test)

train_view1 = np.zeros((n_pairs_train, 64, 64, 3))
train_view2 = np.zeros((n_pairs_train, 64, 64, 3))
test_view1 = np.zeros((n_pairs_test, 64, 64, 3))
test_view2 = np.zeros((n_pairs_test, 64, 64, 3))

for i in range(n_pairs_train):
    print(i, end="\r")
    train_view1[i] = rotate(
        train_data[train_view1_inds[i]], angle=train_angles[i], reshape=False
    )
    train_view2[i] = rotate(
        train_data[train_view2_inds[i]], angle=train_angles[i], reshape=False
    )

for i in range(n_pairs_test):
    print(i, end="\r")
    test_view1[i] = rotate(
        test_data[test_view1_inds[i]], angle=test_angles[i], reshape=False
    )
    test_view2[i] = rotate(
        test_data[test_view2_inds[i]], angle=test_angles[i], reshape=False
    )

np.savez_compressed(
    f"../../data/sprites/single_pose_train.npz",
    view1=train_view1,
    view2=train_view2,
    angles=train_angles,
    view2_inds=train_view2_inds,
)

np.savez_compressed(
    f"../../data/sprites/single_pose_test.npz",
    view1=test_view1,
    view2=test_view2,
    angles=test_angles,
    view2_inds=test_view2_inds,
)
