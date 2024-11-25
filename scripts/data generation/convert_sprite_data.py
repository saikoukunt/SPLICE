import numpy as np
from scipy.ndimage import rotate


def rot_image(m):
    """
    Returns the image "m" by a random angle [-45,45]deg
    clips it to MNIST size
    and returns it flattened into (28*28,) shape
    """
    angle = np.random.rand() * 360  # will lead to ambiguities because "6" = "9"
    rot_m = rotate(m, angle=angle, reshape=False)

    return rot_m, angle


anims = ["slash", "spellcard", "walk"]
dirs = ["front", "left", "right"]

train_data = np.zeros((0, 64, 64, 3))
test_data = np.zeros((0, 64, 64, 3))

# aggregate animations into single array
for anim in anims:
    for dir in dirs:
        train_file = np.load(
            f"../../data/sprites/npy/{anim}_{dir}_frames_train.npy"
        ).reshape(-1, 64, 64, 3)
        test_file = np.load(
            f"../../data/sprites/npy/{anim}_{dir}_frames_test.npy"
        ).reshape(-1, 64, 64, 3)

        train_data = np.concatenate((train_data, train_file), axis=0)
        test_data = np.concatenate((test_data, test_file), axis=0)


# randomly pair sprites and rotate by a random angle
train_angles = np.random.rand(train_data.shape[0]) * 360
test_angles = np.random.rand(test_data.shape[0]) * 360

train_view1_inds = np.random.permutation(train_data.shape[0])
test_view1_inds = np.random.permutation(test_data.shape[0])
train_view2_inds = np.random.permutation(train_data.shape[0])
test_view2_inds = np.random.permutation(test_data.shape[0])

train_view1 = np.zeros_like(train_data)
train_view2 = np.zeros_like(train_data)
test_view1 = np.zeros_like(test_data)
test_view2 = np.zeros_like(test_data)

for i in range(train_data.shape[0]):
    print(i, train_data.shape[0], end="\r")
    train_view1[i] = rotate(
        train_data[train_view1_inds[i]], angle=train_angles[i], reshape=False
    )
    train_view2[i] = rotate(
        train_data[train_view2_inds[i]], angle=train_angles[i], reshape=False
    )

for i in range(test_data.shape[0]):
    print(i, test_data.shape[0], end="\r")
    test_view1[i] = rotate(
        test_data[test_view1_inds[i]], angle=test_angles[i], reshape=False
    )
    test_view2[i] = rotate(
        test_data[test_view2_inds[i]], angle=test_angles[i], reshape=False
    )

np.savez_compressed(
    f"../../data/sprites/train.npz",
<<<<<<< HEAD
    view1=train_view1,
    view2=train_view2,
=======
    view1=np.clip(train_view1, 0, 1),
    view2=np.clip(train_view2, 0, 1),
>>>>>>> 6bb97c23a9aa83707f5a306cb03c3b1d37406d5a
    angles=train_angles,
    view2_inds=train_view2_inds,
)

np.savez_compressed(
    f"../../data/sprites/test.npz",
<<<<<<< HEAD
    view1=test_view1,
    view2=test_view2,
=======
    view1=np.clip(test_view1, 0, 1),
    view2=np.clip(test_view2, 0, 1),
>>>>>>> 6bb97c23a9aa83707f5a306cb03c3b1d37406d5a
    angles=test_angles,
    view2_inds=test_view2_inds,
)
