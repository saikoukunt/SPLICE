import sys

import numpy as np
from scipy.ndimage import rotate
from sklearn.datasets import fetch_openml


def rot_digit(m, angle=None):
    """
    Returns the digit/image "m" by a random angle deg
    clips it to MNIST size
    and returns it flattened into (28*28,) shape
    """
    if angle is None:
        angle = np.random.rand() * 360 

    m = m.reshape((28, 28))
    tmp = rotate(m, angle=angle)
    xs, ys = tmp.shape
    xs = int(xs / 2)
    ys = int(ys / 2)
    rot_m = tmp[xs - 14 : xs + 14, ys - 14 : ys + 14]
    return rot_m.reshape((28 * 28,)), angle


if __name__ == "__main__":
    shared = "digit" if len(sys.argv) < 2 else sys.argv[1]
    print("Generating data ...")
    mnist, mnist_labels = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    if shared == "digit":
        rotated_mnist = np.zeros_like(mnist)
        angles = np.zeros(mnist.shape[0], dtype="float32")

        for i in range(mnist.shape[0]):
            rotated_mnist[i], angles[i] = rot_digit(mnist[i])

        view1 = mnist.astype("int16").astype("float32") / 255.0
        view2 = rotated_mnist.astype("int16").astype("float32") / 255.0

        np.savez_compressed(
            f"../../data/mnist/mnist_rotated_shared-{shared}.npz",
            view1=view1,
            view2=view2,
            labels=mnist_labels.astype("int16"),
            angles=angles,
        )
    elif shared == "angle":
        view1 = np.zeros_like(mnist)
        view2 = np.zeros_like(mnist)
        angles = np.random.rand(mnist.shape[0]) * 360

        # make sure train, val, and test set have different digits
        train_inds1 = np.random.permutation(50000)
        train_inds2 = np.random.permutation(50000)
        val_inds1 = np.random.permutation(10000) + 50000
        val_inds2 = np.random.permutation(10000) + 50000
        test_inds1 = np.random.permutation(10000) + 60000
        test_inds2 = np.random.permutation(10000) + 60000

        view1_inds = np.concatenate((train_inds1, val_inds1, test_inds1))
        view2_inds = np.concatenate((train_inds2, val_inds2, test_inds2))

        for i in range(mnist.shape[0]):
            view1[i], _ = rot_digit(mnist[view1_inds[i]], angle=angles[i])
            view2[i], _ = rot_digit(mnist[view2_inds[i]], angle=angles[i])

        view1 = view1.astype("int16").astype("float32") / 255.0
        view2 = view2.astype("int16").astype("float32") / 255.0

        np.savez_compressed(
            f"../../data/mnist/mnist_rotated_shared-{shared}.npz",
            view1=view1,
            view2=view2,
            labels=mnist_labels.astype("int16"),
            angles=angles,
            inds=(view1_inds, view2_inds),
        )
            

