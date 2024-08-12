import sys

import numpy as np
from scipy.ndimage import rotate
from sklearn.datasets import fetch_openml


def rot_digit(m, restricted_rotations=True):
    """
    Returns the digit/image "m" by a random angle [-45,45]deg
    clips it to MNIST size
    and returns it flattened into (28*28,) shape
    """
    if restricted_rotations:
        angle = np.random.rand() * 90 - 45
    else:
        angle = np.random.rand() * 360  # will lead to ambiguities because "6" = "9"

    m = m.reshape((28, 28))
    tmp = rotate(m, angle=angle)
    xs, ys = tmp.shape
    xs = int(xs / 2)
    ys = int(ys / 2)
    rot_m = tmp[xs - 14 : xs + 14, ys - 14 : ys + 14]
    return rot_m.reshape((28 * 28,)), angle


if __name__ == "__main__":
    degrees = sys.argv[1]
    print("Generating data ...")
    mnist, mnist_labels = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )

    rotated_mnist = np.zeros_like(mnist)
    angles = np.zeros(mnist.shape[0], dtype="float32")
    if degrees == "360":
        for i in range(mnist.shape[0]):
            rotated_mnist[i], angles[i] = rot_digit(
                mnist[i], restricted_rotations=False
            )
    elif degrees == "90":
        for i in range(mnist.shape[0]):
            rotated_mnist[i], angles[i] = rot_digit(mnist[i], restricted_rotations=True)
    else:
        raise ValueError("Invalid degree value %s. Must be 90 or 360" % degrees)

    view1 = mnist.astype("int16").astype("float32") / 255.0
    view2 = rotated_mnist.astype("int16").astype("float32") / 255.0

    np.savez_compressed(
        f"../data/mnist/mnist_rotated_{degrees}.npz",
        original=view1,
        rotated=view2,
        labels=mnist_labels.astype("int16"),
        angles=angles,
    )
