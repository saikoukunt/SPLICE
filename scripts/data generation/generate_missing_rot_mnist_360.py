import numpy as np
from scipy.ndimage import rotate
from sklearn.datasets import fetch_openml
from tqdm import tqdm, trange


def rot_digit_missing(m, digit, angle=None):
    """
    Returns the digit/image "m" by a random angle deg
    clips it to MNIST size
    and returns it flattened into (28*28,) shape
    """
    if angle is None:
        if digit == "7":
            angle = np.random.rand() * 270
        else:
            angle = np.random.rand() * 360

    m = m.reshape((28, 28))
    tmp = rotate(m, angle=angle)
    xs, ys = tmp.shape
    xs = int(xs / 2)
    ys = int(ys / 2)
    rot_m = tmp[xs - 14 : xs + 14, ys - 14 : ys + 14]
    return rot_m.reshape((28 * 28,)), angle


if __name__ == "__main__":
    tqdm.write("Fetching data ...")
    mnist, mnist_labels = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
    )
    rotated_mnist = np.zeros_like(mnist)
    angles = np.zeros(mnist.shape[0], dtype="float32")

    for i in trange(mnist.shape[0], desc="Generating data ..."):
        rotated_mnist[i], angles[i] = rot_digit_missing(mnist[i], mnist_labels[i])

    view1 = mnist.astype("int16").astype("float32") / 255.0
    view2 = rotated_mnist.astype("int16").astype("float32") / 255.0

    np.savez_compressed(
        "../../data/mnist/mnist_rotated_shared-digit_missing.npz",
        view1=view1,
        view2=view2,
        labels=mnist_labels.astype("int16"),
        angles=angles,
    )
