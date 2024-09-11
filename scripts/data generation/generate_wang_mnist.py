from sklearn.datasets import fetch_openml
from scipy.ndimage import rotate

import numpy as np


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


print("Loading MNIST...")
mnist, mnist_labels = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto"
)

print("Rotating digits...")
rotated_mnist = np.zeros_like(mnist)
angles = np.zeros(mnist.shape[0], dtype="float32")
for i in range(mnist.shape[0]):
    rotated_mnist[i], angles[i] = rot_digit(mnist[i], restricted_rotations=True)

view1 = rotated_mnist.astype("int16").astype("float32") / 255.0
view2 = mnist.astype("int16").astype("float32") / 255 + np.random.uniform(
    0, 1, mnist.shape
)
view2 = np.clip(view2, 0, 1)

print("Shuffling...")
mnist_labels = mnist_labels.astype("int")
label_inds = [[] for _ in range(10)]

for i, label in enumerate(mnist_labels):
    label_inds[label].append(i)

for idx_list in label_inds:
    np.random.shuffle(idx_list)

shuffled_inds = []
for i, label in enumerate(mnist_labels):
    shuffled_inds.append(label_inds[label].pop())

view2 = view2[shuffled_inds]

print("Saving...")
np.savez_compressed(
    "../data/mnist/wang_mnist.npz",
    view1=view1,
    view2=view2,
    labels=mnist_labels,
    angles=angles,
)
