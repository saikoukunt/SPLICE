import os

import imageio
import numpy as np
from scipy.ndimage import rotate

folder = "../../data/sprites/frames/slash"
images = np.zeros((1296, 64, 64, 3))
count = 0

for filename in os.listdir(folder):
    if filename.endswith("0.png") and filename.startswith("front"):
        print(filename)
        with open(folder + "/" + filename, "rb") as f:
            im = imageio.imread(f)
        images[count] = np.asarray(im[:, :, :-1], dtype="f") / 256.0
        count += 1

np.save(f"../../data/sprites/npy/single_pose.npy", images)
