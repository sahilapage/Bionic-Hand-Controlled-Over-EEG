import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ================= PATH =================
DATA_ROOT = Path("dexmv_data/session_001/final_hand")
# =======================================

verts = np.load(DATA_ROOT / "vertices.npy")
frames = json.load(open(DATA_ROOT / "frames.json"))

# pick a frame index
IDX = 0   # change this to scroll through frames


def plot_hand(vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               s=1, c="orange")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1,1,1])
    plt.show()


plot_hand(verts[IDX])
