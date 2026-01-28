import json
import numpy as np
import cv2
from pathlib import Path

# ====================== PATH SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

MASK_DIR = DATA_ROOT / SESSION / "mask"
PALM_DEPTH_PATH = DATA_ROOT / SESSION / "palm_depths.json"
INTRINSICS_PATH = DATA_ROOT / SESSION / "intrinsics.json"
# ======================================================


def load_intrinsics(path):
    with open(path, "r") as f:
        intr = json.load(f)
    return intr["fx"], intr["fy"], intr["cx"], intr["cy"]


def load_palm_depths(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def main():
    fx, fy, cx, cy = load_intrinsics(INTRINSICS_PATH)
    palm_depths = load_palm_depths(PALM_DEPTH_PATH)

    translations = {}
    valid_frames = 0

    for frame_id, Z in palm_depths.items():
        if Z is None:
            translations[frame_id] = None
            continue

        mask_path = MASK_DIR / f"{frame_id}.png"
        if not mask_path.exists():
            translations[frame_id] = None
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        centroid = compute_mask_centroid(mask)

        if centroid is None:
            translations[frame_id] = None
            continue

        u, v = centroid

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        translations[frame_id] = [float(X), float(Y), float(Z)]
        valid_frames += 1

    out_path = DATA_ROOT / SESSION / "translations.json"
    with open(out_path, "w") as f:
        json.dump(translations, f, indent=2)

    print("=== STEP 1.3 SUMMARY ===")
    print(f"Valid frames       : {valid_frames}")
    print(f"Saved to           : {out_path}")
    print("✅ STEP 1.3 completed — translation initialized.")


if __name__ == "__main__":
    main()
