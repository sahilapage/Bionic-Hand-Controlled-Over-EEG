import numpy as np
import cv2
import json
from pathlib import Path

# ====================== PATH SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # dexmv_pipeline/
DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

DEPTH_DIR = DATA_ROOT / SESSION / "depth"
MASK_DIR  = DATA_ROOT / SESSION / "mask"
# ======================================================

# ---- Depth cleanup parameters ----
MIN_DEPTH = 0.1       # meters (adjust if needed)
MAX_DEPTH = 1.5       # meters (adjust if needed)
MIN_PIXELS = 200      # minimum hand pixels to trust frame
LOW_PERC = 5          # percentile for outlier removal
HIGH_PERC = 95
# ---------------------------------


def load_depth(path):
    """Load depth map in meters"""
    depth = np.load(path)
    return depth.astype(np.float32)


def load_mask(path):
    """Load binary mask"""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return mask > 0


def clean_hand_depth(depth, mask):
    """
    Returns:
        depth_pixels (1D np.array)
        valid (bool)
    """
    # Mask depth
    depth_hand = depth[mask]

    # Remove invalid values
    depth_hand = depth_hand[
        np.isfinite(depth_hand) &
        (depth_hand > MIN_DEPTH) &
        (depth_hand < MAX_DEPTH)
    ]

    if depth_hand.size < MIN_PIXELS:
        return None, False

    # Percentile-based outlier removal
    lo = np.percentile(depth_hand, LOW_PERC)
    hi = np.percentile(depth_hand, HIGH_PERC)

    depth_hand = depth_hand[
        (depth_hand >= lo) & (depth_hand <= hi)
    ]

    if depth_hand.size < MIN_PIXELS:
        return None, False

    return depth_hand, True


def main():
    depth_files = sorted(DEPTH_DIR.glob("*.npy"))

    palm_depths = {}
    valid_frames = 0

    print(f"Processing {len(depth_files)} frames...")

    for depth_path in depth_files:
        frame_id = depth_path.stem

        mask_path = MASK_DIR / f"{frame_id}.png"
        if not mask_path.exists():
            palm_depths[frame_id] = None
            continue

        depth = load_depth(depth_path)
        mask = load_mask(mask_path)

        depth_pixels, valid = clean_hand_depth(depth, mask)

        if not valid:
            palm_depths[frame_id] = None
            continue

        Z_palm = float(np.median(depth_pixels))
        palm_depths[frame_id] = Z_palm
        valid_frames += 1

    # Save palm depths for next step
    out_path = DATA_ROOT / SESSION / "palm_depths.json"
    with open(out_path, "w") as f:
        json.dump(palm_depths, f, indent=2)

    print("=== STEP 1.2 SUMMARY ===")
    print(f"Total frames       : {len(depth_files)}")
    print(f"Valid hand frames  : {valid_frames}")
    print(f"Saved to           : {out_path}")
    print("✅ STEP 1.2 completed — depth cleaned.")


if __name__ == "__main__":
    main()
