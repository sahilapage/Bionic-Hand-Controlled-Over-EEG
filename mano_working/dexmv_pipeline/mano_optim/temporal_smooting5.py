import json
import numpy as np
from pathlib import Path

# ================= PATH SETUP =================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

POSE_IN  = DATA_ROOT / SESSION / "hand_pose_final.json"
TRANS_IN = DATA_ROOT / SESSION / "translations_final.json"

POSE_OUT  = DATA_ROOT / SESSION / "hand_pose_smoothed.json"
TRANS_OUT = DATA_ROOT / SESSION / "translations_smoothed.json"
# =============================================

ALPHA = 0.8   # smoothing factor


def smooth_sequence(seq_dict):
    keys = sorted(seq_dict.keys())
    prev = None
    out = {}

    for k in keys:
        curr = seq_dict[k]
        if curr is None:
            out[k] = None
            continue

        curr = np.array(curr, dtype=np.float32)

        if prev is None:
            smoothed = curr
        else:
            smoothed = ALPHA * curr + (1 - ALPHA) * prev

        out[k] = smoothed.tolist()
        prev = smoothed

    return out


def main():
    pose = json.load(open(POSE_IN))
    trans = json.load(open(TRANS_IN))

    pose_s = smooth_sequence(pose)
    trans_s = smooth_sequence(trans)

    json.dump(pose_s, open(POSE_OUT, "w"), indent=2)
    json.dump(trans_s, open(TRANS_OUT, "w"), indent=2)

    print("✅ STEP 2.6 completed — temporal smoothing applied.")


if __name__ == "__main__":
    main()
