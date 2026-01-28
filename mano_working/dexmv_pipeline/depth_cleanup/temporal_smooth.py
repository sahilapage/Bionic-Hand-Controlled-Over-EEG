import json
import numpy as np
from pathlib import Path

# ====================== PATH SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

IN_PATH = DATA_ROOT / SESSION / "translations.json"
OUT_PATH = DATA_ROOT / SESSION / "translations_smooth.json"
# ======================================================

ALPHA = 0.8   # smoothing factor (higher = smoother)


def main():
    with open(IN_PATH, "r") as f:
        translations = json.load(f)

    frame_ids = sorted(translations.keys())
    smoothed = {}

    prev_T = None

    for fid in frame_ids:
        T = translations[fid]

        if T is None:
            smoothed[fid] = None
            continue

        T = np.array(T, dtype=np.float32)

        if prev_T is None:
            T_smooth = T
        else:
            T_smooth = ALPHA * T + (1 - ALPHA) * prev_T

        smoothed[fid] = T_smooth.tolist()
        prev_T = T_smooth

    with open(OUT_PATH, "w") as f:
        json.dump(smoothed, f, indent=2)

    valid = sum(v is not None for v in smoothed.values())

    print("=== STEP 1.5 SUMMARY ===")
    print(f"Valid frames       : {valid}")
    print(f"Saved to           : {OUT_PATH}")
    print("✅ STEP 1.5 completed — translation smoothed.")


if __name__ == "__main__":
    main()
