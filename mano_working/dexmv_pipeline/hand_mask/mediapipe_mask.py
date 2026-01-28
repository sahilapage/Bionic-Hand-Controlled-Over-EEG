import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# ====================== PATH SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # dexmv_pipeline/
DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

RGB_DIR = DATA_ROOT / SESSION / "rgb"
MASK_DIR = DATA_ROOT / SESSION / "mask"
# ======================================================

MAX_HANDS = 1

# ---- Sanity check config ----
VISUALIZE = True          # show overlay for first VIS_FRAMES
VIS_FRAMES = 300
MIN_COVERAGE = 0.01
MAX_COVERAGE = 0.60
# -----------------------------


def ensure_dirs():
    MASK_DIR.mkdir(exist_ok=True)


def init_mediapipe():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,     # IMPORTANT for offline dataset
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5
    )


def hand_mask_from_landmarks(rgb_shape, hand_landmarks):
    H, W, _ = rgb_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    pts = []
    for lm in hand_landmarks.landmark:
        pts.append([int(lm.x * W), int(lm.y * H)])

    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def compute_coverage(mask):
    H, W = mask.shape
    return mask.sum() / (255 * H * W)


def visualize_overlay(bgr, mask):
    overlay = bgr.copy()
    overlay[mask > 0] = (
        0.6 * overlay[mask > 0] +
        0.4 * np.array([0, 255, 0])
    ).astype(np.uint8)

    cv2.imshow("Sanity Check | RGB + Hand Mask", overlay)
    cv2.waitKey(30)


def main():
    ensure_dirs()
    hands = init_mediapipe()

    rgb_files = sorted(RGB_DIR.glob("*.png"))
    print(f"Processing {len(rgb_files)} frames...")

    coverages = []

    for i, rgb_path in enumerate(rgb_files):
        frame_id = rgb_path.stem

        bgr = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            mask = hand_mask_from_landmarks(
                rgb.shape,
                results.multi_hand_landmarks[0]
            )
            mask = clean_mask(mask)
        else:
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        coverage = compute_coverage(mask)
        coverages.append(coverage)

        # ---- warnings only for partial / weird cases ----
        if 0 < coverage < MIN_COVERAGE:
            print(f"[WARN] {frame_id}: very small mask ({coverage:.3f})")

        if coverage > MAX_COVERAGE:
            print(f"[WARN] {frame_id}: very large mask ({coverage:.3f})")

        # ---- sampled visualization ----
        if VISUALIZE and i < VIS_FRAMES:
            visualize_overlay(bgr, mask)

        cv2.imwrite(str(MASK_DIR / f"{frame_id}.png"), mask)

    hands.close()
    cv2.destroyAllWindows()

    # ================= DATASET-LEVEL SANITY SUMMARY =================
    coverages = np.array(coverages)

    total = len(coverages)
    empty = (coverages == 0).sum()
    valid = (coverages > 0).sum()

    print("\n=== SANITY SUMMARY ===")
    print(f"Total frames        : {total}")
    print(f"Frames with hand    : {valid}")
    print(f"Empty frames        : {empty}")

    if valid > 0:
        print(f"Min coverage        : {coverages[coverages > 0].min():.3f}")
        print(f"Max coverage        : {coverages.max():.3f}")
        print(f"Mean coverage       : {coverages[coverages > 0].mean():.3f}")

    print("✅ STEP 1.1 completed — hand masks saved.")


if __name__ == "__main__":
    main()
