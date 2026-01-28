import cv2
import json
import mediapipe as mp
from pathlib import Path

# ================= PATH SETUP =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

RGB_DIR = DATA_ROOT / SESSION / "rgb"
OUT_DIR = DATA_ROOT / SESSION / "joints2d"
# =============================================

OUT_DIR.mkdir(exist_ok=True)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def main():
    rgb_files = sorted(RGB_DIR.glob("*.png"))
    print(f"Extracting 2D joints from {len(rgb_files)} frames")

    for rgb_path in rgb_files:
        frame_id = rgb_path.stem

        img = cv2.imread(str(rgb_path))
        if img is None:
            continue

        H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        if not result.multi_hand_landmarks:
            joints = None
        else:
            lm = result.multi_hand_landmarks[0].landmark
            joints = [[int(p.x * W), int(p.y * H)] for p in lm]

        with open(OUT_DIR / f"{frame_id}.json", "w") as f:
            json.dump({"joints": joints}, f)

    mp_hands.close()
    print("✅ 2D joint extraction completed.")


if __name__ == "__main__":
    main()
