import json
import torch
import smplx
from pathlib import Path

# ====================== PATH SETUP ======================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # dexmv_pipeline/

DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

TRANS_PATH = DATA_ROOT / SESSION / "translations_smooth.json"

# IMPORTANT: path should contain the "mano/" folder, not end with it
MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
MANO_SIDE = "right"   # or "left"
# ======================================================

DEVICE = "cpu"


def load_translations(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    translations = load_translations(TRANS_PATH)

    # -------- Load MANO model --------
    mano = smplx.create(
        model_path=str(MANO_MODEL_DIR),
        model_type="mano",
        side=MANO_SIDE,
        use_pca=False,
        flat_hand_mean=True,
        hand_pose_dim=45
    ).to(DEVICE)

    # -------- Initialize parameters --------
    betas = torch.zeros(1, 10, device=DEVICE)
    hand_pose = torch.zeros(1, 45, device=DEVICE)
    global_orient = torch.zeros(1, 3, device=DEVICE)

    print("✅ MANO model loaded successfully.")
    print("Testing MANO forward pass on first valid frame...\n")

    for frame_id, T in translations.items():
        if T is None:
            continue

        transl = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        output = mano(
            betas=betas,
            hand_pose=hand_pose,
            global_orient=global_orient,
            transl=transl
        )

        print(f"Frame ID        : {frame_id}")
        print(f"Translation used: {T}")
        print(f"Vertices shape  : {output.vertices.shape}")
        print(f"Joints shape    : {output.joints.shape}")
        break

    print("\n✅ STEP 2.1 completed — MANO initialized correctly.")


if __name__ == "__main__":
    main()
