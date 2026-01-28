import json
import torch
import smplx
import numpy as np
import cv2
from pathlib import Path

# ================= PATH SETUP =================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

DEPTH_DIR = DATA_ROOT / SESSION / "depth"
MASK_DIR  = DATA_ROOT / SESSION / "mask"

TRANS_IN  = DATA_ROOT / SESSION / "translations_smooth.json"
TRANS_OUT = DATA_ROOT / SESSION / "translations_refined.json"
INTR_PATH = DATA_ROOT / SESSION / "intrinsics.json"

MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
# =============================================

DEVICE = "cpu"
LR = 5e-3
ITERS = 100


def load_intrinsics(path):
    with open(path, "r") as f:
        K = json.load(f)
    return K["fx"], K["fy"], K["cx"], K["cy"]


def project(points, fx, fy, cx, cy):
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return torch.stack([u, v], dim=1)


def main():
    # ---------- Load data ----------
    with open(TRANS_IN) as f:
        translations = json.load(f)

    fx, fy, cx, cy = load_intrinsics(INTR_PATH)

    mano = smplx.create(
        model_path=str(MANO_MODEL_DIR),
        model_type="mano",
        side="right",
        use_pca=False,
        flat_hand_mean=True
    ).to(DEVICE)

    betas = torch.zeros(1, 10, device=DEVICE)
    hand_pose = torch.zeros(1, 45, device=DEVICE)
    global_orient = torch.zeros(1, 3, device=DEVICE)

    refined = {}

    # ---------- Optimize frame-by-frame ----------
    for frame_id, T in translations.items():
        if T is None:
            refined[frame_id] = None
            continue

        print(f"\nOptimizing frame {frame_id}")

        depth = np.load(DEPTH_DIR / f"{frame_id}.npy")
        mask  = cv2.imread(str(MASK_DIR / f"{frame_id}.png"), 0) > 0

        depth = torch.tensor(depth, dtype=torch.float32, device=DEVICE)
        mask  = torch.tensor(mask, device=DEVICE)

        transl = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        transl.requires_grad_(True)

        optimizer = torch.optim.Adam([transl], lr=LR)

        for it in range(ITERS):
            optimizer.zero_grad()

            out = mano(
                betas=betas,
                hand_pose=hand_pose,
                global_orient=global_orient,
                transl=transl
            )

            verts = out.vertices[0]   # (778,3)

            uv = project(verts, fx, fy, cx, cy).long()

            valid = (
                (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) &
                (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            )

            uv = uv[valid]
            z_pred = verts[valid, 2]
            z_obs  = depth[uv[:, 1], uv[:, 0]]
            m      = mask[uv[:, 1], uv[:, 0]]

            loss = ((z_pred - z_obs) ** 2 * m).mean()

            loss.backward()
            optimizer.step()

            if it % 20 == 0:
                print(f"  iter {it:03d} | depth loss {loss.item():.6f}")

        refined[frame_id] = transl.detach().cpu().numpy().flatten().tolist()

    with open(TRANS_OUT, "w") as f:
        json.dump(refined, f, indent=2)

    print("\n✅ STEP 2.2 completed — refined translations saved.")


if __name__ == "__main__":
    main()
