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

TRANS_IN  = DATA_ROOT / SESSION / "translations_refined.json"
POSE_OUT  = DATA_ROOT / SESSION / "hand_pose_depth.json"
TRANS_OUT = DATA_ROOT / SESSION / "translations_pose_refined.json"
INTR_PATH = DATA_ROOT / SESSION / "intrinsics.json"

MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
# =============================================

DEVICE = "cpu"
LR = 3e-3
ITERS = 150


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
    global_orient = torch.zeros(1, 3, device=DEVICE)

    refined_trans = {}
    refined_pose = {}

    for frame_id, T in translations.items():
        if T is None:
            refined_trans[frame_id] = None
            refined_pose[frame_id] = None
            continue

        print(f"\nOptimizing pose for frame {frame_id}")

        depth = np.load(DEPTH_DIR / f"{frame_id}.npy")
        mask  = cv2.imread(str(MASK_DIR / f"{frame_id}.png"), 0) > 0

        depth = torch.tensor(depth, dtype=torch.float32, device=DEVICE)
        mask  = torch.tensor(mask, device=DEVICE)

        transl = torch.tensor(T, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        transl.requires_grad_(True)

        hand_pose = torch.zeros(1, 45, device=DEVICE, requires_grad=True)

        optimizer = torch.optim.Adam([transl, hand_pose], lr=LR)

        for it in range(ITERS):
            optimizer.zero_grad()

            out = mano(
                betas=betas,
                hand_pose=hand_pose,
                global_orient=global_orient,
                transl=transl
            )

            verts = out.vertices[0]

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

            if it % 30 == 0:
                print(f"  iter {it:03d} | depth loss {loss.item():.6f}")

        refined_trans[frame_id] = transl.detach().cpu().numpy().flatten().tolist()
        refined_pose[frame_id] = hand_pose.detach().cpu().numpy().flatten().tolist()

    with open(TRANS_OUT, "w") as f:
        json.dump(refined_trans, f, indent=2)

    with open(POSE_OUT, "w") as f:
        json.dump(refined_pose, f, indent=2)

    print("\n✅ STEP 2.3 completed — pose + translation refined.")


if __name__ == "__main__":
    main()
