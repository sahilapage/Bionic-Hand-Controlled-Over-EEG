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
JOINT2D_DIR = DATA_ROOT / SESSION / "joints2d"

POSE_IN  = DATA_ROOT / SESSION / "hand_pose_depth.json"
TRANS_IN = DATA_ROOT / SESSION / "translations_pose_refined.json"

POSE_OUT  = DATA_ROOT / SESSION / "hand_pose_rgb_depth.json"
TRANS_OUT = DATA_ROOT / SESSION / "translations_rgb_depth.json"

INTR_PATH = DATA_ROOT / SESSION / "intrinsics.json"
MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
# =============================================

DEVICE = "cpu"
LR = 2e-3
ITERS = 120
LAMBDA_DEPTH = 1e-3

# MediaPipe → MANO joint mapping
MP_TO_MANO = [
    0, 5, 6, 7,
    9, 10, 11,
    13, 14, 15,
    17, 18, 19,
    1, 2, 3
]


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
    with open(POSE_IN) as f:
        pose_dict = json.load(f)

    with open(TRANS_IN) as f:
        trans_dict = json.load(f)

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

    out_pose = {}
    out_trans = {}

    for frame_id, pose in pose_dict.items():
        if pose is None:
            out_pose[frame_id] = None
            out_trans[frame_id] = None
            continue

        print(f"\n[STEP 2.4] Optimizing frame {frame_id}")

        depth = torch.tensor(
            np.load(DEPTH_DIR / f"{frame_id}.npy"),
            dtype=torch.float32,
            device=DEVICE
        )

        mask = torch.tensor(
            cv2.imread(str(MASK_DIR / f"{frame_id}.png"), 0) > 0,
            device=DEVICE
        )

        with open(JOINT2D_DIR / f"{frame_id}.json") as f:
            joints2d = np.array(json.load(f)["joints"], dtype=np.float32)

        joints2d = joints2d[MP_TO_MANO]
        joints2d = torch.tensor(joints2d, device=DEVICE)

        hand_pose = torch.tensor(pose, device=DEVICE).view(1, -1)
        hand_pose.requires_grad_(True)

        transl = torch.tensor(trans_dict[frame_id], device=DEVICE).view(1, 3)
        transl.requires_grad_(True)

        optimizer = torch.optim.Adam([hand_pose, transl], lr=LR)

        for it in range(ITERS):
            optimizer.zero_grad()

            out = mano(
                betas=betas,
                hand_pose=hand_pose,
                global_orient=global_orient,
                transl=transl
            )

            verts = out.vertices[0]
            joints3d = out.joints[0]

            # ---------- 2D reprojection loss ----------
            proj_joints = project(joints3d, fx, fy, cx, cy)
            reproj_loss = ((proj_joints - joints2d) ** 2).mean()

            # ---------- depth loss ----------
            uv = project(verts, fx, fy, cx, cy).long()

            valid = (
                (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) &
                (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
            )

            uv = uv[valid]
            z_pred = verts[valid, 2]
            z_obs = depth[uv[:, 1], uv[:, 0]]
            m = mask[uv[:, 1], uv[:, 0]]

            depth_loss = ((z_pred - z_obs) ** 2 * m).mean()

            loss = reproj_loss + LAMBDA_DEPTH * depth_loss
            loss.backward()
            optimizer.step()

            if it % 30 == 0:
                print(
                    f" iter {it:03d} | "
                    f"reproj {reproj_loss.item():.4f} | "
                    f"depth {depth_loss.item():.4f}"
                )

        out_pose[frame_id] = hand_pose.detach().cpu().numpy().flatten().tolist()
        out_trans[frame_id] = transl.detach().cpu().numpy().flatten().tolist()

    with open(POSE_OUT, "w") as f:
        json.dump(out_pose, f, indent=2)

    with open(TRANS_OUT, "w") as f:
        json.dump(out_trans, f, indent=2)

    print("\n✅ STEP 2.4 completed — RGB + depth optimization done.")


if __name__ == "__main__":
    main()
