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

POSE_IN  = DATA_ROOT / SESSION / "hand_pose_rgb_depth.json"
TRANS_IN = DATA_ROOT / SESSION / "translations_rgb_depth.json"

POSE_OUT  = DATA_ROOT / SESSION / "hand_pose_final.json"
TRANS_OUT = DATA_ROOT / SESSION / "translations_final.json"
BETA_OUT  = DATA_ROOT / SESSION / "shape_beta.json"

INTR_PATH = DATA_ROOT / SESSION / "intrinsics.json"
MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
# =============================================

DEVICE = "cpu"
LR = 1e-3
ITERS = 80
LAMBDA_DEPTH = 1e-3
MU_BETA = 1e-2

MP_TO_MANO = [0,5,6,7, 9,10,11, 13,14,15, 17,18,19, 1,2,3]


def load_intrinsics(path):
    with open(path) as f:
        K = json.load(f)
    return K["fx"], K["fy"], K["cx"], K["cy"]


def project(points, fx, fy, cx, cy):
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return torch.stack([u, v], dim=1)


def main():
    pose_dict = json.load(open(POSE_IN))
    trans_dict = json.load(open(TRANS_IN))
    fx, fy, cx, cy = load_intrinsics(INTR_PATH)

    mano = smplx.create(
        model_path=str(MANO_MODEL_DIR),
        model_type="mano",
        side="right",
        use_pca=False,
        flat_hand_mean=True
    ).to(DEVICE)

    # -------- shared shape --------
    betas = torch.zeros(1, 10, device=DEVICE, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=DEVICE)

    params = [betas]
    per_frame = {}

    for fid, pose in pose_dict.items():
        if pose is None:
            continue
        per_frame[fid] = {
            "pose": torch.tensor(pose, device=DEVICE).view(1, -1),
            "trans": torch.tensor(trans_dict[fid], device=DEVICE).view(1, 3)
        }
        per_frame[fid]["pose"].requires_grad_(True)
        per_frame[fid]["trans"].requires_grad_(True)
        params += [per_frame[fid]["pose"], per_frame[fid]["trans"]]

    optim = torch.optim.Adam(params, lr=LR)

    for it in range(ITERS):
        optim.zero_grad()
        total_loss = 0.0

        for fid, data in per_frame.items():
            depth = torch.tensor(
                np.load(DEPTH_DIR / f"{fid}.npy"),
                device=DEVICE, dtype=torch.float32
            )
            mask = torch.tensor(
                cv2.imread(str(MASK_DIR / f"{fid}.png"), 0) > 0,
                device=DEVICE
            )

            joints2d = json.load(open(JOINT2D_DIR / f"{fid}.json"))["joints"]
            if joints2d is None:
                continue

            joints2d = torch.tensor(np.array(joints2d)[MP_TO_MANO],
                                    device=DEVICE)

            out = mano(
                betas=betas,
                hand_pose=data["pose"],
                transl=data["trans"],
                global_orient=global_orient
            )

            verts = out.vertices[0]
            joints3d = out.joints[0]

            # RGB reprojection
            proj = project(joints3d, fx, fy, cx, cy)
            reproj = ((proj - joints2d) ** 2).mean()

            # Depth loss
            uv = project(verts, fx, fy, cx, cy).long()
            valid = (
                (uv[:,0] >= 0) & (uv[:,0] < depth.shape[1]) &
                (uv[:,1] >= 0) & (uv[:,1] < depth.shape[0])
            )
            uv = uv[valid]
            z_pred = verts[valid, 2]
            z_obs = depth[uv[:,1], uv[:,0]]
            m = mask[uv[:,1], uv[:,0]]

            depth_loss = ((z_pred - z_obs) ** 2 * m).mean()

            total_loss += reproj + LAMBDA_DEPTH * depth_loss

        total_loss += MU_BETA * (betas ** 2).mean()
        total_loss.backward()
        optim.step()

        if it % 20 == 0:
            print(f"iter {it:03d} | total loss {total_loss.item():.4f}")

    # -------- save outputs --------
    json.dump(
        {k: v["pose"].detach().cpu().numpy().flatten().tolist()
         for k, v in per_frame.items()},
        open(POSE_OUT, "w"), indent=2
    )

    json.dump(
        {k: v["trans"].detach().cpu().numpy().flatten().tolist()
         for k, v in per_frame.items()},
        open(TRANS_OUT, "w"), indent=2
    )

    json.dump(
        betas.detach().cpu().numpy().flatten().tolist(),
        open(BETA_OUT, "w"), indent=2
    )

    print("\n✅ STEP 2.5 completed — shape β optimized.")


if __name__ == "__main__":
    main()
