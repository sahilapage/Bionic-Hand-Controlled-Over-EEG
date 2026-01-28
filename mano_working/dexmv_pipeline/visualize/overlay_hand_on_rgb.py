import json
import torch
import smplx
import numpy as np
import cv2
from pathlib import Path

# ================= PATH SETUP =================
PROJECT_ROOT = Path(".")
DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

RGB_DIR = DATA_ROOT / SESSION / "rgb"

POSE = DATA_ROOT / SESSION / "hand_pose_rgb_depth.json"   # STEP 2.4 output
TRANS = DATA_ROOT / SESSION / "translations_smoothed.json"
GLOBAL_ORIENT = DATA_ROOT / SESSION / "global_orient.json"
BETA = DATA_ROOT / SESSION / "shape_beta.json"
INTR = DATA_ROOT / SESSION / "intrinsics.json"

MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
# =============================================

DEVICE = "cpu"


def project(points, fx, fy, cx, cy):
    """
    points: (N,3)
    returns: (N,2)
    """
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return np.stack([u, v], axis=1)


def main():
    pose = json.load(open(POSE))
    trans = json.load(open(TRANS))
    global_orient = json.load(open(GLOBAL_ORIENT))
    beta = torch.tensor(json.load(open(BETA)), dtype=torch.float32).view(1, -1)

    K = json.load(open(INTR))
    fx, fy, cx, cy = K["fx"], K["fy"], K["cx"], K["cy"]

    mano = smplx.create(
        model_path=str(MANO_MODEL_DIR),
        model_type="mano",
        side="right",
        use_pca=False,
        flat_hand_mean=True
    ).to(DEVICE)

    for fid in pose.keys():
        if pose[fid] is None:
            continue
        if fid not in global_orient or global_orient[fid] is None:
            continue

        img_path = RGB_DIR / f"{fid}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))

        hand_pose = torch.tensor(
            pose[fid], dtype=torch.float32
        ).view(1, -1).to(DEVICE)

        transl = torch.tensor(
            trans[fid], dtype=torch.float32
        ).view(1, 3).to(DEVICE)

        g_orient = torch.tensor(
            global_orient[fid], dtype=torch.float32
        ).view(1, 3).to(DEVICE)

        with torch.no_grad():
            out = mano(
                global_orient=g_orient,
                hand_pose=hand_pose,
                transl=transl,
                betas=beta
            )

        verts = out.vertices[0].cpu().numpy()
        uv = project(verts, fx, fy, cx, cy)

        for (u, v) in uv.astype(int):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, (0, 255, 0), -1)

        cv2.imshow("MANO Overlay (RGB + Depth + Orientation)", img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
