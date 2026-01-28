import json
import torch
import smplx
from pathlib import Path

# ================= PATH SETUP =================
PROJECT_ROOT = Path(".")
DATA_ROOT = PROJECT_ROOT / "dexmv_data" / "session_001"

POSE_FILE = DATA_ROOT / "hand_pose_rgb_depth.json"      # output of STEP 2.4
TRANS_FILE = DATA_ROOT / "translations_smoothed.json"   # STEP 1.5
JOINT2D_DIR = DATA_ROOT / "joints2d"
INTR_FILE = DATA_ROOT / "intrinsics.json"

OUT_FILE = DATA_ROOT / "global_orient.json"

MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
# ==============================================

DEVICE = "cpu"
ITERS = 120
LR = 5e-3


# ------------------ utils ------------------

def project(points, fx, fy, cx, cy):
    """
    points: (N,3)
    return: (N,2)
    """
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return torch.stack([u, v], dim=1)


def load_joints2d(path):
    """
    Robust loader for ALL joints2d formats you have seen so far.

    Supported:
      - {"joints": [[x,y], ...]}
      - {"joints": {"0": [x,y], ...}}
      - {"0": [x,y], "1": [x,y], ...}
      - [[x,y], ...]

    Returns:
      Tensor (16,2)  -- MANO joint count
    """
    raw = json.load(open(path))

    # unwrap "joints" key if present
    if isinstance(raw, dict) and "joints" in raw:
        raw = raw["joints"]

    # dict case
    if isinstance(raw, dict):
        numeric_keys = [k for k in raw.keys() if k.isdigit()]
        joints = [raw[k] for k in sorted(numeric_keys, key=int)]

    # list case
    elif isinstance(raw, list):
        joints = raw

    else:
        raise ValueError(f"Unsupported joints2d format in {path}")

    joints = torch.tensor(joints, dtype=torch.float32)

    # MANO uses 16 joints
    if joints.shape[0] > 16:
        joints = joints[:16]

    if joints.shape[0] != 16:
        print(f"[WARN] {path.name}: expected 16 joints, got {joints.shape[0]}")

    return joints


# ------------------ main ------------------

def main():
    pose = json.load(open(POSE_FILE))
    trans = json.load(open(TRANS_FILE))
    intr = json.load(open(INTR_FILE))

    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

    mano = smplx.create(
        model_path=str(MANO_MODEL_DIR),
        model_type="mano",
        side="right",
        use_pca=False,
        flat_hand_mean=True
    ).to(DEVICE)

    global_orient_out = {}

    for frame_id in pose.keys():

        if pose[frame_id] is None:
            global_orient_out[frame_id] = None
            continue

        joints2d_path = JOINT2D_DIR / f"{frame_id}.json"
        if not joints2d_path.exists():
            global_orient_out[frame_id] = None
            continue

        print(f"\n[STEP 2.4.5] Optimizing global orientation for {frame_id}")

        j2d = load_joints2d(joints2d_path).to(DEVICE)

        hand_pose = torch.tensor(
            pose[frame_id], dtype=torch.float32
        ).view(1, -1).to(DEVICE)

        transl = torch.tensor(
            trans[frame_id], dtype=torch.float32
        ).view(1, 3).to(DEVICE)

        global_orient = torch.zeros(
            1, 3, device=DEVICE, requires_grad=True
        )

        optimizer = torch.optim.Adam([global_orient], lr=LR)

        for i in range(ITERS):
            optimizer.zero_grad()

            out = mano(
                global_orient=global_orient,
                hand_pose=hand_pose,
                transl=transl
            )

            joints3d = out.joints[0]   # (16,3)
            proj = project(joints3d, fx, fy, cx, cy)

            loss = ((proj - j2d) ** 2).mean()
            loss.backward()
            optimizer.step()

            if i % 30 == 0:
                print(f" iter {i:03d} | reproj {loss.item():.2f}")

        global_orient_out[frame_id] = (
            global_orient.detach().cpu().numpy().tolist()[0]
        )

    json.dump(global_orient_out, open(OUT_FILE, "w"), indent=2)
    print("\n✅ STEP 2.4.5 completed — global orientation optimized.")


if __name__ == "__main__":
    main()
