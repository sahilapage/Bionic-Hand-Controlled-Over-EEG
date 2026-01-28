import json
import torch
import smplx
import numpy as np
from pathlib import Path

# ================= PATH SETUP =================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_ROOT = PROJECT_ROOT / "dexmv_data"
SESSION = "session_001"

POSE_PATH  = DATA_ROOT / SESSION / "hand_pose_smoothed.json"
TRANS_PATH = DATA_ROOT / SESSION / "translations_smoothed.json"
BETA_PATH  = DATA_ROOT / SESSION / "shape_beta.json"

OUT_DIR = DATA_ROOT / SESSION / "final_hand"
OBJ_DIR = OUT_DIR / "meshes"

MANO_MODEL_DIR = PROJECT_ROOT / "mano" / "MANO" / "models"
# =============================================

DEVICE = "cpu"


def save_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(
                f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
            )


def main():
    pose = json.load(open(POSE_PATH))
    trans = json.load(open(TRANS_PATH))
    beta = torch.tensor(json.load(open(BETA_PATH)),
                        device=DEVICE).view(1, -1)

    OUT_DIR.mkdir(exist_ok=True)
    OBJ_DIR.mkdir(exist_ok=True)

    mano = smplx.create(
        model_path=str(MANO_MODEL_DIR),
        model_type="mano",
        side="right",
        use_pca=False,
        flat_hand_mean=True
    ).to(DEVICE)

    faces = mano.faces
    all_verts = []
    all_joints = []
    frame_ids = []

    for fid in sorted(pose.keys()):
        if pose[fid] is None:
            continue

        theta = torch.tensor(pose[fid], device=DEVICE).view(1, -1)
        r = torch.tensor(trans[fid], device=DEVICE).view(1, 3)

        with torch.no_grad():
            out = mano(
                hand_pose=theta,
                transl=r,
                betas=beta
            )

        verts = out.vertices[0].cpu().numpy()
        joints = out.joints[0].cpu().numpy()

        all_verts.append(verts)
        all_joints.append(joints)
        frame_ids.append(fid)

        save_obj(OBJ_DIR / f"{fid}.obj", verts, faces)

    np.save(OUT_DIR / "vertices.npy", np.stack(all_verts))
    np.save(OUT_DIR / "joints3d.npy", np.stack(all_joints))
    json.dump(frame_ids, open(OUT_DIR / "frames.json", "w"), indent=2)

    print("✅ STEP 2.7 completed — final hand meshes exported.")


if __name__ == "__main__":
    main()
