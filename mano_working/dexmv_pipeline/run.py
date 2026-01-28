import json
import torch
import smplx
import numpy as np
import cv2
from pathlib import Path

# ================= CONFIG =================
SESSION = "session_001"
DEVICE = "cpu"

STAGE_CFG = [
    # (vars, iters, lr)
    ("trans",  50, 1e-2),
    ("orient", 80, 5e-3),
    ("pose",  150, 1e-3),
    ("shape",  80, 1e-4),
    ("joint",  80, 5e-4),
]
# ========================================

# ================= PATHS =================
ROOT = Path(".")
DATA = ROOT / "dexmv_data" / SESSION

RGB_DIR   = DATA / "rgb"
DEPTH_DIR = DATA / "depth"
MASK_DIR  = DATA / "mask"
J2D_DIR   = DATA / "joints2d"

OUT_DIR = DATA / "output"
OUT_DIR.mkdir(exist_ok=True)

INTR = json.load(open(DATA / "intrinsics.json"))

MANO_DIR = ROOT / "mano" / "MANO" / "models"
# ========================================


def load_joints2d(path):
    raw = json.load(open(path))
    if isinstance(raw, dict) and "joints" in raw:
        raw = raw["joints"]
    if isinstance(raw, dict):
        raw = [raw[k] for k in sorted(raw.keys(), key=int)]
    j = torch.tensor(raw, dtype=torch.float32)
    return j[:16]   # MANO uses 16 joints


def project(points, fx, fy, cx, cy):
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return torch.stack([u, v], dim=1)


class DexMVHandSystem:
    def __init__(self, mano, intr):
        self.mano = mano
        self.fx, self.fy, self.cx, self.cy = (
            intr["fx"], intr["fy"], intr["cx"], intr["cy"]
        )

        # parameters
        self.r = torch.zeros(1, 3, requires_grad=True)
        self.R = torch.zeros(1, 3, requires_grad=True)
        self.theta = torch.zeros(1, 45, requires_grad=True)
        self.beta = torch.zeros(1, 10, requires_grad=True)

    def forward(self):
        return self.mano(
            global_orient=self.R,
            hand_pose=self.theta,
            transl=self.r,
            betas=self.beta
        )

    def reprojection_loss(self, j2d):
        j3d = self.forward().joints[0]
        proj = project(j3d, self.fx, self.fy, self.cx, self.cy)
        return ((proj - j2d) ** 2).mean()

    def depth_loss(self, depth, mask):
        verts = self.forward().vertices[0]
        Z = verts[:, 2]
        return ((Z.mean() - depth[mask > 0].mean()) ** 2)

    def step(self, params, loss_fn, iters, lr):
        for p in [self.r, self.R, self.theta, self.beta]:
            p.requires_grad = False
        for p in params:
            p.requires_grad = True

        optim = torch.optim.Adam(params, lr=lr)

        for _ in range(iters):
            optim.zero_grad()
            loss = loss_fn()
            loss.backward()
            optim.step()


def main():
    mano = smplx.create(
        model_path=str(MANO_DIR),
        model_type="mano",
        side="right",
        use_pca=False,
        flat_hand_mean=True
    ).to(DEVICE)

    frames = sorted(RGB_DIR.glob("*.png"))

    for rgb_path in frames:
        fid = rgb_path.stem
        print(f"\n=== Processing frame {fid} ===")

        depth = np.load(DEPTH_DIR / f"{fid}.npy")
        mask = cv2.imread(str(MASK_DIR / f"{fid}.png"), 0) > 0
        j2d = load_joints2d(J2D_DIR / f"{fid}.json").to(DEVICE)

        depth = torch.tensor(depth, dtype=torch.float32)
        mask = torch.tensor(mask)

        system = DexMVHandSystem(mano, INTR)

        # --- staged optimization ---
        for stage, iters, lr in STAGE_CFG:
            if stage == "trans":
                system.step(
                    [system.r],
                    lambda: system.depth_loss(depth, mask),
                    iters, lr
                )
            elif stage == "orient":
                system.step(
                    [system.r, system.R],
                    lambda: system.reprojection_loss(j2d),
                    iters, lr
                )
            elif stage == "pose":
                system.step(
                    [system.theta],
                    lambda: system.reprojection_loss(j2d),
                    iters, lr
                )
            elif stage == "shape":
                system.step(
                    [system.beta],
                    lambda: system.reprojection_loss(j2d),
                    iters, lr
                )
            elif stage == "joint":
                system.step(
                    [system.r, system.R, system.theta, system.beta],
                    lambda: system.reprojection_loss(j2d),
                    iters, lr
                )

        # --- final output ---
        with torch.no_grad():
            out = system.forward()

        np.save(OUT_DIR / f"{fid}_verts.npy", out.vertices[0].cpu().numpy())
        np.save(OUT_DIR / f"{fid}_joints.npy", out.joints[0].cpu().numpy())

        params = {
            "r": system.r.detach().cpu().numpy().tolist()[0],
            "R": system.R.detach().cpu().numpy().tolist()[0],
            "theta": system.theta.detach().cpu().numpy().tolist()[0],
            "beta": system.beta.detach().cpu().numpy().tolist()[0],
        }

        json.dump(params, open(OUT_DIR / f"{fid}_params.json", "w"), indent=2)

        print(f"✓ Frame {fid} done")

    print("\n✅ SINGLE-SYSTEM DexMV pipeline finished.")


if __name__ == "__main__":
    main()
