"""
Human Hand TSV → Amazing Hand (Pollen Robotics) Retargeting + MuJoCo Simulation
================================================================================
Pipeline (inspired by DexMV & Robotic Telekinesis papers):
  1. Load TSV files (5 fingertips × 3 xyz, palm-relative vectors)
  2. Drop pinky; map remaining 4 fingers to robot tip sites:
       Human Index  → tip1
       Human Middle → tip2
       Human Ring   → tip3
       Human Thumb  → tip4
  3. Retarget using a Jacobian-based IK approach
  4. Drive the 8 MuJoCo actuators of the Amazing Hand and launch viewer.

Amazing Hand actuator layout (from the MJCF):
  finger1_motor1, finger1_motor2  →  tip1  (index)
  finger2_motor1, finger2_motor2  →  tip2  (middle)
  finger3_motor1, finger3_motor2  →  tip3  (ring)
  finger4_motor1, finger4_motor2  →  tip4  (thumb)

== Root-cause analysis of "motors don't move" bug ==

The original code failed because of a coordinate frame mismatch:
  - Human HAMER frame:   fingers point in   -Y  from palm
  - Robot world frame:   fingers point in   +Z  from palm
  Without remapping, the optimizer tried to push tips to completely
  unreachable targets (energy landscape flat at the joint bounds).

The fix has two parts:
  1. Coordinate remap:  human(X, Y, Z) → robot(X, Z, -Y)
  2. Jacobian-based IK: auto-calibrate per-motor tip sensitivity at startup,
     then solve analytically per finger (2-motor 3-eq least-squares).
     This replaces scipy.minimize which was sensitive to scale/frame issues.

Usage:
    python tsv_retarget_sim.py --tsv_folder demo_out --model_path mjcf/scene.xml
    python tsv_retarget_sim.py --tsv_file demo_out/000070_0_tsv.npy --model_path mjcf/scene.xml

Dependencies:
    pip install numpy scipy mujoco
"""

import argparse
import time
import os
import glob
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# TSV finger order from HAMER: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
HUMAN_FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# human finger index → robot tip site name (pinky dropped)
HUMAN_TO_ROBOT = {1: "tip1", 2: "tip2", 3: "tip3", 0: "tip4"}

ACTUATOR_NAMES = [
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
]

# qpos indices for the 8 actuated joints (verified from MJCF joint tree)
MOTOR_QPOS_ADDR = [0, 12, 17, 29, 34, 46, 51, 63]

CTRL_LOW  = -np.pi / 2
CTRL_HIGH =  np.pi / 2

# Calibration: each motor is swept to this value to measure tip sensitivity
CALIB_ANGLE = 0.5   # rad — small enough to stay in linear regime
CALIB_STEPS = 300   # physics steps to let passive 4-bar linkage settle

# Simulation
SIM_ALPHA = 0.08    # EMA smoothing factor for ctrl interpolation

# ─────────────────────────────────────────────────────────────────────────────
# Coordinate remapping
# ─────────────────────────────────────────────────────────────────────────────

def remap_human_to_robot(v: np.ndarray) -> np.ndarray:
    """
    Remap a human HAMER palm-relative fingertip vector to robot world frame.

    Human HAMER frame : fingers point in -Y from palm
    Robot world frame : fingers point in +Z from palm

    Mapping:  human(X, Y, Z)  →  robot(X, Z, -Y)
    i.e.      robot_X = human_X   (lateral spread — same axis)
              robot_Y = human_Z   (depth/anterior)
              robot_Z = -human_Y  (extension — main finger direction)
    """
    return np.array([v[0], v[2], -v[1]], dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
# Physics settle helper
# ─────────────────────────────────────────────────────────────────────────────

def settle(model: mujoco.MjModel, data: mujoco.MjData,
           ctrl: np.ndarray, steps: int = CALIB_STEPS) -> None:
    """Reset, set ctrl, run `steps` physics steps so passive joints settle."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = np.clip(ctrl, CTRL_LOW, CTRL_HIGH)
    for _ in range(steps):
        mujoco.mj_step(model, data)


def get_tip_vectors(model: mujoco.MjModel, data: mujoco.MjData,
                    palm_id: int, site_ids: list) -> np.ndarray:
    """Return (4,3) array of palm-relative tip vectors from current data."""
    palm = data.xpos[palm_id].copy()
    return np.array([data.site_xpos[s] - palm for s in site_ids])

# ─────────────────────────────────────────────────────────────────────────────
# Auto-calibration: build Jacobian at startup
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(model: mujoco.MjModel, data: mujoco.MjData,
              palm_id: int, site_ids: list) -> tuple:
    """
    Sweep each of the 8 motors to ±CALIB_ANGLE, settle physics,
    record tip sensitivity (Δtip / Δangle) for its own finger.

    Returns:
        tips0  : (4,3) baseline tip vectors at ctrl=0
        jac    : dict  tip_idx → {"m1": (3,), "m2": (3,)} sensitivity per radian
    """
    print("[CALIB] Running FK calibration (measuring motor sensitivities)…")

    # Baseline at all-zero ctrl
    settle(model, data, np.zeros(8))
    tips0 = get_tip_vectors(model, data, palm_id, site_ids)  # (4,3)
    print(f"[CALIB] Baseline tip lengths: "
          f"{np.linalg.norm(tips0, axis=1).round(4)}")

    # Each finger: motor1 = ctrl[2i], motor2 = ctrl[2i+1], tip = tips[i]
    jac = {}
    for fi in range(4):
        ctrl_m1 = np.zeros(8)
        ctrl_m1[fi * 2] = CALIB_ANGLE
        settle(model, data, ctrl_m1)
        tips_m1 = get_tip_vectors(model, data, palm_id, site_ids)
        sens_m1 = (tips_m1[fi] - tips0[fi]) / CALIB_ANGLE

        ctrl_m2 = np.zeros(8)
        ctrl_m2[fi * 2 + 1] = CALIB_ANGLE
        settle(model, data, ctrl_m2)
        tips_m2 = get_tip_vectors(model, data, palm_id, site_ids)
        sens_m2 = (tips_m2[fi] - tips0[fi]) / CALIB_ANGLE

        jac[fi] = {"m1": sens_m1, "m2": sens_m2}
        print(f"[CALIB]   finger{fi+1}: m1 sens={sens_m1.round(4)}  "
              f"m2 sens={sens_m2.round(4)}")

    # Restore baseline
    settle(model, data, np.zeros(8))
    return tips0, jac

# ─────────────────────────────────────────────────────────────────────────────
# Per-finger Jacobian IK
# ─────────────────────────────────────────────────────────────────────────────

def solve_finger_ik(human_v: np.ndarray,
                    tip0: np.ndarray,
                    sens_m1: np.ndarray,
                    sens_m2: np.ndarray) -> tuple:
    """
    Solve for (q_m1, q_m2) such that:
        tip0 + q_m1 * sens_m1 + q_m2 * sens_m2  ≈  target

    where target = human_v (remapped, scaled to robot finger length).

    Returns: (q_m1, q_m2) clamped to [CTRL_LOW, CTRL_HIGH]
    """
    # Scale human direction to robot finger length
    robot_len = float(np.linalg.norm(tip0))
    human_len = float(np.linalg.norm(human_v))
    if human_len < 1e-9:
        return 0.0, 0.0

    target = human_v / human_len * robot_len   # direction from human, length from robot
    delta  = target - tip0                     # desired Δtip

    # Least-squares: [sens_m1 | sens_m2] * [q1, q2]^T = delta
    A = np.column_stack([sens_m1, sens_m2])   # (3, 2)
    q, _, _, _ = np.linalg.lstsq(A, delta, rcond=None)

    q_m1 = float(np.clip(q[0], CTRL_LOW, CTRL_HIGH))
    q_m2 = float(np.clip(q[1], CTRL_LOW, CTRL_HIGH))
    return q_m1, q_m2

# ─────────────────────────────────────────────────────────────────────────────
# TSV → robot ctrl
# ─────────────────────────────────────────────────────────────────────────────

def retarget_tsv(tsv: np.ndarray,
                 tips0: np.ndarray,
                 jac: dict) -> np.ndarray:
    """
    Convert a (5,3) HAMER TSV to 8 robot ctrl values.

    Finger order in tsv: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
    Finger order in robot: tip1=Index, tip2=Middle, tip3=Ring, tip4=Thumb

    Steps:
      1. Drop pinky, reorder to match robot tip order
      2. Remap each vector: human(X,Y,Z) → robot(X,Z,-Y)
      3. Solve per-finger 2-motor Jacobian IK
    """
    assert tsv.shape == (5, 3)

    # Robot tip order: index(1), middle(2), ring(3), thumb(0)
    human_order = [1, 2, 3, 0]
    ctrl = np.zeros(8)

    for fi, human_idx in enumerate(human_order):
        v_human_remapped = remap_human_to_robot(tsv[human_idx])
        q_m1, q_m2 = solve_finger_ik(
            v_human_remapped, tips0[fi],
            jac[fi]["m1"], jac[fi]["m2"]
        )
        ctrl[fi * 2]     = q_m1
        ctrl[fi * 2 + 1] = q_m2

    return ctrl

# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_tsv_info(tsv: np.ndarray, label: str = ""):
    w = 60
    print(f"\n{'='*w}")
    if label:
        print(f"File: {label}")
    print(f"TSV shape: {tsv.shape}  (5 fingertips × 3 xyz)")
    print(f"{'='*w}")
    print(f"{'Finger':<14} {'X':>10} {'Y':>10} {'Z':>10} {'Length':>10}")
    print(f"{'-'*w}")
    for i, name in enumerate(HUMAN_FINGER_NAMES):
        v = tsv[i]
        suffix = "  ← dropped" if i == 4 else ""
        print(f"{name:<14} {v[0]:>10.4f} {v[1]:>10.4f} {v[2]:>10.4f} "
              f"{np.linalg.norm(v):>10.4f}{suffix}")
    print(f"{'='*w}")


def print_ctrl(ctrl: np.ndarray):
    print(f"\n{'Actuator':<24} {'Value (rad)':>12}")
    print("-" * 38)
    for name, val in zip(ACTUATOR_NAMES, ctrl):
        bar = "█" * int(abs(val) / CTRL_HIGH * 10)
        sign = "+" if val >= 0 else "-"
        print(f"{name:<24} {val:>+.4f}  {sign}{bar}")

# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(tsv_files: list, model_path: str, hold_time: float = 2.0):
    print(f"\n[INFO] Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    # Resolve IDs
    palm_id  = model.body("r_wrist_interface").id
    site_ids = [model.site(n).id for n in ("tip1", "tip2", "tip3", "tip4")]
    print(f"[INFO] Found palm body id={palm_id}, tip site ids={site_ids}")

    # Verify actuators
    for name in ACTUATOR_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) == -1:
            raise RuntimeError(f"Actuator '{name}' not found in model.")
    print(f"[INFO] All 8 actuators found.\n")

    # ── Auto-calibrate Jacobian ───────────────────────────────────────────────
    tips0, jac = calibrate(model, data, palm_id, site_ids)

    # ── Retarget all TSV files ────────────────────────────────────────────────
    print(f"\n[INFO] Retargeting {len(tsv_files)} TSV file(s)…")
    retargeted = []
    for fpath in tsv_files:
        tsv = np.load(fpath, allow_pickle=True)
        if isinstance(tsv, np.ndarray) and tsv.dtype == object:
            tsv = tsv.item()
        tsv = np.array(tsv, dtype=np.float64)
        print_tsv_info(tsv, label=os.path.basename(fpath))

        ctrl = retarget_tsv(tsv, tips0, jac)
        print_ctrl(ctrl)
        retargeted.append((fpath, ctrl))

    # ── Launch viewer ─────────────────────────────────────────────────────────
    print(f"\n[INFO] Launching MuJoCo viewer (hold_time={hold_time}s per pose)…")
    print("[INFO] Close the window to exit.\n")

    steps_per_hold = max(1, int(hold_time / model.opt.timestep))
    n_poses = len(retargeted)
    pose_idx   = 0
    step_count = 0
    current_ctrl = np.zeros(8)
    target_ctrl  = retargeted[0][1].copy()

    # Start from settled baseline
    settle(model, data, np.zeros(8))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Smooth EMA interpolation toward target
            current_ctrl = (1 - SIM_ALPHA) * current_ctrl + SIM_ALPHA * target_ctrl
            data.ctrl[:] = current_ctrl

            for _ in range(5):
                mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1

            if step_count >= steps_per_hold:
                step_count = 0
                pose_idx   = (pose_idx + 1) % n_poses
                target_ctrl = retargeted[pose_idx][1].copy()
                fname = os.path.basename(retargeted[pose_idx][0])
                print(f"[SIM] → Pose {pose_idx+1}/{n_poses}: {fname}")

            time.sleep(model.opt.timestep)

    print("\n[INFO] Done.")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Human TSV → Amazing Hand retargeting + MuJoCo simulation"
    )
    parser.add_argument("--model_path", type=str, default="mjcf/scene.xml")
    parser.add_argument("--tsv_file",   type=str, default=None)
    parser.add_argument("--tsv_folder", type=str, default=None)
    parser.add_argument("--hold_time",  type=float, default=2.0)
    args = parser.parse_args()

    tsv_files = []
    if args.tsv_file:
        tsv_files = [args.tsv_file]
    elif args.tsv_folder:
        tsv_files = sorted(glob.glob(os.path.join(args.tsv_folder, "*_tsv.npy")))
        if not tsv_files:
            raise FileNotFoundError(f"No *_tsv.npy in: {args.tsv_folder}")
    else:
        # Demo mode
        import tempfile
        print("[INFO] Demo mode — using sample TSVs.\n")
        samples = {
            "000070_0_tsv.npy": np.array([
                [-0.0353, -0.0869, -0.0025],
                [ 0.0103, -0.0698,  0.0131],
                [ 0.0002, -0.0593,  0.0118],
                [-0.0121, -0.0395,  0.0133],
                [-0.0291, -0.0321,  0.0076],
            ], dtype=np.float64),
            "000095_0_tsv.npy": np.array([
                [-0.0545, -0.0664, -0.0392],
                [-0.0034, -0.1708, -0.0046],
                [-0.0626, -0.1626,  0.0106],
                [-0.0147, -0.0412, -0.0171],
                [-0.0272, -0.0351, -0.0158],
            ], dtype=np.float64),
        }
        tmp = tempfile.mkdtemp(prefix="tsv_demo_")
        for name, arr in samples.items():
            p = os.path.join(tmp, name)
            np.save(p, arr)
            tsv_files.append(p)

    if not os.path.isfile(args.model_path):
        cands = list(Path(".").rglob("scene.xml"))
        hint  = ("\nFound:\n  " + "\n  ".join(str(c) for c in cands[:5])) if cands else ""
        raise FileNotFoundError(f"Model not found: {args.model_path}{hint}")

    run_simulation(tsv_files, args.model_path, hold_time=args.hold_time)


if __name__ == "__main__":
    main()