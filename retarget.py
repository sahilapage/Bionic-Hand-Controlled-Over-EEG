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
  3. Retarget using a keyvector-energy minimisation approach
     (DexPilot / Robotic-Telekinesis style) via scipy.optimize
  4. Drive the 8 MuJoCo actuators of the Amazing Hand to the
     retargeted joint angles and launch a passive viewer.

Amazing Hand actuator layout (from the MJCF):
  finger1_motor1, finger1_motor2  →  tip1  (index)
  finger2_motor1, finger2_motor2  →  tip2  (middle)
  finger3_motor1, finger3_motor2  →  tip3  (ring)
  finger4_motor1, finger4_motor2  →  tip4  (thumb)

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
from scipy.optimize import minimize
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants & mappings
# ─────────────────────────────────────────────────────────────────────────────

# TSV finger order from the HAMER extraction script
#   0 = Thumb, 1 = Index, 2 = Middle, 3 = Ring, 4 = Pinky
HUMAN_FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Mapping: human finger index → Amazing Hand tip site name
# Pinky (4) is dropped
HUMAN_TO_ROBOT = {
    1: "tip1",   # Index  → finger1
    2: "tip2",   # Middle → finger2
    3: "tip3",   # Ring   → finger3
    0: "tip4",   # Thumb  → finger4
}

# Actuator names in order (must match MJCF <actuator> block)
ACTUATOR_NAMES = [
    "finger1_motor1", "finger1_motor2",   # tip1 (index)
    "finger2_motor1", "finger2_motor2",   # tip2 (middle)
    "finger3_motor1", "finger3_motor2",   # tip3 (ring)
    "finger4_motor1", "finger4_motor2",   # tip4 (thumb)
]

# Joint actuator ranges (from MJCF): all ±π/2
CTRL_LOW  = -np.pi / 2   # -1.5708
CTRL_HIGH =  np.pi / 2   #  1.5708

# qpos indices for the 8 actuated motor joints (from working RL env)
# Order matches actuator order: f1m1, f1m2, f2m1, f2m2, f3m1, f3m2, f4m1, f4m2
MOTOR_QPOS_ADDR = [0, 12, 17, 29, 34, 46, 51, 63]

# qpos indices for the 8 actuated motor joints (from working RL env)
# Order: finger1_m1, finger1_m2, finger2_m1, finger2_m2,
#        finger3_m1, finger3_m2, finger4_m1, finger4_m2
MOTOR_QPOS_ADDR = [0, 12, 17, 29, 34, 46, 51, 63]

# Scaling constant for keyvector matching (Robotic Telekinesis §III-A)
# Accounts for size difference between human and robot hand.
# The Amazing Hand is roughly half the size of a human hand.
KV_SCALE = 0.5

# Physics steps to hold a pose for visualisation
SIM_HOLD_STEPS = 300   # ~1 second at 300 Hz default MuJoCo timestep

# ─────────────────────────────────────────────────────────────────────────────
# Forward kinematics helper (MuJoCo)
# ─────────────────────────────────────────────────────────────────────────────

def get_tip_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """
    Return the world-space 3-D positions of the four tip sites after
    calling mj_kinematics / mj_fwdPosition.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_fwdPosition(model, data)
    positions = {}
    for name in ["tip1", "tip2", "tip3", "tip4"]:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id == -1:
            raise RuntimeError(f"Site '{name}' not found in model. "
                               f"Check that scene.xml exposes these site names.")
        positions[name] = data.site_xpos[site_id].copy()
    return positions


def get_palm_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Approximate palm centre as the mean of the four tip base positions.
    We use the r_wrist_interface body origin as the palm/wrist reference.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_interface")
    if body_id == -1:
        # Fallback: mean of tip positions is a reasonable palm estimate
        tips = get_tip_positions(model, data)
        return np.mean(list(tips.values()), axis=0)
    # mj_forward already called by compute_robot_keyvectors before this runs
    return data.xpos[body_id].copy()

# ─────────────────────────────────────────────────────────────────────────────
# Energy function (DexPilot / Robotic Telekinesis style)
# ─────────────────────────────────────────────────────────────────────────────

def compute_robot_keyvectors(
    ctrl: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
) -> dict[str, np.ndarray]:
    """
    Set robot joint angles via qpos addresses (same as working RL env),
    run mj_forward, return palm-relative tip vectors. Restores state after.
    Returns dict: tip_name → 3-D vector from palm.
    """
    # Save state
    q_bak    = data.qpos.copy()
    ctrl_bak = data.ctrl.copy()

    # Set via qpos addresses (not ctrl[actuator_ids]) — this is the fix
    for i, addr in enumerate(MOTOR_QPOS_ADDR):
        data.qpos[addr] = np.clip(ctrl[i], CTRL_LOW, CTRL_HIGH)

    mujoco.mj_forward(model, data)

    palm = get_palm_position(model, data)
    tips = get_tip_positions(model, data)
    result = {name: tip - palm for name, tip in tips.items()}

    # Restore state
    data.qpos[:] = q_bak
    data.ctrl[:] = ctrl_bak
    mujoco.mj_forward(model, data)

    return result


def energy(
    ctrl: np.ndarray,
    human_kvs: dict[str, np.ndarray],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
    scale: float = KV_SCALE,
) -> float:
    """
    Keyvector energy (Eq. 1 of Robotic Telekinesis / DexPilot):

        E = Σ_i ‖ v_h_i  −  c · v_r_i ‖²

    where v_h is the human palm-relative tip vector and
    v_r is the robot palm-relative tip vector.

    A velocity-damping term penalises large joint changes from the
    current position (smoothness prior from DexMV demo translation).
    """
    robot_kvs = compute_robot_keyvectors(ctrl, model, data, actuator_ids)
    loss = 0.0
    for tip_name, v_human in human_kvs.items():
        v_robot = robot_kvs[tip_name]
        loss += np.sum((v_human - scale * v_robot) ** 2)
    # Smoothness regularisation (keeps joints near mid-range)
    loss += 1e-3 * np.sum(ctrl ** 2)
    return float(loss)


def energy_gradient(
    ctrl: np.ndarray,
    human_kvs: dict[str, np.ndarray],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
    scale: float = KV_SCALE,
    eps: float = 1e-4,
) -> np.ndarray:
    """Finite-difference gradient of the energy w.r.t. ctrl."""
    grad = np.zeros_like(ctrl)
    f0 = energy(ctrl, human_kvs, model, data, actuator_ids, scale)
    for i in range(len(ctrl)):
        ctrl_plus = ctrl.copy()
        ctrl_plus[i] += eps
        grad[i] = (energy(ctrl_plus, human_kvs, model, data, actuator_ids, scale) - f0) / eps
    return grad

# ─────────────────────────────────────────────────────────────────────────────
# TSV → human keyvectors
# ─────────────────────────────────────────────────────────────────────────────

def tsv_to_human_keyvectors(tsv: np.ndarray) -> dict[str, np.ndarray]:
    """
    Convert a (5,3) TSV array (palm-relative tip vectors, HAMER order:
    thumb=0, index=1, middle=2, ring=3, pinky=4) into a dict keyed by
    Amazing Hand tip site names, dropping pinky.

    The TSV vectors ARE already palm-relative (fingertip − palm),
    so they map directly to keyvectors.
    """
    assert tsv.shape == (5, 3), f"Expected TSV shape (5,3), got {tsv.shape}"
    human_kvs = {}
    for human_idx, tip_name in HUMAN_TO_ROBOT.items():
        human_kvs[tip_name] = tsv[human_idx].copy()
    return human_kvs

# ─────────────────────────────────────────────────────────────────────────────
# Retargeting solver
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_scale(
    human_kvs: dict[str, np.ndarray],
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> float:
    """
    Compute scale = robot_mean_tip_length / human_mean_tip_length.
    This adapts to the actual robot hand size rather than a fixed constant.
    """
    # Robot tip lengths from rest pose (all-zero qpos)
    q_bak = data.qpos.copy()
    for addr in MOTOR_QPOS_ADDR:
        data.qpos[addr] = 0.0
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_interface")
    palm = data.xpos[body_id].copy()

    robot_lengths = []
    for tip_name in ["tip1", "tip2", "tip3", "tip4"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
        robot_lengths.append(np.linalg.norm(data.site_xpos[sid] - palm))

    data.qpos[:] = q_bak
    mujoco.mj_forward(model, data)

    robot_mean = float(np.mean(robot_lengths))
    human_mean = float(np.mean([np.linalg.norm(v) for v in human_kvs.values()]))
    return robot_mean / human_mean if human_mean > 1e-9 else 1.0


def retarget_tsv(
    tsv: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: list[int],
    x0: np.ndarray | None = None,
    scale: float | None = None,
    method: str = "L-BFGS-B",
    max_iter: int = 200,
    n_restarts: int = 3,
) -> np.ndarray:
    """
    Given a single (5,3) TSV array, find the 8 motor angles that minimise
    the keyvector energy between human and robot hand.

    Returns: ctrl (8,) in radians, clamped to [CTRL_LOW, CTRL_HIGH].
    """
    human_kvs = tsv_to_human_keyvectors(tsv)

    # Adaptive scale: robot hand size / human hand size
    if scale is None:
        scale = compute_adaptive_scale(human_kvs, model, data)
        print(f"  [retarget] adaptive scale = {scale:.4f}")

    bounds = [(CTRL_LOW, CTRL_HIGH)] * len(actuator_ids)

    # Multiple restarts to avoid local minima
    inits = [np.zeros(8)]                                        # open hand
    if x0 is not None:
        inits.insert(0, x0.copy())                               # warm start
    for _ in range(n_restarts):
        inits.append(np.random.uniform(-0.5, 0.5, size=8))      # random

    best_ctrl, best_val = np.zeros(8), np.inf
    for init in inits:
        result = minimize(
            fun=energy,
            x0=init,
            args=(human_kvs, model, data, actuator_ids, scale),
            jac=energy_gradient,
            bounds=bounds,
            method=method,
            options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6},
        )
        if result.fun < best_val:
            best_val  = result.fun
            best_ctrl = result.x.copy()

    print(f"  [retarget] final energy = {best_val:.6f}")
    return np.clip(best_ctrl, CTRL_LOW, CTRL_HIGH)

# ─────────────────────────────────────────────────────────────────────────────
# Smooth interpolation (from RL env alpha smoothing)
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_ctrl(current: np.ndarray, target: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average interpolation for smooth transitions."""
    return (1 - alpha) * current + alpha * target

# ─────────────────────────────────────────────────────────────────────────────
# Print diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_tsv_info(tsv: np.ndarray, label: str = ""):
    print(f"\n{'='*56}")
    if label:
        print(f"File: {label}")
    print(f"TSV shape: {tsv.shape}  (5 fingertips × 3 xyz)")
    print(f"{'='*56}")
    print(f"{'Finger':<14} {'X':>10} {'Y':>10} {'Z':>10} {'Length':>10}")
    print(f"{'-'*56}")
    for i, name in enumerate(HUMAN_FINGER_NAMES):
        v = tsv[i]
        length = float(np.linalg.norm(v))
        suffix = "  ← dropped (pinky)" if i == 4 else ""
        print(f"{name:<14} {v[0]:>10.4f} {v[1]:>10.4f} {v[2]:>10.4f} {length:>10.4f}{suffix}")
    print(f"{'='*56}")


def print_ctrl(ctrl: np.ndarray):
    print("\nRetargeted robot controls (radians):")
    print(f"{'Actuator':<22} {'Value':>10}")
    print("-" * 34)
    for name, val in zip(ACTUATOR_NAMES, ctrl):
        print(f"{name:<22} {val:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(tsv_files: list[str], model_path: str, hold_time: float = 2.0):
    """
    Load the Amazing Hand MuJoCo model, retarget each TSV file, and
    visualise the resulting hand pose.

    Args:
        tsv_files:   List of .npy file paths (each (5,3) TSV array).
        model_path:  Path to scene.xml.
        hold_time:   Seconds to hold each pose before moving to the next.
    """
    print(f"\n[INFO] Loading MuJoCo model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    # Resolve actuator IDs
    actuator_ids = []
    for name in ACTUATOR_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid == -1:
            raise RuntimeError(f"Actuator '{name}' not found in model.")
        actuator_ids.append(aid)
    print(f"[INFO] Found {len(actuator_ids)} actuators: {ACTUATOR_NAMES}")

    # Verify tip sites exist
    for tip in ["tip1", "tip2", "tip3", "tip4"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip)
        if sid == -1:
            raise RuntimeError(
                f"Site '{tip}' not found in model. "
                f"Ensure scene.xml includes the robot MJCF with tip sites."
            )
    print("[INFO] Tip sites tip1–tip4 found.\n")

    # Pre-compute retargeted controls for all TSV files
    print(f"[INFO] Retargeting {len(tsv_files)} TSV file(s)...")
    retargeted = []
    prev_ctrl = np.zeros(len(actuator_ids))

    for fpath in tsv_files:
        tsv = np.load(fpath, allow_pickle=True)
        if isinstance(tsv, np.ndarray) and tsv.dtype == object:
            tsv = tsv.item()  # handle dict-wrapped saves
        tsv = np.array(tsv, dtype=np.float64)
        print_tsv_info(tsv, label=os.path.basename(fpath))

        ctrl = retarget_tsv(tsv, model, data, actuator_ids, x0=prev_ctrl.copy())
        print_ctrl(ctrl)
        retargeted.append((fpath, tsv, ctrl))
        prev_ctrl = ctrl.copy()

    # ── Simulation viewer ────────────────────────────────────────────────────
    print("\n[INFO] Launching MuJoCo viewer…")
    print("[INFO] Press ESC or close the window to exit.")
    print("[INFO] The hand will cycle through each pose.\n")

    steps_per_hold = max(1, int(hold_time / model.opt.timestep))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        pose_idx   = 0
        step_count = 0
        n_poses    = len(retargeted)

        # Current smoothed control (starts at zero = open hand)
        current_ctrl = np.zeros(len(actuator_ids))
        target_ctrl  = retargeted[0][2].copy()

        while viewer.is_running():
            # ── Smooth interpolation toward target ───────────────────────────
            current_ctrl = interpolate_ctrl(current_ctrl, target_ctrl, alpha=0.08)

            # Set ctrl directly (same as reference script, not via actuator_ids)
            data.ctrl[:] = current_ctrl

            # Step physics (5 substeps, same as RL env)
            for _ in range(5):
                mujoco.mj_step(model, data)
            viewer.sync()

            step_count += 1

            # ── Advance to next pose after hold_time ─────────────────────────
            if step_count >= steps_per_hold:
                step_count = 0
                pose_idx   = (pose_idx + 1) % n_poses
                target_ctrl = retargeted[pose_idx][2].copy()
                fname = os.path.basename(retargeted[pose_idx][0])
                print(f"[SIM] Transitioning to pose {pose_idx+1}/{n_poses}: {fname}")

            # ── Slight rate control ───────────────────────────────────────────
            time.sleep(model.opt.timestep)

    print("\n[INFO] Viewer closed. Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Human TSV → Amazing Hand retargeting + MuJoCo simulation"
    )
    parser.add_argument(
        "--model_path", type=str, default="mjcf/scene.xml",
        help="Path to the Amazing Hand scene.xml MuJoCo model."
    )
    parser.add_argument(
        "--tsv_file", type=str, default=None,
        help="Path to a single *_tsv.npy file."
    )
    parser.add_argument(
        "--tsv_folder", type=str, default=None,
        help="Folder containing *_tsv.npy files (all will be loaded in order)."
    )
    parser.add_argument(
        "--hold_time", type=float, default=2.0,
        help="Seconds to hold each retargeted pose in the viewer (default: 2.0)."
    )
    parser.add_argument(
        "--scale", type=float, default=KV_SCALE,
        help=(
            "Keyvector scale factor c (Robotic Telekinesis Eq.1). "
            "Accounts for size ratio between human and robot hand. "
            f"Default: {KV_SCALE}"
        )
    )
    args = parser.parse_args()

    # ── Collect TSV files ─────────────────────────────────────────────────────
    tsv_files = []

    if args.tsv_file:
        if not os.path.isfile(args.tsv_file):
            raise FileNotFoundError(f"TSV file not found: {args.tsv_file}")
        tsv_files = [args.tsv_file]

    elif args.tsv_folder:
        pattern = os.path.join(args.tsv_folder, "*_tsv.npy")
        tsv_files = sorted(glob.glob(pattern))
        if not tsv_files:
            raise FileNotFoundError(f"No *_tsv.npy files found in: {args.tsv_folder}")

    else:
        # Demo mode: create synthetic TSV data matching the sample in the prompt
        print("[INFO] No --tsv_file or --tsv_folder given. Running in DEMO mode.")
        print("[INFO] Using the two sample TSVs from the problem statement.\n")

        sample_tsvs = {
            "000070_0_tsv_demo": np.array([
                [-0.0353, -0.0869, -0.0025],  # Thumb
                [ 0.0103, -0.0698,  0.0131],  # Index
                [ 0.0002, -0.0593,  0.0118],  # Middle
                [-0.0121, -0.0395,  0.0133],  # Ring
                [-0.0291, -0.0321,  0.0076],  # Pinky
            ], dtype=np.float64),
            "000095_0_tsv_demo": np.array([
                [-0.0545, -0.0664, -0.0392],  # Thumb
                [-0.0034, -0.1708, -0.0046],  # Index
                [-0.0626, -0.1626,  0.0106],  # Middle
                [-0.0147, -0.0412, -0.0171],  # Ring
                [-0.0272, -0.0351, -0.0158],  # Pinky
            ], dtype=np.float64),
        }

        # Save to temp files so the pipeline is identical to the real case
        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix="amazing_hand_demo_")
        for name, arr in sample_tsvs.items():
            fpath = os.path.join(tmp_dir, f"{name}.npy")
            np.save(fpath, arr)
            tsv_files.append(fpath)
        print(f"[INFO] Demo TSVs saved to: {tmp_dir}\n")

    # ── Validate model path ───────────────────────────────────────────────────
    if not os.path.isfile(args.model_path):
        # Try to help the user find their model
        candidates = list(Path(".").rglob("scene.xml")) + list(Path(".").rglob("mjcf/*.xml"))
        hint = (
            f"\nCandidates found nearby:\n  " + "\n  ".join(str(c) for c in candidates[:5])
            if candidates else ""
        )
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            f"Use --model_path to point to your scene.xml.{hint}"
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    run_simulation(tsv_files, args.model_path, hold_time=args.hold_time)


if __name__ == "__main__":
    main()