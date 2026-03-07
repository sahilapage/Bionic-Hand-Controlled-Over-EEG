import argparse
import glob
import os
import time
import tempfile
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


HUMAN_FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
HUMAN_ORDER = [1, 2, 3, 0]

ACTUATOR_NAMES = [
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
]

MOTOR_QPOS_ADDR = [0, 12, 17, 29, 34, 46, 51, 63]

CTRL_LOW  = -np.pi / 2   
CTRL_HIGH =  np.pi / 2   

CALIB_STEPS  = 300
CALIB_ANGLE  = 0.5   


HUMAN_Z_MIN  = 0.05
HUMAN_Z_MAX  = 0.18

CURL_GAMMA   = 1.5   
THUMB_X_MAX  = 0.10  
SIM_ALPHA    = 0.08  


def remap(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], v[2], -v[1]], dtype=np.float64)


def settle(model, data, ctrl_vals, steps=CALIB_STEPS):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = np.clip(ctrl_vals, CTRL_LOW, CTRL_HIGH)
    for _ in range(steps):
        mujoco.mj_step(model, data)

def get_tip_vecs(model, data, palm_id, site_ids) -> np.ndarray:
    palm = data.xpos[palm_id].copy()
    return np.array([data.site_xpos[s] - palm for s in site_ids])


def calibrate(model, data, palm_id, site_ids):
    print("[CALIB] Running FK calibration…")
    settle(model, data, np.zeros(8))
    tips0 = get_tip_vecs(model, data, palm_id, site_ids)
    print(f"[CALIB] Baseline tip lengths: {np.linalg.norm(tips0, axis=1).round(4)}")

    jac = {}
    z_curl = np.zeros(4)
    z_ext  = np.zeros(4)

    for fi in range(4):
        c = np.zeros(8); c[fi*2] = CALIB_ANGLE
        settle(model, data, c)
        t_m1p = get_tip_vecs(model, data, palm_id, site_ids)
        c = np.zeros(8); c[fi*2] = -CALIB_ANGLE
        settle(model, data, c)
        t_m1n = get_tip_vecs(model, data, palm_id, site_ids)

        sens_m1 = (t_m1p[fi] - t_m1n[fi]) / (2 * CALIB_ANGLE)

        c = np.zeros(8); c[fi*2+1] = CALIB_ANGLE
        settle(model, data, c)
        t_m2p = get_tip_vecs(model, data, palm_id, site_ids)
        c = np.zeros(8); c[fi*2+1] = -CALIB_ANGLE
        settle(model, data, c)
        t_m2n = get_tip_vecs(model, data, palm_id, site_ids)

        sens_m2 = (t_m2p[fi] - t_m2n[fi]) / (2 * CALIB_ANGLE)

        jac[fi] = {"m1": sens_m1, "m2": sens_m2}

        z_curl[fi] = tips0[fi][2] + sens_m1[2] * CTRL_HIGH
        z_ext [fi] = tips0[fi][2] + sens_m1[2] * CTRL_LOW

        print(f"[CALIB]   finger{fi+1}: "
              f"m1={sens_m1.round(4)}  m2={sens_m2.round(4)}  "
              f"Z_curl={z_curl[fi]:.4f}  Z_ext={z_ext[fi]:.4f}")

    settle(model, data, np.zeros(8))
    return tips0, jac, z_curl, z_ext


HUMAN_X_MAX = 0.40

def retarget_tsv(tsv: np.ndarray,
                 tips0: np.ndarray,
                 jac: dict,
                 z_curl: np.ndarray,
                 z_ext:  np.ndarray) -> np.ndarray:

    assert tsv.shape == (5, 3), f"Expected (5,3), got {tsv.shape}"
    ctrl = np.zeros(8)

    for fi, hi in enumerate(HUMAN_ORDER):
        v = remap(tsv[hi])
        human_z = v[2]   
        human_x = v[0]  

        is_thumb = (fi == 3)   

        if not is_thumb:
            ext_ratio = float(np.clip(
                (human_z - HUMAN_Z_MIN) / (HUMAN_Z_MAX - HUMAN_Z_MIN), 0.0, 1.0))
            ext_ratio = ext_ratio ** CURL_GAMMA  

            target_z = z_curl[fi] + ext_ratio * (z_ext[fi] - z_curl[fi])
            m1 = (target_z - tips0[fi][2]) / jac[fi]["m1"][2]

            m2 = human_x / HUMAN_X_MAX * CTRL_HIGH

        else:
            m1 = -(tsv[hi][0] / THUMB_X_MAX) * CTRL_HIGH

            m2 = tsv[hi][2] / 0.05 * 0.4

        ctrl[fi * 2]     = float(np.clip(m1, CTRL_LOW, CTRL_HIGH))
        ctrl[fi * 2 + 1] = float(np.clip(m2, CTRL_LOW, CTRL_HIGH))

    return ctrl


def print_tsv_info(tsv, label=""):
    w = 60
    print(f"\n{'='*w}")
    if label: print(f"File: {label}")
    print(f"TSV: {tsv.shape}")
    print(f"{'='*w}")
    print(f"{'Finger':<14} {'X':>10} {'Y':>10} {'Z':>10} {'Len':>8}")
    print("-"*w)
    for i, name in enumerate(HUMAN_FINGER_NAMES):
        v = tsv[i]; sfx = "  ← dropped" if i == 4 else ""
        print(f"{name:<14} {v[0]:>10.4f} {v[1]:>10.4f} {v[2]:>10.4f} "
              f"{np.linalg.norm(v):>8.4f}{sfx}")
    print("="*w)


def print_ctrl(ctrl):
    print(f"\n  {'Actuator':<24} {'rad':>7}  bar")
    print("  " + "-"*44)
    for name, val in zip(ACTUATOR_NAMES, ctrl):
        pct = int(abs(val) / CTRL_HIGH * 10)
        sign = "+" if val >= 0 else "-"
        print(f"  {name:<24} {val:>+7.4f}  {sign}{'█'*pct}")


def run_simulation(tsv_files, model_path, hold_time=2.0):
    print(f"\n[INFO] Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    palm_id  = model.body("r_wrist_interface").id
    site_ids = [model.site(n).id for n in ("tip1","tip2","tip3","tip4")]

    for name in ACTUATOR_NAMES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) == -1:
            raise RuntimeError(f"Actuator '{name}' not found.")
    print(f"[INFO] palm_id={palm_id}  site_ids={site_ids}\n")

    tips0, jac, z_curl, z_ext = calibrate(model, data, palm_id, site_ids)

    print(f"\n[INFO] Retargeting {len(tsv_files)} file(s)…")
    retargeted = []
    for fpath in tsv_files:
        tsv = np.load(fpath, allow_pickle=True)
        if isinstance(tsv, np.ndarray) and tsv.dtype == object:
            tsv = tsv.item()
        tsv = np.asarray(tsv, dtype=np.float64)
        print_tsv_info(tsv, os.path.basename(fpath))
        ctrl = retarget_tsv(tsv, tips0, jac, z_curl, z_ext)
        print_ctrl(ctrl)
        retargeted.append((os.path.basename(fpath), ctrl))

    print(f"\n[INFO] Launching viewer (hold={hold_time}s/pose)…")
    steps_per_hold = max(1, int(hold_time / model.opt.timestep))
    pose_idx, step_count = 0, 0
    current_ctrl = np.zeros(8)
    target_ctrl  = retargeted[0][1].copy()

    settle(model, data, np.zeros(8))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            current_ctrl = (1 - SIM_ALPHA) * current_ctrl + SIM_ALPHA * target_ctrl
            data.ctrl[:] = current_ctrl
            for _ in range(5):
                mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1
            if step_count >= steps_per_hold:
                step_count = 0
                pose_idx = (pose_idx + 1) % len(retargeted)
                target_ctrl = retargeted[pose_idx][1].copy()
                print(f"[SIM] → {pose_idx+1}/{len(retargeted)}: {retargeted[pose_idx][0]}")
            time.sleep(model.opt.timestep)

    print("\n[INFO] Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="mjcf/scene.xml")
    p.add_argument("--tsv_file",   default=None)
    p.add_argument("--tsv_folder", default=None)
    p.add_argument("--hold_time",   type=float, default=2.0)
    global HUMAN_Z_MAX
    p.add_argument("--human_z_max", type=float, default=HUMAN_Z_MAX,
                   help="Nominal max human fingertip extension in metres (default 0.18).")
    args = p.parse_args()

    HUMAN_Z_MAX = args.human_z_max

    tsv_files = []
    if args.tsv_file:
        tsv_files = [args.tsv_file]
    elif args.tsv_folder:
        tsv_files = sorted(glob.glob(os.path.join(args.tsv_folder, "*_tsv.npy")))
        if not tsv_files:
            raise FileNotFoundError(f"No *_tsv.npy in: {args.tsv_folder}")
    else:
        print("[INFO] Demo mode.\n")
        samples = {
            "000070_0_tsv.npy": np.array([[-0.0353,-0.0869,-0.0025],[0.0103,-0.0698,0.0131],[0.0002,-0.0593,0.0118],[-0.0121,-0.0395,0.0133],[-0.0291,-0.0321,0.0076]]),
            "000095_0_tsv.npy": np.array([[-0.0545,-0.0664,-0.0392],[-0.0034,-0.1708,-0.0046],[-0.0626,-0.1626,0.0106],[-0.0147,-0.0412,-0.0171],[-0.0272,-0.0351,-0.0158]]),
            "000114_0_tsv.npy": np.array([[0.0980,-0.0786,-0.0357],[0.0120,-0.1714,0.0042],[-0.0361,-0.1704,0.0177],[-0.0690,-0.1462,0.0290],[-0.1043,-0.0946,-0.0016]]),
        }
        tmp = tempfile.mkdtemp(prefix="tsv_demo_")
        for name, arr in samples.items():
            path = os.path.join(tmp, name)
            np.save(path, arr)
            tsv_files.append(path)

    if not os.path.isfile(args.model_path):
        cands = list(Path(".").rglob("scene.xml"))
        hint  = ("\n  " + "\n  ".join(str(c) for c in cands[:5])) if cands else ""
        raise FileNotFoundError(f"Model not found: {args.model_path}{hint}")

    run_simulation(tsv_files, args.model_path, args.hold_time)


if __name__ == "__main__":
    main()