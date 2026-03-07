import sys
import numpy as np
import mujoco

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "mjcf/scene.xml"

MOTOR_QPOS_ADDR = [0, 12, 17, 29, 34, 46, 51, 63]
ACTUATOR_NAMES  = [
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
]
SETTLE_STEPS = 300

model    = mujoco.MjModel.from_xml_path(MODEL_PATH)
data     = mujoco.MjData(model)
palm_id  = model.body("r_wrist_interface").id
site_ids = [model.site(n).id for n in ("tip1", "tip2", "tip3", "tip4")]

print(f"[INFO] Model: {MODEL_PATH}")
print(f"[INFO] palm_id={palm_id}  site_ids={site_ids}")

def settle(ctrl_vals, steps=SETTLE_STEPS):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = np.array(ctrl_vals, dtype=float)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    palm   = data.xpos[palm_id].copy()
    tips   = np.array([data.site_xpos[s] - palm for s in site_ids])
    motors = np.array([data.qpos[a] for a in MOTOR_QPOS_ADDR])
    return tips, motors

tips0, q0 = settle([0] * 8)
print("\n" + "=" * 60)
print(f"BASELINE (ctrl=0, {SETTLE_STEPS} steps)")
print(f"  qpos: {np.round(q0, 4)}")
for i, name in enumerate(("tip1", "tip2", "tip3", "tip4")):
    t = tips0[i]
    print(f"  {name}: {t.round(4)}  len={np.linalg.norm(t):.4f}")

print("\n" + "=" * 60)
print("SWEEP finger1_motor1 (ctrl[0]) at +0.3, +0.6, +1.0, -0.3, -0.6, -1.0")
for v in [0.3, 0.6, 1.0, -0.3, -0.6, -1.0]:
    tips, q = settle([v, 0, 0, 0, 0, 0, 0, 0])
    d = tips[0] - tips0[0]
    print(f"  ctrl[0]={v:+.1f}  qpos={q[0]:+.4f}  "
          f"tip1={tips[0].round(4)}  delta={d.round(4)}  |delta|={np.linalg.norm(d):.4f}")

print("\n" + "=" * 60)
print("SWEEP finger1_motor2 (ctrl[1]) at +0.3, +0.6, +1.0, -0.3, -0.6, -1.0")
for v in [0.3, 0.6, 1.0, -0.3, -0.6, -1.0]:
    tips, q = settle([0, v, 0, 0, 0, 0, 0, 0])
    d = tips[0] - tips0[0]
    print(f"  ctrl[1]={v:+.1f}  qpos={q[1]:+.4f}  "
          f"tip1={tips[0].round(4)}  delta={d.round(4)}  |delta|={np.linalg.norm(d):.4f}")

print("\n" + "=" * 60)
known_good = [-0.157, 0.848, -0.581, 0.456, 1.57, -1.57, 1.57, -1.57]
print(f"KNOWN-GOOD POSE from RL env: {known_good}")
tips_g, q_g = settle(known_good)
print(f"  settled qpos: {np.round(q_g, 4)}")
for i, name in enumerate(("tip1", "tip2", "tip3", "tip4")):
    print(f"  {name}: {tips_g[i].round(4)}  len={np.linalg.norm(tips_g[i]):.4f}")

print("\n" + "=" * 60)
print("CTRL vs QPOS after settling (do they match?)")
test = [0.5, -0.3, 0.7, 0.2, -0.5, 0.8, 0.4, -0.6]
_, q_t = settle(test)
print(f"  ctrl = {np.round(test, 3)}")
print(f"  qpos = {np.round(q_t, 4)}")
print(f"  diff = {np.round(q_t - np.array(test), 4)}")

print("\n" + "=" * 60)
print("SIGN CHECK: positive motor1 = curl (Z decrease) or extend (Z increase)?")
for fi in range(4):
    ci = fi * 2
    ctrl_p = [0.0] * 8; ctrl_p[ci] =  1.0
    ctrl_n = [0.0] * 8; ctrl_n[ci] = -1.0
    tips_p, _ = settle(ctrl_p)
    tips_n, _ = settle(ctrl_n)
    d_pos = tips_p[fi] - tips0[fi]
    d_neg = tips_n[fi] - tips0[fi]
    print(f"  finger{fi+1}_motor1 +1.0: tip{fi+1} delta={d_pos.round(4)}")
    print(f"  finger{fi+1}_motor1 -1.0: tip{fi+1} delta={d_neg.round(4)}")

print("\n" + "=" * 60)
CALIB = 0.5
print(f"FULL JACOBIAN (sensitivity at +/-{CALIB} rad, {SETTLE_STEPS} steps settle)")
for mi, name in enumerate(ACTUATOR_NAMES):
    fi = mi // 2
    ctrl_p = [0.0] * 8; ctrl_p[mi] =  CALIB
    ctrl_n = [0.0] * 8; ctrl_n[mi] = -CALIB
    tips_p, _ = settle(ctrl_p)
    tips_n, _ = settle(ctrl_n)
    sens = (tips_p[fi] - tips_n[fi]) / (2 * CALIB)
    print(f"  {name:<24}  sens={sens.round(4)}  |sens|={np.linalg.norm(sens):.4f} m/rad")