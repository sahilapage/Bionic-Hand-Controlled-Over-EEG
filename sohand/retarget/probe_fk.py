"""Forward-kinematics diagnostic: how does each motor move its fingertip?

    python -m sohand.retarget.probe_fk
    python -m sohand.retarget.probe_fk --model-path mjcf/cube/scene.xml

Sweeps every actuator and reports the resulting fingertip displacement, which
is the sensitivity (m/rad) the retargeting Jacobian is built from. Run it after
any change to the linkage geometry or the actuator gains -- if these numbers
move, the constants in `sohand.retarget.retarget` are stale.

Every measurement settles physics first. The fingers are passive four-bar
linkages, so their constrained pose does not exist until the solver has run;
reading `site_xpos` straight after `mj_forward` gives the unconstrained one.
That is not a subtlety -- it is why an early version reported that the motors
did not move at all.
"""

import argparse

import mujoco
import numpy as np

from sohand.hand import JOINTS, TIP_SITES
from sohand.paths import HAND_SCENE

SETTLE_STEPS = 300
CALIB_ANGLE = 0.5
RULE = "=" * 60

# A grasp pose taken from the RL environment, used as a sanity check that the
# model reaches a sensible configuration under a realistic command.
KNOWN_GOOD_POSE = [-0.157, 0.848, -0.581, 0.456, 1.57, -1.57, 1.57, -1.57]


class Probe:
    def __init__(self, model_path, settle_steps=SETTLE_STEPS):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.settle_steps = settle_steps
        self.palm_id = self.model.body("r_wrist_interface").id
        self.site_ids = [self.model.site(n).id for n in TIP_SITES]
        self.qpos_addr = [self.model.joint(n).qposadr[0] for n in JOINTS]
        if self.model.nu != len(JOINTS):
            raise SystemExit(
                f"model has {self.model.nu} actuators, expected {len(JOINTS)}")

    def settle(self, ctrl):
        """Apply a command, run physics to convergence, return (tips, qpos)."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = np.asarray(ctrl, dtype=float)
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.data)
        palm = self.data.xpos[self.palm_id].copy()
        tips = np.array([self.data.site_xpos[s] - palm for s in self.site_ids])
        return tips, np.array([self.data.qpos[a] for a in self.qpos_addr])

    def one_motor(self, index, value):
        ctrl = np.zeros(len(JOINTS))
        ctrl[index] = value
        return self.settle(ctrl)


def report(probe):
    tips0, q0 = probe.settle(np.zeros(len(JOINTS)))

    print(f"\n{RULE}\nBASELINE (ctrl = 0, {probe.settle_steps} settle steps)")
    print(f"  qpos: {np.round(q0, 4)}")
    for i, name in enumerate(TIP_SITES):
        print(f"  {name}: {tips0[i].round(4)}  len={np.linalg.norm(tips0[i]):.4f}")

    for motor, label in ((0, "finger1_motor1 (flexion)"),
                         (1, "finger1_motor2 (abduction)")):
        print(f"\n{RULE}\nSWEEP {label}")
        for v in (0.3, 0.6, 1.0, -0.3, -0.6, -1.0):
            tips, q = probe.one_motor(motor, v)
            d = tips[0] - tips0[0]
            print(f"  ctrl[{motor}]={v:+.1f}  qpos={q[motor]:+.4f}  "
                  f"tip1={tips[0].round(4)}  delta={d.round(4)}  "
                  f"|delta|={np.linalg.norm(d):.4f}")

    print(f"\n{RULE}\nCTRL vs QPOS after settling -- do the servos reach the command?")
    test = np.array([0.5, -0.3, 0.7, 0.2, -0.5, 0.8, 0.4, -0.6])
    _, q_test = probe.settle(test)
    print(f"  ctrl = {np.round(test, 3)}")
    print(f"  qpos = {np.round(q_test, 4)}")
    print(f"  diff = {np.round(q_test - test, 4)}")

    print(f"\n{RULE}\nKNOWN-GOOD GRASP POSE")
    tips_g, q_g = probe.settle(KNOWN_GOOD_POSE)
    print(f"  settled qpos: {np.round(q_g, 4)}")
    for i, name in enumerate(TIP_SITES):
        print(f"  {name}: {tips_g[i].round(4)}  len={np.linalg.norm(tips_g[i]):.4f}")

    print(f"\n{RULE}\nSIGN CHECK -- does positive motor1 curl (Z down) or extend?")
    for finger in range(len(TIP_SITES)):
        tips_p, _ = probe.one_motor(finger * 2, 1.0)
        tips_n, _ = probe.one_motor(finger * 2, -1.0)
        print(f"  finger{finger + 1}_motor1 +1.0: "
              f"tip delta={(tips_p[finger] - tips0[finger]).round(4)}")
        print(f"  finger{finger + 1}_motor1 -1.0: "
              f"tip delta={(tips_n[finger] - tips0[finger]).round(4)}")

    print(f"\n{RULE}\nFULL JACOBIAN (central difference at +/-{CALIB_ANGLE} rad)")
    for motor, name in enumerate(JOINTS):
        finger = motor // 2
        tips_p, _ = probe.one_motor(motor, CALIB_ANGLE)
        tips_n, _ = probe.one_motor(motor, -CALIB_ANGLE)
        sens = (tips_p[finger] - tips_n[finger]) / (2 * CALIB_ANGLE)
        print(f"  {name:<18}  sens={sens.round(4)}  "
              f"|sens|={np.linalg.norm(sens):.4f} m/rad")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=HAND_SCENE)
    p.add_argument("--settle-steps", type=int, default=SETTLE_STEPS,
                   help="physics steps per measurement; too few and the "
                        "linkage has not resolved")
    args = p.parse_args()

    probe = Probe(args.model_path, args.settle_steps)
    print(f"model: {args.model_path}")
    print(f"palm body: r_wrist_interface   tip sites: {', '.join(TIP_SITES)}")
    report(probe)


if __name__ == "__main__":
    main()
