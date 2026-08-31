"""Amazing Hand model wiring and the constants shared across environments.

Four fingers, two servos each: eight actuated degrees of freedom. Each finger
is a passive four-bar linkage driven by two horns, so the fingertip pose is a
nonlinear function of both motor angles and the linkage joints must be settled
by physics -- `mj_forward` alone leaves them unconstrained.

Two MuJoCo models ship with this repo and they name things differently:

    mjcf/hand/   actuators are named after the joints they drive (`JOINTS`)
    mjcf/cube/   actuators carry the `motor_` prefix (`ACTUATORS`)

`JOINTS` is correct for both; `ACTUATORS` is correct only for `mjcf/cube/`.
"""

import numpy as np

# Actuator names in the manipulation model (mjcf/cube/robot.xml).
ACTUATORS = [
    "motor_finger1_1", "motor_finger1_2", "motor_finger2_1", "motor_finger2_2",
    "motor_finger3_1", "motor_finger3_2", "motor_finger4_1", "motor_finger4_2",
]

# Driven joint names. Identical in both models, and also the actuator names in
# mjcf/hand/robot.xml. motor1 flexes the finger, motor2 abducts it.
JOINTS = [
    "finger1_motor1", "finger1_motor2", "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2", "finger4_motor1", "finger4_motor2",
]

# Fingertip sites, in finger order. Finger 4 is the thumb.
TIP_SITES = ["tip1", "tip2", "tip3", "tip4"]

N_JOINTS = 8
N_FINGERS = 4

# --------------------------------------------------------------------------
# Cube geometry (mjcf/cube)
# --------------------------------------------------------------------------
# The cube mesh is not centred on its body origin. This is the <inertial pos>
# from scene.xml and is the true geometric centre in the cube's local frame --
# 3.5 cm from the origin, which is why `qvel[0:3]` (the *origin's* velocity)
# must not be read as the cube's linear velocity.
CUBE_LOCAL_CENTER = np.array([-0.0028, -0.035, 0.0011], dtype=np.float32)

# Half-extent of the shipped 4.7 cm cube. `scene_spin.xml` scales the mesh, so
# environments measure this from the loaded model instead of trusting it.
CUBE_HALF = 0.0235

# --------------------------------------------------------------------------
# Observation normalisation
# --------------------------------------------------------------------------
JVEL_SCALE = 5.0     # rad/s -> [-1, 1]
REACH_NORM = 0.10    # m, fingertip-to-cube vectors -> ~[-1, 1]

# --------------------------------------------------------------------------
# Grasp and reset shaping
# --------------------------------------------------------------------------
HAND_CLOSE_FRAC = 0.50    # fraction of joint range the hand closes to on reset
HAND_CLOSE_JITTER = 0.02
CLOSE_PROBE_FRAC = 0.3
GRASP_OPEN_FRAC = 0.05

# Settle length in *control* steps, not raw physics steps. The original code
# ran 30 raw mj_step calls -- 60 ms -- far too short for the servos to track
# the commanded close. Measured result: the hand reached frac 0.178 instead of
# 0.50, and since `grasp_frac` is captured from the achieved pose and becomes
# the *centre* of the action band, the policy's reachable command range
# silently became [-0.32, +0.68] instead of [0, 1].
SETTLE_CTRL_STEPS = 60    # x frame_skip x 2 ms = 1.2 s

# Peak joint speed, as a fraction of range per control step. 0.16 frac/step is
# 0.223 rad per 20 ms = 11.2 rad/s, well beyond what the hand's serial-bus
# servos can execute -- a policy trained against it commands motion the real
# hand cannot follow. 0.08 is about 5.6 rad/s.
MAX_CTRL_RATE_FRAC = 0.08

# --------------------------------------------------------------------------
# Cube faces
# --------------------------------------------------------------------------
# Face index -> outward normal in the cube's local frame. `f ^ 1` is the
# opposite face under this ordering; the other four are a 90 deg roll away.
FACE_NORMALS = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0],
                         [0, -1, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.float32)
FACE_NAMES = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
