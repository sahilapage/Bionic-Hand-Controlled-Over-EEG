import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize
import time

MODEL_PATH = "mjcf/scene.xml"

# From your working environment
motor_qpos_addr = [0, 12, 17, 29, 34, 46, 51, 63]

CTRL_LOW = -1.57
CTRL_HIGH = 1.57

# -----------------------------
# Load Model
# -----------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

palm_body_id = model.body("r_wrist_interface").id

tip_sites = [
    model.site("tip1").id,
    model.site("tip2").id,
    model.site("tip3").id,
    model.site("tip4").id,
]

# -----------------------------
# HUMAN TSV (example peace sign)
# Replace this later with HAMER output
# -----------------------------
human_tsv = np.array([
    [-0.05134989, -0.06822547, -0.04434989],
    [ 0.00448461, -0.17077546, -0.00089004],
    [-0.06126514, -0.16255069,  0.00843796],
    [-0.01218381, -0.04114483, -0.01920276]
])


# -----------------------------
# Scaling
# -----------------------------
def scale_human_tsv(human_tsv):
    mujoco.mj_forward(model, data)
    palm_pos = data.xpos[palm_body_id]

    robot_lengths = []
    for site_id in tip_sites:
        tip_pos = data.site_xpos[site_id]
        robot_lengths.append(np.linalg.norm(tip_pos - palm_pos))

    robot_mean = np.mean(robot_lengths)
    human_mean = np.mean(np.linalg.norm(human_tsv, axis=1))

    scale = robot_mean / human_mean
    return human_tsv * scale


# -----------------------------
# Robot TSV
# -----------------------------
def compute_robot_tsv(q):
    q_backup = data.qpos.copy()

    for i, addr in enumerate(motor_qpos_addr):
        data.qpos[addr] = q[i]

    mujoco.mj_forward(model, data)

    palm_pos = data.xpos[palm_body_id]
    robot_tsv = []

    for site_id in tip_sites:
        tip_pos = data.site_xpos[site_id]
        robot_tsv.append(tip_pos - palm_pos)

    data.qpos[:] = q_backup
    mujoco.mj_forward(model, data)

    return np.array(robot_tsv)


# -----------------------------
# Energy
# -----------------------------
def energy(q, human_scaled):
    robot_tsv = compute_robot_tsv(q)
    return np.sum((human_scaled - robot_tsv) ** 2)


# -----------------------------
# Solve IK
# -----------------------------
def solve_ik(human_tsv):
    human_scaled = scale_human_tsv(human_tsv)

    q_init = np.random.uniform(-0.5, 0.5, size=8)

    bounds = [(CTRL_LOW, CTRL_HIGH)] * 8

    res = minimize(
        energy,
        q_init,
        args=(human_scaled,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 40}
    )

    print("Energy:", res.fun)
    return res.x


# -----------------------------
# Solve once
# -----------------------------
print("Solving IK...")
q_solution = solve_ik(human_tsv)
print("Final q:", np.round(q_solution, 3))

# -----------------------------
# Apply solution continuously
# -----------------------------
while True:
    data.ctrl[:] = q_solution

    for _ in range(5):
        mujoco.mj_step(model, data)

    viewer.sync()
    time.sleep(0.01)