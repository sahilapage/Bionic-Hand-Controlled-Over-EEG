import mujoco
import mujoco.viewer
import numpy as np

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_PATH = "mjcf/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

tip_names = ["tip1", "tip2", "tip3", "tip4"]
tip_ids = [model.site(name).id for name in tip_names]
palm_body_id = model.body("r_wrist_interface").id

nu = model.nu
print("Number of actuators:", nu)


# -----------------------------
# HUMAN TSV (10x3 → select 4 tips)
# -----------------------------
human_tsv_full = np.array([
 [ 0.11699297 ,-0.05389309, -0.0532191 ],
 [ 0.08337262 ,-0.04067587 ,-0.04084402],
 [ 0.08497161 ,-0.15235056, -0.04798556],
 [ 0.07180147 ,-0.12855552 ,-0.04240038],
 [ 0.0392772  ,-0.17336158, -0.05059738],
 [ 0.03336631 ,-0.1467817 , -0.04575389],
 [-0.00080065, -0.16531998 ,-0.05478768],
 [-0.00212543 ,-0.13935578 ,-0.04708304],
 [-0.04975762 ,-0.12892559 ,-0.06462509],
 [-0.04211351 ,-0.11017343, -0.05133494]
])

# Select only TIP rows
human_tsv = human_tsv_full[[0, 2, 4, 6]]  # (4,3)

# -----------------------------
# Normalize human direction
# -----------------------------
human_len = np.linalg.norm(human_tsv, axis=1, keepdims=True)
human_len[human_len < 1e-6] = 1e-6
human_dir = human_tsv / human_len


# -----------------------------
# ROBOT TSV FUNCTION
# -----------------------------
def compute_robot_tsv():
    palm_pos = data.xpos[palm_body_id]
    palm_rot = data.xmat[palm_body_id].reshape(3,3)

    tsvs = []
    for tid in tip_ids:
        tip_pos = data.site_xpos[tid]
        v_world = tip_pos - palm_pos
        v_local = palm_rot.T @ v_world
        tsvs.append(v_local)

    return np.stack(tsvs)


# -----------------------------
# MULTI-OBJECTIVE LOSS
# -----------------------------
def total_loss():

    robot = compute_robot_tsv()

    # ---- Direction loss ----
    robot_norm = np.linalg.norm(robot, axis=1, keepdims=True)
    robot_norm[robot_norm < 1e-6] = 1e-6
    robot_dir = robot / robot_norm

    cos = np.sum(robot_dir * human_dir, axis=1)
    L_dir = 1 - np.mean(cos)

    # ---- Magnitude (relative curl) ----
    human_len_flat = human_len.flatten()
    robot_len = robot_norm.flatten()

    human_len_norm = human_len_flat / human_len_flat.max()
    robot_len_norm = robot_len / robot_len.max()

    L_mag = np.mean((human_len_norm - robot_len_norm) ** 2)

    # ---- Spread (X-axis difference) ----
    L_spread = np.mean((robot[:, 0] - human_tsv[:, 0]) ** 2)

    # ---- Tunable weights ----
    w_dir =0
    w_mag = -0.00001
    w_spread = 0

    return w_dir * L_dir + w_mag * L_mag + w_spread * L_spread


# -----------------------------
# RANDOM SEARCH WITH REFINEMENT
# -----------------------------
best_ctrl = np.zeros(nu)
best_loss = 1e9

for _ in range(4000):

    # small noise around best solution
    ctrl = best_ctrl + np.random.normal(0, 0.3, size=nu)

    # clamp to reasonable motor limits
    ctrl = np.clip(ctrl, -1.5, 1.5)

    data.ctrl[:] = ctrl
    for _ in range(25):
        mujoco.mj_step(model, data)

    loss = total_loss()

    if loss < best_loss:
        best_loss = loss
        best_ctrl = ctrl.copy()

print("Final Loss:", best_loss)


# -----------------------------
# SET BEST POSE
# -----------------------------
data.ctrl[:] = best_ctrl
for _ in range(150):
    mujoco.mj_step(model, data)


# -----------------------------
# VISUALIZE RESULT
# -----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
