import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "mjcf/scene.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

tip_names = ["tip1", "tip2", "tip3", "tip4"]
tip_ids = [model.site(name).id for name in tip_names]

palm_body_id = model.body("r_wrist_interface").id


def compute_robot_tsv(model, data):
    palm_pos = data.xpos[palm_body_id]
    palm_rot = data.xmat[palm_body_id].reshape(3, 3)

    tsvs = []
    for tid in tip_ids:
        tip_pos = data.site_xpos[tid]
        v_world = tip_pos - palm_pos
        v_local = palm_rot.T @ v_world
        tsvs.append(v_local)

    return np.stack(tsvs)  # (4,3)


# ---------------------------------
# SET YOUR MOTOR TARGETS (8 only!)
# ---------------------------------
# Order matches actuator section in XML

desired_ctrl = np.array([
    -0.786,   # finger1_motor1
     1.12,    # finger1_motor2
    -0.613,   # finger2_motor1
     0.337,   # finger2_motor2
     1.57,    # finger3_motor1
    -1.57,    # finger3_motor2
     1.57,    # finger4_motor1
    -1.57     # finger4_motor2
])

# Apply motor targets
data.ctrl[:] = desired_ctrl


# Let physics settle
for _ in range(500):
    mujoco.mj_step(model, data)


# Compute TSV
robot_tsv = compute_robot_tsv(model, data)
print("\nRobot TSV:\n", robot_tsv)


# ---------------------------------
# VISUALIZE STATIC POSE
# ---------------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
