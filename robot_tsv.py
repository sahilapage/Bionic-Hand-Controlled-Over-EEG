import mujoco
import mujoco.viewer
import numpy as np
import time

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


output_file = "robot_tsv.txt"
num_frames = 1000

with mujoco.viewer.launch_passive(model, data) as viewer:
    with open(output_file, "w") as f:

        for frame in range(num_frames):

            # -------------------
            # Apply random control
            # -------------------
            data.ctrl[:] = np.random.uniform(
                low=-0.5, high=0.5, size=model.nu
            )

            mujoco.mj_step(model, data)
            viewer.sync()

            robot_tsv = compute_robot_tsv(model, data)
            flat = robot_tsv.flatten()
            f.write(" ".join(map(str, flat)) + "\n")

            time.sleep(model.opt.timestep)

print("TSV data saved to robot_tsv.txt")
