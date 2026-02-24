import time
import mujoco
import numpy as np
import mujoco.viewer

# ────────────────────────────────────────
MODEL_PATH = "mjcf/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

tip_names = ["tip1", "tip2", "tip3", "tip4"]
tip_ids   = [model.site(name).id for name in tip_names]

palm_body_id = model.body("r_wrist_interface").id

# Flag for printing
print_requested = False


# ────────────────────────────────────────
def compute_robot_tsv():
    palm_pos = data.xpos[palm_body_id]
    palm_rot = data.xmat[palm_body_id].reshape(3, 3)

    tsvs = []
    tip_world_positions = []

    for tid in tip_ids:
        tip_pos = data.site_xpos[tid]
        tip_world_positions.append(np.round(tip_pos, 4))

        v_world = tip_pos - palm_pos
        v_local = palm_rot.T @ v_world
        tsvs.append(np.round(v_local, 4))

    return np.array(tsvs), np.array(tip_world_positions)


# ────────────────────────────────────────
def print_state():
    tsv, tip_world = compute_robot_tsv()

    print("\n" + "="*60)
    print("CURRENT ACTUATOR CTRL:")
    print(np.round(data.ctrl, 4))

    print("\nCURRENT JOINT POSITIONS (qpos):")
    print(np.round(data.qpos, 4))

    print("\nFINGERTIP WORLD POSITIONS:")
    print(tip_world)

    print("\nTSV (local wrt palm):")
    print(tsv)
    print("="*60 + "\n")


# ────────────────────────────────────────
def keyboard_callback(keycode):
    global print_requested

    # GLFW keycode for P is 80
    if keycode == 80:
        print_requested = True


# ────────────────────────────────────────
# Initial motor angles
data.ctrl[:] = np.zeros(model.nu)
mujoco.mj_forward(model, data)

print("\nInstructions:")
print("1. Change joint values in MuJoCo right panel.")
print("2. Press 'P' to print ctrl, qpos, tip positions, TSV.\n")

# ────────────────────────────────────────
with mujoco.viewer.launch_passive(
    model=model,
    data=data,
    show_left_ui=False,
    show_right_ui=True,
    key_callback=keyboard_callback
) as viewer:

    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()

        if print_requested:
            print_state()
            print_requested = False

        time.sleep(0.002)

print("Viewer closed.")