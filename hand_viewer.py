import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mjcf\scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch(model, data):
    while True:
        mujoco.mj_step(model, data)
