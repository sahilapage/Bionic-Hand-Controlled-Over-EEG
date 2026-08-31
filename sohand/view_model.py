"""Open a MuJoCo model in the interactive viewer, with nothing driving it.

    python -m sohand.view_model                       # mjcf/hand/scene.xml
    python -m sohand.view_model --model cube          # mjcf/cube/scene.xml
    python -m sohand.view_model --model spin
    python -m sohand.view_model --model path/to/scene.xml
    python -m sohand.view_model --ctrl 0.5 0 0.5 0 0.5 0 0.5 0

For inspecting geometry, contacts and joint ranges after editing an XML. To
watch a *policy* instead, use `python -m sohand.rl.view`.
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from sohand import paths

MODELS = {
    "hand": paths.HAND_SCENE,
    "cube": paths.CUBE_SCENE,
    "spin": paths.SPIN_SCENE,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="hand",
                   help=f"one of {', '.join(MODELS)}, or a path to a scene xml")
    p.add_argument("--ctrl", type=float, nargs="*", default=None,
                   help="constant actuator commands to hold, one per actuator")
    args = p.parse_args()

    path = MODELS.get(args.model, args.model)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    if args.ctrl is not None:
        if len(args.ctrl) != model.nu:
            raise SystemExit(f"--ctrl needs {model.nu} values, got {len(args.ctrl)}")
        data.ctrl[:] = np.asarray(args.ctrl)

    print(f"{path}\n  nq={model.nq}  nv={model.nv}  nu={model.nu}  "
          f"nbody={model.nbody}  timestep={model.opt.timestep}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            # Pace to wall-clock, or the sim runs as fast as the CPU allows and
            # the motion is unreadable.
            lag = model.opt.timestep - (time.time() - step_start)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
