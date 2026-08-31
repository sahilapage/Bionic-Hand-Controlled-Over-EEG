"""Minimal MuJoCo/Gymnasium base class.

Deliberately thinner than `gymnasium.envs.mujoco.MujocoEnv`: it owns model
loading, the physics step and the three rendering paths, and leaves the
observation, reset and reward entirely to the subclass.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np


class MujocoEnv(gym.Env):
    """Loads a MuJoCo model and steps it `frame_skip` times per action."""

    def __init__(self, model_path: str, frame_skip: int,
                 render_mode: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        self._viewer = None
        self._renderer = None
        self._cam_renderer = None

    # -- subclass contract --------------------------------------------------
    def reset_model(self) -> None:
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        raise NotImplementedError

    # -- Gymnasium API ------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.reset_model()
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def do_simulation(self, ctrl: np.ndarray, n_frames: int) -> None:
        """Apply a full-length ctrl vector and step physics `n_frames` times."""
        self.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            mujoco.mj_step(self.model, self.data)

    # -- rendering ----------------------------------------------------------
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # The scene ships a `tracking_camera` framed on the cube. Falling
                # back to the free camera is fine; a missing camera is not worth
                # aborting a rollout over.
                try:
                    cam_id = self.model.camera("tracking_camera").id
                    with self._viewer.lock():
                        self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        self._viewer.cam.fixedcamid = cam_id
                except KeyError:
                    pass
            if self._viewer.is_running():
                self._viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

        return None

    def render_camera_frame(self, camera_id: int, width: int = 240,
                            height: int = 240) -> np.ndarray:
        """Offscreen render from a named camera, for vision-in-the-loop work."""
        if self._cam_renderer is None:
            self._cam_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._cam_renderer.update_scene(self.data, camera=camera_id)
        return self._cam_renderer.render()

    def viewer_running(self):
        """False once a human-render viewer window has been closed.

        Scripts that drive a rollout need this: without it, closing the window
        leaves the loop simulating into a dead renderer until the episode
        happens to end.
        """
        return self._viewer is None or self._viewer.is_running()

    def close(self):
        for attr in ("_viewer", "_renderer", "_cam_renderer"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.close()
                setattr(self, attr, None)
