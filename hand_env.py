import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

# --------------------------------------------------
# Config
# --------------------------------------------------
MODEL_PATH = "mjcf/scene.xml"

CTRL_LOW = -1.57079633   # -pi/2
CTRL_HIGH = 1.57079633  # +pi/2
CTRL_STEPS = 5          # physics steps per action

motor_qpos_addr = [0, 12, 17, 29, 34, 46, 51, 63]
motor_qvel_addr = [0, 10, 14, 24, 28, 38, 42, 52]

N_MOTORS = len(motor_qpos_addr)

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def scale_action(action):
    action = np.clip(action, -1.0, 1.0)
    return CTRL_LOW + (action + 1.0) * 0.5 * (CTRL_HIGH - CTRL_LOW)

# --------------------------------------------------
# Environment
# --------------------------------------------------
class HandEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render
        self.viewer = None

        # Action space (8 motors)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(N_MOTORS,),
            dtype=np.float32
        )

        # Observation space: qpos + qvel + error
        obs_dim = N_MOTORS * 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.max_steps = 150
        self.step_count = 0

        # Peace sign target (✌️)
        self.target_qpos = np.array([
            -0.157, 0.848, -0.581, 0.456,
             1.57, -1.57, 1.57, -1.57
        ], dtype=np.float32)

    # --------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Neutral pose
        for idx in motor_qpos_addr:
            self.data.qpos[idx] = 0.0

        self.step_count = 0
        return self._get_obs(), {}

    # --------------------------------------------------
    def step(self, action):
        alpha = 0.2

        # Smooth control
        self.data.ctrl[:] = (
            (1 - alpha) * self.data.ctrl[:] +
            alpha * scale_action(action)
        )

        # Physics stepping
        for _ in range(CTRL_STEPS):
            mujoco.mj_step(self.model, self.data)

        # Velocity damping (post-physics)
        self.data.qvel[motor_qvel_addr] *= 0.95

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps

        if self.render_mode:
            self.render()

        return obs, reward, terminated, truncated, {}

    # --------------------------------------------------
    def _get_obs(self):
        qpos = self.data.qpos[motor_qpos_addr].astype(np.float32)
        qvel = self.data.qvel[motor_qvel_addr].astype(np.float32)
        error = (self.target_qpos - qpos).astype(np.float32)
        return np.concatenate([qpos, qvel, error])

    # --------------------------------------------------
    def _compute_reward(self):
        pos = self.data.qpos[motor_qpos_addr]
        vel = self.data.qvel[motor_qvel_addr]

        error = np.abs(pos - self.target_qpos)
        l2_error = np.linalg.norm(error)
        worst_error = np.max(error)

        # Dense shaping
        position_reward = 5.0 * np.exp(-0.5 * l2_error)

        # Success bonus
        bonus = 5.0 if worst_error < 0.05 else 0.0

        # Smoothness penalty
        velocity_penalty = -0.001 * np.sum(vel ** 2)

        return position_reward + bonus + velocity_penalty

    # --------------------------------------------------
    def _is_terminated(self):
        error = np.abs(self.data.qpos[motor_qpos_addr] - self.target_qpos)

        # Success
        if np.max(error) < 0.05:
            return True

        # Safety failure
        if np.any(np.abs(self.data.qpos[motor_qpos_addr]) > 3.0):
            return True

        return False

    # --------------------------------------------------
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data
            )
        self.viewer.sync()

    # --------------------------------------------------
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# --------------------------------------------------
# Quick sanity test (optional)
# --------------------------------------------------
if __name__ == "__main__":
    env = HandEnv(render=False)

    obs, _ = env.reset()
    print("Observation shape:", obs.shape)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        print(f"reward: {reward:.3f}")

        if done or trunc:
            obs, _ = env.reset()

    env.close()
