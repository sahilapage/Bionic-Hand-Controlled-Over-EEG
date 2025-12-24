import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

# -----------------------------
# Constants
# -----------------------------
MODEL_PATH = "mjcf/scene.xml"
CTRL_LOW = -1.57079633
CTRL_HIGH = 1.57079633
CTRL_STEPS = 10   # physics steps per action

# -----------------------------
# Utility: Action Scaling
# -----------------------------
def scale_action(action):
    return CTRL_LOW + (action + 1.0) * 0.5 * (CTRL_HIGH - CTRL_LOW)

# -----------------------------
# Hand Gym Environment
# -----------------------------
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
            shape=(self.model.nu,),
            dtype=np.float32
        )

        # Observation space (qpos + qvel)
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    # -------------------------
    # Reset
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Small random init (important for RL)
        self.data.qpos[:] += np.random.uniform(-0.05, 0.05, size=self.model.nq)

        obs = self._get_obs()
        return obs, {}

    # -------------------------
    # Step
    # -------------------------
    def step(self, action):
        # Scale action
        ctrl = scale_action(action)
        self.data.ctrl[:] = ctrl

        # Step physics
        for _ in range(CTRL_STEPS):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = False

        if self.render_mode:
            self.render()

        return obs, reward, terminated, truncated, {}

    # -------------------------
    # Observation
    # -------------------------
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()
        ])

    # -------------------------
    # Reward (temporary)
    # -------------------------
    def _compute_reward(self):
        # Penalize high velocities (smooth motion)
        return -np.sum(self.data.qvel ** 2)

    # -------------------------
    # Render
    # -------------------------
    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    # -------------------------
    # Close
    # -------------------------
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


# -----------------------------
# Test Environment
# -----------------------------
if __name__ == "__main__":
    env = HandEnv(render=True)

    obs, _ = env.reset()
    print("Observation shape:", obs.shape)

    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        time.sleep(0.01)

        if step % 100 == 0:
            print(f"step={step}, reward={reward:.4f}")

    env.close()
