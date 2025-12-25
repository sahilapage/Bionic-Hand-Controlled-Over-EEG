import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym


MODEL_PATH = "mjcf/scene.xml"
CTRL_LOW = -1.57079633
CTRL_HIGH = 1.57079633
CTRL_STEPS = 10   # physics steps per action

def scale_action(action):
    action = np.clip(action, -1.0, 1.0)
    return CTRL_LOW + (action + 1.0) * 0.5 * (CTRL_HIGH - CTRL_LOW)


class HandEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render
        self.viewer = None

        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )

        # Observation space
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.step_count = 0
        self.max_steps = 200

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial pose slightly
        self.data.qpos[:] += np.random.uniform(
            -0.05, 0.05, size=self.model.nq
        )

        self.step_count = 0

        obs = self._get_obs()
        return obs, {}


    def _is_terminated(self):
        if np.any(np.abs(self.data.qpos) > 3.0):
            return True
        return False


    def step(self, action):
        # Scale action
        self.data.ctrl[:] = scale_action(action)

        # Step physics
        for _ in range(CTRL_STEPS):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps

        if self.render_mode:
            self.render()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos.copy(),
            self.data.qvel.copy()

    def _compute_reward(self):
        vel_penalty = np.sum(self.data.qvel ** 2)
        ctrl_penalty = np.sum(self.data.ctrl ** 2)
        return -0.001 * vel_penalty - 0.0001 * ctrl_penalty

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data
            )
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



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

