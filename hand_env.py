import time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

MODEL_PATH = "AmazingHand/Demo/AHSimulation/AHSimulation/AH_Right/mjcf/scene.xml"
CTRL_LOW = -1.57079633  # -pi/2
CTRL_HIGH = 1.57079633  # pi/2
CTRL_STEPS = 5   # physics steps per action

motor_qpos_addr = [0, 12, 17, 29, 34, 46, 51, 63]
motor_qvel_addr = [0, 10, 14, 24, 28, 38, 42, 52]

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
        self.n_act = len(motor_qpos_addr)
        obs_dim = self.n_act * 3

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

        for idx in motor_qpos_addr:
            self.data.qpos[idx] = 0.0

        self.target_qpos = np.zeros(len(motor_qpos_addr), dtype=np.float32)

        self.step_count = 0
        return self._get_obs(), {}

    def _is_terminated(self):
        finger_positions = self.data.qpos[motor_qpos_addr]
        if np.any(np.abs(finger_positions) > 3.0):
            return True
        return False

    def step(self, action):
        # Scale action
        alpha = 0.2
        self.data.ctrl[:] = (1 - alpha) * self.data.ctrl[:] + alpha * scale_action(action)

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
        qpos = self.data.qpos[motor_qpos_addr].astype(np.float32)
        qvel = self.data.qvel[motor_qvel_addr].astype(np.float32)
        error = (self.target_qpos - qpos).astype(np.float32)

        return np.concatenate([qpos, qvel, error]).astype(np.float32)

    def _compute_reward(self):

        self.data.qvel[motor_qvel_addr] *= 0.95

        correct_position = np.array([-0.157, 0.848, -0.581, 0.456, 1.57, -1.57, 1.57, -1.57])

        current_position = self.data.qpos[motor_qpos_addr]

        current_velocity = self.data.qvel[motor_qvel_addr]

        error_per_finger = np.abs(current_position - correct_position)
        l2_normalised_error = np.linalg.norm(error_per_finger)
        worst_finger_error = np.max(error_per_finger)
        max_velocity = np.max(current_velocity)
        reward_for_position = 10 * np.exp(0.5 * -l2_normalised_error)   # 0.5 is the "harshness", max value for position reward is 10
        
        if worst_finger_error < 0.05:
            bonus = 50
        elif worst_finger_error < 0.1:
            bonus = 20.0
        else:
            bonus = 0.0

        velocity_penalty = -0.0001 * np.sum(np.square(current_velocity)) 

        total_reward = bonus + reward_for_position + velocity_penalty
        
        return total_reward

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
    episode_reward = 0.0
    for step in range(1000):    
        action = env.action_space.sample()
        obs, reward, done, trunc, _ = env.step(action)
        episode_reward += reward
        # print(reward)
        
        time.sleep(0.01)

        if done or trunc:
            print("\n -------- process has been reset due to end of episode or unexpected movement -------- ")
            print(f"episode reward: {episode_reward} | number of steps: {step}")
            episode_reward = 0 
            obs, _ = env.reset()

        # if step % 100 == 0:
        #     print(f"step={step}, reward={reward:.4f}")

    env.close()