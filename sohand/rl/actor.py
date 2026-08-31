"""Replay a trained SAC actor with nothing but NumPy.

A deterministic SAC action is `tanh(mu(latent_pi(obs)))` -- two hidden layers
and a squash. Exporting those five arrays to an `.npz` (see
`sohand.rl.export_actor`) means the deployment target needs neither torch nor
stable-baselines3: a Raspberry Pi evaluates this in microseconds.

The observation must be built in exactly the order and scaling that
`sohand.envs.spin.AmazingHandSpinEnv._get_obs` uses, and the action must be put
through the same pipeline the env applies (grasp band -> low-pass -> slew
limit). Both are part of the policy, not of the simulator.
"""

import os

import numpy as np

from sohand import paths

# run -> (actor weights, scene, grasp closure). Each run has its own geometry
# and closure, so they travel together: pointing run 2's policy at run 1's
# scene feeds it a cube of the wrong size in the wrong place.
RUNS = {
    1: ("actor_run1.npz", None, None),          # 4.7 cm cube, the baseline
    2: ("actor_run2.npz", "scene_spin.xml", 0.20),   # 6.1 cm, shifted to finger1
}


def load_actor(path):
    """Read an exported actor `.npz` into a list of (weight, bias) pairs."""
    z = np.load(path)
    return [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]


class NumpyActor:
    """Deterministic SAC policy: ReLU MLP with a tanh-squashed output."""

    def __init__(self, path):
        self.path = path
        self.layers = load_actor(path)
        self.obs_dim = int(self.layers[0][0].shape[1])
        self.act_dim = int(self.layers[-1][0].shape[0])

    @classmethod
    def for_run(cls, run):
        """Load the actor shipped for `run`, plus its scene and grasp closure."""
        name, scene, close = RUNS[run]
        actor = cls(paths.require_checkpoint(name))
        scene_path = os.path.join(paths.MJCF_DIR, "cube", scene) if scene else None
        return actor, scene_path, close

    def __call__(self, obs, step=None):
        x = np.asarray(obs, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)      # ReLU on the hidden layers only
        return np.tanh(x).astype(np.float32)   # SAC squashes the mean


class SinusoidGait:
    """The open-loop CEM baseline: one sinusoid per joint, 24 parameters.

    The floor a closed-loop policy has to clear. The shipped gait holds the cube
    almost indefinitely (drop rate 0.05) and turns it +0.061 +- 0.009
    revolutions per 20 s episode -- 0.014 rad/s, n=60, measured with
    `sohand.rl.evaluate --gait`. That is 20x a do-nothing policy and 1/22 of the
    0.314 rad/s one full revolution requires.

    It is well below the 0.10-0.12 rad/s reported when the gait was searched;
    the search evidently ran against a configuration this repository does not
    ship. The measured number is the one to compare against.
    """

    def __init__(self, params, dt):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (24,):
            raise ValueError(f"expected 24 gait parameters, got {params.shape}")
        self.amp = np.clip(params[0:8], 0.0, 1.0)
        self.freq = np.clip(params[8:16], 0.03, 2.0)
        self.phase = params[16:24]
        self.dt = dt

    @classmethod
    def load(cls, path, dt):
        return cls(np.load(path), dt)

    def __call__(self, obs, step):
        phase = 2 * np.pi * self.freq * (step * self.dt) + self.phase
        return np.clip(self.amp * np.sin(phase), -1.0, 1.0).astype(np.float32)
