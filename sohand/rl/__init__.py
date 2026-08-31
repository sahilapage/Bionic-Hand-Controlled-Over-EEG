"""Training, evaluation and deployment for the in-hand rotation policy.

`sohand.rl.actor` is deliberately torch-free: a deterministic SAC action is
`tanh(mu(latent_pi(obs)))`, an MLP and a squash, so the control loop on the
robot runs from a plain `.npz` without stable-baselines3 installed.
"""

from sohand.rl.actor import NumpyActor, SinusoidGait, load_actor

__all__ = ["NumpyActor", "SinusoidGait", "load_actor"]
