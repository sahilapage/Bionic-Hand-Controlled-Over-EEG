"""SAC for continuous in-hand cube rotation on the Amazing Hand.

    python -m sohand.rl.train_sac --n-envs 12 --timesteps 8000000
    python -m sohand.rl.train_sac --scene mjcf/cube/scene_spin.xml --close-frac 0.20
    python -m sohand.rl.train_sac --resume runs/spin/models/checkpoints/resume


WHAT IS DIFFERENT FROM THE THREE PREVIOUS RUNS, AND WHY
-------------------------------------------------------
Run 1 (SAC, face targets, gamma 0.995, ent_coef auto): critic loss went
0.0155 -> 2.47e11, the auto-tuned entropy coefficient chased it to 2098, and Q
reached 1.16e7 against an achievable return bound of ~49. Solve rate 0.060.

Run 2 (SAC, face targets, gamma 0.98, ent_coef fixed 0.05, terminate on
success): critic loss still went 0.017 -> 2.4e7, Q ~5.65e4. Solve rate 0.070.

Fixing gamma and pinning the entropy coefficient slowed the divergence by four
orders of magnitude but did not stop it, because neither was the cause. Three
things are changed here, in the order they matter:

1. THE TASK AND THE REWARD (in `sohand/envs/spin.py`). The old reward paid
   freezing better than rotating -- measured, -0.48 for do-nothing against
   -1.41 for a rollout that genuinely turned the cube 71 deg -- and the sparse
   +-20 events it was built around are what the critic had to represent. The
   new reward is dense, bounded at +-0.5 per step, and structurally incapable
   of preferring a stationary policy.

2. LAYERNORM IN THE CRITIC. Bootstrapped value estimates diverge when the Q
   network extrapolates confidently off-distribution; LayerNorm bounds the
   pre-activation scale of every hidden layer and is the single cheapest known
   fix (it is what CrossQ, BRO and DroQ all rely on). SB3's stock critic has
   none, which is why runs 1 and 2 both diverged with different gammas.

3. A BOUNDED RETURN. The rotation term is an increment, so the max per-step
   reward is exactly w_rot * angvel_clip * dt = 8 * 0.8 * 0.02 = 0.128, every
   penalty only subtracts, and every episode ends within 1000 steps. At
   gamma 0.98 that bounds |Q| at 6.4 -- a number the critic can actually
   represent. Runs 1 and 2 were trying to fit an unbounded chain of +-20
   sparse events.

Everything else is deliberately conventional. The failure was never the
optimiser.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                CheckpointCallback)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.sac.policies import SACPolicy

from sohand.envs import AmazingHandSpinEnv, SPIN_CFG, make_spin_cfg
from sohand.paths import CUBE_SCENE

DEFAULT_RUN_DIR = "runs/spin"

EVAL_EVERY_STEPS = 100_000
VIDEO_EVERY_STEPS = 250_000
CKPT_EVERY_STEPS = 50_000
RESUME_CKPT_EVERY_STEPS = 200_000
DIAG_EVERY_STEPS = 10_000

# Monitor snapshots only the *final* step's info dict, which is exactly where
# the env writes its ep_* episode totals.
INFO_KEYWORDS = (
    "ep_rotation_rad", "ep_revolutions", "ep_rot_per_sec", "ep_dropped",
    "ep_steps", "ep_reset_tries", "success", "min_finger_ema",
    "reached_quarter_turn", "reached_half_turn", "reached_full_turn",
    "reached_two_turns",
    "ep_r_rot", "ep_r_offaxis", "ep_r_linvel", "ep_r_drift", "ep_r_pose",
    "ep_r_work", "ep_r_torque", "ep_r_action", "ep_r_contact", "ep_r_finger",
    "ep_r_drop",
)
EP_REWARD_KEYS = tuple(k for k in INFO_KEYWORDS if k.startswith("ep_r_"))


# ---------------------------------------------------------------------------
# LayerNorm critic
# ---------------------------------------------------------------------------
def _layernorm_mlp(in_dim: int, net_arch: List[int], activation_fn) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for h in net_arch:
        layers += [nn.Linear(last, h), nn.LayerNorm(h), activation_fn()]
        last = h
    layers.append(nn.Linear(last, 1))
    return nn.Sequential(*layers)


class LayerNormContinuousCritic(ContinuousCritic):
    """SB3's twin Q network with LayerNorm after every hidden layer.

    Rebuilding the q networks after `super().__init__` (rather than copying the
    parent's body) keeps every other piece of SB3's critic contract intact --
    feature extractor sharing, `n_critics`, image normalisation -- and
    `add_module` under the same `qf{i}` name replaces the stock network so no
    orphan parameters reach the optimiser.
    """

    def __init__(self, observation_space, action_space, net_arch, features_extractor,
                 features_dim, activation_fn=nn.ReLU, normalize_images=True,
                 n_critics=2, share_features_extractor=True):
        super().__init__(observation_space, action_space, net_arch, features_extractor,
                         features_dim, activation_fn, normalize_images, n_critics,
                         share_features_extractor)
        action_dim = int(np.prod(action_space.shape))
        self.q_networks = []
        for i in range(n_critics):
            q = _layernorm_mlp(features_dim + action_dim, net_arch, activation_fn)
            self.add_module(f"qf{i}", q)
            self.q_networks.append(q)


class LayerNormSACPolicy(SACPolicy):
    def make_critic(self, features_extractor=None):
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormContinuousCritic(**kwargs).to(self.device)


SAC_HPARAMS = dict(
    learning_rate=3e-4,
    buffer_size=600_000,      # 74-dim obs -> ~400 MB; the box has ~13 GB free
    learning_starts=25_000,
    batch_size=512,
    tau=0.005,
    # Return bound with max per-step reward 0.5: 0.5 / (1 - 0.98) = 25.
    # The effective horizon, 50 steps = 1 s, is about one gait cycle, which is
    # the credit-assignment span this task actually needs -- the rotation
    # reward is paid the instant the cube turns, not at the end of a sequence.
    gamma=0.98,
    # Auto-tuning is kept: it is scale-robust, and run 1's alpha blow-up was a
    # *consequence* of the critic diverging, not its cause. The target is
    # raised from SB3's default -dim(A) = -8 because 8 near-maximally-entropic
    # joint commands on a grasped object is not exploration, it is dropping the
    # cube; -6 still keeps the policy stochastic.
    ent_coef="auto_0.05",
    target_entropy=-6.0,
    policy_kwargs=dict(net_arch=[512, 512], activation_fn=nn.ReLU),
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class SpinDiagnosticCallback(BaseCallback):
    """Rollout statistics in the units the task is defined in: revolutions."""

    def __init__(self, every_steps: int = DIAG_EVERY_STEPS):
        super().__init__()
        self.every_steps = every_steps

    def _on_step(self) -> bool:
        freq = max(self.every_steps // self.training_env.num_envs, 1)
        if self.n_calls % freq != 0:
            return True
        buf = self.model.ep_info_buffer
        if not buf:
            return True

        def m(key):
            vals = [ep[key] for ep in buf if key in ep]
            return float(safe_mean(vals)) if vals else float("nan")

        stats = {
            "success_rate": m("success"),                 # >= 1 full revolution
            "revolutions": m("ep_revolutions"),
            "rot_deg": np.degrees(m("ep_rotation_rad")),
            "rad_per_s": m("ep_rot_per_sec"),
            "quarter_turn": m("reached_quarter_turn"),
            "half_turn": m("reached_half_turn"),
            "two_turns": m("reached_two_turns"),
            "drop_rate": m("ep_dropped"),
            "min_finger_ema": m("min_finger_ema"),
            "ep_steps": m("ep_steps"),
            "reset_tries": m("ep_reset_tries"),
        }
        print(f"[DIAG {self.num_timesteps}] "
              + " ".join(f"{k}={v:.3f}" for k, v in stats.items())
              + " | " + " ".join(f"{k[5:]}={m(k):.2f}" for k in EP_REWARD_KEYS))
        for k, v in stats.items():
            self.logger.record(f"rollout/{k}", v)
        for k in EP_REWARD_KEYS:
            self.logger.record(f"reward/{k}", m(k))
        return True


class SpinEvalCallback(BaseCallback):
    """Deterministic evaluation on nominal physics with clean sensors.

    Separate from the training statistics on purpose: the training envs run
    domain randomisation and sensor noise, so their numbers measure robustness,
    not whether the skill was learned.
    """

    def __init__(self, eval_env, every_steps: int, save_dir: str,
                 n_episodes: int = 12):
        super().__init__()
        self.eval_env = eval_env
        self.every_steps = every_steps
        self.save_dir = save_dir
        self.n_episodes = n_episodes
        self.best = -np.inf

    def _on_step(self) -> bool:
        freq = max(self.every_steps // self.training_env.num_envs, 1)
        if self.n_calls % freq != 0:
            return True
        revs, drops, rets = [], [], []
        for i in range(self.n_episodes):
            obs = self.eval_env.reset()
            done, ret = False, 0.0
            info = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, dones, infos = self.eval_env.step(action)
                ret += float(r[0])
                done, info = bool(dones[0]), infos[0]
            revs.append(float(info.get("ep_revolutions", 0.0)))
            drops.append(float(info.get("ep_dropped", 0.0)))
            rets.append(ret)
        rev = float(np.mean(revs))
        drop = float(np.mean(drops))
        ret = float(np.mean(rets))
        succ = float(np.mean([r >= SPIN_CFG.success_revolutions for r in revs]))
        print(f"[EVAL {self.num_timesteps}] revolutions={rev:.3f} "
              f"success={succ:.3f} drop={drop:.2f} return={ret:.2f} "
              f"(best {max(self.best, rev):.3f})")
        for k, v in (("revolutions", rev), ("success_rate", succ),
                     ("drop_rate", drop), ("mean_reward", ret)):
            self.logger.record(f"eval/{k}", v)
        if rev > self.best:
            self.best = rev
            os.makedirs(self.save_dir, exist_ok=True)
            self.model.save(os.path.join(self.save_dir, "best_model"))
            print(f"[EVAL] new best ({rev:.3f} rev) saved")
        return True


class ResumeCheckpointCallback(BaseCallback):
    """One rolling checkpoint that includes the replay buffer, so a resume
    continues rather than restarting SAC with an empty buffer."""

    def __init__(self, save_path: str, every_steps: int):
        super().__init__()
        self.save_path = save_path
        self.every_steps = every_steps

    def _on_step(self) -> bool:
        freq = max(self.every_steps // self.training_env.num_envs, 1)
        if self.n_calls % freq != 0:
            return True
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save(os.path.join(self.save_path, "resume"))
        self.model.save_replay_buffer(os.path.join(self.save_path, "resume_buffer"))
        with open(os.path.join(self.save_path, "resume_state.txt"), "w") as fh:
            fh.write(f"{self.num_timesteps}\n")
        print(f"[Resume] checkpoint + buffer saved @ {self.num_timesteps}")
        return True


class CriticHealthCallback(BaseCallback):
    """Abort on critic divergence instead of burning GPU-days on a dead run.

    Both previous runs kept training for millions of steps after the value
    function had already blown past any achievable return. The bound here is
    the real one: max per-step reward / (1 - gamma), with 20x slack.
    """

    def __init__(self, every_steps: int = 20_000):
        super().__init__()
        self.every_steps = every_steps
        # The reward is an *increment*: max per step = w_rot * clip * dt.
        step_max = SPIN_CFG.w_rot * SPIN_CFG.angvel_clip * 0.02
        self.limit = 20.0 * step_max / (1.0 - SAC_HPARAMS["gamma"])

    def _on_step(self) -> bool:
        freq = max(self.every_steps // self.training_env.num_envs, 1)
        if self.n_calls % freq != 0 or self.model.replay_buffer.size() < 1000:
            return True
        data = self.model.replay_buffer.sample(256, env=self.model._vec_normalize_env)
        with th.no_grad():
            q = th.cat(self.model.critic(data.observations, data.actions), dim=1)
        qmax = float(q.abs().max())
        self.logger.record("train/q_abs_max", qmax)
        if not np.isfinite(qmax) or qmax > self.limit:
            print(f"[ABORT] |Q| = {qmax:.3e} exceeds the achievable-return bound "
                  f"{self.limit:.1f}. The critic has diverged; stopping.")
            return False
        return True


def _init_wandb(project, config):
    try:
        import wandb
        from wandb.integration.sb3 import WandbCallback
    except ImportError:
        print("[wandb] not installed -- continuing without it")
        return None
    for mode in ("online", "offline"):
        try:
            os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", mode)
            wandb.init(project=project, config=config,
                       sync_tensorboard=True, resume="allow")
            return WandbCallback(verbose=1)
        except Exception as e:
            print(f"[wandb] {mode} init failed ({e})")
            os.environ["WANDB_MODE"] = "offline"
    return None


# ---------------------------------------------------------------------------
def main(args) -> None:
    run_dir = args.run_dir
    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    scene = os.path.abspath(args.scene) if args.scene else CUBE_SCENE
    if not os.path.exists(scene):
        raise FileNotFoundError(f"MuJoCo scene not found: {scene}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    n_envs = args.n_envs
    hparams = dict(SAC_HPARAMS,
                   train_freq=args.train_freq,
                   gradient_steps=args.gradient_steps)
    if args.buffer_size:
        hparams["buffer_size"] = args.buffer_size
    if args.learning_starts is not None:
        hparams["learning_starts"] = args.learning_starts

    wandb_cb = _init_wandb(args.project, {**{k: str(v) for k, v in hparams.items()},
                                          "n_envs": n_envs,
                                          "total_timesteps": args.timesteps,
                                          "randomize": args.randomize,
                                          "spin_axis": SPIN_CFG.spin_axis,
                                          "seed": args.seed})

    monitor_kwargs = dict(info_keywords=INFO_KEYWORDS)
    # Geometry is a run-level choice, not a code edit. `scene_spin.xml` moves
    # the cube 1.7 cm toward finger1 and grows it to 6.1 cm; measured over 20
    # resets that takes thumb contact from 0.00 to 0.85 and fingers-in-contact
    # from 1.75 to 2.15. It needs a shallower settle -- at close_frac 0.50 the
    # grasp fails its own validation on every attempt.
    env_kw = dict(randomize=args.randomize, sensor_noise=args.randomize,
                  model_path=scene)
    # THE TRADE RUN 2 MADE. Measured at n=120 on nominal physics
    # (`python -m sohand.rl.evaluate --run N --episodes 120`):
    #
    #            revolutions        rad/s   drop rate   steps alive
    #   run 1    2.02 +- 0.08       0.67       0.13         924
    #   run 2    3.21 +- 0.14       1.23       0.39         809
    #
    # Run 2 turns the cube half again as far and nearly twice as fast, and
    # drops it three times as often. At drop_penalty=10 against ~55 return per
    # episode a drop costs under 20%, so the policy correctly concluded that
    # spinning fast and occasionally dropping beats spinning carefully. Raise
    # --drop-penalty if you want the other trade.
    cfg_kw = {}
    if args.close_frac is not None:
        cfg_kw["hand_close_frac"] = args.close_frac
    if args.drop_penalty is not None:
        cfg_kw["drop_penalty"] = args.drop_penalty
    if cfg_kw:
        env_kw["cfg"] = make_spin_cfg(**cfg_kw)
    eval_kw = dict(env_kw, randomize=False, sensor_noise=False)

    train_env = make_vec_env(
        AmazingHandSpinEnv, n_envs=n_envs, seed=args.seed,
        vec_env_cls=SubprocVecEnv, env_kwargs=env_kw, monitor_dir=log_dir,
        monitor_kwargs=monitor_kwargs,
    )
    eval_env = make_vec_env(
        AmazingHandSpinEnv, n_envs=1, seed=args.seed + 10_000,
        vec_env_cls=DummyVecEnv, env_kwargs=eval_kw,
        monitor_kwargs=monitor_kwargs,
    )

    if args.resume and os.path.exists(args.resume + ".zip"):
        model = SAC.load(args.resume, env=train_env, tensorboard_log=log_dir,
                         device=args.device)
        buf = args.resume + "_buffer"
        if os.path.exists(buf + ".pkl"):
            model.load_replay_buffer(buf)
            # size() counts buffer *slots*; each slot holds one transition
            # per env, so the true count is size() * n_envs. Printing the
            # raw number reads like a 12x smaller buffer than was restored.
            n_slots = model.replay_buffer.size()
            print(f"[Resume] replay buffer restored ({n_slots * n_envs} "
                  f"transitions in {n_slots} slots x {n_envs} envs)")
        else:
            print("[Resume] no replay buffer -- SAC restarts with an empty one")
    else:
        policy = "MlpPolicy" if args.no_layernorm else LayerNormSACPolicy
        model = SAC(policy, train_env, device=args.device,
                    tensorboard_log=log_dir, verbose=1, seed=args.seed,
                    **hparams)

    callbacks = [
        SpinDiagnosticCallback(every_steps=args.diag_every),
        SpinEvalCallback(eval_env, args.eval_every, model_dir,
                         n_episodes=args.eval_episodes),
        CriticHealthCallback(),
        CheckpointCallback(save_freq=max(args.ckpt_every // n_envs, 1),
                           save_path=ckpt_dir, name_prefix="sac_spin",
                           save_replay_buffer=False, save_vecnormalize=False),
        ResumeCheckpointCallback(ckpt_dir, args.resume_ckpt_every),
    ]
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    ratio = args.gradient_steps / (args.train_freq * n_envs)
    print(f"[Setup] n_envs={n_envs} replay_ratio={ratio:.2f} "
          f"axis={SPIN_CFG.spin_axis} randomize={args.randomize} "
          f"layernorm_critic={not args.no_layernorm}")

    # progress_bar defaults OFF: SB3's rich progress bar opens a Live display
    # that swallows every print() made during learn(), hiding the [DIAG]/[EVAL]
    # lines -- the entire console signal for a run.
    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                progress_bar=args.progress, reset_num_timesteps=not bool(args.resume),
                log_interval=10)
    model.save(os.path.join(model_dir, "sac_spin_final"))
    print(f"Training complete. Artifacts in {run_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--train-freq", type=int, default=32,
                   help="VecEnv steps collected between update phases")
    p.add_argument("--gradient-steps", type=int, default=64,
                   help="updates per phase; replay ratio = this/(train_freq*n_envs)")
    p.add_argument("--timesteps", type=int, default=8_000_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--project", type=str, default="sohand-cube-spin")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=12)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY_STEPS)
    p.add_argument("--ckpt-every", type=int, default=CKPT_EVERY_STEPS)
    p.add_argument("--resume-ckpt-every", type=int, default=RESUME_CKPT_EVERY_STEPS)
    p.add_argument("--diag-every", type=int, default=DIAG_EVERY_STEPS)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--randomize", dest="randomize", action="store_true", default=True)
    p.add_argument("--no-randomize", dest="randomize", action="store_false")
    p.add_argument("--no-layernorm", action="store_true",
                   help="stock SB3 critic; for an ablation against runs 1-2")
    p.add_argument("--scene", type=str, default=None,
                   help="alternate scene xml, e.g. mjcf/cube/scene_spin.xml")
    p.add_argument("--close-frac", type=float, default=None,
                   help="grasp settle closure; 0.20 for the v2 geometry")
    p.add_argument("--drop-penalty", type=float, default=None,
                   help="terminal cost of dropping; default 10, 30 makes a "
                        "drop cost ~20%% of a good episode instead of 7%%")
    p.add_argument("--run-dir", type=str, default=DEFAULT_RUN_DIR,
                   help="output root; models, logs and checkpoints go here")
    p.add_argument("--progress", action="store_true")
    main(p.parse_args())
