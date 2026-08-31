"""How much does a trained policy's episode outcome depend on the exact seed?

    python -m sohand.rl.sensitivity --run 1 --episodes 16

Runs each seed twice, the second time with the cube's start position moved by
`--epsilon` metres (1 nm by default -- far below any physical or sensing
resolution), and reports how far the two outcomes diverge.

This exists because contact-rich dynamics are chaotic, and that has a direct
consequence for how results from this repo may be reported: if a perturbation
five orders of magnitude below machine-relevant scales changes a single
episode's revolution count by more than the difference you are claiming, then a
handful of episodes -- or one demo video -- measures nothing. The distribution
is stable; individual episodes are not.
"""

import argparse

import mujoco
import numpy as np

from sohand.envs import AmazingHandSpinEnv, make_spin_cfg
from sohand.envs.spin import SPIN_AXES
from sohand.rl.actor import NumpyActor, RUNS
from sohand.rl.evaluate import MAX_TRACKABLE_SWING
from sohand.rotations import twist_and_swing, unwrap_delta


def episode(env, policy, axis, seed, epsilon):
    obs, _ = env.reset(seed=seed)
    if epsilon:
        # Perturb the cube's x start position, then re-derive the observation
        # so the policy's first action sees the perturbed state.
        env.data.qpos[env.cube_qpos] += epsilon
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()

    prev, swing = twist_and_swing(env.cube_rotation(), axis)
    net, worst, done, k, info = 0.0, swing, False, 0, {}
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, k))
        yaw, swing = twist_and_swing(env.cube_rotation(), axis)
        net += unwrap_delta(yaw, prev)
        prev, worst = yaw, max(worst, swing)
        done, k = term or trunc, k + 1
    # Episodes that tip past the trackable swing report a meaningless twist, so
    # they are marked rather than silently compared.
    return net / (2 * np.pi), float(info.get("ep_dropped", 0.0)), worst


def main(args):
    actor, scene, close = NumpyActor.for_run(args.run)
    kw = {}
    if scene:
        kw["model_path"] = scene
    if close is not None:
        kw["cfg"] = make_spin_cfg(hand_close_frac=close)
    env = AmazingHandSpinEnv(randomize=False, sensor_noise=False, **kw)
    axis = SPIN_AXES[env.cfg.spin_axis]

    base, perturbed = [], []
    print(f"run {args.run} ({RUNS[args.run][0]})   perturbation {args.epsilon:g} m\n")
    for i in range(args.episodes):
        seed = args.seed + i
        rev_a, drop_a, sw_a = episode(env, actor, axis, seed, 0.0)
        rev_b, drop_b, sw_b = episode(env, actor, axis, seed, args.epsilon)
        if max(sw_a, sw_b) > MAX_TRACKABLE_SWING:
            print(f"  seed {seed}: skipped -- the cube tipped past "
                  f"{np.degrees(MAX_TRACKABLE_SWING):.0f} deg, "
                  f"so net twist is not defined")
            continue
        base.append(rev_a)
        perturbed.append(rev_b)
        print(f"  seed {seed}: {rev_a:+.3f} vs {rev_b:+.3f} rev   "
              f"change {rev_b - rev_a:+.3f}   drop {drop_a:.0f}/{drop_b:.0f}")
    env.close()

    a, b = np.array(base), np.array(perturbed)
    n = len(a)
    if n < 2:
        raise SystemExit("too few trackable episodes to compare")
    print(f"\n  mean          {a.mean():+.3f} vs {b.mean():+.3f}")
    print(f"  sd            {a.std(ddof=1):.3f} vs {b.std(ddof=1):.3f}")
    print(f"  |change|      mean {np.abs(b - a).mean():.3f} rev, "
          f"max {np.abs(b - a).max():.3f} rev")
    print(f"  correlation   {np.corrcoef(a, b)[0, 1]:+.3f} across {n} seeds")
    print(f"\n  Per-episode outcomes are {'NOT ' if np.abs(b - a).mean() > 0.1 else ''}"
          f"reproducible at this perturbation scale.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, default=1, choices=tuple(RUNS))
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--epsilon", type=float, default=1e-9,
                   help="cube start-position perturbation, metres")
    main(p.parse_args())
