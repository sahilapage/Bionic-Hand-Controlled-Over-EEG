"""Evaluate a rotation policy, in the units the task is defined in.

    python -m sohand.rl.evaluate --run 1 --episodes 120
    python -m sohand.rl.evaluate --run 2 --randomize
    python -m sohand.rl.evaluate --model runs/spin/models/best_model
    python -m sohand.rl.evaluate --gait                  # open-loop CEM baseline
    python -m sohand.rl.evaluate                         # do-nothing floor

Two rotation numbers are reported and they are not the same quantity:

  credited   the integral of omega . k_hat that the reward pays for. Cyclic
             wobble accumulates a nonzero geometric phase in this integral, so
             it can be farmed by a policy that shakes the cube in place.
  swing-twist  the twist angle of the cube's *orientation* about k_hat, a pure
             function of pose, so wobble contributes exactly zero to it.

Their ratio is the honesty check on the headline figure: well below 1.0 means
the policy is being paid for motion it is not converting into net rotation.

The check is only available on episodes where it is well conditioned. The twist
about a fixed axis becomes undefined as the object flips -- swing -> pi -- and
this hand rolls the cube face over face rather than spinning it cleanly, so a
minority of episodes pass close enough to a flip that accumulated twist is
meaningless there (observed: a single episode reporting -4.4 revolutions on a
rollout that genuinely turned +2.7). Those episodes are excluded from the
swing-twist figure and counted separately rather than averaged in.

Nominal physics with clean sensors measures skill; `--randomize` measures
robustness. They answer different questions and conflating them is how a policy
that only works in the nominal model gets called solved.
"""

import argparse

import numpy as np

from sohand import paths
from sohand.envs import AmazingHandSpinEnv, SPIN_CFG, make_spin_cfg
from sohand.envs.spin import SPIN_AXES
from sohand.rl.actor import NumpyActor, RUNS, SinusoidGait
from sohand.rotations import twist_and_swing, unwrap_delta


# Beyond this swing angle the twist about a fixed axis is too ill-conditioned
# to accumulate: n = cos(swing / 2) is the divisor, so at 150 deg a 1 mrad
# orientation change can move the reported twist by degrees.
MAX_TRACKABLE_SWING = np.radians(150.0)


def rollout(env, policy, axis, seed, frames=None):
    """One episode -> (return, info, net twist in radians, max swing seen)."""
    obs, _ = env.reset(seed=seed)
    prev, swing = twist_and_swing(env.cube_rotation(), axis)
    net, ret, k, worst_swing = 0.0, 0.0, 0, swing
    done, info = False, {}
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs, k))
        ret += r
        yaw, swing = twist_and_swing(env.cube_rotation(), axis)
        net += unwrap_delta(yaw, prev)
        prev = yaw
        worst_swing = max(worst_swing, swing)
        if frames is not None:
            img = env.render()
            if img is not None:
                frames.append(img)
        done, k = term or trunc, k + 1
    return ret, info, net, worst_swing


def build(args):
    """Environment, policy and a label."""
    scene, close = args.scene, args.close_frac
    if args.run is not None:
        _, run_scene, run_close = RUNS[args.run]
        if scene is None and run_scene:
            scene = f"{paths.MJCF_DIR}/cube/{run_scene}"
        if close is None:
            close = run_close

    kw = dict(randomize=args.randomize, sensor_noise=args.randomize,
              render_mode="rgb_array" if args.video else None)
    if scene:
        kw["model_path"] = scene
    if close is not None:
        kw["cfg"] = make_spin_cfg(hand_close_frac=close)
    env = AmazingHandSpinEnv(**kw)

    if args.gait:
        gait_file = args.gait_file or paths.require_checkpoint("gait_cem.npy")
        return env, SinusoidGait.load(gait_file, env.dt), "open-loop CEM gait"
    if args.run is not None:
        actor, _, _ = NumpyActor.for_run(args.run)
        return env, actor, f"run {args.run} ({RUNS[args.run][0]})"
    if args.actor:
        return env, NumpyActor(args.actor), args.actor
    if args.model:
        from stable_baselines3 import SAC
        model = SAC.load(args.model, device="cpu")

        def policy(obs, k):
            return model.predict(obs, deterministic=not args.stochastic)[0]
        return env, policy, args.model
    return env, (lambda obs, k: np.zeros(8, np.float32)), "do-nothing"


def main(args):
    env, policy, label = build(args)
    axis = SPIN_AXES[env.cfg.spin_axis]
    frames = [] if args.video else None

    credited, twist, swings = [], [], []
    rates, drops, rets, steps = [], [], [], []
    for i in range(args.episodes):
        ret, info, net, swing = rollout(
            env, policy, axis, seed=args.seed + i,
            frames=frames if (args.video and i == 0) else None)
        credited.append(float(info.get("ep_revolutions", 0.0)))
        twist.append(net / (2 * np.pi))
        swings.append(swing)
        rates.append(float(info.get("ep_rot_per_sec", 0.0)))
        drops.append(float(info.get("ep_dropped", 0.0)))
        steps.append(int(info.get("ep_steps", 0)))
        rets.append(ret)
    cube_cm = env.cube_half * 200
    max_steps = env.cfg.max_steps
    env.close()

    credited = np.array(credited)
    twist, swings = np.array(twist), np.array(swings)
    n = len(credited)
    sem = credited.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    ok = swings <= MAX_TRACKABLE_SWING

    print(f"\npolicy: {label}   episodes={n}   randomize={args.randomize}   "
          f"cube {cube_cm:.1f} cm")
    print(f"  revolutions (credited)   {credited.mean():+.3f} +- {sem:.3f} SEM"
          f"   [{credited.min():+.3f} .. {credited.max():+.3f}]")
    if ok.any():
        ratio = (f"   ratio {twist[ok].mean() / credited[ok].mean():.3f}"
                 if abs(credited[ok].mean()) > 1e-9 else "")
        print(f"  revolutions (swing-twist){twist[ok].mean():+.3f}{ratio}"
              f"   over {int(ok.sum())}/{n} trackable episodes")
    else:
        print(f"  revolutions (swing-twist)  n/a -- every episode passed within "
              f"{np.degrees(MAX_TRACKABLE_SWING):.0f} deg of a flip")
    if not ok.all():
        print(f"  (excluded {int((~ok).sum())} episodes where the cube tipped "
              f"past {np.degrees(MAX_TRACKABLE_SWING):.0f} deg and the twist "
              f"stops being defined)")
    print(f"  degrees                  "
          f"{np.degrees(credited.mean() * 2 * np.pi):+.1f}")
    print(f"  rad/s                    {np.mean(rates):+.4f}")
    print(f"  return                   {np.mean(rets):+.2f}")
    print(f"  drop rate                {np.mean(drops):.3f}"
          f"   mean steps {np.mean(steps):.0f} / {max_steps}")
    for bar in (0.25, 0.5, 1.0, 2.0):
        print(f"  >= {bar:>4.2f} rev             {float(np.mean(credited >= bar)):.3f}")
    print(f"  SUCCESS (>= {SPIN_CFG.success_revolutions:.1f} rev)      "
          f"{float(np.mean(credited >= SPIN_CFG.success_revolutions)):.3f}")

    if args.video and frames:
        import imageio
        imageio.mimwrite(args.video, frames, fps=50, quality=8)
        print(f"  video -> {args.video}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run", type=int, default=None, choices=tuple(RUNS),
                   help="a released checkpoint, with its own scene and closure")
    g.add_argument("--actor", default=None, help="an exported actor .npz")
    g.add_argument("--model", default=None, help="an SB3 .zip (needs torch)")
    g.add_argument("--gait", action="store_true", help="open-loop CEM baseline")
    p.add_argument("--gait-file", default=None)
    p.add_argument("--scene", default=None)
    p.add_argument("--close-frac", type=float, default=None)
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--randomize", action="store_true",
                   help="domain randomisation + sensor noise (robustness)")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--video", default=None)
    main(p.parse_args())
