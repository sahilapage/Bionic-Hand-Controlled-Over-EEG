"""Watch a policy rotate the cube in the MuJoCo viewer.

    python -m sohand.rl.view --run 1              # 4.7 cm cube, the baseline
    python -m sohand.rl.view --run 2              # 6.1 cm cube, shifted to finger1
    python -m sohand.rl.view --run 1 --randomize  # + domain randomisation
    python -m sohand.rl.view --gait               # the open-loop CEM baseline

Needs only numpy, mujoco and gymnasium -- the actor is replayed from a plain
`.npz` rather than dragging torch onto a machine that only wants to watch the
cube turn. No VecNormalize was used in training, so observations are fed raw.

The net yaw printed on screen comes from the swing-twist decomposition of the
cube's orientation about the spin axis, not from integrating angular velocity:
it is a pure function of orientation, so cyclic wobble contributes exactly zero
and the number cannot be inflated by the cube shaking in place. It is marked
`?` once the cube has tipped more than 150 degrees in an episode, because past
that the twist about a fixed axis stops being defined -- see
`sohand.rotations.twist_and_swing`.
"""

import argparse
import time

import numpy as np

from sohand import paths
from sohand.envs import AmazingHandSpinEnv, make_spin_cfg
from sohand.envs.spin import SPIN_AXES
from sohand.rl.actor import NumpyActor, SinusoidGait
from sohand.rl.evaluate import MAX_TRACKABLE_SWING
from sohand.rotations import twist_and_swing, unwrap_delta


def build_env(args):
    """Environment, policy and a label, wired consistently for the chosen run."""
    kw = dict(render_mode="human", randomize=args.randomize,
              sensor_noise=args.randomize)

    if args.gait:
        env = AmazingHandSpinEnv(**kw)
        gait_file = args.gait_file or paths.require_checkpoint("gait_cem.npy")
        return env, SinusoidGait.load(gait_file, env.dt), "open-loop CEM gait"

    if args.actor:
        actor, scene, close = NumpyActor(args.actor), args.scene, args.close_frac
    else:
        actor, scene, close = NumpyActor.for_run(args.run)
        scene = args.scene or scene
        close = args.close_frac if args.close_frac is not None else close

    if scene:
        kw["model_path"] = scene
    if close is not None:
        kw["cfg"] = make_spin_cfg(hand_close_frac=close)

    env = AmazingHandSpinEnv(**kw)
    label = (f"run {args.run}  cube {env.cube_half * 200:.1f} cm  "
             f"close {close if close is not None else 0.50}")
    return env, actor, label


def main(args):
    env, policy, label = build_env(args)
    axis = SPIN_AXES[env.cfg.spin_axis]
    print(f"policy: {label}   randomize={args.randomize}   "
          f"(close the viewer or press Ctrl-C to stop)\n")

    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            obs, _ = env.reset(seed=args.seed + episode)
            prev, swing = twist_and_swing(env.cube_rotation(), axis)
            net, worst_swing = 0.0, swing
            done, k, t0 = False, 0, time.time()
            info = {}

            while not done:
                obs, _, term, trunc, info = env.step(policy(obs, k))
                done = term or trunc

                yaw, swing = twist_and_swing(env.cube_rotation(), axis)
                net += unwrap_delta(yaw, prev)
                prev = yaw
                worst_swing = max(worst_swing, swing)
                if k % 25 == 0:
                    touching = int(env.fingers_touching().sum())
                    flag = "?" if worst_swing > MAX_TRACKABLE_SWING else " "
                    print(f"\r  ep {episode}  t={k * env.dt:5.1f}s   "
                          f"net yaw {np.degrees(net):+8.1f} deg{flag}"
                          f"({net / (2 * np.pi):+.2f} rev)   "
                          f"fingers on cube {touching}", end="", flush=True)

                # The sim runs at 50 Hz; pace it or the motion blurs past.
                lag = t0 + (k + 1) * env.dt - time.time()
                if lag > 0 and not args.fast:
                    time.sleep(lag)
                # Closing the viewer should end the script, not leave it
                # simulating into a dead renderer until the episode times out.
                if not env.viewer_running():
                    raise KeyboardInterrupt
                k += 1

            outcome = "DROPPED" if info.get("ep_dropped") else "held"
            note = ("   (cube tipped past 150 deg -- net yaw unreliable, see "
                    "sohand.rl.evaluate)" if worst_swing > MAX_TRACKABLE_SWING
                    else "")
            print(f"\r  ep {episode}: {net / (2 * np.pi):+.2f} revolutions in "
                  f"{k * env.dt:.1f}s   {outcome}{note}{' ' * 8}")
            episode += 1
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, default=1, choices=(1, 2),
                   help="1 = 4.7 cm baseline, 2 = 6.1 cm repositioned")
    p.add_argument("--actor", default=None, help="an exported actor .npz")
    p.add_argument("--scene", default=None, help="override the run's scene xml")
    p.add_argument("--close-frac", type=float, default=None,
                   help="override the run's grasp settle closure")
    p.add_argument("--gait", action="store_true",
                   help="the open-loop CEM baseline instead of a policy")
    p.add_argument("--gait-file", default=None)
    p.add_argument("--episodes", type=int, default=0, help="0 = loop forever")
    p.add_argument("--seed", type=int, default=5006)
    p.add_argument("--randomize", action="store_true",
                   help="domain randomisation + sensor noise")
    p.add_argument("--fast", action="store_true", help="run flat out, no pacing")
    main(p.parse_args())
