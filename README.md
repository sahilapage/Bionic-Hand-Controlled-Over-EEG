# so-hand

Dexterous manipulation on the [Pollen Robotics Amazing
Hand](https://github.com/pollen-robotics/AmazingHand) — four fingers, eight
motors, one MuJoCo model, four ways of driving it.

<p align="center">
  <img src="assets/image.png" width="55%" />
</p>

Four independent components, sharing the robot model and one set of
conventions:

| | what it does | docs |
|---|---|---|
| **`sohand.envs` / `sohand.rl`** | continuous in-hand cube rotation, learned with SAC in MuJoCo | [in-hand rotation](docs/in-hand-rotation.md) |
| **`sohand.retarget`** | maps a human hand pose onto the eight motors | [retargeting](docs/retargeting.md) |
| **`sohand.perception`** | 3D hand pose from RGB, and RGB-D capture | [hand pose](docs/hand-pose.md) |
| **`sohand.eeg`** | single-channel EEG acquisition and state classification | [EEG](docs/eeg.md) |

They are deliberately independent: nothing imports across a component
boundary, so a missing optional dependency in one — torch, depthai, PyQt5 —
never breaks another. The rotation policy is the finished piece; retargeting
and hand-pose extraction feed each other; the EEG front end is a separate
exploration and drives nothing yet.

---

## Quick start

```bash
git clone https://github.com/sahilapage/so-hand.git
cd so-hand
pip install -e .                 # numpy, mujoco, gymnasium
```

Watch a trained policy rotate the cube:

```bash
python -m sohand.rl.fetch        # trained weights, from the release
python -m sohand.rl.view --run 1
```

That needs nothing beyond the base install. The policy is replayed from a
1.2 MB `.npz` — a deterministic SAC action is `tanh(mu(latent_pi(obs)))`, an MLP
and a squash, so neither torch nor stable-baselines3 has to be installed to run
one. The weights live on the [release](https://github.com/sahilapage/so-hand/releases)
rather than in the tree; `fetch` verifies them against a pinned SHA-256.

Reproduce the numbers:

```bash
python -m sohand.rl.evaluate --run 1 --episodes 120
```

Install the rest as you need it:

```bash
pip install -e ".[train]"        # stable-baselines3, torch, tensorboard, wandb
pip install -e ".[perception]"   # opencv, mediapipe, depthai (+ HAMER separately)
pip install -e ".[eeg]"          # pyserial, pylsl, scipy, PyQt5
```

Install **editable**. The MuJoCo models are repository data, not package data,
so a non-editable install leaves them behind; `SOHAND_ROOT` overrides the
location if you need to.

---

## In-hand rotation

The headline result. Keep rotating a cube about one fixed axis, for as many
revolutions as possible, without dropping it — the task the in-hand
manipulation literature actually solves, and the one this hand can reach.

| policy | n | revolutions / 20 s | success | drop rate |
|---|---:|---:|---:|---:|
| **run 1** — 4.7 cm cube | 120 | **+2.02 ± 0.08** | 0.775 | 0.133 |
| **run 2** — 6.1 cm cube | 120 | **+3.21 ± 0.14** | 0.867 | 0.392 |
| open-loop CEM baseline | 60 | +0.06 ± 0.01 | 0.000 | 0.050 |
| do-nothing | 60 | +0.00 ± 0.00 | 0.000 | 0.000 |

Mean ± SEM, deterministic policy, nominal physics, 20 s episodes. `success` =
at least one full revolution. Simulation only — see **Status** below.

Three things are worth reading the [full write-up](docs/in-hand-rotation.md)
for:

**The reward has exactly one positive term.** `r = w_rot · (ω·k̂) · dt` minus
penalties, where *nothing but the rotation term is ever positive*. Doing nothing
scores ≤ 0 and any net rotation beats it. The term is an exact increment, so an
episode's rotation reward is precisely `w_rot ×` net radians turned — signed, so
turning one way and back nets zero. No weighting can make freezing optimal,
which is the failure mode that killed three earlier runs on a face-target
formulation.

**Weights were measured, not chosen.** Running do-nothing, random and a CEM
rotator through the environment and printing each term's unweighted episode sum
showed that three plausible Hora-derived penalties were *anti-task* on this
hand — they charged a real rotator 100–1000× what they charged a frozen policy.
Copying a reward from a paper without measuring its terms is how you get a
policy that freezes.

**Rotation is reported two ways, and the check is enforced in code.** The
credited `∫ω·k̂ dt` that the reward pays for, and the swing-twist angle of the
cube's actual orientation. A path integral picks up geometric phase, so wobble
can inflate the first and cannot touch the second — their ratio says whether
the headline number is real. It comes out at ≈1.00. The pose-only measure has
its own hard limit, since twist about a fixed axis is undefined through a flip,
so `evaluate` measures the swing too and drops the episodes where the check
cannot be made instead of averaging nonsense into the mean.

Train it:

```bash
python -m sohand.rl.train_sac --n-envs 12 --timesteps 8000000
```

---

## Repository layout

```
sohand/
├── paths.py          filesystem layout, resolved from the package not the cwd
├── hand.py           model wiring and the constants every component shares
├── rotations.py      quaternion / swing-twist helpers, NumPy only
├── view_model.py     open any of the models in the viewer, nothing driving it
├── envs/             MuJoCo environments
│   ├── mujoco_env.py base class: model loading, stepping, the three render paths
│   └── spin.py       AmazingHandSpinEnv — 74-dim observation, 8-dim action
├── rl/               train_sac, evaluate, view, actor (NumPy replay),
│                     export_actor, fetch, make_scene, sensitivity
├── retarget/         retarget, probe_fk
├── perception/       hand_pose (HAMER + MediaPipe), record_rgbd (OAK-D)
└── eeg/              bands (shared DSP), bridge, classify, visualizer

mjcf/                 two MuJoCo models — see mjcf/README.md
checkpoints/          CEM baseline; trained actors land here via rl.fetch
firmware/             Arduino sketch for the EEG front end
docs/                 the long-form write-ups
assets/               media for these READMEs
```

Every module runs as `python -m sohand.<module>` and carries its own
`--help`.

---

## Hardware

The Amazing Hand is Pollen Robotics' open-source 4-finger, 8-DoF hand: each
finger is a passive four-bar linkage driven by two STS3215 serial-bus servos.

One consequence shows up everywhere in this repo: **the linkage's constrained
pose does not exist until the physics solver has run.** `mj_forward` alone
leaves every fingertip at its unconstrained position, so forward kinematics, a
Jacobian sweep and a grasp settle must each step physics first. That was the
original cause of "the motors do not move".

---

## Status

The rotation policy is **simulation-only**. It is state-based — it receives
ground-truth cube pose, which is strictly more than a camera would give — so
"no vision" would be a misleading way to describe it. 70 of the 74 observation
dimensions are recoverable on hardware from one camera with AprilTags plus the
joint encoders; only the 4 contact bits need sensing that is not already there.
The deployment path, and what is likely to break on the way, are written up in
[docs/in-hand-rotation.md](docs/in-hand-rotation.md#deploying-it).

---

## License

MIT — see [LICENSE](LICENSE). The Amazing Hand mechanical design is Pollen
Robotics'; HAMER and MANO carry their own licenses.
