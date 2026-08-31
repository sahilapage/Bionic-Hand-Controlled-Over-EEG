# Continuous in-hand cube rotation

Keep rotating a cube about one fixed axis, for as many revolutions as possible,
without dropping it. Four fingers, eight motors, 50 Hz, SAC in MuJoCo.

This is the task the in-hand manipulation literature actually solves
([Hora](https://haozhi.io/hora/), CoRL 2022;
[AnyRotate](https://anyrotate.github.io/), CoRL 2024) and it replaces an earlier
face-target formulation that three separate runs failed to learn. Understanding
*why* it failed is most of the design, so that comes first.

---

## Why the previous task did not work

The face-target environment asked for a discrete reorientation: bring a named
cube face to +Z and hold it. PPO and two SAC runs all converged to the same
degenerate policy — freeze and hold — at solve rates of 0.027, 0.060 and 0.070.
Three measurements explain it, and none of them is "needs more exploration":

**The reward paid freezing better than rotating.** On the run-2 weights, a
do-nothing rollout scored −0.48 per episode; a rollout that genuinely turned the
cube 70.9° scored −1.41. Motion cost more than progress paid.

**Undirected rotation earned nothing.** During that same 70.9° rotation, the
face-alignment angle moved 91.4° → 88.3°, so the potential-based shaping term
telescoped to −0.24. Almost a full quarter-turn of real manipulation was worth
less than zero.

**The task had no easy instance.** Face-to-face targets make the *minimum* task
a 90° roll, which is at the edge of what this hand can do in one motion. There
was no partial credit and no gentle end of the distribution to learn from.

---

## Why +Z

Measured, not assumed. 120 open-loop random per-joint sinusoids were pushed
through the real action pipeline:

```
survived the full 8 s: 116/120          ← the hand does hold the cube
accumulated rotation, world frame, degrees
    axis      mean       p90       max
      +X      -0.3      10.7      33.9
      +Y      -0.1      10.1      68.4
      +Z     +12.4      38.3      88.0
principal axis of accumulated spin (SVD): [-0.03, 0.01, 1.00]
singular values: 4.84 (Z) vs 2.70, 1.68
```

Rotation about world +Z is the axis this hand drives, by about 2× over the
others, and **random actions already produce a positive mean about it**. The
reward signal is reachable by exploration on step one — exactly what the
face-target task never offered.

Geometrically this is unsurprising. Fingers 1–3 are spread along y and press
along +x, so differential finger force is a torque about z; the thumb presses
down from +z and acts as the pivot.

---

## The reward

```
r = w_rot · clip(ω_f · k̂, ±angvel_clip) · dt   ← the task
  − w_offaxis     · min(‖ω_f − (ω_f·k̂)k̂‖, offaxis_clip)
  − w_linvel      · ‖v_centre‖₁
  − w_drift       · ‖c − c_start‖
  − w_work · (Στq̇)²  − w_torque · ‖τ‖²  − w_action_rate · ‖Δa‖²
  − w_finger_idle · (1 − min over fingers of a contact EMA)
  − drop_penalty                                ← terminal, episode ends
```

with `w_rot = 8`, `angvel_clip = 0.8 rad/s`, `dt = 0.02 s`. ω_f is the
low-pass-filtered angular velocity — raw instantaneous ω has roughly a 1:25
signal-to-noise ratio against a reachable spin rate of 0.1–0.5 rad/s.

Two invariants hold, and both are verified by measurement rather than asserted:

**Nothing except the rotation term is ever positive.** A policy that does
nothing scores ≤ 0, and any net rotation beats it. There is no configuration of
the weights that restores freeze-is-optimal — which is the failure mode that
killed the previous three runs.

**The rotation term is an exact increment**, so an episode's rotation reward is
`w_rot ×` (net radians turned about k̂) — exactly, while the spin rate stays
under `angvel_clip`, and an underestimate above it. At `w_rot = 8` one
revolution is worth +50.3. The term is signed, so turning one way and back nets
zero: it pays for progress, not for motion. Above the clip the reward
saturates, so there is nothing to be gained from spinning the cube faster than
the task needs.

The *reported* rotation metric is computed from the unfiltered ω, not the
filtered one the reward uses: "the cube rotated N degrees" has to mean the cube
actually rotated N degrees, not that a filtered estimate of it did.

### Weights are measured, not chosen

A calibration sweep runs do-nothing, uniform random and a CEM-optimised rotator
gait through the environment and reports each term's **unweighted** per-episode
sum. The first pass rejected a set of plausible-looking Hora-derived weights
outright:

| term | do-nothing | uniform random | CEM rotator |
|---|---:|---:|---:|
| `rot_clipped` | 0.75 | 2.20 | **4.51** |
| `offaxis` | 2.58 | 202.6 | **505.1** |
| `pose_sq` | 0.002 | 157.6 | **681.7** |
| `work_sq` | 0.41 | 23527.0 | **14805.4** |
| `torque_sq` | 0.84 | 2730.6 | 1779.2 |
| `act_rate_sq` | 0.00 | 5228.9 | 33.2 |
| `contact_loss` | 621.8 | 647.0 | 679.6 |

Three of those terms are *anti-task* on this hand. `pose_sq`, `work_sq` and
`offaxis` each charge the rotator 100–1000× what they charge a frozen policy,
because on an 8-DoF hand whose entire action band is ±0.5 around the grasp,
**moving away from the grasp pose is the task**. Hora can afford
`poseDiffPenaltyScale = −0.3` because a 16-DoF Allegro gaits *around* a
canonical grasp; this hand cannot. And `contact_loss` came out at ≈0.63 for
every policy alike — a constant offset carrying no gradient — so it is off.

Copying a reward from a paper without measuring its terms on your own hand is
how you get a policy that freezes.

---

## The two rotation numbers

`sohand.rl.evaluate` reports rotation twice, and the difference matters:

- **credited** — the integral `∫ ω·k̂ dt` the reward actually pays for.
- **swing-twist** — the twist angle of the cube's *orientation* about k̂, a pure
  function of pose.

They are not the same quantity. A path integral picks up geometric (Berry)
phase: cyclic wobble that returns the cube to its starting orientation still
accumulates a nonzero value, so in principle the credited figure can be
inflated by a cube shaking in place. A pose function cannot be. Their ratio is
therefore the check on whether the headline number is real.

**It comes out at ≈1.00.** The policy is not being paid for wobble; the
credited figure is net rotation.

### The check has a hard limit, and the code enforces it

The twist about a fixed axis is exact but becomes *ill-conditioned* as the
object tips over. With `q = (w, v)`, the twist is `2·atan2(v·k̂, w)` and the
divisor is `n = hypot(w, v·k̂) = cos(swing/2)`. At a swing of 179° that divisor
is 0.009, so a milliradian of orientation change can move the reported twist by
degrees — and accumulating it across a trajectory produces nonsense.

This is not hypothetical here. This hand rolls the cube face over face rather
than spinning it cleanly, so a minority of episodes pass close to a flip. One
such episode reported **−4.4 revolutions on a rollout that genuinely turned
+2.7** before the guard was added.

`sohand.rotations.twist_and_swing` therefore returns the swing alongside the
twist, and `evaluate` excludes any episode whose swing exceeds 150°, reporting
how many it dropped rather than averaging them in. `sohand.rl.view` marks such
an episode's on-screen yaw with `?`; `sohand.rl.sensitivity` skips it.

No pure function of orientation can track rotation about a fixed axis through a
flip — that is a property of SO(3), not of this implementation. The honest
response is to detect the condition and say so, which is what these tools do.

---

## Results

All figures below are reproducible from a clean clone:

```bash
python -m sohand.rl.evaluate --run 1 --episodes 120
python -m sohand.rl.evaluate --run 1 --episodes 120 --randomize
python -m sohand.rl.evaluate --run 2 --episodes 120
python -m sohand.rl.evaluate --gait --episodes 60
python -m sohand.rl.evaluate --episodes 60
```

| policy | n | revolutions / 20 s | success | drop rate | rad/s | steps alive |
|---|---:|---:|---:|---:|---:|---:|
| **run 1** — 4.7 cm cube, nominal | 120 | **+2.018 ± 0.083** | 0.775 | 0.133 | +0.674 | 924 |
| run 1 — randomised physics + sensor noise | 120 | **+2.044 ± 0.070** | 0.900 | 0.083 | +0.668 | 954 |
| **run 2** — 6.1 cm cube, nominal | 120 | **+3.209 ± 0.139** | 0.867 | 0.392 | +1.227 | 809 |
| open-loop CEM gait | 60 | **+0.061 ± 0.009** | 0.000 | 0.050 | +0.014 | 960 |
| do-nothing | 60 | **+0.003 ± 0.001** | 0.000 | 0.000 | +0.001 | 1000 |

Mean ± SEM, deterministic policy, 20 s (1000-step) episodes. The learned policy
turns the cube **33× further than the open-loop gait**, which in turn is
20× a do-nothing policy.

**The wobble check passes.** Over the episodes where swing-twist is well
conditioned, the ratio of pose-measured to credited rotation is 1.003 for run 1
(95/120 episodes), 1.009 randomised (98/120), and 0.998 for run 2
(86/120). The credited figure is net rotation, not accumulated wobble.
The 21–28% of episodes that get excluded are themselves informative: this
hand tips the cube past 150° often, which is the face-over-face rolling the
geometry section describes rather than a clean spin.

**Run 2 is the interesting negative result.** The larger, repositioned cube
buys 1.6× the revolutions and 1.8× the spin rate, and pays with a drop rate
of 0.39 against run 1's 0.13 — episodes end at 809 of 1000 steps instead of
924. At `drop_penalty = 10` against a return of about 57, a drop costs under
20%, so the policy correctly concluded that spinning fast and occasionally
dropping beats spinning carefully. Raise `--drop-penalty` for the other trade.

**Randomisation does not hurt, and slightly helps**: 0.900 success against
0.775, drop 0.083 against 0.133. The policy trained with domain
randomisation on, so nominal physics is mildly *out* of distribution for it.

**On the CEM baseline.** The shipped gait measures 0.014 rad/s here, an order of
magnitude below the 0.10–0.12 rad/s recorded when it was searched — it was
evidently tuned against a configuration this repository does not ship. The
measured figure is the one in the table; treat it as a floor, not as the best
open-loop control achievable on this hand.

**On per-episode variance.** Contact-rich dynamics are chaotic, and the effect
is not subtle. Moving the cube's start position by **one nanometre** — five
orders of magnitude below any physical or sensing resolution — changes a single
episode's outcome by a full revolution or more, while leaving the distribution
intact. Measure it yourself:

```bash
python -m sohand.rl.sensitivity --run 1 --episodes 20
```

```
  mean          +2.645 vs +2.185
  sd            0.327 vs 0.686
  |change|      mean 0.563 rev, max 1.796 rev
  correlation   +0.333 across 16 seeds

  Per-episode outcomes are NOT reproducible at this perturbation scale.
```

The consequence for reporting: any claim about this policy needs n ≥ 100. A
handful of episodes measures nothing, and a demo video is one sample.

---

## The trainer

Runs 1 and 2 both diverged. Run 1: critic loss 0.0155 → 2.47e11, auto-tuned α
chased it to 2098, and |Q| reached 1.16e7 against an achievable return of ~49.
Run 2 (γ 0.98, α pinned at 0.05) still went 0.017 → 2.4e7. Fixing γ and pinning
α slowed divergence by four orders of magnitude without stopping it, because
neither was the cause.

**LayerNorm critic.** Bootstrapped value estimates diverge when the Q network
extrapolates confidently off-distribution. LayerNorm bounds the pre-activation
scale of every hidden layer and is the cheapest known fix — it is what CrossQ,
BRO and DroQ all rely on. SB3's stock critic has none. `--no-layernorm` runs the
stock critic as an ablation.

**A bounded return.** The rotation term is an increment, so the max per-step
reward is exactly `w_rot · angvel_clip · dt = 8 · 0.8 · 0.02 = 0.128`, every
penalty only subtracts, and every episode ends within 1000 steps. At γ = 0.98
that bounds |Q| at 6.4 — a number a critic can represent. Runs 1–2 were fitting
an unbounded chain of ±20 sparse events.

**`CriticHealthCallback`** aborts the run once |Q| passes 20× the achievable
bound, instead of burning GPU-days on a dead critic.

**`target_entropy = −6`** rather than SB3's default −8. Eight
near-maximally-entropic joint commands on a grasped cube is not exploration; it
is dropping the cube.

```bash
python -m sohand.rl.train_sac --n-envs 12 --timesteps 8000000
python -m sohand.rl.train_sac --scene mjcf/cube/scene_spin.xml --close-frac 0.20
python -m sohand.rl.train_sac --resume runs/spin/models/checkpoints/resume
```

Everything lands under `runs/spin/` — models, tensorboard logs and a rolling
resume checkpoint that includes the replay buffer.

---

## The observation, and what it means for hardware

74 dimensions:

| block | dims | source on the real rig |
|---|---:|---|
| `jpos`, `jvel` | 16 | joint encoders |
| `last_action`, `prev_action`, `filtered_ctrl` | 24 | commanded setpoints — free |
| `tilt_cos`, `tilt_vec`, `phase4` | 6 | tracked cube orientation |
| `angvel`, `linvel`, `center_offset` | 9 | tracked cube pose, differentiated |
| `finger_to_cube` | 12 | forward kinematics + tracked cube position |
| `contacts` | 4 | tactile pads, or a motor-current threshold |
| `axis` | 3 | constant, the commanded spin axis |

**This is a state-based policy, not a proprioception-only one.** It receives
ground-truth cube pose from the simulator — strictly more information than a
camera gives. 70 of the 74 dimensions are recoverable on hardware from one
camera with AprilTags plus the encoders; only the 4 contact bits need sensing
that is not already there.

Pose features are chosen so the observation is stationary in the coordinate the
task is periodic in. Absolute yaw about k̂ is irrelevant here and, for a cube,
only meaningful modulo 90°, so the policy sees `(cos 4ψ, sin 4ψ)` — a feature
that repeats every quarter turn exactly as the cube does — rather than a raw
rotation matrix.

Angular velocity is low-pass filtered before it reaches the observation: the raw
instantaneous ω has roughly a 1:25 signal-to-noise ratio against a reachable
spin rate of 0.1–0.5 rad/s.

### Deploying it

1. **Export the actor.** `python -m sohand.rl.export_actor <model> --out actor.npz`.
   The deterministic policy is `tanh(mu(latent_pi(obs)))` — two 512-unit layers,
   microseconds on a Pi. The exporter verifies its output against SB3 before it
   returns.
2. **Build the observation in exactly the training order and scaling.** The
   divisions by `ANGVEL_OBS_SCALE` and `REACH_NORM` and the `clip(-3, 3)` are
   part of the contract; get one wrong and the policy sees garbage.
3. **Reproduce the action pipeline.** Grasp band 0.50 → low-pass 0.42 → slew
   limit 0.05 · range/step. That pipeline is part of the policy, not of the
   simulator.
4. **Track the cube at ≥ 50 Hz**, and budget the tracker's latency. Training
   assumed a 5 Hz filter on a clean estimate, not 100 ms of lag.

Known risks, in order: the fingers occlude the AprilTags exactly when the grasp
matters most; contact bits faked from motor current are a much noisier signal
than the simulated ones; and MuJoCo's position servo is not an STS3215.
Domain randomisation (friction ±30%, mass ±20%, damping ±25%, servo gain ±15%,
control rate ±20%, 2% action dropout, plus sensor noise on every observation
block) covers the first two partially and does not cover a wrong servo model at
all.

If occlusion proves fatal, the fallback is to retrain proprioception-only: drop
all 27 tracker-derived dimensions (the pose features, the velocities, the
centre offset and `finger_to_cube`) and keep the 47 that come from encoders,
commanded setpoints and contacts. That is Hora's setup and is known to work, at
a cost in sample efficiency.

---

## Geometry

The shipped 4.7 cm cube is never touched by the fingertips. On
`mjcf/cube/scene.xml` the tips sit centimetres from its surface and every
contact is borne by mid-chain linkage bodies — for finger 1, by `rotule_ball_2`
and `ball_link`, which is the knuckle. The cube rests in the knuckle pocket
because the fingers curl well past it.

A kinematic sweep over size × position, recorded in
`sohand/rl/make_scene.py`, found that size alone is not the fix. Growing the
cube stops helping past about 6 cm — it drives fingers 2 and 3 into penetration
while finger 1 and the thumb fall further behind. The missing axis was y: the
finger row spans y = −0.020 … +0.052 (centre +0.016) while the cube sat at
y = −0.001, biased toward finger 3.

`mjcf/cube/scene_spin.xml` is the result: the cube scaled to 6.1 cm and shifted
+1.7 cm in y, generated by `python -m sohand.rl.make_scene`. Measured over 20
resets after the grasp settle, with each scene at its own closure:

| | tip → cube surface, cm | worst | spread | fingers in contact |
|---|---|---:|---:|---:|
| `scene.xml`, closure 0.50 | 6.6 · 4.6 · 3.6 · 3.7 | 6.6 | 3.0 | 1.75 / 4 |
| `scene_spin.xml`, closure 0.20 | 5.1 · 3.8 · 3.8 · 2.9 | 5.1 | 2.2 | 2.15 / 4 |

The thumb is the difference that matters: its contact rate goes from 0.00 to
0.85. Evenness, not raw closeness, was the selection criterion — a
configuration with a smaller worst-case gap but a 1.4 cm spread between best
and worst finger performs worse, because one finger doing all the work is how
the cube gets dropped.

The cube is modelled as a hollow shell so its mass stays at 71 g. A solid cube
at 1.3× scale would be 156 g, which is a great deal for 3.23 N·m servos.
Regenerate with different geometry and you have a different task — the run 2
checkpoint is tied to this one.

Reproduce the table with:

```bash
python -m sohand.rl.evaluate --run 1 --episodes 120     # scene.xml
python -m sohand.rl.evaluate --run 2 --episodes 120     # scene_spin.xml
```

---

## Prior work

- Qi, Kumar, Calandra, Ma & Malik, **In-Hand Object Rotation via Rapid Motor
  Adaptation**, CoRL 2022 — the reward skeleton (clipped ω·k̂ plus penalties),
  z-axis-only rotation, proprioception-history policy, validated grasp cache.
- Yang, Church, Lin & Lepora, **AnyRotate**, CoRL 2024 — multi-axis,
  gravity-invariant, the rotation / contact / stability / termination
  decomposition.
- Chen, Xu & Agrawal, **A System for General In-Hand Object Re-Orientation**,
  CoRL 2021 — gravity curriculum, teacher–student.
- OpenAI et al., **Learning Dexterous In-Hand Manipulation**, IJRR 2020 — the
  original domain-randomised Shadow Hand result.
- Khandate et al., **Sampling-based Exploration for RL of Dexterous
  Manipulation**, RSS 2023 — G-RRT reset distributions; the principled version
  of the CEM gait search used here as a baseline.
