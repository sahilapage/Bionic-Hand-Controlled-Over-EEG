# Human hand pose → Amazing Hand

Real-time retargeting of a human hand pose onto the Amazing Hand's eight
motors, in MuJoCo.

| Peace sign | Closed fist | Open hand |
|:---:|:---:|:---:|
| ![](../assets/retarget-peace.jpg) | ![](../assets/retarget-fist.jpg) | ![](../assets/retarget-open.jpg) |

*Left of each pair: the HAMER mesh over the human hand. Right: the resulting
robot pose.*

```
camera → HAMER → TSV (5×3) → retarget_tsv() → 8 motor commands → MuJoCo
```

The approach follows [DexMV](https://yzqin.github.io/dexmv/) and
[Robotic Telekinesis](https://robotic-telekinesis.github.io/), but uses a
measured Jacobian inverse rather than energy minimisation, which makes it fast
enough to run per frame.

## Input

A **Task Space Vector**: a `(5, 3)` array of fingertip positions relative to the
palm, one row per finger in the order thumb, index, middle, ring, pinky.
`sohand.perception.hand_pose` writes exactly this. It is pose-invariant — it
encodes finger configuration without the hand's global position or orientation —
which is what makes it transferable to a hand with different proportions.

## Finger mapping

The Amazing Hand has four fingers; HAMER outputs five. The pinky is dropped.

| Human | Robot tip | Motors |
|---|---|---|
| Index | `tip1` | `finger1_motor1`, `finger1_motor2` |
| Middle | `tip2` | `finger2_motor1`, `finger2_motor2` |
| Ring | `tip3` | `finger3_motor1`, `finger3_motor2` |
| Thumb | `tip4` | `finger4_motor1`, `finger4_motor2` |
| Pinky | — | dropped |

Per finger, `motor1` flexes (positive curls toward the palm) and `motor2`
abducts.

## How it works

**1. Frame remap.** HAMER's fingers point along −Y; the robot's along +Z. Fixed
remap `human(x, y, z) → robot(x, z, −y)`, so `robot_Z = −human_Y` becomes the
extension signal and `robot_X = human_X` the lateral one.

**2. Auto-calibration.** At startup each motor is swept to ±0.5 rad and physics
settled for 300 steps, giving a per-finger tip sensitivity in m/rad:

$$\mathbf{J}_i^{m} = \frac{\mathbf{t}_i(+\delta) - \mathbf{t}_i(-\delta)}{2\delta}$$

Nothing is hardcoded — the numbers come from whichever model you load. Measured
on `mjcf/hand/scene.xml`:

```
finger1_motor1  sens=[ 0.0270  0.0165 -0.0343]  |sens|=0.0467 m/rad
finger1_motor2  sens=[-0.0248  0.0167  0.0319]  |sens|=0.0437 m/rad
finger4_motor1  sens=[-0.0266 -0.0272  0.0270]  |sens|=0.0466 m/rad   (thumb)
```

`motor1` drives mostly Z (≈ −0.034 m/rad), `motor2` mostly X (≈ −0.025 m/rad).
Reproduce with `python -m sohand.retarget.probe_fk`.

The settle is not optional. The fingers are passive four-bar linkages, so their
constrained pose does not exist until the solver has run — `mj_forward` alone
leaves every tip where the linkage has not yet pulled it.

**3. Retargeting, fingers 1–3.** Normalise the remapped Z to an extension ratio,

$$r = \mathrm{clip}\left(\frac{z - z_{\min}}{z_{\max} - z_{\min}},\ 0,\ 1\right),
\qquad r' = r^{\gamma},\ \gamma = 1.5$$

with `z_min = 0.05 m` (fully curled) and `z_max = 0.18 m` (fully extended).
γ > 1 compresses the middle of the range toward the curled end, so a fist
reaches full curl without over-extending an open hand. Map `r'` onto the
finger's reachable Z range and invert through the Jacobian for `motor1`;
`motor2` takes the lateral component, scaled down because the robot's abduction
range is far smaller than a human's.

**4. Thumb.** Handled separately: a human thumb curls *across* the palm, along
X rather than Y, so `motor1 ← −human_X` and `motor2 ← human_Z`.

**5. Playback.** Commands are EMA-smoothed (α = 0.08) before reaching `data.ctrl`,
because stepping between poses instantly makes the linkage overshoot.

## Usage

```bash
python -m sohand.retarget.retarget                            # three demo poses
python -m sohand.retarget.retarget --tsv-folder poses/        # cycle a folder
python -m sohand.retarget.retarget --tsv-file pose.npy
python -m sohand.retarget.probe_fk                            # FK diagnostic
```

| Flag | Default | Effect |
|---|---|---|
| `--model-path` | `mjcf/hand/scene.xml` | which model to retarget onto |
| `--human-z-max` | `0.18` | max fingertip extension, m — lower it for a small hand |
| `--hold-time` | `2.0` | seconds per pose in the viewer |

## Limits

- The Jacobian is linear, measured at 0.5 rad. Accuracy degrades at large motor
  angles but is adequate for retargeting.
- `z_min` / `z_max` are anatomical estimates for an average adult hand. For a
  different subject, capture one open-hand and one fist pose, read the TSV Z
  values, and set the flags from those.
- Four fingers cannot represent a five-finger pose. Dropping the pinky is a
  choice, not a bug, and gestures that depend on it will not survive.

## References

- Qin et al., **DexMV: Imitation Learning for Dexterous Manipulation from Human
  Videos**, CVPR 2022 — the Task Space Vector representation.
- Pavlakos et al., **Reconstructing Hands in 3D with Transformers** (HAMER).
- Sivakumar et al., **Robotic Telekinesis**, RSS 2022.
