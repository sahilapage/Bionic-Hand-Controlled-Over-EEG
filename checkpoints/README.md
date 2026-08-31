# Checkpoints

Trained policy weights are **not tracked in git**. They are published as assets
on the [release](https://github.com/sahilapage/so-hand/releases) that produced
them, and fetched on demand:

```bash
python -m sohand.rl.fetch          # both actors, digest-verified
python -m sohand.rl.view --run 1
```

Weights are opaque binaries that never change after training. Committing them
makes every clone pay for them permanently — git stores each version whole,
diffs nothing, and cannot ever forget one — while a release asset is versioned
against the tag it belongs to and costs a clone nothing.

| asset | source | what it is |
|---|---|---|
| `actor_run1.npz` | release | SAC actor, 4.7 cm cube (`mjcf/cube/scene.xml`), grasp closure 0.50 |
| `actor_run2.npz` | release | SAC actor, 6.1 cm cube (`mjcf/cube/scene_spin.xml`), grasp closure 0.20 |
| `gait_cem.npy` | tracked | 24 open-loop sinusoid parameters — the CEM baseline a policy must beat (+0.061 ± 0.009 rev / 20 s, n=60) |

`gait_cem.npy` stays in git at 320 bytes. It is a handful of coefficients, not
a learned artefact, and the baseline is meaningless if it drifts away from the
code that quotes its score.

Each actor is `{W0, b0, W1, b1, W2, b2}` — a 74→512→512→8 ReLU MLP with a tanh
squash, which is the whole deterministic SAC policy. 1.2 MB each, and
`sohand.rl.actor.NumpyActor` evaluates them without torch.

A run's geometry and grasp closure travel with its weights, so
`sohand.rl.actor.RUNS` keeps all three together. Pointing run 2's policy at run
1's scene hands it a cube of the wrong size in the wrong place.

## Training your own

Full SB3 `.zip` checkpoints (actor, twin critics, replay buffer) run to
hundreds of megabytes and are never published. Training writes them to `runs/`,
which is gitignored:

```bash
python -m sohand.rl.train_sac --n-envs 12 --timesteps 8000000
python -m sohand.rl.export_actor runs/spin/models/best_model \
    --out checkpoints/actor_run3.npz
```

The exporter verifies its own output against SB3's deterministic action before
it returns. Anything matching `checkpoints/*.npz` is ignored by git, so a new
export will not accidentally be committed.

## Publishing a release

```bash
gh release create v0.2.0 checkpoints/actor_run*.npz \
    --title "v0.2.0" --notes "Trained in-hand rotation policies."
```

`sohand/rl/fetch.py` pins the SHA-256 of each asset. Re-uploading different
weights under the same name will fail verification on the next fetch — update
`ASSETS` in the same commit as the upload.
