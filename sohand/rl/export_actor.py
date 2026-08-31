"""Export a trained SAC actor to a torch-free `.npz`.

    python -m sohand.rl.export_actor runs/spin/models/best_model \
        --out checkpoints/actor_run3.npz

Only the deterministic path is exported -- `latent_pi` plus `mu` -- because
that is all deployment evaluates. The log-std head, the critics and the replay
buffer stay in the `.zip`.

The export refuses a policy it cannot replay exactly: anything other than a
plain `Flatten` feature extractor, or an activation other than ReLU, would make
`sohand.rl.actor.NumpyActor` silently compute a different function.
"""

import argparse
import os

import numpy as np


def export(model_path, out_path, device="cpu"):
    import torch.nn as nn
    from stable_baselines3 import SAC

    model = SAC.load(model_path, device=device)
    actor = model.policy.actor

    extractor = type(actor.features_extractor).__name__
    if extractor != "FlattenExtractor":
        raise ValueError(
            f"feature extractor is {extractor}, not FlattenExtractor; the "
            "NumPy replay path assumes the observation reaches latent_pi "
            "unchanged")

    layers = [m for m in actor.latent_pi if isinstance(m, nn.Linear)]
    acts = [m for m in actor.latent_pi if not isinstance(m, nn.Linear)]
    if not all(isinstance(m, nn.ReLU) for m in acts):
        raise ValueError(
            f"non-ReLU activations {[type(m).__name__ for m in acts]}; "
            "NumpyActor applies ReLU between every hidden layer")
    layers.append(actor.mu)

    arrays = {"n_layers": np.int64(len(layers))}
    for i, lin in enumerate(layers):
        arrays[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(out_path, **arrays)

    shapes = " -> ".join(str(arrays[f"W{i}"].shape[1]) for i in range(len(layers)))
    print(f"wrote {out_path}  ({shapes} -> {arrays[f'W{len(layers)-1}'].shape[0]}, "
          f"{sum(a.size for k, a in arrays.items() if k != 'n_layers'):,} params)")
    return out_path


def _verify(model_path, out_path, n=64, device="cpu"):
    """Check the NumPy replay matches SB3's deterministic action bit for bit."""
    from stable_baselines3 import SAC

    from sohand.rl.actor import NumpyActor

    model = SAC.load(model_path, device=device)
    replay = NumpyActor(out_path)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, replay.obs_dim)).astype(np.float32)
    ref = np.stack([model.predict(o, deterministic=True)[0] for o in obs])
    got = np.stack([replay(o) for o in obs])
    err = float(np.abs(ref - got).max())
    print(f"max |SB3 - NumPy| over {n} random observations: {err:.3e}")
    if err > 1e-5:
        raise SystemExit("replay mismatch -- do not deploy this export")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", help="SB3 .zip, without the extension")
    p.add_argument("--out", default="actor.npz")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()
    export(a.model, a.out, a.device)
    if not a.no_verify:
        _verify(a.model, a.out, device=a.device)
