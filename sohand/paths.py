"""Filesystem layout, resolved from this file rather than the working directory.

These scripts used to build model paths relative to the working directory,
which meant each one ran from exactly one place. Resolving from `__file__`
makes every entry point runnable from anywhere; `SOHAND_ROOT` overrides it when
the package is installed away from the models.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.abspath(os.environ.get("SOHAND_ROOT", os.path.join(_HERE, os.pardir)))

MJCF_DIR = os.path.join(ROOT, "mjcf")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
ASSET_DIR = os.path.join(ROOT, "assets")

# The hand on its own, as exported from Onshape: position actuators named
# `finger{i}_motor{j}`, no object. Used by the retargeting pipeline.
HAND_SCENE = os.path.join(MJCF_DIR, "hand", "scene.xml")

# Hand + graspable cube + AprilTag decals, with contact parameters and grasp
# pads tuned for manipulation. Actuators are named `motor_finger{i}_{j}` and
# the cube adds a free joint, so the two models are NOT interchangeable.
CUBE_SCENE = os.path.join(MJCF_DIR, "cube", "scene.xml")

# Same model with the cube grown to 6.1 cm and shifted into the finger row.
SPIN_SCENE = os.path.join(MJCF_DIR, "cube", "scene_spin.xml")


def checkpoint(name):
    """Absolute path to a file in `checkpoints/`, whether or not it exists."""
    return os.path.join(CHECKPOINT_DIR, name)


def require_checkpoint(name):
    """Absolute path to a checkpoint, failing with the fix if it is absent.

    The trained weights are not tracked in git -- they are opaque binaries that
    every clone would pay for forever -- so a fresh checkout has the code but
    not the `.npz` files. Rather than let numpy raise a bare `FileNotFoundError`
    from inside the actor loader, say where they come from.
    """
    path = checkpoint(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{name} is not in {CHECKPOINT_DIR}.\n"
            "Trained weights are published with the release, not tracked in "
            "git. Fetch them with:\n"
            "    python -m sohand.rl.fetch\n"
            "or train your own and export it -- see checkpoints/README.md.")
    return path


def require_models():
    """Fail early, with the fix, if the MuJoCo models are not where we expect.

    The models are repo data rather than package data, so a non-editable
    `pip install .` copies the code and leaves them behind. Install with
    `pip install -e .`, or point `SOHAND_ROOT` at the checkout.
    """
    if not os.path.isdir(MJCF_DIR):
        raise FileNotFoundError(
            f"MuJoCo models not found at {MJCF_DIR}.\n"
            "Install editable (`pip install -e .`) or set SOHAND_ROOT to the "
            "repository checkout.")
    return MJCF_DIR
