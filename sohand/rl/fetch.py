"""Download the trained policy weights into `checkpoints/`.

    python -m sohand.rl.fetch              # every actor, skipping what is present
    python -m sohand.rl.fetch --run 1      # just run 1
    python -m sohand.rl.fetch --force      # re-download and re-verify

The weights are published as release assets rather than tracked in git: they
are opaque binaries that never change, so committing them makes every clone
pay for them forever and every diff step over them. The code is versioned
here, the artefacts are versioned alongside the tag that produced them.

Downloads are checked against the SHA-256 recorded below, so a truncated
transfer or a replaced asset fails loudly instead of surfacing later as a
policy that quietly does not rotate the cube. Standard library only -- no
`requests`, nothing added to the base install.
"""

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request

from sohand import paths

RELEASE = "v0.2.0"
BASE_URL = f"https://github.com/sahilapage/so-hand/releases/download/{RELEASE}"

# name -> (sha256, description). The digests pin the exact weights the numbers
# in docs/in-hand-rotation.md were measured from.
ASSETS = {
    "actor_run1.npz": (
        "45be307b35b78c6e3f39ccd97c9ec6a01e9a94ee45eb965a0eb4460eeba6c0a5",
        "SAC actor, 4.7 cm cube, +2.02 rev / 20 s",
    ),
    "actor_run2.npz": (
        "28da1943a556838e6a231af3a34e4aea40c5ad218ffccf3c47c9ae2a8f2d461f",
        "SAC actor, 6.1 cm cube, +3.21 rev / 20 s",
    ),
}

RUN_ASSETS = {1: "actor_run1.npz", 2: "actor_run2.npz"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(name, dest, expect):
    """Fetch one asset to `dest`, verifying it before it lands at that path."""
    url = f"{BASE_URL}/{name}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Download to a temporary file in the same directory and rename only after
    # the digest matches, so an interrupted run never leaves a half-written
    # `.npz` that looks like a valid checkpoint.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    os.close(fd)
    try:
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    block = r.read(1 << 16)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    if total:
                        print(f"\r  {name}  {100 * done // total:3d}%",
                              end="", flush=True)
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"{url}\n  HTTP {e.code}. The {RELEASE} release may not carry "
                f"this asset yet.") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"{url}\n  {e.reason}") from e

        got = sha256(tmp)
        if got != expect:
            raise RuntimeError(
                f"{name} failed verification.\n"
                f"  expected {expect}\n  got      {got}\n"
                "  The asset was replaced or the transfer was truncated.")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"\r  {name}  ok  ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", type=int, choices=(1, 2), default=None,
                   help="fetch one run's actor instead of all of them")
    p.add_argument("--force", action="store_true",
                   help="re-download even if the file is already present")
    args = p.parse_args()

    wanted = ([RUN_ASSETS[args.run]] if args.run else list(ASSETS))
    print(f"{BASE_URL}\n")

    failed = []
    for name in wanted:
        expect, what = ASSETS[name]
        dest = paths.checkpoint(name)
        if os.path.isfile(dest) and not args.force:
            state = "ok" if sha256(dest) == expect else "PRESENT BUT WRONG DIGEST"
            print(f"  {name}  {state}, skipping   ({what})")
            continue
        try:
            download(name, dest, expect)
        except RuntimeError as e:
            print(f"\r  {name}  FAILED\n    {e}")
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} of {len(wanted)} failed.")
        return 1
    print(f"\nInto {paths.CHECKPOINT_DIR}. "
          "Try: python -m sohand.rl.view --run 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
