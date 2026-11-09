"""Fetch the external repos we build on into third_party/, pinned to known commits.

Usage (from the repo root):
    python setup/clone_third_party.py             # fetch all
    python setup/clone_third_party.py GMR         # fetch one or more by name

Re-running is safe: existing clones are fetched and checked out at the pin.
To move a pin, update PINS below and commit the change so everyone matches.
"""

import subprocess
import sys
from pathlib import Path

PINS = {
    "booster_train": (
        "https://github.com/BoosterRobotics/booster_train.git",
        "83037fde138c49cd8d6459a4b592538e6e979ffb",
    ),
    "booster_deploy": (
        "https://github.com/BoosterRobotics/booster_deploy.git",
        "7bb1462e2ac742eba5c01659b815e2e99fbe617f",
    ),
    "booster_assets": (
        "https://github.com/BoosterRobotics/booster_assets.git",
        "12a97516f57469078707afb587a5f84c1cb110d1",
    ),
    "GMR": (
        "https://github.com/YanjieZe/GMR.git",
        "25f56db79ac411258b377b6d4ccba0c8ff3372ac",
    ),
}

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party"


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch(name):
    url, commit = PINS[name]
    dest = THIRD_PARTY / name
    if not (dest / ".git").exists():
        print(f"== cloning {name}")
        git("clone", url, str(dest))
    else:
        print(f"== updating {name}")
        git("fetch", "origin", cwd=dest)
    git("checkout", "--detach", commit, cwd=dest)
    print(f"   {name} @ {commit[:10]}")


def main():
    names = sys.argv[1:] or list(PINS)
    unknown = [n for n in names if n not in PINS]
    if unknown:
        sys.exit(f"Unknown repo(s): {unknown}. Choose from {list(PINS)}")
    THIRD_PARTY.mkdir(exist_ok=True)
    for name in names:
        fetch(name)


if __name__ == "__main__":
    main()
