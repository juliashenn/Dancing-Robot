"""Run booster_train's RSL-RL train/play scripts with our k1_dance tasks registered.

    python train/run_rsl_rl.py train --task K1-Dance-Pop1-v0 --headless --num_envs 4096
    python train/run_rsl_rl.py play  --task K1-Dance-Pop1-v0-Play --num_envs 1 --headless --checkpoint <model.pt>

All arguments after train/play are passed through unchanged (including Hydra overrides such as
agent.save_interval=250). Logs go to logs/rsl_rl/<experiment_name>/ in the repo root.
"""

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "third_party" / "booster_train" / "scripts" / "rsl_rl"

if len(sys.argv) < 2 or sys.argv[1] not in ("train", "play"):
    sys.exit("Usage: python train/run_rsl_rl.py {train|play} --task <task> [args...]")
mode = sys.argv.pop(1)

sys.path.insert(0, str(REPO_ROOT / "train"))
import k1_dance  # noqa: E402,F401  (registers our Gym task ids)

script = SCRIPTS / f"{mode}.py"
sys.path.insert(0, str(SCRIPTS))  # booster_train's scripts import cli_args from their own folder
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
