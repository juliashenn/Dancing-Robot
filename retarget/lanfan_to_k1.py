"""Retarget a LAFAN1 BVH clip onto the Booster K1 with GMR.

GMR (pinned commit) has no LAFAN1 config for the K1, so this registers ours
(retarget/ik_configs/bvh_lafan1_to_k1.json) at runtime and then runs GMR's own
scripts/bvh_to_robot.py unchanged. third_party/GMR stays untouched.

Run with GMR's Python, from the repo root:
    third_party/GMR/.venv/Scripts/python retarget/lafan1_to_k1.py \
        --bvh_file motions/raw/lafan1/dance1_subject3.bvh \
        --save_path motions/gmr_pkl/lafan_dance1_s3.pkl
Then convert (adds joint-limit clipping and the OK/REVIEW report):
    python retarget/gmr_to_k1_csv.py motions/gmr_pkl/lafan_dance1_s3.pkl

Any other bvh_to_robot.py flag (e.g. --rate_limit, --record_video) passes through.
"""
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "retarget" / "ik_configs" / "bvh_lafan1_to_k1.json"
GMR_SCRIPT = REPO_ROOT / "third_party" / "GMR" / "scripts" / "bvh_to_robot.py"

from general_motion_retargeting import IK_CONFIG_DICT  # noqa: E402

IK_CONFIG_DICT["bvh_lafan1"]["booster_k1"] = CONFIG

args = sys.argv[1:]
if "--robot" not in args:
    args += ["--robot", "booster_k1"]
if "--format" not in args:
    args += ["--format", "lafan1"]
sys.argv = [str(GMR_SCRIPT), *args]
runpy.run_path(str(GMR_SCRIPT), run_name="__main__")