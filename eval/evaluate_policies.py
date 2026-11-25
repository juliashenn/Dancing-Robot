"""Evaluate every k1_dance / k1_routine task in MuJoCo and write one results table.

Runs deploy/record_mujoco.py --no-video for each task (about 20 s per 12 s dance), from booster_deploy's default
stance and from the K1 walking stance (what the real robot starts from), and collects the JSON summaries.

    python eval/evaluate_policies.py                     # all tasks whose model files exist
    python eval/evaluate_policies.py k1_dance_lock1 k1_routine_lock_pop_hiphop   # re-run some, keep the rest
    python eval/evaluate_policies.py --table-only

Writes eval/results/mujoco_results.csv and eval/results/mujoco_results.md (and per-run JSON in
eval/results/mujoco_runs/). Run with any Python; it calls booster_deploy's venv for the simulations.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_PY = REPO_ROOT / "third_party" / "booster_deploy" / ".venv" / "Scripts" / "python.exe"
if not DEPLOY_PY.exists():
    DEPLOY_PY = REPO_ROOT / "third_party" / "booster_deploy" / ".venv" / "bin" / "python"
RECORD = REPO_ROOT / "deploy" / "record_mujoco.py"
TASK_PKG = REPO_ROOT / "deploy" / "k1_dance_deploy"
RUNS = REPO_ROOT / "eval" / "results" / "mujoco_runs"

# Tasks that exist only for testing, and robot-gain variants (the walk-stance + _robot check is in the README).
SKIP = re.compile(r"(_robot$|^k1_dance_seq_test$)")
COLUMNS = ["task", "start_stance", "completed_motion", "motion_seconds", "mean_joint_error_rad",
           "max_root_xy_drift_m", "mean_heading_error_deg", "max_heading_error_deg", "min_trunk_height_m",
           "worst_handoff_joint_error_rad"]


def available_tasks():
    """Registered k1 tasks whose model files exist, read via booster_deploy's venv."""
    code = (
        "import sys, os, pkgutil; os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1]);"
        "import tasks;"
        "[__import__(m.name) for m in pkgutil.walk_packages(tasks.__path__, prefix='tasks.')];"
        "sys.path.insert(0, sys.argv[2]); import k1_dance_deploy;"
        "from booster_deploy.utils.registry import list_tasks;"
        "pol = lambda c: getattr(c.policy, 'dances', None) or [(c.policy.checkpoint_path, c.policy.motion_path)];"
        "bm = os.path.join(sys.argv[1], 'tasks', 'beyond_mimic');"
        "print('\\n'.join(n for n, c in list_tasks().items() if n.startswith(('k1_dance', 'k1_routine'))"
        " and all(os.path.exists(os.path.join(bm, p)) for p, _ in pol(c))))"
    )
    out = subprocess.run([str(DEPLOY_PY), "-c", code, str(DEPLOY_PY.parents[2]), str(REPO_ROOT / "deploy")],
                         capture_output=True, text=True, check=True).stdout
    return [t for t in out.split() if not SKIP.search(t)]


def run(task, stance):
    out = RUNS / f"{task}_{stance}"
    subprocess.run([str(DEPLOY_PY), str(RECORD), "--task", task, "--no-video", "--start-stance", stance,
                    "--out", str(out)], capture_output=True, text=True, check=True)
    return load(out.with_suffix(".json"))


def load(path):
    d = json.loads(path.read_text())
    d["worst_handoff_joint_error_rad"] = max((p["handoff_joint_error_rad"] for p in d.get("per_dance", [])[1:]),
                                             default=None)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="*", help="task names (default: every k1_dance/k1_routine task with models)")
    p.add_argument("--stances", nargs="+", default=["default", "walk"], choices=["default", "walk"])
    p.add_argument("--table-only", action="store_true", help="just rebuild the table from the saved runs")
    args = p.parse_args()

    current = available_tasks()
    tasks = [] if args.table_only else (args.tasks or current)
    RUNS.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for stance in args.stances:
            try:
                d = run(task, stance)
            except subprocess.CalledProcessError as e:
                print(f"{task} ({stance}): FAILED\n{e.stderr[-800:]}", file=sys.stderr)
                continue
            print(f"{task:34} {stance:7} completed={d['completed_motion']!s:5} err={d['mean_joint_error_rad']} "
                  f"drift={d['max_root_xy_drift_m']} heading={d['mean_heading_error_deg']}")

    # The table covers every current task with a saved run (so evaluating a few tasks doesn't drop the rest).
    rows = [{k: d.get(k) for k in COLUMNS}
            for task in current for stance in ("default", "walk")
            if (f := RUNS / f"{task}_{stance}.json").exists() and (d := load(f))]
    out = REPO_ROOT / "eval" / "results" / "mujoco_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    md = ["| Task | Start | Completed | Length (s) | Joint error (rad) | Max drift (m) | Heading error (deg) | Worst handoff (rad) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| `{r['task']}` | {r['start_stance']} | {'yes' if r['completed_motion'] else '**no**'} | "
                  f"{r['motion_seconds']} | {r['mean_joint_error_rad']} | {r['max_root_xy_drift_m']} | "
                  f"{r['mean_heading_error_deg']} | {r['worst_handoff_joint_error_rad'] or ''} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"\nSaved {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
