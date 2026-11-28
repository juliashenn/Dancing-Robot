"""Measure the policy handoff between every ordered pair of library dances, then pick routine orders.

A handoff's quality depends on the two dances that meet: how the first one ends and how the second one's
policy takes over from that state. In MuJoCo the same pair scores the same wherever it sits in a routine,
so one pass over all ordered pairs is enough to choose the order of any routine.

    python eval/handoff_matrix.py                  # measure all pairs (walk stance), then rank routine orders
    python eval/handoff_matrix.py --rank-only      # re-rank from eval/results/handoff_matrix.csv

Handoff score = mean joint error (rad) over the first 1 s after the switch (record_mujoco's per-dance metric).
For each routine in deploy/k1_dance_deploy/routines.yaml it prints the current order and the order with the
smallest worst handoff (ties broken by the total). Run with any Python; simulations use booster_deploy's venv.
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))
from evaluate_policies import DEPLOY_PY, RECORD  # noqa: E402

ROUTINES = REPO_ROOT / "deploy" / "k1_dance_deploy" / "routines.yaml"
OUT = REPO_ROOT / "eval" / "results" / "handoff_matrix.csv"
RUNS = REPO_ROOT / "eval" / "results" / "handoff_runs"


def library():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import os; os.chdir(sys.argv[2]); sys.path.insert(0, '.');"
            "import k1_dance_deploy as k; print(' '.join(n for n, d in k.DANCE_LIBRARY.items()"
            " if os.path.exists(os.path.join(k._HERE, d[0]))))")
    out = subprocess.run([str(DEPLOY_PY), "-c", code, str(REPO_ROOT / "deploy"), str(DEPLOY_PY.parents[2])],
                         capture_output=True, text=True, check=True).stdout
    return out.split()


def measure(dances, stance):
    env = dict(os.environ, K1_DANCE_PAIR_TASKS="1")
    RUNS.mkdir(parents=True, exist_ok=True)
    rows = []
    for a, b in itertools.permutations(dances, 2):
        out = RUNS / f"{a}_{b}_{stance}"
        subprocess.run([str(DEPLOY_PY), str(RECORD), "--task", f"k1_pair_{a}_{b}", "--no-video",
                        "--start-stance", stance, "--out", str(out)], capture_output=True, text=True, check=True, env=env)
        d = json.loads(out.with_suffix(".json").read_text())
        second = d["per_dance"][1]
        rows.append({"from": a, "to": b, "handoff_joint_error_rad": second["handoff_joint_error_rad"],
                     "handoff_max_heading_error_deg": second["handoff_max_heading_error_deg"],
                     "completed": d["completed_motion"]})
        print(f"{a:>7} -> {b:<7} {second['handoff_joint_error_rad']:.3f} rad"
              f"{'' if d['completed_motion'] else '  DID NOT COMPLETE'}")
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {OUT}")


def best_order(dances, h, score):
    """Order with the smallest worst handoff, ties broken by the total. Exact for any routine length."""
    if len(dances) <= 8:
        return min(itertools.permutations(dances), key=score)
    # Too many orders to list: for each threshold T (ascending, binary search over the pair scores), find the
    # cheapest order that only uses handoffs <= T (Held-Karp over subsets); the first feasible T is optimal.
    n = len(dances)
    w = [[h[(a, b)] if a != b else 0.0 for b in dances] for a in dances]

    def cheapest(t):
        inf = float("inf")
        cost = {(1 << i, i): (0.0, None) for i in range(n)}
        for mask in range(1, 1 << n):
            for last in range(n):
                if (mask, last) not in cost:
                    continue
                c = cost[(mask, last)][0]
                for nxt in range(n):
                    if mask & (1 << nxt) or w[last][nxt] > t:
                        continue
                    key = (mask | (1 << nxt), nxt)
                    if c + w[last][nxt] < cost.get(key, (inf, None))[0]:
                        cost[key] = (c + w[last][nxt], last)
        full = (1 << n) - 1
        ends = [(cost[(full, i)][0], i) for i in range(n) if (full, i) in cost]
        if not ends:
            return None
        order, mask, last = [], full, min(ends)[1]
        while last is not None:
            order.append(dances[last])
            prev = cost[(mask, last)][1]
            mask, last = mask & ~(1 << last), prev
        return order[::-1]

    thresholds = sorted({w[i][j] for i in range(n) for j in range(n) if i != j})
    lo, hi, best = 0, len(thresholds) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        found = cheapest(thresholds[mid])
        if found:
            best, hi = found, mid - 1
        else:
            lo = mid + 1
    return best


def rank():
    with open(OUT) as f:
        h = {(r["from"], r["to"]): float(r["handoff_joint_error_rad"]) for r in csv.DictReader(f)}
    score = lambda order: (max(h[p] for p in zip(order, order[1:])), sum(h[p] for p in zip(order, order[1:])))
    routines = yaml.safe_load(ROUTINES.read_text()) or {}
    for name, dances in routines.items():
        if len(dances) < 2 or not all((a, b) in h for a, b in itertools.permutations(set(dances), 2)):
            continue
        best = best_order(dances, h, score)
        cur, bst = score(dances), score(best)
        print(f"{name}: current {dances} worst {cur[0]:.3f} / total {cur[1]:.3f}\n"
              f"{'':{len(name)}}  best    {list(best)} worst {bst[0]:.3f} / total {bst[1]:.3f}")
    names = sorted({a for a, _ in h})
    print("\nHandoff matrix (rad, row = first dance, column = next dance):")
    print(f"{'':>8}" + "".join(f"{n:>8}" for n in names))
    for a in names:
        print(f"{a:>8}" + "".join(f"{h[(a, b)]:8.3f}" if a != b else f"{'-':>8}" for b in names))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rank-only", action="store_true")
    p.add_argument("--stance", default="walk", choices=["default", "walk"],
                   help="start stance (walk = what the real robot starts from)")
    args = p.parse_args()
    if not args.rank_only:
        measure(library(), args.stance)
    rank()


if __name__ == "__main__":
    main()
