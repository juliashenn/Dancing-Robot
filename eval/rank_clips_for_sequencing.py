"""Rank retargeted AIST++ clips by how easy they should be to train and to start/end in the neutral stance.

Uses the per-clip CSV + JSON written by retarget/gmr_to_k1_csv.py (motions/k1_csv/g*_c*.csv/.json) and
only considers clips the feasibility filter marked OK.

Difficulty (lower = easier to train), from what we've seen in training:
  * turning (Jazz needed slowing down + 20000 iterations; Lock/Pop barely turn and took ~5000),
  * travel (the policy doesn't observe position, so travel becomes drift),
  * trunk tilt and leg speed (balance and motor limits),
  * leg clipping (retargeting artifacts).
Neutral fit (lower = easier blend in/out of the neutral stance): how far the first and last frames are
from booster_train's default stance (joints 0, arms down), in joint angle, trunk tilt and root height.

    third_party/GMR/.venv/Scripts/python eval/rank_clips_for_sequencing.py
Writes eval/results/sequencing_candidates.csv and prints the best clips per genre.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "retarget"))
from gmr_to_k1_csv import K1_JOINT_NAMES, joint_group  # noqa: E402

GENRES = {"gBR": "Break", "gPO": "Pop", "gLO": "Lock", "gMH": "Middle Hip-hop", "gLH": "LA Hip-hop",
          "gHO": "House", "gWA": "Waack", "gKR": "Krump", "gJS": "Street Jazz", "gJB": "Ballet Jazz"}
NEUTRAL_ROOT_Z = 0.523  # retarget/make_routine.py's neutral stance height (feet on the floor)
LEGS = np.array([joint_group(n) == "legs" for n in K1_JOINT_NAMES])
ARMS = np.array([joint_group(n) == "arms" for n in K1_JOINT_NAMES])


def neutral_joints():
    q = np.zeros(len(K1_JOINT_NAMES))
    q[K1_JOINT_NAMES.index("left_shoulder_roll_joint")] = -1.3
    q[K1_JOINT_NAMES.index("right_shoulder_roll_joint")] = 1.3
    return q


def pose_distance(row, neutral):
    """Distance of one CSV frame from the neutral stance: legs count double (balance), arms once,
    plus trunk tilt and root-height difference."""
    joints = row[7:]
    x, y = row[3], row[4]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x**2 + y**2), -1.0, 1.0)))
    leg = float(np.sqrt(np.mean((joints[LEGS] - neutral[LEGS]) ** 2)))
    arm = float(np.sqrt(np.mean((joints[ARMS] - neutral[ARMS]) ** 2)))
    dz = abs(float(row[2]) - NEUTRAL_ROOT_Z)
    return 2 * leg + arm + tilt / 30 + dz / 0.05, leg, arm, tilt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips", type=Path, default=REPO_ROOT / "motions" / "k1_csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results" / "sequencing_candidates.csv")
    p.add_argument("--per-genre", type=int, default=3)
    args = p.parse_args()

    neutral = neutral_joints()
    rows = []
    for meta_path in sorted(args.clips.glob("g*_c*.json")):
        m = json.loads(meta_path.read_text())
        if m["verdict"] != "OK":
            continue
        motion = np.loadtxt(meta_path.with_suffix(".csv"), delimiter=",", skiprows=1)
        legs = {k: v for k, v in m["joint_speeds"].items() if joint_group(k) == "legs"}
        leg_p99 = max(v["p99_rad_s"] for v in legs.values())
        # Each term is ~1 for a "typical hard but feasible" value.
        difficulty = (m["yaw_rate_p95_deg_s"] / 150 + m["travel_span_m"] / 0.5 + m["max_trunk_tilt_deg"] / 30
                      + leg_p99 / 8 + m["frames_with_leg_clipping_pct"] / 5)
        start, s_leg, s_arm, s_tilt = pose_distance(motion[0], neutral)
        end, e_leg, e_arm, e_tilt = pose_distance(motion[-1], neutral)
        rows.append({
            "clip": meta_path.stem, "genre": GENRES.get(meta_path.stem[:3], meta_path.stem[:3]),
            "duration_s": m["duration_s"],
            "score": round(difficulty + (start + end) / 2, 2),
            "difficulty": round(difficulty, 2), "neutral_gap": round((start + end) / 2, 2),
            "yaw_rate_p95_deg_s": m["yaw_rate_p95_deg_s"], "travel_span_m": m["travel_span_m"],
            "max_trunk_tilt_deg": m["max_trunk_tilt_deg"], "leg_speed_p99": leg_p99,
            "leg_clip_pct": m["frames_with_leg_clipping_pct"],
            "start_leg_rms_rad": round(s_leg, 2), "end_leg_rms_rad": round(e_leg, 2),
            "start_tilt_deg": round(s_tilt, 1), "end_tilt_deg": round(e_tilt, 1),
        })
    if not rows:
        sys.exit(f"No OK clips with CSVs in {args.clips}")
    rows.sort(key=lambda r: r["score"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} OK clips ranked (lower score = easier). Best {args.per_genre} per genre:\n")
    print(f"{'clip':30} {'genre':15} {'dur':>5} {'score':>5} {'diff':>5} {'gap':>5} {'yaw/s':>5} {'trvl':>5} "
          f"{'tilt':>5} {'legv':>5}")
    for genre in sorted({r["genre"] for r in rows}, key=lambda g: min(r["score"] for r in rows if r["genre"] == g)):
        for r in [r for r in rows if r["genre"] == genre][:args.per_genre]:
            print(f"{r['clip']:30} {r['genre']:15} {r['duration_s']:5.1f} {r['score']:5.2f} {r['difficulty']:5.2f} "
                  f"{r['neutral_gap']:5.2f} {r['yaw_rate_p95_deg_s']:5.0f} {r['travel_span_m']:5.2f} "
                  f"{r['max_trunk_tilt_deg']:5.1f} {r['leg_speed_p99']:5.1f}")
    print(f"\nSaved {args.out}")
    print("For reference, clips we've trained:")
    for stem in ("gLO_sBM_cAll_d13_mLO0_ch01", "gPO_sBM_cAll_d10_mPO0_ch02", "gLH_sBM_cAll_d16_mLH0_ch01"):
        hit = [(i + 1, r) for i, r in enumerate(rows) if r["clip"] == stem]
        if hit:
            i, r = hit[0]
            print(f"  {stem} ({r['genre']}): rank {i}/{len(rows)}, score {r['score']}")


if __name__ == "__main__":
    main()
