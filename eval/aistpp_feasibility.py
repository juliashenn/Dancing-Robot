"""Summarize K1 retargeting feasibility for AIST++ clips, by dance genre.

Reads the per-clip JSON reports written by retarget/gmr_to_k1_csv.py and prints
(and saves) per-genre tables: OK vs REVIEW counts, clipping, which joints hit
their limits, and joint speeds. Needs only the standard library + numpy.

    python eval/aistpp_feasibility.py                       # reads motions/k1_csv/g*.json
    python eval/aistpp_feasibility.py --out eval/results/aistpp_feasibility.csv
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GENRES = {"gBR": "Break", "gPO": "Pop", "gLO": "Lock", "gMH": "Middle Hip-hop", "gLH": "LA Hip-hop",
          "gHO": "House", "gWA": "Waack", "gKR": "Krump", "gJS": "Street Jazz", "gJB": "Ballet Jazz"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", type=Path, default=REPO_ROOT / "motions" / "k1_csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results" / "aistpp_feasibility.csv")
    args = p.parse_args()

    clips = []
    for f in sorted(args.reports.glob("g*_c*.json")):
        m = json.loads(f.read_text())
        parts = f.stem.split("_")
        speeds = m.get("max_joint_speed_rad_s", {})
        clips.append({
            "clip": f.stem, "genre": parts[0], "style": "basic" if parts[1] == "sBM" else "advanced",
            "verdict": m["verdict"], "duration_s": m["duration_s"],
            "clip_pct": m["frames_with_any_clipping_pct"],
            "worst_rad": max((j["worst_violation_rad"] for j in m["clipped_joints"].values()), default=0.0),
            "clipped_joints": list(m["clipped_joints"]),
            "max_speed": max(speeds.values(), default=0.0),
            "root_z_min": m["root_height_min_m"],
        })
    if not clips:
        raise SystemExit(f"No AIST++ reports in {args.reports}")

    print(f"{len(clips)} clips, {sum(c['duration_s'] for c in clips) / 60:.1f} min\n")
    header = f"{'Genre':15} {'clips':>5} {'OK':>4} {'OK%':>5} {'OK min':>6} {'med clip%':>9} {'med worst':>9} {'med speed':>9}"
    print(header + "\n" + "-" * len(header))
    rows = []
    by_genre = collections.defaultdict(list)
    for c in clips:
        by_genre[c["genre"]].append(c)
    for g, cs in sorted(by_genre.items(), key=lambda kv: -np.mean([c["verdict"] == "OK" for c in kv[1]])):
        ok = [c for c in cs if c["verdict"] == "OK"]
        row = {
            "genre": GENRES.get(g, g), "clips": len(cs), "ok": len(ok),
            "ok_pct": round(100 * len(ok) / len(cs)),
            "ok_minutes": round(sum(c["duration_s"] for c in ok) / 60, 1),
            "median_clip_pct": round(float(np.median([c["clip_pct"] for c in cs])), 1),
            "median_worst_rad": round(float(np.median([c["worst_rad"] for c in cs])), 2),
            "median_max_speed_rad_s": round(float(np.median([c["max_speed"] for c in cs])), 1),
        }
        rows.append(row)
        print(f"{row['genre']:15} {row['clips']:5} {row['ok']:4} {row['ok_pct']:4}% {row['ok_minutes']:6} "
              f"{row['median_clip_pct']:8}% {row['median_worst_rad']:9} {row['median_max_speed_rad_s']:9}")
    total_ok = sum(c["verdict"] == "OK" for c in clips)
    print(f"\nAll: {total_ok}/{len(clips)} OK ({100 * total_ok / len(clips):.0f}%), "
          f"{sum(c['duration_s'] for c in clips if c['verdict'] == 'OK') / 60:.1f} min of OK motion")
    for style in ("basic", "advanced"):
        cs = [c for c in clips if c["style"] == style]
        print(f"  {style:8}: {sum(c['verdict'] == 'OK' for c in cs)}/{len(cs)} OK")

    joint_counts = collections.Counter(j for c in clips for j in c["clipped_joints"])
    print("\nJoints that hit their limits (share of clips):")
    for j, n in joint_counts.most_common(8):
        print(f"  {j:28} {100 * n / len(clips):5.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    clip_out = args.out.with_name(args.out.stem + "_clips.csv")
    with open(clip_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in clips[0] if k != "clipped_joints"] + ["clipped_joints"])
        w.writeheader()
        for c in clips:
            w.writerow({**c, "clipped_joints": ";".join(c["clipped_joints"])})
    print(f"\nSaved {args.out} and {clip_out}")


if __name__ == "__main__":
    main()
