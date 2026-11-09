"""Summarize K1 retargeting feasibility for AIST++ clips, by dance genre.

Reads the per-clip JSON reports written by retarget/gmr_to_k1_csv.py and prints
(and saves) a per-genre table: OK vs REVIEW, why clips fail, trunk tilt and
leg speeds. Needs only the standard library + numpy.

    python eval/aistpp_feasibility.py
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
REASON_KINDS = ("leg clipping", "too fast", "tilt", "low")


def reason_kind(reason):
    if "clipped" in reason:
        return "leg clipping"
    if "too fast" in reason:
        return "too fast"
    if "tilts" in reason:
        return "tilt"
    return "low"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", type=Path, default=REPO_ROOT / "motions" / "k1_csv")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "results" / "aistpp_feasibility.csv")
    args = p.parse_args()

    clips = []
    for f in sorted(args.reports.glob("g*_c*.json")):
        m = json.loads(f.read_text())
        parts = f.stem.split("_")
        legs = {k: v for k, v in m["joint_speeds"].items() if any(s in k for s in ("hip", "knee", "ankle"))}
        clips.append({
            "clip": f.stem, "genre": parts[0], "style": "basic" if parts[1] == "sBM" else "advanced",
            "verdict": m["verdict"], "duration_s": m["duration_s"],
            "reasons": sorted({reason_kind(r) for r in m["reasons"]}),
            "leg_clip_pct": m["frames_with_leg_clipping_pct"],
            "max_trunk_tilt_deg": m["max_trunk_tilt_deg"],
            "max_leg_speed_p99": max(v["p99_rad_s"] for v in legs.values()),
            "root_z_min": m["root_height_min_m"],
        })
    if not clips:
        raise SystemExit(f"No AIST++ reports in {args.reports}")

    print(f"{len(clips)} clips, {sum(c['duration_s'] for c in clips) / 60:.1f} min\n")
    header = (f"{'Genre':15} {'clips':>5} {'OK':>4} {'OK%':>5} {'OK min':>6} | REVIEW because: "
              f"{'legclip':>7} {'fast':>5} {'tilt':>5} {'low':>4} | {'med tilt':>8} {'med leg p99':>11}")
    print(header + "\n" + "-" * len(header))
    by_genre = collections.defaultdict(list)
    for c in clips:
        by_genre[c["genre"]].append(c)
    rows = []
    for g, cs in sorted(by_genre.items(), key=lambda kv: -np.mean([c["verdict"] == "OK" for c in kv[1]])):
        ok = [c for c in cs if c["verdict"] == "OK"]
        counts = collections.Counter(r for c in cs for r in c["reasons"])
        row = {
            "genre": GENRES.get(g, g), "clips": len(cs), "ok": len(ok),
            "ok_pct": round(100 * len(ok) / len(cs)),
            "ok_minutes": round(sum(c["duration_s"] for c in ok) / 60, 1),
            **{f"fail_{k.replace(' ', '_')}": counts[k] for k in REASON_KINDS},
            "median_max_tilt_deg": round(float(np.median([c["max_trunk_tilt_deg"] for c in cs])), 1),
            "median_leg_speed_p99": round(float(np.median([c["max_leg_speed_p99"] for c in cs])), 1),
        }
        rows.append(row)
        print(f"{row['genre']:15} {row['clips']:5} {row['ok']:4} {row['ok_pct']:4}% {row['ok_minutes']:6} |"
              f"{'':16}{counts['leg clipping']:7} {counts['too fast']:5} {counts['tilt']:5} {counts['low']:4} |"
              f" {row['median_max_tilt_deg']:8} {row['median_leg_speed_p99']:11}")
    total_ok = sum(c["verdict"] == "OK" for c in clips)
    print(f"\nAll: {total_ok}/{len(clips)} OK ({100 * total_ok / len(clips):.0f}%), "
          f"{sum(c['duration_s'] for c in clips if c['verdict'] == 'OK') / 60:.1f} min of OK motion")
    for style in ("basic", "advanced"):
        cs = [c for c in clips if c["style"] == style]
        print(f"  {style:8}: {sum(c['verdict'] == 'OK' for c in cs)}/{len(cs)} OK")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    clip_out = args.out.with_name(args.out.stem + "_clips.csv")
    with open(clip_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(clips[0]))
        w.writeheader()
        for c in clips:
            w.writerow({**c, "reasons": ";".join(c["reasons"])})
    print(f"\nSaved {args.out} and {clip_out}")


if __name__ == "__main__":
    main()
