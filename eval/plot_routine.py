"""Poster figure: tracking over a whole routine, with each dance and each policy handoff marked.

    third_party/booster_deploy/.venv/Scripts/python deploy/record_mujoco.py --task k1_routine_everything \
        --no-video --frames --out eval/results/routine_everything
    python eval/plot_routine.py eval/results/routine_everything      # -> routine_everything.png / .svg

Reads <run>.json (dance start times and names) and <run>_frames.csv (per-frame metrics).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BAND = "#efeeea"
SERIES = "#2a78d6"
NAMES = {"lock1n": "Lock", "pop1n": "Pop", "hiphop1n": "Hip-hop", "sjazz1n": "Street Jazz",
         "krump1n": "Krump", "pop2n": "Pop 2"}

PANELS = [  # one measure per panel (no shared y-axis between different units)
    ("joint_error_rad", "Joint error (rad)"),
    ("heading_error_deg", "Heading error (°)"),
    ("root_xy_drift_m", "Drift (m)"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="record_mujoco --out path (without extension)")
    p.add_argument("--smooth", type=float, default=0.5, help="moving-average window in seconds (0 = raw)")
    args = p.parse_args()

    summary = json.loads(args.run.with_suffix(".json").read_text())
    with open(f"{args.run}_frames.csv") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t_s"]) for r in rows])
    starts = summary["dance_start_times_s"] + [t[-1]]
    names = [NAMES.get(Path(m).stem, Path(m).stem) for m in summary["motion"]]
    dt = t[1] - t[0]
    win = max(1, int(round(args.smooth / dt)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(12, 6.8), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.18})
    for ax, (key, label) in zip(axes, PANELS):
        y = np.array([float(r[key]) for r in rows])
        if win > 1:
            y = np.convolve(np.pad(y, (win // 2, win - 1 - win // 2), mode="edge"), np.ones(win) / win, mode="valid")
        ax.set_facecolor(SURFACE)
        for i, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
            if i % 2 == 0:
                ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
        for s in starts[1:-1]:
            ax.axvline(s, color=INK_2, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.plot(t, y, color=SERIES, lw=2, solid_capstyle="round", zorder=2)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, max(y) * 1.15)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(x=0)
    for a, b, name in zip(starts[:-1], starts[1:], names):
        axes[0].text((a + b) / 2, 1.02, name, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     color=INK, fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("Time (s)")
    stance = summary.get("start_stance", "default")
    fig.suptitle(f"{len(names)} dances, one policy each, switching in the neutral stance "
                 f"({summary['motion_seconds']:.0f} s in MuJoCo, {stance} start).\n"
                 "Dashed line = policy handoff; "
                 "drift is measured from where the current dance started.",
                 x=0.07, y=0.99, ha="left", color=INK_2, fontsize=10)
    for ext in ("png", "svg"):
        fig.savefig(args.run.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {args.run.with_suffix('.png')} and .svg")


if __name__ == "__main__":
    main()
