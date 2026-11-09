"""Convert GMR retargeted K1 motions (.pkl) to Booster's motion CSV format.

For each clip this:
  1. writes a CSV that booster_train's csv_to_npz.py accepts (header row, root
     position, root quaternion x/y/z/w, then the 22 K1 joints in Booster's order);
  2. clips every joint angle to the real robot's limits (from Booster's K1 MJCF),
     since GMR's K1 model allows wider ranges than the hardware;
  3. reports how much clipping was needed, as a first feasibility check.

Usage (from the repo root; needs only numpy):
    python retarget/gmr_to_k1_csv.py path/to/clip.pkl [more.pkl or folders ...]
    python retarget/gmr_to_k1_csv.py motions/gmr_pkl/ --out-dir motions/k1_csv

Outputs per clip:  <out-dir>/<name>.csv  and  <out-dir>/<name>.json  (fps + report)
Then convert with csv_to_npz.py using --input_fps from the JSON.
"""

import argparse
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIMITS_XML = REPO_ROOT / "third_party" / "booster_assets" / "robots" / "K1" / "K1_22dof.xml"

# Booster's CSV joint order (booster_assets/src/booster_assets/motions.py, K1_JOINT_NAMES).
# GMR's K1 model (assets/booster_k1/K1_serial.xml) uses the same order under different names.
K1_JOINT_NAMES = [
    "aahead_yaw_joint", "aahead_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
ROOT_COLUMNS = ["root_pos_x", "root_pos_y", "root_pos_z",
                "root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"]


def load_joint_limits(xml_path):
    """Return (lower, upper) arrays in K1_JOINT_NAMES order from a MuJoCo XML."""
    ranges = {}
    for joint in ET.parse(xml_path).getroot().iter("joint"):
        name, rng = joint.get("name"), joint.get("range")
        if name and rng:
            ranges[name.removeprefix("aa")] = [float(v) for v in rng.split()]
    missing = [n for n in K1_JOINT_NAMES if n.removeprefix("aa") not in ranges]
    if missing:
        sys.exit(f"Joint limits missing from {xml_path}: {missing}")
    limits = np.array([ranges[n.removeprefix("aa")] for n in K1_JOINT_NAMES])
    return limits[:, 0], limits[:, 1]


def load_gmr_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    for key in ("root_pos", "root_rot", "dof_pos", "fps"):
        if key not in data:
            raise ValueError(f"{path}: missing '{key}' (is this a GMR output .pkl?)")
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)  # GMR saves x, y, z, w
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    n = dof_pos.shape[0]
    if dof_pos.shape != (n, len(K1_JOINT_NAMES)):
        raise ValueError(f"{path}: dof_pos shape {dof_pos.shape}, expected (frames, 22). Retargeted with robot booster_k1?")
    if root_pos.shape != (n, 3) or root_rot.shape != (n, 4):
        raise ValueError(f"{path}: root shapes {root_pos.shape} / {root_rot.shape} don't match {n} frames")
    return root_pos, root_rot, dof_pos, float(data["fps"])


def clip_report(dof_pos, lower, upper, fps, clip_frac_warn, clip_rad_warn):
    below = np.clip(lower - dof_pos, 0.0, None)
    above = np.clip(dof_pos - upper, 0.0, None)
    violation = np.maximum(below, above)            # rad outside the limit, 0 if inside
    clipped = violation > 1e-9
    clipped_dof = np.clip(dof_pos, lower, upper)
    vel = np.abs(np.diff(clipped_dof, axis=0)) * fps if len(dof_pos) > 1 else np.zeros_like(dof_pos)

    joints = {}
    for j, name in enumerate(K1_JOINT_NAMES):
        if clipped[:, j].any():
            joints[name] = {
                "frames_clipped_pct": round(100.0 * clipped[:, j].mean(), 2),
                "worst_violation_rad": round(float(violation[:, j].max()), 3),
            }
    worst_frac = max((v["frames_clipped_pct"] for v in joints.values()), default=0.0) / 100.0
    worst_rad = max((v["worst_violation_rad"] for v in joints.values()), default=0.0)
    verdict = "OK" if worst_frac <= clip_frac_warn and worst_rad <= clip_rad_warn else "REVIEW"
    report = {
        "verdict": verdict,
        "frames_with_any_clipping_pct": round(100.0 * clipped.any(axis=1).mean(), 2),
        "clipped_joints": joints,
        "max_joint_speed_rad_s": {n: round(float(v), 2) for n, v in zip(K1_JOINT_NAMES, vel.max(axis=0))} if len(vel) else {},
    }
    return clipped_dof, report


def convert(pkl_path, out_dir, lower, upper, args):
    root_pos, root_rot, dof_pos, fps = load_gmr_pkl(pkl_path)
    norms = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError(f"{pkl_path}: zero-length root quaternion")
    root_rot = root_rot / norms

    clipped_dof, report = clip_report(dof_pos, lower, upper, fps, args.clip_frac_warn, args.clip_rad_warn)
    motion = np.hstack([root_pos, root_rot, clipped_dof])

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{pkl_path.stem}.csv"
    np.savetxt(csv_path, motion, delimiter=",", fmt="%.8f",
               header=",".join(ROOT_COLUMNS + K1_JOINT_NAMES), comments="")

    meta = {
        "source": str(pkl_path),
        "fps": fps,
        "frames": int(motion.shape[0]),
        "duration_s": round(motion.shape[0] / fps, 2),
        "root_height_min_m": round(float(root_pos[:, 2].min()), 3),
        "root_height_max_m": round(float(root_pos[:, 2].max()), 3),
        **report,
    }
    (out_dir / f"{pkl_path.stem}.json").write_text(json.dumps(meta, indent=2))

    print(f"[{report['verdict']:6}] {pkl_path.name}: {meta['frames']} frames @ {fps:g} fps "
          f"({meta['duration_s']} s), clipping in {report['frames_with_any_clipping_pct']}% of frames")
    for name, stats in sorted(report["clipped_joints"].items(), key=lambda kv: -kv[1]["frames_clipped_pct"]):
        print(f"           {name:28} {stats['frames_clipped_pct']:6.2f}% of frames, worst {stats['worst_violation_rad']:.3f} rad")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="GMR .pkl files or folders of them")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "motions" / "k1_csv")
    p.add_argument("--limits-xml", type=Path, default=DEFAULT_LIMITS_XML,
                   help="K1 MJCF with the real joint limits (default: booster_assets K1_22dof.xml)")
    p.add_argument("--clip-frac-warn", type=float, default=0.05,
                   help="Flag REVIEW if any joint is clipped in more than this fraction of frames (default 0.05)")
    p.add_argument("--clip-rad-warn", type=float, default=0.2,
                   help="Flag REVIEW if any joint exceeds its limit by more than this many rad (default 0.2)")
    args = p.parse_args()

    if not args.limits_xml.exists():
        sys.exit(f"{args.limits_xml} not found. Run setup/clone_third_party.py first.")
    lower, upper = load_joint_limits(args.limits_xml)

    pkls = []
    for inp in args.inputs:
        pkls += sorted(inp.glob("*.pkl")) if inp.is_dir() else [inp]
    if not pkls:
        sys.exit("No .pkl files found.")

    results, failed = [], []
    for pkl in pkls:
        try:
            results.append(convert(pkl, args.out_dir, lower, upper, args))
        except (ValueError, OSError, pickle.UnpicklingError) as e:
            failed.append(pkl)
            print(f"[ERROR ] {e}")
    ok = sum(r["verdict"] == "OK" for r in results)
    print(f"\n{len(results)} converted ({ok} OK, {len(results) - ok} REVIEW), {len(failed)} failed -> {args.out_dir}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
