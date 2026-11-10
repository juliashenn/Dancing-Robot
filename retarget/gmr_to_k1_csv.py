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

# Motor speed limits (rad/s) from booster_train assets/robots/actuator.py + booster_k1.py:
# (velocity_limit: torque drops to zero, knee_point_velocity: full torque up to here).
MOTOR_SPEED = {
    "hip_pitch": (14.66, 1.88),   # E6408
    "hip_roll": (12.15, 8.38),    # E4315
    "hip_yaw": (17.59, 7.85),     # E4310
    "knee": (12.57, 2.09),        # E6416
    "ankle": (17.59, 7.85),       # E4310 (parallel linkage, ratio 1.0)
    "shoulder": (33.51, 5.24),    # R14
    "elbow": (33.51, 5.24),       # R14
    "head": (7.85, 7.85),         # HT4438
}


SMALL_CLIP_RAD = 0.05  # ~3 deg: clipping smaller than this is ignored by the frequency rule


def joint_group(name):
    if "head" in name:
        return "head"
    if any(s in name for s in ("hip", "knee", "ankle")):
        return "legs"
    return "arms"


def speed_limit(name):
    return next(v for k, v in MOTOR_SPEED.items() if k in name)


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
    """Clip to joint limits and judge feasibility.

    Verdict rules (what matters for balance):
      * legs (hip/knee/ankle): REVIEW if clipped by more than SMALL_CLIP_RAD in > clip_frac_warn of frames,
        or by > clip_rad_warn rad in any frame,
        or if the 99th-percentile speed exceeds the motor's hard speed limit;
      * arms: clipping is reported but doesn't fail the clip (a less-bent elbow looks nearly the same);
        REVIEW only if the 99th-percentile speed exceeds the motor's hard limit;
      * head: reported only.
    Speeds use the 99th percentile over frames, so single-frame retargeting glitches don't dominate;
    glitch frames (speed above the hard limit) are counted separately.
    """
    below = np.clip(lower - dof_pos, 0.0, None)
    above = np.clip(dof_pos - upper, 0.0, None)
    violation = np.maximum(below, above)            # rad outside the limit, 0 if inside
    clipped = violation > 1e-9
    clipped_dof = np.clip(dof_pos, lower, upper)
    if len(dof_pos) > 2:
        vel = np.abs(np.gradient(clipped_dof, 1.0 / fps, axis=0))
    else:
        vel = np.zeros_like(dof_pos)

    joints, speeds, reasons = {}, {}, []
    for j, name in enumerate(K1_JOINT_NAMES):
        group = joint_group(name)
        hard, full_torque = speed_limit(name)
        p99 = float(np.percentile(vel[:, j], 99))
        over_hard = int((vel[:, j] > hard).sum())
        speeds[name] = {"p99_rad_s": round(p99, 2), "max_rad_s": round(float(vel[:, j].max()), 2),
                        "frames_over_motor_limit": over_hard,
                        "frames_over_full_torque_pct": round(100.0 * float((vel[:, j] > full_torque).mean()), 1)}
        if clipped[:, j].any():
            joints[name] = {
                "group": group,
                "frames_clipped_pct": round(100.0 * clipped[:, j].mean(), 2),
                "worst_violation_rad": round(float(violation[:, j].max()), 3),
            }
            # Violations under SMALL_CLIP_RAD (e.g. the 1.4 deg gap between GMR's and Booster's hip-roll
            # limits) don't count toward the frequency rule.
            significant = (violation[:, j] > SMALL_CLIP_RAD).mean()
            if group == "legs" and (significant > clip_frac_warn or violation[:, j].max() > clip_rad_warn):
                reasons.append(f"{name} clipped ({joints[name]['frames_clipped_pct']}% of frames, "
                               f"worst {joints[name]['worst_violation_rad']} rad)")
        if group != "head" and p99 > hard:
            reasons.append(f"{name} too fast (p99 {p99:.1f} rad/s > motor limit {hard})")

    report = {
        "verdict": "OK" if not reasons else "REVIEW",
        "reasons": reasons,
        "frames_with_any_clipping_pct": round(100.0 * clipped.any(axis=1).mean(), 2),
        "frames_with_leg_clipping_pct": round(100.0 * clipped[:, [joint_group(n) == "legs" for n in K1_JOINT_NAMES]].any(axis=1).mean(), 2),
        "clipped_joints": joints,
        "joint_speeds": speeds,
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

    # Whole-body checks: big trunk tilt or a very low trunk (floor work) is hard for the K1 to balance through.
    x, y = root_rot[:, 0], root_rot[:, 1]
    tilt_deg = np.degrees(np.arccos(np.clip(1 - 2 * (x**2 + y**2), -1.0, 1.0)))  # trunk z-axis vs. vertical
    report["max_trunk_tilt_deg"] = round(float(tilt_deg.max()), 1)
    if tilt_deg.max() > args.max_tilt_deg:
        report["reasons"].append(f"trunk tilts {tilt_deg.max():.0f} deg (> {args.max_tilt_deg:g})")
    if root_pos[:, 2].min() < args.min_root_height:
        report["reasons"].append(f"trunk drops to {root_pos[:, 2].min():.2f} m (< {args.min_root_height:g} m)")

    # Turning and travel: the deployed policy can't see its position or heading in the room, so fast
    # spins and traveling steps don't transfer (Jazz1 spun ~500 deg/s for 12 s and traveled 4.5 m; in
    # MuJoCo the policy kept facing forward and ended 1.7 m from the reference).
    w, z = root_rot[:, 3], root_rot[:, 2]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2)))
    yaw_rate = np.abs(np.gradient(np.degrees(yaw), 1.0 / fps)) if len(yaw) > 2 else np.zeros(1)
    span = float(np.max(np.ptp(root_pos[:, :2], axis=0)))
    report["yaw_rate_p95_deg_s"] = round(float(np.percentile(yaw_rate, 95)), 1)
    report["total_turn_deg"] = round(float(np.degrees(np.abs(np.diff(yaw)).sum())), 1)
    report["travel_span_m"] = round(span, 2)
    if report["yaw_rate_p95_deg_s"] > args.max_yaw_rate:
        report["reasons"].append(f"spins too fast (p95 {report['yaw_rate_p95_deg_s']:.0f} deg/s > {args.max_yaw_rate:g})")
    if span > args.max_travel:
        report["reasons"].append(f"travels {span:.2f} m (> {args.max_travel:g} m)")
    report["verdict"] = "OK" if not report["reasons"] else "REVIEW"

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
          f"({meta['duration_s']} s), leg clipping in {report['frames_with_leg_clipping_pct']}% of frames, "
          f"any clipping in {report['frames_with_any_clipping_pct']}%")
    for reason in report["reasons"]:
        print(f"           - {reason}")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="GMR .pkl files or folders of them")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "motions" / "k1_csv")
    p.add_argument("--limits-xml", type=Path, default=DEFAULT_LIMITS_XML,
                   help="K1 MJCF with the real joint limits (default: booster_assets K1_22dof.xml)")
    p.add_argument("--clip-frac-warn", type=float, default=0.05,
                   help="Flag REVIEW if any leg joint is clipped in more than this fraction of frames (default 0.05)")
    p.add_argument("--clip-rad-warn", type=float, default=0.2,
                   help="Flag REVIEW if any leg joint exceeds its limit by more than this many rad (default 0.2)")
    p.add_argument("--max-tilt-deg", type=float, default=45.0,
                   help="Flag REVIEW if the trunk tilts more than this from vertical (default 45)")
    p.add_argument("--min-root-height", type=float, default=0.40,
                   help="Flag REVIEW if the trunk drops below this height in m; standing is ~0.55 (default 0.40)")
    p.add_argument("--max-yaw-rate", type=float, default=300.0,
                   help="Flag REVIEW if the 95th-percentile turning speed exceeds this, deg/s (default 300; Pop1 127 and MJ 168 worked, Jazz1 665 did not)")
    p.add_argument("--max-travel", type=float, default=1.0,
                   help="Flag REVIEW if the trunk travels over a span larger than this, m (default 1.0)")
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
