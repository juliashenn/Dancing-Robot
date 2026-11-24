"""Chain K1 dance clips into one routine that passes through a neutral standing pose between clips.

Each clip eases in from the neutral pose over its first --blend seconds and eases back out over its last
--blend seconds. The clip keeps playing while the blend weight changes, so joint velocities stay smooth.
The robot holds the neutral pose for --hold seconds between clips and --hold-ends seconds at the start and
end. Each clip is rotated and shifted to start where the previous one ended, facing the same way.

    third_party/GMR/.venv/Scripts/python retarget/make_routine.py --name routine1 \
        gLO_sBM_cAll_d13_mLO0_ch01 gPO_sBM_cAll_d10_mPO0_ch02 gLH_sBM_cAll_d16_mLH0_ch01

Reads motions/k1_csv/selected/<clip>.csv (and .json for its fps). Writes motions/k1_csv/selected/<name>.csv
and <name>.json (segment start/end times, transition joint speeds, feasibility report).
Then convert with booster_train's csv_to_npz.py --input_fps <fps>.

The neutral pose is booster_train's K1 default (all joints 0, arms down: shoulder roll -/+1.3 rad),
lowered until the feet rest on the floor. Needs numpy, scipy and mujoco (GMR's venv has them).
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gmr_to_k1_csv import (  # noqa: E402
    DEFAULT_LIMITS_XML, K1_JOINT_NAMES, ROOT_COLUMNS, clip_report, load_joint_limits, speed_limit,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTED = REPO_ROOT / "motions" / "k1_csv" / "selected"
GMR_K1_XML = REPO_ROOT / "third_party" / "GMR" / "assets" / "booster_k1" / "K1_serial.xml"


def neutral_joints():
    q = np.zeros(len(K1_JOINT_NAMES))
    q[K1_JOINT_NAMES.index("left_shoulder_roll_joint")] = -1.3
    q[K1_JOINT_NAMES.index("right_shoulder_roll_joint")] = 1.3
    return q


def standing_height(joints):
    """Root height at which the K1's lowest collision geom touches the floor, trunk upright."""
    model = mj.MjModel.from_xml_path(str(GMR_K1_XML))
    data = mj.MjData(model)
    floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
    floor_body = model.geom_bodyid[floor]
    geoms = [g for g in range(model.ngeom)
             if g != floor and model.geom_bodyid[g] != floor_body
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    z0 = 1.0
    data.qpos[:3] = [0.0, 0.0, z0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = joints
    mj.mj_kinematics(model, data)
    fromto = np.zeros(6)
    gap = min(mj.mj_geomDistance(model, data, floor, g, 2.0, fromto) for g in geoms)
    return z0 - gap


def yaw_of(rot):
    """Heading (rad) of each rotation: angle of the body x-axis in the floor plane."""
    x_axis = rot.apply(np.array([1.0, 0.0, 0.0]))
    x_axis = np.atleast_2d(x_axis)
    return np.arctan2(x_axis[:, 1], x_axis[:, 0])


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def load_clip(stem):
    csv_path = SELECTED / f"{stem}.csv"
    if not csv_path.exists():
        sys.exit(f"{csv_path} not found (copy the clip from motions/k1_csv/ into selected/ first)")
    motion = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    meta_path = csv_path.with_suffix(".json")
    fps = json.loads(meta_path.read_text())["fps"] if meta_path.exists() else 30.0
    return motion, float(fps)


def place_clip(motion, start_xy, start_yaw):
    """Rotate/shift a clip so its first frame is at start_xy facing start_yaw."""
    pos, rot, joints = motion[:, :3].copy(), R.from_quat(motion[:, 3:7]), motion[:, 7:].copy()
    turn = R.from_euler("z", start_yaw - yaw_of(rot[0])[0])
    pos[:, :2] = turn.apply(pos - pos[0])[:, :2] + start_xy
    return pos, turn * rot, joints


def blend_with_neutral(pos, rot, joints, weight, neutral_q, neutral_z):
    """weight 1 = clip pose, 0 = neutral pose (same xy and heading as the clip, trunk upright)."""
    w = weight[:, None]
    joints = w * joints + (1 - w) * neutral_q
    pos = pos.copy()
    pos[:, 2] = weight * pos[:, 2] + (1 - weight) * neutral_z
    upright = R.from_euler("z", yaw_of(rot)[:, None])
    tilt = (upright.inv() * rot).as_rotvec()             # roll/pitch (and nothing else) relative to upright
    rot = upright * R.from_rotvec(tilt * w)
    return pos, rot, joints


def neutral_hold(n, xy, yaw, neutral_q, neutral_z):
    pos = np.tile([xy[0], xy[1], neutral_z], (n, 1))
    rot = R.from_euler("z", np.full((n, 1), yaw))
    return pos, rot, np.tile(neutral_q, (n, 1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="+", help="clip stems in motions/k1_csv/selected/, in routine order")
    p.add_argument("--name", required=True, help="output stem, e.g. routine1")
    p.add_argument("--blend", type=float, default=1.0, help="seconds to ease into / out of each clip")
    p.add_argument("--hold", type=float, default=0.5, help="seconds standing in the neutral pose between clips")
    p.add_argument("--hold-ends", type=float, default=1.0, help="seconds standing at the start and end")
    args = p.parse_args()

    neutral_q = neutral_joints()
    neutral_z = standing_height(neutral_q)
    clips = [(stem, *load_clip(stem)) for stem in args.clips]
    fps = clips[0][2]
    if any(c[2] != fps for c in clips):
        sys.exit(f"Clips have different frame rates: {[(c[0], c[2]) for c in clips]}")
    nb, nh, ne = round(args.blend * fps), round(args.hold * fps), round(args.hold_ends * fps)

    parts, segments, transition = [], [], []
    xy, yaw, t = np.zeros(2), 0.0, 0

    def add(pos, rot, joints, label, is_transition):
        nonlocal t
        parts.append(np.hstack([pos, rot.as_quat(), joints]))
        transition.append(np.asarray(is_transition, dtype=bool) & np.ones(len(pos), dtype=bool))
        segments.append({"part": label, "start_s": round(t / fps, 2), "end_s": round((t + len(pos)) / fps, 2)})
        t += len(pos)

    add(*neutral_hold(ne, xy, yaw, neutral_q, neutral_z), "neutral (start)", True)
    for i, (stem, motion, _) in enumerate(clips):
        n = len(motion)
        if n <= 2 * nb:
            sys.exit(f"{stem}: {n} frames is too short for {nb}-frame blends at both ends")
        pos, rot, joints = place_clip(motion, xy, yaw)
        k = np.arange(n)
        weight = np.minimum(smootherstep(k / nb), smootherstep((n - 1 - k) / nb))
        pos, rot, joints = blend_with_neutral(pos, rot, joints, weight, neutral_q, neutral_z)
        add(pos, rot, joints, stem, weight < 1.0)
        xy, yaw = pos[-1, :2].copy(), float(yaw_of(rot[-1])[0])
        if i < len(clips) - 1:
            add(*neutral_hold(nh, xy, yaw, neutral_q, neutral_z), "neutral", True)
    add(*neutral_hold(ne, xy, yaw, neutral_q, neutral_z), "neutral (end)", True)

    motion = np.vstack(parts)
    in_transition = np.concatenate(transition)

    lower, upper = load_joint_limits(DEFAULT_LIMITS_XML)
    clipped, report = clip_report(motion[:, 7:], lower, upper, fps, 0.05, 0.2)
    motion[:, 7:] = clipped
    vel = np.abs(np.gradient(clipped, 1.0 / fps, axis=0))
    ratio = np.array([vel[:, j] / speed_limit(name)[0] for j, name in enumerate(K1_JOINT_NAMES)]).T
    worst = int(np.argmax(ratio[in_transition].max(axis=0)))
    yaw_all = np.unwrap(yaw_of(R.from_quat(motion[:, 3:7])))

    out_csv = SELECTED / f"{args.name}.csv"
    np.savetxt(out_csv, motion, delimiter=",", fmt="%.8f", header=",".join(ROOT_COLUMNS + K1_JOINT_NAMES), comments="")
    meta = {
        "clips": args.clips,
        "fps": fps,
        "frames": int(len(motion)),
        "duration_s": round(len(motion) / fps, 2),
        "blend_s": args.blend, "hold_s": args.hold, "hold_ends_s": args.hold_ends,
        "neutral_root_height_m": round(float(neutral_z), 3),
        "segments": segments,
        "transition_max_speed": {
            "joint": K1_JOINT_NAMES[worst],
            "rad_s": round(float(vel[in_transition, worst].max()), 2),
            "pct_of_motor_limit": round(100 * float(ratio[in_transition, worst].max()), 1),
        },
        "total_turn_deg": round(float(np.degrees(np.abs(np.diff(yaw_all)).sum())), 1),
        "travel_span_m": round(float(np.max(np.ptp(motion[:, :2], axis=0))), 2),
        **report,
    }
    out_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    print(f"{out_csv.name}: {meta['frames']} frames @ {fps:g} fps ({meta['duration_s']} s), "
          f"neutral stance at {meta['neutral_root_height_m']} m")
    for s in segments:
        print(f"  {s['start_s']:6.2f} - {s['end_s']:6.2f} s  {s['part']}")
    tm = meta["transition_max_speed"]
    print(f"Fastest joint during blends: {tm['joint']} {tm['rad_s']} rad/s ({tm['pct_of_motor_limit']}% of motor limit)")
    print(f"Verdict: {report['verdict']}" + "".join(f"\n  - {r}" for r in report["reasons"]))


if __name__ == "__main__":
    main()
