"""Retarget LAFAN1 BVH clips onto the Booster K1 with GMR (headless, CPU).
Usage (repo root): third_party/GMR/.venv/bin/python retarget/lafan1_to_k1.py motions/raw/lafan1/dance1_subject3.bvh
Outputs motions/gmr_pkl/lafan_<clip>.pkl and motions/k1_csv/lafan_<clip>.csv. LAFAN1 is 30 fps.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CONFIG = HERE / "ik_configs" / "bvh_lafan1_to_k1.json"
LAFAN1_FPS = 30
sys.path.insert(0, str(HERE))

from general_motion_retargeting import IK_CONFIG_DICT  # noqa: E402
from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_bvh_file  # noqa: E402

import gmr_to_k1_csv  # noqa: E402
from retarget_to_k1 import ground_offset  # noqa: E402

IK_CONFIG_DICT["bvh_lafan1"]["booster_k1"] = CONFIG


def retarget_bvh(bvh, pkl_out):
    frames, human_height = load_bvh_file(str(bvh), format="lafan1")
    retargeter = GMR(src_human="bvh_lafan1", tgt_robot="booster_k1",
                     actual_human_height=human_height, verbose=False)
    qpos_all = np.array([retargeter.retarget(f).copy() for f in frames])
    gap = ground_offset(retargeter.model, qpos_all)
    qpos_all[:, 2] -= gap
    qpos_all[:, :2] -= qpos_all[0, :2]
    motion = {
        "fps": float(LAFAN1_FPS),
        "root_pos": qpos_all[:, :3],
        "root_rot": qpos_all[:, [4, 5, 6, 3]],  # w,x,y,z -> x,y,z,w
        "dof_pos": qpos_all[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    pkl_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_out, "wb") as f:
        pickle.dump(motion, f)
    return len(qpos_all), gap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--glob", default="*.bvh")
    p.add_argument("--pkl-dir", type=Path, default=REPO_ROOT / "motions" / "gmr_pkl")
    p.add_argument("--csv-dir", type=Path, default=REPO_ROOT / "motions" / "k1_csv")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not CONFIG.exists():
        sys.exit(f"Missing IK config: {CONFIG}")
    sources = []
    for inp in args.inputs:
        sources += sorted(inp.glob(args.glob)) if inp.is_dir() else [inp]
    if not sources:
        sys.exit("No .bvh files found.")
    lower, upper = gmr_to_k1_csv.load_joint_limits(gmr_to_k1_csv.DEFAULT_LIMITS_XML)
    csv_args = argparse.Namespace(clip_frac_warn=0.05, clip_rad_warn=0.2, max_tilt_deg=45.0, min_root_height=0.40)
    failed = []
    for i, src in enumerate(sources, 1):
        pkl_out = args.pkl_dir / f"lafan_{src.stem}.pkl"
        try:
            if pkl_out.exists() and not args.overwrite:
                print(f"({i}/{len(sources)}) {src.name}: already retargeted, converting only")
            else:
                t0 = time.time()
                n, gap = retarget_bvh(src, pkl_out)
                print(f"({i}/{len(sources)}) {src.name}: {n} frames in {time.time() - t0:.0f}s "
                      f"(shifted {-gap:+.3f} m to the floor)")
            gmr_to_k1_csv.convert(pkl_out, args.csv_dir, lower, upper, csv_args)
        except Exception as e:
            failed.append(src)
            print(f"[ERROR ] {src.name}: {type(e).__name__}: {e}")
    print(f"\nDone: {len(sources) - len(failed)} ok, {len(failed)} failed. CSVs in {args.csv_dir}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
