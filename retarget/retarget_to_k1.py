"""Retarget human dance motion (AIST++ or AMASS) onto the Booster K1 with GMR.

Runs one clip at a time, on CPU, with no viewer: works on Windows laptops.
Writes GMR-format .pkl files, then converts them to Booster CSVs with
gmr_to_k1_csv.py (joint-limit clipping + feasibility report).

Run with GMR's Python (set up per docs/setup.md):
    third_party/GMR/.venv/Scripts/python retarget/retarget_to_k1.py motions/raw/aistpp/gBR_sBM_cAll_d04_mBR0_ch01.pkl
    third_party/GMR/.venv/Scripts/python retarget/retarget_to_k1.py motions/raw/aistpp/ --limit 5

Supported inputs:
  * AIST++ motion .pkl  (keys smpl_poses, smpl_trans, smpl_scaling; SMPL, y-up, 60 fps)
  * AMASS SMPL-X .npz   (keys pose_body, root_orient, trans, betas, gender, mocap_frame_rate; z-up)
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
GMR_ROOT = REPO_ROOT / "third_party" / "GMR"
BODY_MODELS = GMR_ROOT / "assets" / "body_models"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco as mj  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    get_smplx_data_offline_fast,
    load_smplx_file,
)
import gmr_to_k1_csv  # noqa: E402

AISTPP_FPS = 60
Y_UP_TO_Z_UP = R.from_euler("x", 90, degrees=True)


def aistpp_to_amass_npz(pkl_path, out_npz):
    """Convert an AIST++ SMPL motion to the AMASS SMPL-X layout GMR reads."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    poses = np.asarray(d["smpl_poses"], dtype=np.float64)          # (N, 72): 24 joints x axis-angle
    trans = np.asarray(d["smpl_trans"], dtype=np.float64) / float(np.ravel(d["smpl_scaling"])[0])
    # AIST++ is y-up; AMASS/GMR expect z-up.
    root_orient = (Y_UP_TO_Z_UP * R.from_rotvec(poses[:, :3])).as_rotvec()
    trans = Y_UP_TO_Z_UP.apply(trans)
    # SMPL joints 1..21 are the same body joints, in the same order, as SMPL-X body_pose.
    pose_body = poses[:, 3:66]
    np.savez(out_npz, pose_body=pose_body, root_orient=root_orient, trans=trans,
             betas=np.zeros(16), gender=np.array("neutral"), mocap_frame_rate=np.array(AISTPP_FPS))


def ground_offset(model, qpos_all):
    """Signed gap (m) between the robot's lowest collision geom and the floor, over the whole clip."""
    data = mj.MjData(model)
    floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ground")
    floor_body = model.geom_bodyid[floor]
    geoms = [g for g in range(model.ngeom)
             if g != floor and model.geom_bodyid[g] != floor_body
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    lowest = np.inf
    fromto = np.zeros(6)
    for qpos in qpos_all:
        data.qpos[:] = qpos
        mj.mj_kinematics(model, data)
        for g in geoms:
            lowest = min(lowest, mj.mj_geomDistance(model, data, floor, g, 1.0, fromto))
    return lowest


def retarget_clip(src, pkl_out, tgt_fps):
    if src.suffix == ".pkl":
        # Converted copy is kept (not a temp file): GMR holds the .npz open, which blocks deletion on Windows.
        npz = pkl_out.parent / "_smplx" / f"{src.stem}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        aistpp_to_amass_npz(src, npz)
    else:
        npz = src
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(str(npz), str(BODY_MODELS))
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)

    retargeter = GMR(src_human="smplx", tgt_robot="booster_k1", actual_human_height=human_height, verbose=False)
    qpos_all = np.array([retargeter.retarget(f).copy() for f in frames])

    gap = ground_offset(retargeter.model, qpos_all)
    qpos_all[:, 2] -= gap                       # lowest point of the clip rests on the floor
    qpos_all[:, :2] -= qpos_all[0, :2]          # start at the origin

    motion = {
        "fps": float(fps),
        "root_pos": qpos_all[:, :3],
        "root_rot": qpos_all[:, [4, 5, 6, 3]],  # MuJoCo w,x,y,z -> x,y,z,w
        "dof_pos": qpos_all[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    pkl_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_out, "wb") as f:
        pickle.dump(motion, f)
    return len(qpos_all), fps, gap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="AIST++ .pkl / AMASS .npz files, or folders of them")
    p.add_argument("--pkl-dir", type=Path, default=REPO_ROOT / "motions" / "gmr_pkl")
    p.add_argument("--csv-dir", type=Path, default=REPO_ROOT / "motions" / "k1_csv")
    p.add_argument("--fps", type=int, default=30, help="Frame rate GMR resamples to (default 30)")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N clips")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not (BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz").exists():
        sys.exit(f"SMPL-X body models not found in {BODY_MODELS / 'smplx'} (see docs/setup.md).")

    sources = []
    for inp in args.inputs:
        sources += sorted([*inp.glob("*.pkl"), *inp.glob("*.npz")]) if inp.is_dir() else [inp]
    sources = sources[: args.limit]
    if not sources:
        sys.exit("No input clips found.")

    lower, upper = gmr_to_k1_csv.load_joint_limits(gmr_to_k1_csv.DEFAULT_LIMITS_XML)
    csv_args = argparse.Namespace(clip_frac_warn=0.05, clip_rad_warn=0.2)
    failed = []
    for i, src in enumerate(sources, 1):
        pkl_out = args.pkl_dir / f"{src.stem}.pkl"
        try:
            if pkl_out.exists() and not args.overwrite:
                print(f"({i}/{len(sources)}) {src.name}: already retargeted, converting only")
            else:
                t0 = time.time()
                n, fps, gap = retarget_clip(src, pkl_out, args.fps)
                print(f"({i}/{len(sources)}) {src.name}: {n} frames @ {fps:g} fps in {time.time() - t0:.0f}s "
                      f"(shifted {-gap:+.3f} m to the floor)")
            gmr_to_k1_csv.convert(pkl_out, args.csv_dir, lower, upper, csv_args)
        except Exception as e:  # keep going through a batch
            failed.append(src)
            print(f"[ERROR ] {src.name}: {type(e).__name__}: {e}")
    print(f"\nDone: {len(sources) - len(failed)} ok, {len(failed)} failed. CSVs in {args.csv_dir}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
