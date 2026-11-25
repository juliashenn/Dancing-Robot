"""Run a dance policy in MuJoCo with booster_deploy and record an mp4 (sim-to-sim check).

Uses booster_deploy's own MuJoCo controller (same observations, PD gains, action scaling and
physics loop it uses before going to the real robot), but renders offscreen to a video instead
of opening a viewer. The translucent green robot is the reference motion; the solid robot is
the policy under MuJoCo physics.

Run with booster_deploy's Python (see docs/setup.md):
    third_party/booster_deploy/.venv/Scripts/python deploy/record_mujoco.py --task k1_mj2
    third_party/booster_deploy/.venv/Scripts/python deploy/record_mujoco.py --task k1_mj2 \
        --checkpoint path/to/exported/policy.pt --motion path/to/motion.npz --out videos/ours.mp4

Writes <out>.mp4 and <out>.json (duration completed, whether the safety stop fired, tracking errors).
"""

import argparse
import json
import os
import pkgutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOSTER_DEPLOY = REPO_ROOT / "third_party" / "booster_deploy"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--task", default="k1_mj2", help="booster_deploy task to base the run on (default k1_mj2)")
p.add_argument("--checkpoint", type=Path, help="Policy file (.pt TorchScript or .onnx) to use instead of the task's")
p.add_argument("--motion", type=Path, help="Reference motion .npz to use instead of the task's")
p.add_argument("--out", type=Path, help="Output path without extension (default videos/<task>_mujoco)")
p.add_argument("--max-seconds", type=float, default=None,
               help="Stop after this long even if the motion continues (default: motion length + 2 s)")
p.add_argument("--width", type=int, default=960)
p.add_argument("--height", type=int, default=720)
p.add_argument("--no-ghost", action="store_true", help="Don't draw the reference robot")
p.add_argument("--no-video", action="store_true", help="Only compute the metrics (JSON); much faster")
p.add_argument("--start-stance", choices=["default", "walk"], default="default",
               help="Initial pose: booster_deploy's default (= our neutral stance) or the K1 walking policy's "
                    "stance, which is what the real robot is in when the dance starts (prepare_mode='walking')")
args = p.parse_args()
# Resolve user paths before changing directory.
args.checkpoint = args.checkpoint.resolve() if args.checkpoint else None
args.motion = args.motion.resolve() if args.motion else None
if args.out and not args.out.is_absolute():
    args.out = Path.cwd() / args.out

# booster_deploy expects to run from its own folder (task paths, `import tasks`).
os.chdir(BOOSTER_DEPLOY)
sys.path.insert(0, str(BOOSTER_DEPLOY))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import tasks as tasks_pkg  # noqa: E402
for mod in pkgutil.walk_packages(tasks_pkg.__path__, prefix="tasks."):
    __import__(mod.name)
sys.path.insert(0, str(REPO_ROOT / "deploy"))
import k1_dance_deploy  # noqa: E402,F401  (registers our k1_dance_* tasks)
from booster_deploy.controllers.mujoco_controller import MujocoController  # noqa: E402
from booster_deploy.utils.registry import get_task  # noqa: E402


def yaw_of(q):
    """Yaw (rad) of a MuJoCo quaternion (w, x, y, z)."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def per_dance_metrics(switch_frames, frames, fps, dances, joint_err, xy_err, heading_err, handoff_s=1.0):
    """Metrics for each dance of a sequence, plus the first `handoff_s` seconds after each policy took over
    (where a bad handoff would show up). Drift is measured from where that dance started."""
    bounds = list(switch_frames) + [frames]
    result = []
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        if end <= start:
            continue
        h = slice(start, min(end, start + int(handoff_s * fps)))
        result.append({
            "dance": Path(dances[i][1]).stem,
            "start_s": round(start / fps, 2), "end_s": round(end / fps, 2),
            "mean_joint_error_rad": round(float(joint_err[start:end].mean()), 3),
            "max_root_xy_drift_m": round(float(xy_err[start:end].max()), 3),
            "mean_heading_error_deg": round(float(heading_err[start:end].mean()), 1),
            "handoff_joint_error_rad": round(float(joint_err[h].mean()), 3),
            "handoff_max_heading_error_deg": round(float(heading_err[h].max()), 1),
        })
    return result


# K1 walking policy's standing pose (booster_deploy tasks/locomotion/robots/k1, real joint order): on the robot,
# the walking policy holds this stance until A is pressed and the dance policy takes over.
WALK_STANCE = [0, 0, 0.2, -1.25, 0, -0.5, 0.2, 1.25, 0, 0.5,
               -0.15, 0, 0, 0.3, -0.15, 0, -0.15, 0, 0, 0.3, -0.15, 0]


def set_stance(model, data, joints):
    """Put the robot in `joints` with its lowest geom resting on the floor plane."""
    data.qpos[7:] = joints
    mujoco.mj_forward(model, data)
    planes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    fromto = np.zeros(6)
    gap = min(mujoco.mj_geomDistance(model, data, planes[0], g, 1.0, fromto)
              for g in range(model.ngeom) if g not in planes and model.geom_bodyid[g] != 0)
    data.qpos[2] -= gap
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def task_dir(cfg):
    module = sys.modules[type(cfg).__module__]
    return Path(module.__file__).resolve().parent


def main():
    cfg = get_task(args.task)
    cfg.policy.device = "cpu"
    cfg.policy.stop_at_motion_end = True
    if args.checkpoint:
        cfg.policy.checkpoint_path = str(args.checkpoint.resolve())
    if args.motion:
        # booster_deploy joins motion_path onto the task folder, so pass it relative to that folder.
        cfg.policy.motion_path = os.path.relpath(args.motion.resolve(), task_dir(cfg))

    out = args.out or REPO_ROOT / "videos" / f"{args.task}_mujoco"
    out = out if out.is_absolute() else (REPO_ROOT / out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ctrl = MujocoController(cfg)
    model, data, ghost = ctrl.mj_model, ctrl.mj_data, ctrl._ghost_mj_data
    if args.start_stance == "walk":
        set_stance(model, data, WALK_STANCE)
    fps = round(1.0 / cfg.policy_dt)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.2, -15.0, 135.0
    ghost_opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    ghost_rgba = np.array(cfg.mujoco.ghost_rgba, dtype=np.float32)

    joint_err, root_xy_err, heights, frames = [], [], [], 0
    yaw, ref_yaw = [], []
    max_steps = (int(args.max_seconds * fps) if args.max_seconds
                 else int(ctrl.policy.motion.time_step_total) + 2 * fps)
    writer = None if args.no_video else imageio.get_writer(
        f"{out}.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=8)
    t0 = time.time()
    ctrl.update_state()
    ctrl.start()
    while ctrl.is_running and frames < max_steps:
        ctrl.update_state()
        dof_targets = ctrl.policy_step()
        ctrl.ctrl_step(dof_targets)
        frames += 1

        if writer:
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, cam)
            if not args.no_ghost and ctrl._reference_qpos is not None:
                n0 = renderer.scene.ngeom
                mujoco.mjv_addGeoms(model, ghost, ghost_opt, pert, int(mujoco.mjtCatBit.mjCAT_DYNAMIC), renderer.scene)
                for i in range(n0, renderer.scene.ngeom):
                    renderer.scene.geoms[i].rgba[:] = ghost_rgba
            writer.append_data(renderer.render())

        if ctrl._reference_qpos is not None:
            ref = ctrl._reference_qpos
            joint_err.append(float(np.mean(np.abs(data.qpos[7:] - ref[7:]))))
            root_xy_err.append(float(np.linalg.norm(data.qpos[:2] - ref[:2])))
            yaw.append(yaw_of(data.qpos[3:7]))
            ref_yaw.append(yaw_of(ref[3:7]))
        heights.append(float(data.qpos[2]))
    if writer:
        writer.close()
    # Heading (yaw) tracking, i.e. "does it do the spins": error per frame and total degrees turned.
    yaw, ref_yaw = np.unwrap(yaw), np.unwrap(ref_yaw)
    heading_err = np.degrees(np.abs((yaw - ref_yaw + np.pi) % (2 * np.pi) - np.pi))

    motion_frames = int(ctrl.policy.motion.time_step_total)
    completed = frames >= motion_frames
    summary = {
        "task": args.task,
        "start_stance": args.start_stance,
        "checkpoint": cfg.policy.checkpoint_path,
        # Sequences (k1_dance_deploy/sequence.py): when each dance's policy took over.
        **({"dance_start_times_s": [round(f / fps, 2) for f in ctrl.policy.switch_frames]}
           if hasattr(ctrl.policy, "switch_frames") else {}),
        "motion": getattr(cfg.policy, "motion_path", None) or [m for _, m in cfg.policy.dances],
        "motion_seconds": round(motion_frames / fps, 2),
        "ran_seconds": round(frames / fps, 2),
        "completed_motion": completed,
        "safety_stop": (not completed) and frames < max_steps,
        "mean_joint_error_rad": round(float(np.mean(joint_err)), 3) if joint_err else None,
        "max_root_xy_drift_m": round(float(np.max(root_xy_err)), 3) if root_xy_err else None,
        "mean_heading_error_deg": round(float(np.mean(heading_err)), 1) if len(heading_err) else None,
        "max_heading_error_deg": round(float(np.max(heading_err)), 1) if len(heading_err) else None,
        "robot_total_turn_deg": round(float(np.degrees(np.sum(np.abs(np.diff(yaw))))))  if len(yaw) > 1 else None,
        "reference_total_turn_deg": round(float(np.degrees(np.sum(np.abs(np.diff(ref_yaw))))))  if len(ref_yaw) > 1 else None,
        "min_trunk_height_m": round(float(np.min(heights)), 3),
        "wall_seconds": round(time.time() - t0, 1),
    }
    if hasattr(ctrl.policy, "switch_frames") and len(joint_err) == frames:
        summary["per_dance"] = per_dance_metrics(ctrl.policy.switch_frames, frames, fps, cfg.policy.dances,
                                                 np.array(joint_err), np.array(root_xy_err), heading_err)
    Path(f"{out}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if writer:
        print(f"video: {out}.mp4")


if __name__ == "__main__":
    main()
