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
p.add_argument("--max-seconds", type=float, default=60.0, help="Stop after this long even if the motion continues")
p.add_argument("--width", type=int, default=960)
p.add_argument("--height", type=int, default=720)
p.add_argument("--no-ghost", action="store_true", help="Don't draw the reference robot")
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
    fps = round(1.0 / cfg.policy_dt)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.2, -15.0, 135.0
    ghost_opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    ghost_rgba = np.array(cfg.mujoco.ghost_rgba, dtype=np.float32)

    joint_err, root_xy_err, heights, frames = [], [], [], 0
    max_steps = int(args.max_seconds * fps)
    writer = imageio.get_writer(f"{out}.mp4", fps=fps, codec="libx264", quality=8, macro_block_size=8)
    t0 = time.time()
    ctrl.update_state()
    ctrl.start()
    while ctrl.is_running and frames < max_steps:
        ctrl.update_state()
        dof_targets = ctrl.policy_step()
        ctrl.ctrl_step(dof_targets)
        frames += 1

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
        heights.append(float(data.qpos[2]))
    writer.close()

    motion_frames = int(ctrl.policy.motion.time_step_total)
    completed = frames >= motion_frames
    summary = {
        "task": args.task,
        "checkpoint": cfg.policy.checkpoint_path,
        "motion": cfg.policy.motion_path,
        "motion_seconds": round(motion_frames / fps, 2),
        "ran_seconds": round(frames / fps, 2),
        "completed_motion": completed,
        "safety_stop": (not completed) and frames < max_steps,
        "mean_joint_error_rad": round(float(np.mean(joint_err)), 3) if joint_err else None,
        "max_root_xy_drift_m": round(float(np.max(root_xy_err)), 3) if root_xy_err else None,
        "min_trunk_height_m": round(float(np.min(heights)), 3),
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(f"{out}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"video: {out}.mp4")


if __name__ == "__main__":
    main()
