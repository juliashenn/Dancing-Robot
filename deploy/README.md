# deploy/

Our booster_deploy task folder(s), modeled on Booster's K1 dance tasks (`third_party/booster_deploy/tasks/beyond_mimic/robots/k1/`, tasks `k1_mj2` and `k1_fight`).

Each task needs:
- the exported policy (`models/*.pt` or `.onnx`)
- the matching reference motion (`motions/*.npz`)
- a controller config registered with booster_deploy

Workflow:
1. Run in MuJoCo first: `python scripts/deploy.py --task <our_task> --mujoco`
2. Only policies that dance cleanly in MuJoCo go to the robot.
3. On the robot: gantry first, course staff present.
