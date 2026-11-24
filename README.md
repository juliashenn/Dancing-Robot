# Dancing-Robot

CS498 Robotics, Team 9 (Booster K1). Alexander Lauinger, Julia Shen, Matt Pankow, Matthew Shang.

We teach the Booster K1 humanoid to perform a dance routine retargeted from public human motion-capture data (LAFAN1, AMASS, AIST++). Policies are trained in Isaac Lab with Booster's BeyondMimic port ([booster_train](https://github.com/BoosterRobotics/booster_train)), checked in MuJoCo, and run on the robot with [booster_deploy](https://github.com/BoosterRobotics/booster_deploy).

## Pipeline

```
AIST++ human dance (SMPL)
  → retarget/      GMR (booster_k1) → K1 joint trajectory (.csv) + feasibility check
  → retarget/make_routine.py   start/end each dance in a neutral stance
  → csv_to_npz     reference motion (.npz)
  → train/         Isaac Lab tracking policy (PPO), one per dance → exported .pt / .onnx
  → deploy/        booster_deploy: MuJoCo check, then the real K1;
                   routines.yaml chains dances by switching policies in the neutral stance
  → eval/          clip ranking, MuJoCo results table
```

See [docs/pipeline.md](docs/pipeline.md) for the full pipeline, how dances are chained, and the runtime flowchart of the robot. Results so far: [deploy/README.md](deploy/README.md#results-so-far) and [eval/results/mujoco_results.md](eval/results/mujoco_results.md).

## Repo layout

```
docs/           setup guide, pipeline notes, decision log
setup/          scripts that fetch pinned external repos into third_party/
third_party/    (gitignored) booster_train, booster_deploy, booster_assets, GMR
retarget/       mocap → K1 joint trajectories (GMR wrappers, format conversion)
motions/
  raw/          (gitignored) downloaded human mocap
  k1_csv/       retargeted K1 clips we keep
  k1_npz/       reference motions used for training
choreography/   segment joining, transitions, music timing
train/          our Isaac Lab tasks (based on booster_train's K1 MJ_Dance_002)
deploy/         our booster_deploy task folder(s) and exported models
eval/           metrics, Isaac vs. MuJoCo comparison, figures
```

## Quickstart

1. Install Isaac Sim 5.1 and Isaac Lab 2.3.2 (see [docs/setup.md](docs/setup.md)).
2. Fetch the external repos at their pinned commits:
   ```bash
   python setup/clone_third_party.py
   ```
3. Follow the per-folder READMEs for each pipeline stage.
