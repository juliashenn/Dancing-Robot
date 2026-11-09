# Dancing-Robot

CS498 Robotics, Team 9 (Booster K1). Alexander Lauinger, Julia Shen, Matt Pankow, Matthew Shang.

We teach the Booster K1 humanoid to perform a dance routine retargeted from public human motion-capture data (LAFAN1, AMASS, AIST++). Policies are trained in Isaac Lab with Booster's BeyondMimic port ([booster_train](https://github.com/BoosterRobotics/booster_train)), checked in MuJoCo, and run on the robot with [booster_deploy](https://github.com/BoosterRobotics/booster_deploy).

## Pipeline

```
Human mocap (BVH / SMPL-X)
  → retarget/      GMR (booster_k1) → K1 joint trajectory (.csv)
  → choreography/  filter clips, join segments, blend transitions, time to music
  → csv_to_npz     reference motion (.npz)
  → train/         Isaac Lab tracking policy (PPO) → exported .pt / .onnx
  → deploy/        booster_deploy: MuJoCo check, then the real K1
  → eval/          tracking metrics and plots
```

See [docs/pipeline.md](docs/pipeline.md) for the runtime flowchart of the robot.

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
