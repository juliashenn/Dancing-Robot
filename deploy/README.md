# deploy/

Running our dance policies with [booster_deploy](https://github.com/BoosterRobotics/booster_deploy): in MuJoCo first (sim-to-sim check), then on the real K1.

| Path | What it is |
|---|---|
| `k1_dance_deploy/` | booster_deploy task package for our policies (copy to `booster_deploy/tasks/k1_dance/` on the robot). Holds the training-matched motor gains, our exported models (`models/`) and reference motions (`motions/`). |
| `record_mujoco.py` | Runs a booster_deploy task in MuJoCo and records an mp4 + JSON summary (completed? safety stop? tracking error, drift). |

## Setup (laptop)

```bash
cd third_party/booster_deploy
uv venv --python 3.11
uv pip install numpy mujoco scipy pyyaml onnxruntime "imageio[ffmpeg]" torch --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install -e ../booster_assets
```
(`evdev` from booster_deploy's requirements is Linux-only and only needed on the robot.)

## Record a policy in MuJoCo

```bash
third_party/booster_deploy/.venv/Scripts/python deploy/record_mujoco.py --task k1_dance_mj2_ours --out videos/mj2_ours
```
Grey robot = policy under MuJoCo physics; translucent green = reference motion.

## Important: motor gains

Booster's `k1_mj2` task uses gains from an older booster_train K1 config (leg Kp 80/30, action scale 0.25·effort/Kp). Our policies are trained with the current booster_train config, so `K1DanceControllerCfg` sets the training-time Kp, Kd, effort limits and per-joint action scale. With `k1_mj2`'s values our MJ policy fell after 2.3 s; with ours it completes the 25 s dance.

**Before the real robot:** the K1 ankles are a parallel linkage; per booster_train's README, recompute the ankle Kd motor-side instead of sending the simulation value.

## Results so far

| Policy | Completed | Mean joint error | Max drift |
|---|---|---|---|
| Booster's shipped MJ model (`k1_mj2`) | 25.2 / 25.2 s | 0.088 rad | 0.30 m |
| Ours, MJ dance, iteration 3000 (`k1_dance_mj2_ours`) | 25.2 / 25.2 s | 0.126 rad | 0.53 m |
