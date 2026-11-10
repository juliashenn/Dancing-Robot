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
| Ours, AIST++ Pop (gPO_sBM_cAll_d10_mPO0_ch02), iteration 2500 | 12.0 / 12.0 s | 0.104 rad | 0.17 m |
| Ours, AIST++ Pop, iteration 12750 (`k1_dance_pop1`) | 12.0 / 12.0 s | 0.092 rad | 0.19 m |
| Ours, AIST++ Ballet Jazz (gJB_sBM_cAll_d08_mJB0_ch01), iteration 4999 (`k1_dance_jazz1`) | 12.0 / 12.0 s | 0.111 rad | **1.70 m** |

Pop training in Isaac Lab plateaued after ~5000 iterations (body error ~2.5–3 cm, <0.5% failures); in MuJoCo, 12750 is only slightly better than 2500, so ~3000–5000 iterations is enough per clip.

**Jazz1 lesson:** the clip is a sequence of traveling turns (~500 deg/s for 12 s, 4.5 m of travel). The policy survived but kept facing forward and didn't follow the spins (training: 7.6 cm body error, 2% failures at 5000 iterations vs Pop's 2.8 cm / 0.3%). Fast continuous spins (~1.4 turns/s on small flat feet) are physically hard and weren't learned in 5000 iterations, and traveling steps add drift because the policy doesn't observe its position in the room (it does observe its orientation). The feasibility filter now rejects clips turning faster than 300 deg/s (95th percentile) or traveling over 1 m; 35 of 41 Ballet Jazz clips fail it.
