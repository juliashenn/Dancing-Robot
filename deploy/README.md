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

**Real robot ankle damping:** the K1 ankles are a parallel linkage, and booster_deploy sends `joint_damping` straight to the motors, so the simulation ankle Kd (2.98 pitch / 0.85 roll, which include booster_train's armature ratio) must not be used on hardware. Every task has a `<task>_robot` version with the motor-side value Kd = 2·ζ·J_motor·2πf = 2·1.5·0.0282528·2π·4 = **2.13** on all four ankle joints (Booster's own k1_mj2 config uses 2.0). In MuJoCo, `k1_dance_lock1_robot` performs the same as `k1_dance_lock1` (0.079 vs 0.081 rad, 0.09 m drift).

## Running on the real K1

Checklist for the session with the robot (from booster_deploy's README plus our setup):

1. **Copy our task into booster_deploy on the robot:** `deploy/k1_dance_deploy/` → `booster_deploy/tasks/k1_dance/` (models and motions are inside it). booster_deploy imports every package under `tasks/`, so nothing else needs registering. Check with `python scripts/deploy.py --list`: the `k1_dance_*` tasks should be listed.
2. **Environment on the robot:** activate booster_deploy's `.venv` (needs torch, numpy, scipy, pyyaml; see its requirements.txt), then `source /opt/booster/BoosterRos2Interface/install/setup.bash`. Firmware must be ≥ v1.7.2.
3. **Run a `_robot` task, never the plain one:** `python scripts/deploy.py --task k1_dance_lock1_robot`. Start with Lock (most accurate, least drift), then Pop; leave Hip-hop and slowed Jazz (more travel/turning) for later. For routines, first run each dance alone with its `k1_dance_lib_<dance>_robot` task (Street Jazz drifts only ~2 cm, so it's the safest first neutral-ended dance), then the routine.
4. **Start-up:** press `X` (remote) to enter Custom mode; the K1's default `prepare_mode="walking"` holds the robot standing with Booster's walking policy. Press `A` (or `r` on the keyboard) to start the dance. The dance then starts from the walking policy's stance (knees bent 0.3 rad), not our straight-legged neutral; `record_mujoco.py --start-stance walk` tests exactly that, and Lock, Pop and Hip-hop all complete from it with unchanged accuracy (0.080 / 0.091 / 0.133 rad). When the motion ends the controller exits to walking mode (default `exit_mode`), so the robot keeps balancing.
5. **Safety:** someone on the gantry/harness, 1–2 m of clear floor (Lock drifts ~0.1 m in sim, Hip-hop ~0.6 m, slowed Jazz ~0.8 m). The policy's safety fallback stops it if the trunk orientation differs a lot from the reference.
6. **Routines:** `k1_routine_<name>_robot` for each routine in `k1_dance_deploy/routines.yaml` (dances switch policies in the neutral stance). Edit that file to build a new routine; it appears once all its dances have trained models. Run its MuJoCo test first.

## Results so far

| Policy | Completed | Mean joint error | Max drift |
|---|---|---|---|
| Booster's shipped MJ model (`k1_mj2`) | 25.2 / 25.2 s | 0.088 rad | 0.30 m |
| Ours, MJ dance, iteration 3000 (`k1_dance_mj2_ours`) | 25.2 / 25.2 s | 0.126 rad | 0.53 m |
| Ours, AIST++ Pop (gPO_sBM_cAll_d10_mPO0_ch02), iteration 2500 | 12.0 / 12.0 s | 0.104 rad | 0.17 m |
| Ours, AIST++ Pop, iteration 12750 (`k1_dance_pop1`) | 12.0 / 12.0 s | 0.092 rad | 0.19 m |
| Ours, AIST++ Ballet Jazz (gJB_sBM_cAll_d08_mJB0_ch01), iteration 4999 (`k1_dance_jazz1`) | 12.0 / 12.0 s | 0.111 rad | **1.70 m** |
| Ours, AIST++ Lock (gLO_sBM_cAll_d13_mLO0_ch01), iteration 4999 (`k1_dance_lock1`) | 12.0 / 12.0 s | **0.081 rad** | 0.09 m |
| Ours, Ballet Jazz played 2.5× slower (30 s), iteration 9999 | 29.9 / 29.9 s | 0.107 rad | 0.34 m |
| Ours, Ballet Jazz 2.5× slower, iteration 19999 (`k1_dance_jazz1slow`) | 29.9 / 29.9 s | 0.106 rad | 0.78 m |
| Ours, AIST++ LA Hip-hop (gLH_sBM_cAll_d16_mLH0_ch01), iteration 9999 (`k1_dance_hiphop1`) | 12.0 / 12.0 s | 0.132 rad | 0.63 m |

Spin tracking for the Ballet Jazz clip (it turns ~4700°, about 13 full turns; `record_mujoco.py` reports heading error and total turn):

| Policy | Mean heading error | Max heading error | Robot turned |
|---|---|---|---|
| Original speed, iteration 4999 | 79.5° | 180° | 428° of 4699° |
| 2.5× slower, iteration 9999 | 23.5° | 180° | 3342° of 4699° |
| 2.5× slower, iteration 19999 | **5.7°** | **31°** | **4725° of 4699°** |

Pop training in Isaac Lab plateaued after ~5000 iterations (body error ~2.5–3 cm, <0.5% failures); in MuJoCo, 12750 is only slightly better than 2500, so ~3000–5000 iterations is enough per clip.

**Jazz1 lesson:** the clip is a sequence of traveling turns (~500 deg/s for 12 s, 4.5 m of travel). The policy survived but kept facing forward and didn't follow the spins (training: 7.6 cm body error, 2% failures at 5000 iterations vs Pop's 2.8 cm / 0.3%). Fast continuous spins (~1.4 turns/s on small flat feet) are physically hard and weren't learned in 5000 iterations, and traveling steps add drift because the policy doesn't observe its position in the room (it does observe its orientation). The feasibility filter now rejects clips turning faster than 300 deg/s (95th percentile) or traveling over 1 m; 35 of 41 Ballet Jazz clips fail it.

**Slowing Jazz1 down fixed the spins:** the same clip resampled 2.5× slower (`csv_to_npz.py --input_fps 12`, ~270 deg/s turns) and trained for 10000 iterations follows the turns. Training orientation error fell from 1.40 rad to 0.35 rad and angular-velocity error from 6.0 to 1.5 rad/s; in MuJoCo it completes all 30 s with 0.34 m max drift instead of 1.70 m. Both changes mattered: at 5000 iterations the slowed clip's orientation error was ~0.9 rad (the original's was 1.38 and only creeping down), and the big drop to ~0.3 rad came between iterations 8000 and 8500. Continuing to 20000 iterations finished the job: training orientation error fell to 0.13 rad (flat after ~16000), and in MuJoCo the robot does every turn (mean heading error 5.7°). Max drift rose to 0.78 m, since the robot now actually travels through the turns while not observing its position. So "3000–5000 iterations per clip" holds for easy clips like Pop and Lock; clips with fast turns need slowing down and ~15000–20000.

## Dance library and routines

Each library dance is a clip that starts and ends in the neutral stance (`retarget/make_routine.py`), with its own policy. Street Jazz, Krump and Pop 2 were picked by `eval/rank_clips_for_sequencing.py` and trained on their neutral-ended clips. For Lock, Pop and Hip-hop we fine-tuned the original policies on the neutral-ended clip (+2000 iterations). That turned out to hurt Lock and Pop, so the library uses their original policies (see the checkpoint sweep below). Every model was picked by MuJoCo joint error. Full table, from both start stances: [eval/results/mujoco_results.md](../eval/results/mujoco_results.md). The numbers below are from the walk stance, which is what the real robot starts from.

| Task | Model | Length | Completed | Mean joint error | Max drift | Worst handoff (first 1 s of a dance) |
|---|---|---|---|---|---|---|
| `k1_dance_lib_lock` | lock1_it4999 (original) | 13.0 s | yes | **0.083 rad** | 0.11 m | |
| `k1_dance_lib_pop` | pop1_it12750 (original) | 13.0 s | yes | 0.095 rad | 0.22 m | |
| `k1_dance_lib_hiphop` | hiphop1n_it11500 | 13.0 s | yes | 0.132 rad | 0.45 m | |
| `k1_dance_lib_sjazz` (Street Jazz, gJS_sBM_cAll_d01_mJS0_ch01) | sjazz1n_it6000 | 13.0 s | yes | 0.134 rad | **0.02 m** | |
| `k1_dance_lib_krump` (gKR_sBM_cAll_d29_mKR0_ch02) | krump1n_it9000 | 13.0 s | yes | 0.101 rad | 0.09 m | |
| `k1_dance_lib_pop2` (gPO_sBM_cAll_d10_mPO1_ch05) | pop2n_it4999 | 11.7 s | yes | 0.110 rad | 0.07 m | |
| `k1_dance_lib_pop3` (gPO_sBM_cAll_d10_mPO1_ch04) | pop3n_it4000 | 11.7 s | yes | 0.093 rad | 0.06 m | |
| `k1_dance_lib_sjazz2` (gJS_sBM_cAll_d03_mJS3_ch02) | sjazz2n_it3500 | 9.7 s | yes | 0.123 rad | 0.14 m | |
| `k1_dance_lib_lock2` (gLO_sBM_cAll_d13_mLO2_ch02) | lock2n_it4500 | 10.6 s | yes | 0.107 rad | 0.07 m | |
| `k1_dance_lib_house` (gHO_sBM_cAll_d20_mHO5_ch04) | house1n_it5000 | 8.1 s | yes | 0.123 rad | 0.13 m | |
| `k1_routine_pop_lock_hiphop` (Pop, Lock, Hip-hop) | | 38.9 s | yes | **0.104 rad** | 0.28 m | 0.134 rad |
| `k1_routine_easy_three` (Pop 2, Street Jazz, Krump) | | 37.6 s | yes | 0.115 rad | 0.09 m | 0.132 rad |
| `k1_routine_everything` (Pop 2, Street Jazz, Krump, Pop, Lock, Hip-hop) | | 76.6 s | yes | 0.109 rad | 0.33 m | 0.134 rad |
| `k1_routine_new_four` (Street Jazz 2, House, Pop 3, Lock 2) | | 40.0 s | yes | 0.110 rad | 0.14 m | 0.116 rad |
| `k1_routine_all_ten` (Pop 2, Lock 2, Street Jazz, Krump, Hip-hop, House, Pop 3, Pop, Street Jazz 2, Lock) | | 116.6 s | yes | 0.110 rad | 0.44 m | 0.133 rad |

The second batch (Pop 3, Street Jazz 2, Lock 2, House) is the next-easiest clips in the ranking, each with a choreography the library didn't have yet (the `ch` number in the clip name). Clips of the same choreography by another dancer look like repeats. Each trained in about 4 h; House got 8000 iterations, but its best checkpoint was at 5000. Figure for the 10-dance routine: [eval/results/routine_all_ten.png](../eval/results/routine_all_ten.png).

**Switching policies in neutral works.** Every routine completes from both start stances, and each dance inside a routine tracks as well as it does alone (e.g. Lock 0.084 vs 0.083 rad). The first second after a switch is worse (0.09–0.15 rad here) while the new policy takes over, but nothing falls.

**Picking checkpoints and orders improved the 6-dance routine a lot.** Compared with the first version (last checkpoint of every run, order picked by hand), mean error went from 0.118 to 0.109 rad, max drift from 0.63 to 0.33 m, and worst handoff from 0.185 to 0.134 rad.

**Order matters.** `eval/handoff_matrix.py` runs every ordered pair of dances (90 two-dance runs for 10 dances, about 1 h) and finds the best order of each routine: the smallest worst handoff, ties broken by the total. It is exact for any length (a subset search above 8 dances, since a 10-dance routine has 3.6 million orders). Some pairs hand off much worse than others. Into Krump, most dances score 0.17–0.23 rad, but Street Jazz and Pop 3 score 0.09. The matrix is a guide, not an exact prediction, because how one dance ends affects the next. For example, from the default stance Pop 2 ends a little differently and the Street Jazz → Krump handoff becomes 0.20. Results: [eval/results/handoff_matrix.csv](../eval/results/handoff_matrix.csv).

**Pick checkpoints by MuJoCo result, not by training length** ([eval/results/checkpoint_sweep.csv](../eval/results/checkpoint_sweep.csv)). Joint error in MuJoCo on the neutral-ended clip, default / walk stance:

| Dance | Original policy (never saw neutral ends) | Neutral-trained checkpoints | Picked |
|---|---|---|---|
| Lock | **0.083 / 0.083** (4999) | 0.094–0.118 (5250–6999) | original |
| Pop | **0.095 / 0.095** (12750) | 0.103–0.119 (13250–15000) | original |
| Hip-hop | 0.135 / 0.134 (9999), drift 1.3–1.6 m | 0.130–0.138 (10250–11999); 11500: 0.131 / 0.132, drift 0.4 m | 11500 |
| Street Jazz | | 4999: 0.145 / 0.143 → 6000: **0.146 / 0.134** → 9999: 0.181 / 0.145 | 6000 |
| Krump | | 4999: 0.140 / 0.113 → 9000: **0.109 / 0.101** → 9999: 0.117 / 0.106 | 9000 |
| Pop 3 | | 2500: 0.133 / 0.117 → 4000: **0.102 / 0.093** → 4999: 0.106 / 0.101 | 4000 |
| Street Jazz 2 | | 2500: 0.145 / 0.134 → 3500: **0.126 / 0.123** → 4999: 0.129 / 0.125 | 3500 |
| Lock 2 | | 2500: 0.135 / 0.123 → 4500: **0.110 / 0.107** → 4999: 0.118 / 0.111 | 4500 |
| House | | 4000: 0.137 / 0.134 → 5000: **0.127 / 0.123** → 7999: 0.128 / 0.127 | 5000 |

- **The original Lock and Pop policies already handle the 0.5 s neutral blends.** Fine-tuning them on the neutral-ended clip made them worse, and they take over from neutral as cleanly as the fine-tuned ones (0.11–0.14 rad handoffs). For Hip-hop the fine-tuning helped, mainly by cutting drift.
- **More training can make a policy worse in MuJoCo.** Street Jazz's Isaac Lab error kept falling from 5000 to 10000 iterations, but in MuJoCo it got worse. The likely reason is that the extra training fit Isaac Lab's physics more closely without carrying over. Checkpoints also zig-zag by about 0.01–0.02 rad.
- **Lesson:** export several checkpoints (every 500–1000 iterations) and keep the best one in MuJoCo, judged from the walk stance.

**First hardware test:** `k1_routine_easy_three_robot`. It drifts the least (≤ 9 cm) and has smooth handoffs. Before it, run each `k1_dance_lib_<dance>_robot` on its own. `k1_routine_pop_lock_hiphop_robot` is the most accurate routine (0.104 rad) but drifts about 0.3 m.
