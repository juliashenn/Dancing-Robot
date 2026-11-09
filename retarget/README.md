# retarget/

Turns human motion capture into K1 joint trajectories.

- **Input:** human mocap in `motions/raw/`.
- **Tool:** [GMR](https://github.com/YanjieZe/GMR) with robot `booster_k1` (22 joints: 2 neck, 4 per arm, 6 per leg).
- **Output:** `.csv` in the format below, in `motions/k1_csv/`.

## Scripts

| Script | Python | What it does |
|---|---|---|
| `retarget_to_k1.py` | GMR's (`third_party/GMR/.venv`) | AIST++ `.pkl` or AMASS `.npz` → GMR retarget onto the K1 → sets the lowest collision geom on the floor → `motions/gmr_pkl/<clip>.pkl`, then runs the converter below |
| `gmr_to_k1_csv.py` | any with numpy | GMR `.pkl` → Booster CSV + `<clip>.json` (fps, duration, clipping report, max joint speeds) |

```bash
third_party/GMR/.venv/Scripts/python retarget/retarget_to_k1.py motions/raw/aistpp/gBR_sBM_cAll_d04_mBR0_ch01.pkl
```

Each clip gets a verdict: **OK**, or **REVIEW** if any joint is clipped in more than 5% of frames or by more than 0.2 rad. REVIEW clips ask for poses the K1 can't reach; check them in the viewer before spending GPU time on them.

## Which input formats GMR supports for the K1

GMR (pinned commit) only ships a K1 IK config for **SMPL-X** input (`smplx_to_k1.json`).

| Dataset | Format | K1 support in GMR |
|---|---|---|
| AMASS | SMPL-X | Yes, directly |
| AIST++ | SMPL | Yes: `retarget_to_k1.py` converts the AIST++ `.pkl` (y-up, 60 fps) to the SMPL-X layout first |
| LAFAN1 | BVH | **No K1 config.** Needs a new `bvh_lafan1_to_k1.json` (adapt `bvh_lafan1_to_t1_29dof.json` with the K1 body names from `smplx_to_k1.json`) |

## Target CSV format (what booster_train's `csv_to_npz.py` expects)

Matches Booster's example `third_party/booster_assets/motions/K1/k1_mj2_seg1.csv`:

- Header row (optional), then one row per frame, comma-separated, 29 columns.
- Columns 1–3: root position x, y, z (meters, world frame).
- Columns 4–7: root orientation quaternion in **x, y, z, w** order.
- Columns 8–29: joint angles (radians) in this order:

```
aahead_yaw_joint, aahead_pitch_joint,
left_shoulder_pitch_joint, left_shoulder_roll_joint, left_elbow_pitch_joint, left_elbow_yaw_joint,
right_shoulder_pitch_joint, right_shoulder_roll_joint, right_elbow_pitch_joint, right_elbow_yaw_joint,
left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint, left_knee_pitch_joint, left_ankle_pitch_joint, left_ankle_roll_joint,
right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint, right_knee_pitch_joint, right_ankle_pitch_joint, right_ankle_roll_joint
```

Frame rate is passed separately to `csv_to_npz.py` (`--input_fps`); it resamples to 50 Hz.

## GMR output vs. that format

- GMR's `.pkl` has `root_pos`, `root_rot` (already x, y, z, w), `dof_pos` and `fps`.
- GMR's K1 model (`GMR/assets/booster_k1/K1_serial.xml`) lists the 22 joints in the **same order** as above, under different names (`AAHead_yaw`, `ALeft_Shoulder_Pitch`, ...). Joint axes match on all 22.
- **Joint limits differ.** GMR's model is looser than Booster's `K1_22dof.xml` (e.g. shoulder pitch −3.316 vs −2.932 rad; elbow yaw ranges differ). The converter must **clip each joint to Booster's limits** and report how often clipping happened. Heavy clipping means the clip is a poor fit for the K1.
