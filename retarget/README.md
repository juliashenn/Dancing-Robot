# retarget/

Turns human motion capture into K1 joint trajectories.

- **Input:** human mocap in `motions/raw/` (LAFAN1 `.bvh`, AMASS SMPL-X `.npz`).
- **Tool:** [GMR](https://github.com/YanjieZe/GMR) with robot `booster_k1` (22 joints: 2 neck, 4 per arm, 6 per leg).
- **Output:** `.csv` in BeyondMimic format in `motions/k1_csv/`: root position, root orientation, then joint positions, one row per frame.

Before training on our own clips, compare our CSV to Booster's example (`k1_mj2_seg1` in booster_assets): joint order, quaternion convention (wxyz vs xyzw) and frame rate must all match.
