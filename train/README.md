# train/

Our Isaac Lab tasks, based on booster_train's K1 dance task (`third_party/booster_train/.../beyond_mimic/robots/k1/mj_dance_002/`).

Each new dance is a copy of that task with `motion_file` pointed at our `.npz` in `motions/k1_npz/`. Use the no-state-estimation variant so the policy can run on the real robot.

First milestone: reproduce Booster's `Booster-K1-MJ_Dance_002-v0` from scratch and record how long training takes.

Training output (`logs/`) is gitignored. Copy exported policies we want to keep into `deploy/`.
