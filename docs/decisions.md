# Decision log

Record each significant choice with the date and the reason, so the poster and final report can explain them.

| Date | Decision | Why |
|---|---|---|
| 2026-11-08 | Train with Isaac Lab locally (RTX 4070 Ti Super) instead of mjlab on DeltaAI | No cluster queue or 4 h job cap; Booster's own booster_train and booster_deploy already cover a K1 dance from training to the robot |
| 2026-11-08 | Base our task on booster_train's `Booster-K1-MJ_Dance_002-v0` (no-state-estimation variant) | The real robot can't measure its own position or speed, so the policy must not depend on them |
| 2026-11-08 | Proposed: one policy for the whole routine (segments joined into one reference motion) | Switching between separate policies mid-dance is a common place for falls |
| 2026-11-09 | Use AIST++ as the main dataset and screen every clip with a feasibility filter | AIST++ has 10 genres with music; the filter (joint limits, motor speed, tilt) keeps clips the K1 can physically do |
| 2026-11-10 | Reject clips spinning > 300°/s or traveling > 1 m | Ballet Jazz (~500°/s spins, 4.5 m travel) survived training but didn't spin and drifted 1.7 m in MuJoCo |
| 2026-11-11 | For spin-heavy clips: slow the clip down (2.5×) and train 15000–20000 iterations | Slowed Jazz at 20000 iterations did all ~13 turns (5.7° mean heading error); neither change alone was enough |
| 2026-11-24 | Chain dances by switching per-dance policies in a shared neutral stance (sequencer + routines.yaml), keep one-policy-per-routine as fallback | Any order of dances without retraining; a Lock → Pop switch test in MuJoCo completed with per-dance accuracy unchanged |
| 2026-11-24 | Robot tasks use motor-side ankle Kd = 2.13 | The ankles are a parallel linkage; booster_deploy sends Kd straight to the motors (booster_train README) |
