# Decision log

Record each significant choice with the date and the reason, so the poster and final report can explain them.

| Date | Decision | Why |
|---|---|---|
| 2026-11-08 | Train with Isaac Lab locally (RTX 4070 Ti Super) instead of mjlab on DeltaAI | No cluster queue or 4 h job cap; Booster's own booster_train and booster_deploy already cover a K1 dance from training to the robot |
| 2026-11-08 | Base our task on booster_train's `Booster-K1-MJ_Dance_002-v0` (no-state-estimation variant) | The real robot can't measure its own position or speed, so the policy must not depend on them |
| 2026-11-08 | Proposed: one policy for the whole routine (segments joined into one reference motion) | Switching between separate policies mid-dance is a common place for falls |
