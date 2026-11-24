# choreography/

Building routines from retargeted clips. The pieces ended up next to the code they extend:

| Step | Where |
|---|---|
| Feasibility filter (joint limits, motor speed, trunk tilt, spins, travel) | `retarget/gmr_to_k1_csv.py`, summary in `eval/aistpp_feasibility.py` |
| Pick clips that are easy to train and to start/end in neutral | `eval/rank_clips_for_sequencing.py` → `eval/results/sequencing_candidates.csv` |
| Blend a clip in from / out to the neutral stance, or chain several clips into one motion | `retarget/make_routine.py` |
| Build a routine from trained dances (policies switch in the neutral stance) | `deploy/k1_dance_deploy/routines.yaml` |

Not done: timing segment boundaries to music.

See [docs/pipeline.md](../docs/pipeline.md#chaining-dances-into-a-routine) for how chaining works.
