# Pipeline

## Offline: from human mocap to a policy

```
AIST++ human dance (SMPL, 60 fps)
  │  retarget/retarget_to_k1.py: GMR (booster_k1), ground the clip, 30 fps
  ▼
K1 joint trajectory (.csv, BeyondMimic format) + feasibility report (.json)
  │  retarget/gmr_to_k1_csv.py: joint limits, motor speed, trunk tilt, spin rate, travel
  │  eval/aistpp_feasibility.py: 275 of 411 clips OK
  │  eval/rank_clips_for_sequencing.py: easiest clips to train and to start/end in neutral
  ▼
retarget/make_routine.py: blend in from / out to the neutral stance (one clip, or several chained)
  ▼                                                  → motions/k1_csv/selected/<name>.csv
csv_to_npz.py (booster_train, on the desktop)        → motions/k1_npz/<name>.npz
  ▼
Isaac Lab training, PPO, 4096 envs (desktop)         ← train/k1_dance (tasks), train/windows (launcher)
  │  ~5000 iterations for easy clips, 10000–20000 with turns; play.py exports the policy (.pt/.onnx)
  ▼
booster_deploy in MuJoCo (laptop)                    ← deploy/record_mujoco.py, eval/evaluate_policies.py
  ▼
booster_deploy on the K1: <task>_robot tasks (motor-side ankle Kd), see deploy/README.md
```

## Chaining dances into a routine

Every dance in the library (`deploy/k1_dance_deploy/__init__.py`, `DANCE_LIBRARY`) starts and ends in the
same neutral stance (booster_train's default pose: legs straight, arms at the sides). Most policies are trained
on the neutral-ended clip; Lock and Pop keep their original policies, which already handle the 0.5 s neutral
blends (fine-tuning made them worse in MuJoCo). A routine is a list of dances in `deploy/k1_dance_deploy/routines.yaml`.
Each dance's checkpoint is picked by a MuJoCo sweep, and each routine's order by `eval/handoff_matrix.py`.

At runtime the sequencer (`deploy/k1_dance_deploy/sequence.py`) runs one dance's policy at a time. When a
dance's motion ends (robot standing in neutral), it resets the next dance's policy and hands over. The reset
makes the new policy treat the robot's current heading as "forward", so heading errors don't carry over; the
policies don't observe position, so nothing else needs re-anchoring.

Fallback: `make_routine.py` can also chain several clips into one long motion and train a single policy on it
(`Routine1` task). That's less flexible (a new ~12 h training per routine) but has no policy handoffs.

## Runtime: what the robot does every 20 ms

```
                    ┌──────────────────────────────┐
                    │  REFERENCE DANCE (.npz)      │
                    │  pre-computed from human     │
                    │  mocap → GMR → K1 joints     │
                    └──────────────┬───────────────┘
                                   │ frame t (advances at 50 Hz)
                                   ▼
┌───────────────────────┐   ┌──────────────────────────────┐
│  ROBOT SENSORS        │   │  TARGET FOR THIS FRAME       │
│  • IMU: body angular  │   │  • reference joint positions │
│    velocity +         │   │  • reference joint velocities│
│    orientation        │   │  • reference trunk           │
│  • 22 joint encoders: │   │    orientation               │
│    positions and      │   └──────────────┬───────────────┘
│    velocities         │                  │
└──────────┬────────────┘                  │
           │                               │
           ▼                               ▼
┌──────────────────────────────────────────────────────────┐
│  BUILD OBSERVATION                                       │
│  • reference joint positions and velocities              │
│  • orientation error (robot trunk vs. reference trunk)   │
│  • body angular velocity                                 │
│  • joint positions/velocities (relative to default pose) │
│  • previous action  ◄───────────────────────────┐        │
└──────────────────────────┬──────────────────────┼────────┘
                           │                      │
                           ▼                      │
┌──────────────────────────────────────────────┐  │
│  POLICY NETWORK (trained in Isaac Lab, PPO)  │  │
│  runs at 50 Hz                               │  │
└──────────────────────────┬───────────────────┘  │
                           │ 22 action values     │
                           ├──────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  JOINT TARGETS                                           │
│  target = default pose + action × action scale           │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  MOTOR PD CONTROL (on the motors, faster than 50 Hz)     │
│  torque = Kp·(target − position) − Kd·velocity           │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  OUTPUT: 22 joints move → K1 performs the dance step     │
│  while balancing                                         │
└──────────────────────────┬───────────────────────────────┘
                           │ new body state
                           └──────► back to ROBOT SENSORS
                                    (repeat every 20 ms until
                                     the reference motion ends)
```

The policy uses only what the robot can measure itself (IMU and joint encoders) plus the reference frame. It never needs the robot's position or speed in the room, which is why the no-state-estimation task variant can run on hardware.
