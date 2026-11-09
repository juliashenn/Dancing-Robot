# Pipeline

## Offline: from human mocap to a policy

```
Human mocap (LAFAN1 BVH / AMASS / AIST++)
  │  GMR (booster_k1)                      ← retarget/
  ▼
K1 joint trajectory (.csv, BeyondMimic format)
  │  feasibility filter, join segments,
  │  blend transitions, time to music      ← choreography/
  ▼
csv_to_npz.py (booster_train)              ← motions/k1_npz/
  ▼
Isaac Lab training (PPO)                   ← train/
  │  play.py → export policy (.pt/.onnx)
  ▼
booster_deploy --mujoco                    ← deploy/  (check in a second simulator)
  ▼
booster_deploy on the K1 (gantry first)
```

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
