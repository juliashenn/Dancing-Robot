"""Our K1 dance tracking tasks, built on booster_train's K1 BeyondMimic task (MJ_Dance_002).

Each dance in DANCES gets two Gym ids:
    K1-Dance-<Name>-v0        training (rough terrain, pushes, no state estimation)
    K1-Dance-<Name>-v0-Play   playback (flat, no pushes, motion plays from the start)

The reference motion is motions/k1_npz/<clip>.npz (made with booster_train's csv_to_npz.py).
To add a dance: add an entry to DANCES here and a matching pair of classes in env_cfgs.py.

This module only calls gym.register (no Isaac Lab imports), so it can be imported before
Isaac Sim starts; env_cfgs.py is imported later, when the task is created.
"""

import gymnasium as gym

# Gym name -> clip stem in motions/k1_npz/
DANCES = {
    "Pop1": "gPO_sBM_cAll_d10_mPO0_ch02",
    "Jazz1": "gJB_sBM_cAll_d08_mJB0_ch01",
    "Lock1": "gLO_sBM_cAll_d13_mLO0_ch01",
    # Jazz1 played 2.5x slower (csv_to_npz --input_fps 12 on the 30 fps CSV) so the spins are learnable.
    "Jazz1Slow": "gJB_sBM_cAll_d08_mJB0_ch01_x2.5slow",
    "HipHop1": "gLH_sBM_cAll_d16_mLH0_ch01",
}

for _name in DANCES:
    gym.register(
        id=f"K1-Dance-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"k1_dance.env_cfgs:{_name}TrainCfg",
            "rsl_rl_cfg_entry_point": f"k1_dance.env_cfgs:{_name}PPOCfg",
        },
    )
    gym.register(
        id=f"K1-Dance-{_name}-v0-Play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"k1_dance.env_cfgs:{_name}PlayCfg",
            "rsl_rl_cfg_entry_point": f"k1_dance.env_cfgs:{_name}PPOCfg",
        },
    )
