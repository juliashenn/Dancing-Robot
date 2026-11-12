"""Environment and PPO configs for our dances (imported after Isaac Sim starts).

Everything (robot, rewards, observations without state estimation, domain randomization,
PPO settings) is inherited unchanged from booster_train's K1 MJ_Dance_002 task; only the
reference motion and the experiment name differ.
"""

from pathlib import Path

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.mj_dance_002.env_cfg import (
    PlayFlatWoStateEstimationEnvCfg,
    RoughWoStateEstimationEnvCfg,
)

from k1_dance import DANCES

MOTION_DIR = Path(__file__).resolve().parents[2] / "motions" / "k1_npz"


def motion_file(name):
    path = MOTION_DIR / f"{DANCES[name]}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Convert the CSV with booster_train's csv_to_npz.py first.")
    return str(path)


# ---- Pop1: AIST++ gPO_sBM_cAll_d10_mPO0_ch02 (12 s pop routine) --------------------------------

@configclass
class Pop1TrainCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Pop1")


@configclass
class Pop1PlayCfg(PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Pop1")


@configclass
class Pop1PPOCfg(BasePPORunnerCfg):
    max_iterations = 20000
    experiment_name = "k1_dance_pop1"


# ---- Jazz1: AIST++ gJB_sBM_cAll_d08_mJB0_ch01 (12 s ballet jazz routine) -----------------------

@configclass
class Jazz1TrainCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Jazz1")


@configclass
class Jazz1PlayCfg(PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Jazz1")


@configclass
class Jazz1PPOCfg(BasePPORunnerCfg):
    # Pop plateaued by ~5000 iterations; see deploy/README.md.
    max_iterations = 5000
    experiment_name = "k1_dance_jazz1"


# ---- Lock1: AIST++ gLO_sBM_cAll_d13_mLO0_ch01 (12 s locking routine) ---------------------------

@configclass
class Lock1TrainCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Lock1")


@configclass
class Lock1PlayCfg(PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Lock1")


@configclass
class Lock1PPOCfg(BasePPORunnerCfg):
    max_iterations = 5000
    experiment_name = "k1_dance_lock1"


# ---- Jazz1Slow: Jazz1 at 2.5x slower (30 s), to learn the spins --------------------------------
# Original spins ~665 deg/s (p95) and weren't learned in 5000 iterations; slowed: ~270 deg/s.

@configclass
class Jazz1SlowTrainCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Jazz1Slow")


@configclass
class Jazz1SlowPlayCfg(PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("Jazz1Slow")


@configclass
class Jazz1SlowPPOCfg(BasePPORunnerCfg):
    max_iterations = 10000
    experiment_name = "k1_dance_jazz1slow"


# ---- HipHop1: AIST++ gLH_sBM_cAll_d16_mLH0_ch01 (12 s LA-style hip-hop) ------------------------
# Energetic footwork (leg p99 7.2 rad/s), mild turning (93 deg/s p95), 0.5 m of travel.

@configclass
class HipHop1TrainCfg(RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("HipHop1")


@configclass
class HipHop1PlayCfg(PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = motion_file("HipHop1")


@configclass
class HipHop1PPOCfg(BasePPORunnerCfg):
    max_iterations = 10000
    experiment_name = "k1_dance_hiphop1"
