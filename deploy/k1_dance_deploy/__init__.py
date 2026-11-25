"""booster_deploy tasks for our K1 dance policies.

This folder is a booster_deploy task package: copy it to booster_deploy/tasks/k1_dance/ on the
robot (it imports booster_deploy's own K1 BeyondMimic config). Locally, deploy/record_mujoco.py
imports it directly.

Why not reuse Booster's k1_mj2 task as-is: its motor gains (Kp 80/30 on the legs) and the action
scale derived from them (0.25 * effort / Kp) belong to an older booster_train K1 config. Policies
trained with the current booster_train (ours) expect the gains below; with k1_mj2's values the leg
actions are ~5x too small and the robot falls within ~2 s in MuJoCo.

Values are copied from booster_train's K1 config as printed at training time
(assets/robots/booster_k1.py: Kp = armature * (2*pi*f)^2, Kd = 2*zeta*armature*(2*pi*f),
action scale = 0.25 * effort_limit / Kp). Joint order (booster_deploy "real" order):
head yaw, head pitch, L arm x4, R arm x4, L hip pitch/roll/yaw, knee, ankle pitch/roll, R leg x6.

REAL ROBOT NOTE: the K1 ankles are a parallel linkage. booster_train's README says not to send
the simulation ankle Kd to the motors; recompute it motor-side before running on hardware.
"""

import copy
import math
import os

import yaml

from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from tasks.beyond_mimic import beyond_mimic as _beyond_mimic
from tasks.beyond_mimic.robots.k1 import K1BeyondMimicControllerCfg

from .sequence import DanceSequencePolicyCfg

_HERE = os.path.dirname(os.path.abspath(__file__))
_POLICY_DIR = os.path.dirname(os.path.abspath(_beyond_mimic.__file__))


def asset(rel_path):
    """Path to a file in this package, relative to booster_deploy's BeyondMimic policy folder
    (booster_deploy joins model/motion paths onto that folder, not onto the task's)."""
    return os.path.relpath(os.path.join(_HERE, rel_path), _POLICY_DIR)

HEAD = dict(kp=3.9478417602100686, kd=0.25132741228, effort=6.0, scale=0.3799544386804864)
ARM = dict(kp=3.9478417602100686, kd=0.25132741228, effort=14.0, scale=0.8865603569211349)
LEG = [  # hip pitch, hip roll, hip yaw, knee, ankle pitch, ankle roll
    dict(kp=30.200989465607023, kd=3.60497756989125, effort=68.0, scale=0.5628954647118317),
    dict(kp=21.447961045805584, kd=2.560161764834957, effort=43.0, scale=0.501213144552139),
    dict(kp=17.846013389258083, kd=2.1302109340993156, effort=38.3, scale=0.5365343951699267),
    dict(kp=60.401978931214046, kd=4.806636759855, effort=112.0, scale=0.46356097093915555),
    dict(kp=24.984418744961317, kd=2.9822953077390415, effort=38.3, scale=0.38323885369280475),
    dict(kp=7.1384053557032345, kd=0.8520843736397263, effort=38.3, scale=1.3413359879248166),
]
JOINTS = [HEAD] * 2 + [ARM] * 8 + LEG + LEG


@configclass
class K1DanceControllerCfg(K1BeyondMimicControllerCfg):
    """K1 dance policy trained with the current booster_train K1 config."""

    def __post_init__(self):
        super().__post_init__()
        self.robot.joint_stiffness = [j["kp"] for j in JOINTS]
        self.robot.joint_damping = [j["kd"] for j in JOINTS]
        self.robot.effort_limit = [j["effort"] for j in JOINTS]
        self.policy.fixed_action_scale = [j["scale"] for j in JOINTS]


@configclass
class K1DanceMJ2OursCfg(K1DanceControllerCfg):
    """Our reproduction of Booster's MJ dance (iteration 3000 of the 2026-11-08 run)."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = asset("motions/k1_mj2_seg1.npz")
        self.policy.checkpoint_path = asset("models/mj2_ours_it3000.pt")


@configclass
class K1DancePop1Cfg(K1DanceControllerCfg):
    """AIST++ Pop clip gPO_sBM_cAll_d10_mPO0_ch02 (12 s), latest exported checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = asset("motions/gPO_sBM_cAll_d10_mPO0_ch02.npz")
        self.policy.checkpoint_path = asset("models/pop1_it12750.pt")


@configclass
class K1DanceJazz1Cfg(K1DanceControllerCfg):
    """AIST++ ballet jazz clip gJB_sBM_cAll_d08_mJB0_ch01 (12 s), iteration 4999."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = asset("motions/gJB_sBM_cAll_d08_mJB0_ch01.npz")
        self.policy.checkpoint_path = asset("models/jazz1_it4999.pt")


@configclass
class K1DanceLock1Cfg(K1DanceControllerCfg):
    """AIST++ locking clip gLO_sBM_cAll_d13_mLO0_ch01 (12 s), iteration 4999."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = asset("motions/gLO_sBM_cAll_d13_mLO0_ch01.npz")
        self.policy.checkpoint_path = asset("models/lock1_it4999.pt")


@configclass
class K1DanceJazz1SlowCfg(K1DanceControllerCfg):
    """The ballet jazz clip played 2.5x slower (30 s) so the spins are learnable, iteration 19999 (does all ~13 turns)."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = asset("motions/gJB_sBM_cAll_d08_mJB0_ch01_x2.5slow.npz")
        self.policy.checkpoint_path = asset("models/jazz1slow_it19999.pt")


@configclass
class K1DanceHipHop1Cfg(K1DanceControllerCfg):
    """AIST++ LA hip-hop clip gLH_sBM_cAll_d16_mLH0_ch01 (12 s), iteration 9999."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = asset("motions/gLH_sBM_cAll_d16_mLH0_ch01.npz")
        self.policy.checkpoint_path = asset("models/hiphop1_it9999.pt")


def sequence(*dances):
    """Controller config that plays (model, motion) pairs back to back, switching policies in neutral."""
    cfg = K1DanceControllerCfg()
    cfg.policy = DanceSequencePolicyCfg(
        template=cfg.policy, dances=[(asset(model), asset(motion)) for model, motion in dances])
    return cfg


# Smoke test of the switching code with the original (not neutral-ended) Lock and Pop policies:
# the handover is abrupt because Lock ends mid-move and Pop starts mid-move.
SEQ_TEST = [("models/lock1_it4999.pt", "motions/gLO_sBM_cAll_d13_mLO0_ch01.npz"),
            ("models/pop1_it12750.pt", "motions/gPO_sBM_cAll_d10_mPO0_ch02.npz")]
# Neutral-ended dances for switching (the *N tasks in train/k1_dance): name -> (model, motion).
DANCE_LIBRARY = {
    "lock": ("models/lock1n_it6999.pt", "motions/lock1n.npz"),        # fine-tuned from Lock1
    "pop": ("models/pop1n_it15000.pt", "motions/pop1n.npz"),          # fine-tuned from Pop1
    "hiphop": ("models/hiphop1n_it11999.pt", "motions/hiphop1n.npz"),  # fine-tuned from HipHop1
    "sjazz": ("models/sjazz1n_it4999.pt", "motions/sjazz1n.npz"),     # Street Jazz, trained neutral-ended
    "krump": ("models/krump1n_it4999.pt", "motions/krump1n.npz"),
    "pop2": ("models/pop2n_it4999.pt", "motions/pop2n.npz"),
}


def load_routines():
    """routines.yaml -> {name: [dance, ...]}, keeping only routines whose dances all have models."""
    with open(os.path.join(_HERE, "routines.yaml")) as f:
        routines = yaml.safe_load(f) or {}
    ready = {}
    for name, dances in routines.items():
        unknown = [d for d in dances if d not in DANCE_LIBRARY]
        if unknown:
            print(f"[k1_dance] routine '{name}': unknown dances {unknown} (library: {list(DANCE_LIBRARY)})")
            continue
        if all(os.path.exists(os.path.join(_HERE, DANCE_LIBRARY[d][0])) for d in dances):
            ready[name] = dances
    return ready

# Real robot: the ankles are a parallel linkage. Simulation Kd includes booster_train's armature_ratio
# for the linkage (K1: 1.4 pitch, 0.4 roll), but robot.joint_damping is sent straight to the motor, so
# booster_train's README says to recompute it with motor-side inertia and no armature_ratio:
#   Kd = 2 * zeta * J_motor * (2*pi*f) = 2 * 1.5 * 0.0282528 (E4310) * 2*pi*4 = 2.130
# (Booster's own k1_mj2 deploy config uses 2.0 on the ankles.) Kp keeps the training value.
ROBOT_ANKLE_KD = 2 * 1.5 * 0.0282528 * (2 * math.pi * 4.0)
ANKLE_INDICES = (14, 15, 20, 21)  # L ankle pitch/roll, R ankle pitch/roll (booster_deploy real order)


def for_robot(cfg):
    """Same task with the motor-side ankle damping for hardware."""
    cfg = copy.deepcopy(cfg)
    damping = list(cfg.robot.joint_damping)
    for i in ANKLE_INDICES:
        damping[i] = ROBOT_ANKLE_KD
    cfg.robot.joint_damping = damping
    return cfg


TASKS = {
    "k1_dance_seq_test": sequence(*SEQ_TEST),
    "k1_dance_mj2_ours": K1DanceMJ2OursCfg(),
    "k1_dance_pop1": K1DancePop1Cfg(),
    "k1_dance_jazz1": K1DanceJazz1Cfg(),
    "k1_dance_lock1": K1DanceLock1Cfg(),
    "k1_dance_jazz1slow": K1DanceJazz1SlowCfg(),
    "k1_dance_hiphop1": K1DanceHipHop1Cfg(),
    # Each library dance on its own (neutral in, neutral out): test these before the routines.
    **{f"k1_dance_lib_{name}": sequence(dance) for name, dance in DANCE_LIBRARY.items()
       if os.path.exists(os.path.join(_HERE, dance[0]))},
    **{f"k1_routine_{name}": sequence(*(DANCE_LIBRARY[d] for d in dances))
       for name, dances in load_routines().items()},
}
# <name> uses the simulation gains (MuJoCo sim-to-sim); <name>_robot is the version to run on hardware.
for _name, _cfg in TASKS.items():
    register_task(_name, _cfg)
    register_task(f"{_name}_robot", for_robot(_cfg))
