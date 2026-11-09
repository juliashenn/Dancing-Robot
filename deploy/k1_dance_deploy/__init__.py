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

import os

from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from tasks.beyond_mimic import beyond_mimic as _beyond_mimic
from tasks.beyond_mimic.robots.k1 import K1BeyondMimicControllerCfg

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


register_task("k1_dance_mj2_ours", K1DanceMJ2OursCfg())
