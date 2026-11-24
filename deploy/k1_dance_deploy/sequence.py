"""Play several dance policies back to back, switching policies while the robot stands in neutral.

Each dance's motion starts and ends in the neutral stance (retarget/make_routine.py --hold-ends 0.5 <clip>),
and its policy was fine-tuned on that motion. DanceSequencePolicy runs dance 1's policy until its motion
ends, then resets dance 2's policy and hands over, and so on.

A reset makes the new policy treat the robot's current heading as the start of its motion (BeyondMimicPolicy
measures orientation relative to the robot's yaw at reset), so heading error from earlier dances doesn't
carry over. Position isn't observed by these policies, so nothing needs re-anchoring for control. Only the
MuJoCo ghost (reference) is moved to where the robot is at each switch, so the video and drift numbers
compare each dance against where it actually started.
"""

from __future__ import annotations

import copy
from dataclasses import MISSING
from types import SimpleNamespace

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import PolicyCfg
from booster_deploy.utils.isaaclab import math as lab_math
from booster_deploy.utils.isaaclab.configclass import configclass
from tasks.beyond_mimic.beyond_mimic import BeyondMimicPolicy, BeyondMimicPolicyCfg


class DanceSequencePolicy(Policy):
    def __init__(self, cfg: DanceSequencePolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.dances = []
        for checkpoint, motion in cfg.dances:
            sub = copy.deepcopy(cfg.template)
            sub.checkpoint_path, sub.motion_path = checkpoint, motion
            sub.device = cfg.device
            sub.enable_safety_fallback = cfg.enable_safety_fallback
            sub.stop_at_motion_end = False          # the sequence decides when to stop
            self.dances.append(BeyondMimicPolicy(sub, controller))
        self.robot = controller.robot
        # Total length, for callers that read policy.motion.time_step_total (deploy/record_mujoco.py).
        self.motion = SimpleNamespace(time_step_total=sum(int(d.motion.time_step_total) for d in self.dances))

    def reset(self) -> None:
        self.index = 0
        self.switch_frames = [0]                    # global frame at which each dance started
        self.frame = 0
        self._start(self.dances[0], last_action=None)

    def _start(self, dance: BeyondMimicPolicy, last_action):
        dance.reset()
        if last_action is not None and self.cfg.carry_last_action:
            dance.last_action = last_action.clone()
        # Where this dance starts in the world, for drawing its reference (ghost) in the right place.
        root_pos = self.robot.data.root_pos_w.to(self.cfg.device)
        self._origin_pos = torch.tensor([root_pos[0], root_pos[1], 0.0], dtype=torch.float32, device=self.cfg.device)
        self._origin_yaw = lab_math.yaw_quat_zxy(self.robot.data.root_quat_w.to(self.cfg.device))

    def inference(self) -> torch.Tensor:
        dance = self.dances[self.index]
        action = dance.inference()
        self._place_ghost(dance)
        self.frame += 1

        if dance.current_frame >= dance.motion.time_step_total:
            if self.index + 1 < len(self.dances):
                self.index += 1
                self.switch_frames.append(self.frame)
                self._start(self.dances[self.index], last_action=dance.last_action)
            elif self.cfg.stop_at_motion_end:
                self.controller.stop()
        return action

    def _place_ghost(self, dance: BeyondMimicPolicy):
        if not hasattr(self.controller, "set_reference_qpos"):
            return
        pos = lab_math.quat_apply(self._origin_yaw, dance.cmd_root_pos_w) + self._origin_pos
        quat = lab_math.quat_mul(self._origin_yaw, dance.cmd_root_quat_w)
        joints = dance.cmd_dof_pos[self.robot.data.sim2real_joint_indexes]
        self.controller.set_reference_qpos(torch.cat([pos, quat, joints], dim=0))  # type: ignore


@configclass
class DanceSequencePolicyCfg(PolicyCfg):
    constructor = DanceSequencePolicy
    checkpoint_path: str = "(sequence)"
    # Shared settings for every dance (action scale, anchor body, ...); checkpoint/motion are filled per dance.
    template: BeyondMimicPolicyCfg = MISSING
    # (checkpoint_path, motion_path) per dance, in order; paths as for BeyondMimicPolicyCfg.
    dances: list = MISSING
    # Give the next policy the previous policy's last action (as mid-episode in training) instead of zeros.
    carry_last_action: bool = True
    stop_at_motion_end: bool = True
