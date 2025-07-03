# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg
 
##
# Pre-defined configs
##
from msr_config.robot import MSR_PSM_CFG, MSR_PSM_HIGH_PD_CFG  # isort: skip
from msr_config.controllers import DifferentialIKWithSoftRCMControllerCfg
from msr_config.actions import DifferentialInverseKinematicsWithSoftRCMActionCfg # TODO: to write customize action

@configclass
class MSRPSMReachEnvCfg(joint_pos_env_cfg.MSRPSMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 4096

        # Set MSR_PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (msr)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_shoulder_Joint", 
                "psm_upper_Joint", 
                "psm_fore_Joint", 
                "psm_wrist1_Joint", 
                "psm_wrist2_Joint", 
                "psm_wrist3_Joint", 
                "psm_insertion_Joint", 
                "psm_roll_Joint", 
                "psm_pitch_Joint", 
                "psm_yaw_Joint"], 
            body_name="psm_tool_tip_Link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            # body_offset=DifferentialInverseKinematicsWithSoftRCMActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),

        )
        self.scene.replicate_physics = True
@configclass
class MSRPSMReachEnvCfgv0(joint_pos_env_cfg.MSRPSMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 4096

        # Set MSR_PSM as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (msr)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "psm_shoulder_Joint", 
                "psm_upper_Joint", 
                "psm_fore_Joint", 
                "psm_wrist1_Joint", 
                "psm_wrist2_Joint", 
                "psm_wrist3_Joint", 
                "psm_insertion_Joint", 
                "psm_roll_Joint", 
                "psm_pitch_Joint", 
                "psm_yaw_Joint"], 
            body_name="psm_tool_tip_Link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            # body_offset=DifferentialInverseKinematicsWithSoftRCMActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        )
        self.scene.replicate_physics = True
@configclass
class MSRPSMReachEnvCfg_PLAY(MSRPSMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
