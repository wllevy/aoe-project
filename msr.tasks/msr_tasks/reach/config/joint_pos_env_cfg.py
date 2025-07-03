# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, DEFORMABLE_TARGET_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.managers import EventTermCfg as EventTerm
##
# Pre-defined configs
##

from msr_config.robot import MSR_PSM_CFG, MSR_PSM_HIGH_PD_CFG  # isort: skip
import numpy as np

##
# Environment configuration
##


@configclass
class MSRPSMReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # self.scene.num_envs = 64

        # switch robot
        self.scene.robot = MSR_PSM_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["psm_tool_tip_Link"]

        # override actions/ JointPositionToLimitsActionCfg/JointPositionActionCfg
        self.actions.arm_action = mdp.JointPositionActionCfg(
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
            scale=0.3, 
            # clip={
            #     "psm_shoulder_Joint": (-np.pi * 2, np.pi * 2), 
            #     "psm_upper_Joint": (-np.pi * 2, np.pi * 2), 
            #     "psm_fore_Joint": (-np.pi * 2, np.pi * 2), 
            #     "psm_wrist1_Joint": (-np.pi * 2, np.pi * 2), 
            #     "psm_wrist2_Joint": (-np.pi * 2, np.pi * 2), 
            #     "psm_wrist3_Joint": (-np.pi * 2, np.pi * 2), 
            #     'psm_insertion_Joint': (-0.005, 0.002),
            #     "psm_roll_Joint": (-np.pi / 2, np.pi / 2), 
            #     "psm_pitch_Joint": (-np.pi / 2, np.pi / 2), 
            #     "psm_yaw_Joint": (-np.pi / 2, np.pi / 2)
            #     },
            use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        # self.commands.ee_pose.body_name = "psm_yaw_Link"
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="psm_tool_tip_Link",
            resampling_time_range=(3.0, 3.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                # pos_x=(0.290, 0.298),#(-0.07, 0.07),
                # pos_y=(0.382, 0.387),#(-0.07, 0.07),
                # pos_z=(0.160, 0.160),#(-0.12, -0.08),
                pos_x=(0.3, 0.5),
                pos_y=(0.2, 0.4),
                pos_z=(0.15, 0.3),
                # roll=(3.14, 3.14),
                # pitch=(0, 0),
                # yaw=(1.57, 1.57),
                roll=(1.60, 2.80),
                pitch=(-0.26, -0),
                yaw=(1.07, 1.57),
                # roll=(2.60, 2.80),
                # pitch=(-0.26, -0),
                # yaw=(1.57, 1.57),
            ),
        )
        # set the scale of the visualization markers to (0.01, 0.01, 0.01)
        self.commands.ee_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        self.commands.ee_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/psm_base_Link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/psm_tool_tip_Link",
                    name="end_effector",
                ),
            ],
        )

        rcm_marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        rcm_marker_cfg.markers["target"].scale = (0.02, 0.02, 0.02)
        rcm_marker_cfg.markers["target"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        # self.scene.rcm_marker = VisualizationMarkers(rcm_marker_cfg.replace(prim_path="/Visuals/rcm_target"))




@configclass
class MSRPSMReachEnvCfg_PLAY(MSRPSMReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
