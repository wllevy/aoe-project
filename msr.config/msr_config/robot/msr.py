# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the modular surgical research platform.

The following configurations are available:

* :obj:`MSR_PSM_CFG`: MSR_PSM robot with gripper
* :obj:`MSR_PSM_HIGH_PD_CFG`: MSR_PSM with gripper with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import numpy as np

##
# Configuration
##

MSR_PSM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/aoe_project/Assets/MSR_model/MSR-URDF-Mortorpack-0522-3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ), # NOTE: here the enabled_self_collisions=False, otherwise will vibrates during simulation.
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        # scale=(10.0, 10.0, 10.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "psm_shoulder_Joint": np.pi/2,
            "psm_upper_Joint": -np.pi/4,
            "psm_fore_Joint": -2./3.*np.pi,
            "psm_wrist1_Joint": np.pi,
            "psm_wrist2_Joint": -np.pi/2,
            "psm_wrist3_Joint": -2./3.*np.pi,
            "psm_insertion_Joint": 0.02,
            "psm_roll_Joint": 0.0, # 0.0
            "psm_pitch_Joint": 0.0,
            "psm_yaw_Joint": 0.0,
            "psm_gripper1_Joint": 0.0,
            "psm_gripper2_Joint": 0.0,
        },
    ),
    actuators={
        "psm_robotarm": ImplicitActuatorCfg(
            joint_names_expr=[
                "psm_shoulder_Joint", 
                "psm_upper_Joint", 
                "psm_fore_Joint", 
                "psm_wrist1_Joint", 
                "psm_wrist2_Joint", 
                "psm_wrist3_Joint"],
            effort_limit=12,
            velocity_limit=1,
            stiffness=800.0,
            damping=40.0,
        ),
        "psm_insertion": ImplicitActuatorCfg(
            joint_names_expr=["psm_insertion_Joint"],
            effort_limit=12,
            velocity_limit=0.2,
            stiffness=800.0,
            damping=40.0,
        ),
        "psm_roll": ImplicitActuatorCfg(
            joint_names_expr=["psm_roll_Joint"],
            effort_limit=12,
            velocity_limit_sim=1,
            stiffness=800.0,
            damping=40.0,
        ), # TODO: check psm_roll joint
        "psm_pitch": ImplicitActuatorCfg(
            joint_names_expr=["psm_pitch_Joint"],
            effort_limit=12,
            velocity_limit_sim=1,
            stiffness=800.0,
            damping=40.0,
        ),
        "psm_yaw": ImplicitActuatorCfg(
            joint_names_expr=["psm_yaw_Joint"],
            effort_limit=12,
            velocity_limit=1,
            stiffness=800.0,
            damping=40.0,
        ),
        "psm_gripper": ImplicitActuatorCfg(
            joint_names_expr=["psm_gripper1_Joint", "psm_gripper2_Joint"],
            effort_limit=0.1,
            velocity_limit=1,
            stiffness=800, #1.7e13, #500,
            damping=40, #1745, # 10, # smaller value will cause unexpected bahavior
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of MSR PSM robot."""


MSR_PSM_HIGH_PD_CFG = MSR_PSM_CFG.copy()
MSR_PSM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
MSR_PSM_HIGH_PD_CFG.actuators["psm_robotarm"].stiffness = 400.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_robotarm"].damping = 320.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_insertion"].stiffness = 400.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_insertion"].damping = 320.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_roll"].stiffness = 400.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_roll"].damping = 320.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_pitch"].stiffness = 400.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_pitch"].damping = 320.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_yaw"].stiffness = 400.0
MSR_PSM_HIGH_PD_CFG.actuators["psm_yaw"].damping = 320.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_robotarm"].stiffness = 8000.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_robotarm"].damping = 400.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_insertion"].stiffness = 8000.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_insertion"].damping = 400.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_roll"].stiffness = 8000.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_roll"].damping = 400.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_pitch"].stiffness = 8000.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_pitch"].damping = 400.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_yaw"].stiffness = 8000.0
# MSR_PSM_HIGH_PD_CFG.actuators["psm_yaw"].damping = 400.0
"""Configuration of MSR PSM robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
