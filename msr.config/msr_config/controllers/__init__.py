# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different controllers and motion-generators.

Controllers or motion generators are responsible for closed-loop tracking of a given command. The
controller can be a simple PID controller or a more complex controller such as impedance control
or inverse kinematics control. The controller is responsible for generating the desired joint-level
commands to be sent to the robot.
"""

from .differential_ik_with_soft_rcm import DifferentialIKWithSoftRCMController
from .differential_ik_with_soft_rcm_cfg import DifferentialIKWithSoftRCMControllerCfg

