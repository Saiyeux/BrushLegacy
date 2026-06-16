"""
Thin wrapper around pyfranka.franka_pybind.

Copied from Cobrush Pro scripts/basic_motion/franka.py and extended with
JointVelocities / JointVelocitiesFinished for conical sweep support.
"""

import math
from pyfranka.franka_pybind import (
    FrankaApi,
    RobotMode,
    MotionGenerator,
    JointPositions,
    JointPositionsMotionFinished,
    JointVelocities,
    JointVelocitiesFinished,
    CartesianVelocities,
    CartesianVelocitiesFinished,
)

_TORQUE_LOWER = [25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0]
_TORQUE_UPPER = [35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0]
_FORCE_LOWER  = [30.0, 30.0, 30.0, 25.0, 25.0, 25.0]
_FORCE_UPPER  = [40.0, 40.0, 40.0, 35.0, 35.0, 35.0]

HOME_JOINTS = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
J7_PIN      = -0.02   # J7 null-space target — held throughout all Cartesian moves


class Franka:
    """
    Minimal Franka controller.

    Usage:
        robot = Franka("192.170.10.200")
        if not robot.wait_ready():
            raise SystemExit("robot not ready")
        robot.go_home()
        robot.robot_control(joint_positions_handle=my_callback)
    """

    def __init__(self, ip: str = "192.170.10.200", log_size: int = 5000):
        self.api = FrankaApi()
        self.api.init_config(ip, log_size=log_size)
        self.api.set_default_behavior()
        self.api.set_collision_behavior(
            _TORQUE_LOWER, _TORQUE_UPPER,
            _TORQUE_LOWER, _TORQUE_UPPER,
            _FORCE_LOWER,  _FORCE_UPPER,
            _FORCE_LOWER,  _FORCE_UPPER,
        )
        print(f"[Franka] connected to {ip}")

    def wait_ready(self, retries: int = 3) -> bool:
        """Return True when robot is in Idle mode; attempt error recovery on Reflex."""
        for i in range(retries):
            state = self.api.readOnce()
            if state.robot_mode == RobotMode.kIdle:
                print("[Franka] ready")
                return True
            print(f"[Franka] not ready (mode={state.robot_mode}), attempt {i+1}/{retries}")
            if state.robot_mode == RobotMode.kReflex:
                self.api.automatic_error_recovery()
        return False

    def read_state(self):
        return self.api.readOnce()

    def go_home(self, speed_factor: float = 0.3):
        """Move to canonical home joint configuration."""
        print("[Franka] moving to home...")
        gen = MotionGenerator(speed_factor, HOME_JOINTS)
        self.api.robot_control(joint_positions_handle=gen.operator)
        print("[Franka] at home")

    def robot_control(self, **kwargs):
        self.api.robot_control(**kwargs)
