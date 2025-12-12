"""Z1 MPC modules (fixed-base, 6-DoF arm).

Public API:
- z1mpc.robot.RobotWrapper
- z1mpc.mpc.MPCConfig, z1mpc.mpc.WholeBodyMPC
- z1mpc.sim.Z1MuJoCoSim
- z1mpc.trajectory.ReferenceTrajectory
- z1mpc.utils (quat helpers)
- z1mpc.viz (viewer helpers)
"""

from .robot import RobotWrapper
from .mpc import MPCConfig, WholeBodyMPC
from .sim import Z1MuJoCoSim
from .trajectory import ReferenceTrajectory
from . import utils
from . import viz

__all__ = [
    "RobotWrapper",
    "MPCConfig",
    "WholeBodyMPC",
    "Z1MuJoCoSim",
    "ReferenceTrajectory",
    "utils",
    "viz",
]

