"""Pinocchio robot wrapper for fixed-base Z1 arm (6-DoF).

Provides CasADi FK functions and access to Pinocchio numeric model/data
for gravity compensation g(q).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import casadi as ca
import numpy as np

import pinocchio as pin
import pinocchio.casadi as cpin


@dataclass
class RobotWrapper:
    urdf_path: str = "robot_description/z1.urdf"
    ee_frame_name: str = "link06"

    def __post_init__(self) -> None:
        # Numeric model/data
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        # CasADi model/data for FK
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.ee_frame_id = self.cmodel.getFrameId(self.ee_frame_name)
        if self.ee_frame_id < 0:
            raise ValueError(f"End-effector frame '{self.ee_frame_name}' not found.")

        # CasADi FK
        q_sym = ca.SX.sym("q", self.model.nq)
        cpin.forwardKinematics(self.cmodel, self.cdata, q_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        placement = self.cdata.oMf[self.ee_frame_id]
        p_local = placement.translation
        R_local = placement.rotation

        self.fk_arm_pos = ca.Function("fk_arm_pos", [q_sym], [p_local])
        self.fk_arm_rot = ca.Function("fk_arm_rot", [q_sym], [R_local])

        if self.model.nq != 6:
            raise RuntimeError(f"Expected arm nq=6, got {self.model.nq}.")

    @property
    def nq_arm(self) -> int:
        return int(self.model.nq)

    def fk_symbolic(self, q_arm: ca.SX) -> Tuple[ca.SX, ca.SX]:
        if int(q_arm.shape[0]) != self.model.nq:
            raise ValueError("fk_symbolic expects 6-dof arm q")
        p = self.fk_arm_pos(q_arm)
        R = self.fk_arm_rot(q_arm)
        return p, R

    # Numeric gravity compensation g(q)
    def gravity(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(self.model.nq)
        pin.computeGeneralizedGravity(self.model, self.data, q)
        return self.data.g.copy()

