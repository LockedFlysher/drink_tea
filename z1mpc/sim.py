"""MuJoCo simulation wrapper for fixed-base Z1 (z1.xml)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import mujoco


@dataclass
class Z1MuJoCoSim:
    xml_path: str = "robot_description/z1.xml"

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Joint indices (qpos and dof) for 6 arm joints
        self.joint_names = [f"joint{i+1}" for i in range(6)]
        self.joint_qpos_indices = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.joint_names
        ]
        self.joint_dof_indices = [
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.joint_names
        ]
        # Cache actuator ids (motor1..motor6) once
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i+1}")
            for i in range(6)
        ]

    def reset_from_x(self, q_arm: np.ndarray) -> None:
        q_arm = np.asarray(q_arm, dtype=float).reshape(6)
        for i, q_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[q_idx] = q_arm[i]
        mujoco.mj_forward(self.model, self.data)

    def set_qpos(self, q_arm: np.ndarray) -> None:
        """Directly set arm joint positions and forward the model (kinematic replay)."""
        self.reset_from_x(q_arm)

    def reset_keyframe(self, name: str = "home") -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            mujoco.mj_forward(self.model, self.data)

    def get_arm_state(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.array([self.data.qpos[idx] for idx in self.joint_qpos_indices], dtype=float)
        v = np.array([self.data.qvel[idx] for idx in self.joint_dof_indices], dtype=float)
        return q, v

    def neutralize_actuators_to_q(self) -> None:
        # Set position servos to current q to avoid fighting external torques
        for i, act_id in enumerate(self.actuator_ids):
            if act_id >= 0:
                q_idx = self.joint_qpos_indices[i]
                self.data.ctrl[act_id] = float(self.data.qpos[q_idx])

    def set_arm_torque(self, tau: np.ndarray) -> None:
        tau = np.asarray(tau, dtype=float).reshape(6)
        # Clear previous
        for dof_idx in self.joint_dof_indices:
            self.data.qfrc_applied[dof_idx] = 0.0
        # Apply new
        for i, dof_idx in enumerate(self.joint_dof_indices):
            self.data.qfrc_applied[dof_idx] = float(tau[i])

    def step_n(self, n: int = 1) -> None:
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)
