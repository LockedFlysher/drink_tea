"""6-DoF arm MPC (fixed-base) using CasADi/Ipopt.

State x=q (6), control u=dq (6), dynamics: q_{k+1} = q_k + dt*u_k.
End-effector tracking cost (position xyz equal weight + orientation),
and hard bounds on q and dq.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import casadi as ca
import numpy as np

from .robot import RobotWrapper
from .utils import rot_to_quat, orientation_error_from_quats


@dataclass
class MPCConfig:
    horizon_steps: int = 20
    dt: float = 0.1
    # Position/orientation tracking weights (legacy-style, separate z weight)
    w_pos_vec: tuple[float, float, float] = (50.0, 50.0, 50.0)
    w_ori: float = 1.0
    # Control weight (L2)
    R_u: float = 0.1
    # Velocity limits fallback (will use URDF limits if available)
    dq_min: float = -1.0
    dq_max: float = 1.0
    # Velocity control gain for torque-level tracking (used outside MPC)
    kd_arm: float = 30.0
    terminal_cost : float = 10.0


class WholeBodyMPC:
    def __init__(self, robot: RobotWrapper, cfg: MPCConfig) -> None:
        self.robot = robot
        self.cfg = cfg
        self.nx = robot.nq_arm
        self.nu = robot.nq_arm
        self.N = cfg.horizon_steps
        self._build_ocp()
        self._X_guess: np.ndarray | None = None
        self._U_guess: np.ndarray | None = None
        self._lam_g: np.ndarray | None = None
        self._lam_x: np.ndarray | None = None

    def _build_ocp(self) -> None:
        N = self.N
        nx = self.nx
        nu = self.nu
        dt = self.cfg.dt

        opti = ca.Opti()
        self.opti = opti

        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        self.X = X
        self.U = U

        x0_param = opti.parameter(nx)
        p_ref_param = opti.parameter(3, N + 1)
        q_ref_param = opti.parameter(4, N + 1)
        self.x0_param = x0_param
        self.p_ref_param = p_ref_param
        self.q_ref_param = q_ref_param

        w_pos_vec = np.array(self.cfg.w_pos_vec, dtype=float).reshape(3)
        w_ori = self.cfg.w_ori
        R_u = self.cfg.R_u * ca.DM.eye(nu)

        # Bounds from URDF
        n_pos_joints = int(self.robot.nq_arm)
        q_lower_np = np.asarray(self.robot.model.lowerPositionLimit, dtype=float).reshape(-1)
        q_upper_np = np.asarray(self.robot.model.upperPositionLimit, dtype=float).reshape(-1)
        if q_lower_np.size != n_pos_joints or q_upper_np.size != n_pos_joints:
            raise RuntimeError("URDF position limits size mismatch")
        q_min_vec = ca.DM(q_lower_np).reshape((n_pos_joints, 1))
        q_max_vec = ca.DM(q_upper_np).reshape((n_pos_joints, 1))

        vel_lim_np = np.asarray(self.robot.model.velocityLimit, dtype=float).reshape(-1)
        if vel_lim_np.size != n_pos_joints:
            raise RuntimeError("URDF velocity limits size mismatch")
        dq_min_vec = ca.DM((-vel_lim_np).reshape((n_pos_joints, 1)))
        dq_max_vec = ca.DM(( vel_lim_np).reshape((n_pos_joints, 1)))

        opti.subject_to(X[:, 0] == x0_param)

        total_cost = 0
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # Discrete dynamics
            opti.subject_to(x_next == x_k + dt * u_k)

            # End-effector FK
            p_ee_k, R_ee_k = self.robot.fk_symbolic(x_k)
            p_ref_k = p_ref_param[:, k]
            q_ref_k = q_ref_param[:, k]

            # Position (xyz weighted)
            pos_err = p_ee_k - p_ref_k
            Wpos = ca.diag(ca.DM(w_pos_vec))
            pos_cost = ca.mtimes([pos_err.T, Wpos, pos_err])

            # Orientation cost (left quaternion error)
            q_ee_k = rot_to_quat(R_ee_k)
            ori_err = orientation_error_from_quats(q_ee_k, q_ref_k)
            ori_cost = w_ori * ca.dot(ori_err, ori_err)

            # Bounds
            q_joint = x_k
            dq_joint = u_k
            opti.subject_to(opti.bounded(q_min_vec, q_joint, q_max_vec))
            opti.subject_to(opti.bounded(dq_min_vec, dq_joint, dq_max_vec))

            effort_k = ca.mtimes([u_k.T, R_u, u_k])
            total_cost += pos_cost + ori_cost + effort_k

            # if k == self.N - 1:
            #     total_cost += (pos_cost + ori_cost + effort_k)*self.cfg.terminal_cost

        opti.minimize(total_cost)

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 80,
            "print_time": 0,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-6,
            "ipopt.warm_start_mult_bound_push": 1e-6,
            "ipopt.mu_init": 1e-3,
        }
        opti.solver("ipopt", opts)

    def solve(
        self,
        x0: np.ndarray,
        p_ref_traj: np.ndarray,
        q_ref_traj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x0 = np.asarray(x0).reshape(self.nx)
        p_ref_traj = np.asarray(p_ref_traj).reshape(3, self.N + 1)
        q_ref_traj = np.asarray(q_ref_traj).reshape(4, self.N + 1)

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.p_ref_param, p_ref_traj)
        self.opti.set_value(self.q_ref_param, q_ref_traj)

        if self._X_guess is None or self._U_guess is None:
            X_init = np.repeat(x0.reshape(-1, 1), self.N + 1, axis=1)
            U_init = np.zeros((self.nu, self.N), dtype=float)
        else:
            X_prev = self._X_guess
            U_prev = self._U_guess
            if X_prev.shape[1] == self.N + 1 and U_prev.shape[1] == self.N:
                X_init = np.hstack([X_prev[:, 1:], X_prev[:, [-1]]])
                U_init = np.hstack([U_prev[:, 1:], U_prev[:, [-1]]])
                X_init[:, 0] = x0
            else:
                X_init = np.repeat(x0.reshape(-1, 1), self.N + 1, axis=1)
                U_init = np.zeros((self.nu, self.N), dtype=float)

        self.opti.set_initial(self.X, X_init)
        self.opti.set_initial(self.U, U_init)
        if self._lam_g is not None:
            try:
                self.opti.set_initial(self.opti.lam_g, self._lam_g)
            except Exception:
                pass
        if self._lam_x is not None:
            try:
                self.opti.set_initial(self.opti.lam_x, self._lam_x)
            except Exception:
                pass

        try:
            sol = self.opti.solve()
        except Exception:
            self._lam_g = None
            self._lam_x = None
            sol = self.opti.solve()

        X_star = np.array(sol.value(self.X))
        U_star = np.array(sol.value(self.U))
        self._X_guess = X_star
        self._U_guess = U_star
        try:
            lam_g_val = np.array(sol.value(self.opti.lam_g))
            lam_x_val = np.array(sol.value(self.opti.lam_x))
            self._lam_g = lam_g_val
            self._lam_x = lam_x_val
        except Exception:
            self._lam_g = None
            self._lam_x = None
        return X_star, U_star
