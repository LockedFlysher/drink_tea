"""
Mobile Manipulator Whole-body MPC Demo
======================================

This script implements a Whole-body Reference Trajectory Planner (MPC) for a
planar mobile base + 6-DOF arm (total 9 DOF) using:
- CasADi: nonlinear optimization (IPOPT)
- Pinocchio + pinocchio.casadi: symbolic forward kinematics inside the OCP
- MuJoCo: simulation and visualization

Mathematical model (as in the problem description)
--------------------------------------------------
State (x ∈ R^9):
    x = [x_base, y_base, φ_base, φ_1, ..., φ_6]^T

Control (u ∈ R^9):
    u = [v_x, v_y, φ̇_base, φ̇_1, ..., φ̇_6]^T

System dynamics (kinematic model):
    ẋ = u
    x_{k+1} = x_k + dt * u_k       # Forward Euler discretization

Cost per stage (discrete-time approximation of Eq. (11)):
    L_k = C_ee(x_k) + L_B(x_k, u_k) + u_k^T R u_k

where
  - C_ee: end-effector position and orientation tracking cost
  - L_B: relaxed log-barrier on joint and velocity limits
  - R: positive definite control-weight matrix

NOTE:
- The URDF and MuJoCo XML paths are given as placeholders and must be adapted.
- The URDF is assumed to expose exactly 9 configuration DOF ordered as:
  [x_base, y_base, φ_base, q1, q2, q3, q4, q5, q6].
  You can adapt the mapping in RobotWrapper if your model differs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Tuple
import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np

try:
    import pinocchio as pin
    import pinocchio.casadi as cpin
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Pinocchio and pinocchio.casadi are required for this demo.\n"
        "Install with, e.g.:\n"
        "  pip install pin==3.*\n"
        "and ensure pinocchio.casadi is available."
    ) from exc


# --------------------------------------------------------------------------- #
# Robot wrapper: Pinocchio + CasADi FK                                        #
# --------------------------------------------------------------------------- #


@dataclass
class RobotWrapper:
    """
    Small helper around Pinocchio for symbolic FK in CasADi.

    The configuration vector q is assumed to be:
      q = [x_base, y_base, φ_base, q1, ..., q6] ∈ R^9
    matching the state x used in the MPC.
    """

    urdf_path: str
    end_effector_frame: str

    def __post_init__(self) -> None:
        """
        Initialize Pinocchio model.

        Priority:
        1) Try loading from URDF path (if provided and valid).
        2) If loading fails (e.g. MJCF file like z1_floating_base.xml),
           fall back to a synthetic planar base + 6-DOF arm model (9 DOF).
        """
        self.model = None
        # Try URDF if file exists
        if self.urdf_path and os.path.isfile(self.urdf_path):
            try:
                self.model = pin.buildModelFromUrdf(self.urdf_path)
            except Exception:
                print(
                    f"[RobotWrapper] Failed to load URDF from '{self.urdf_path}', "
                    "falling back to synthetic planar model."
                )
        else:
            if self.urdf_path:
                print(
                    f"[RobotWrapper] URDF path '{self.urdf_path}' not found, "
                    "falling back to synthetic planar model."
                )

        # Fallback: build a simple planar 3+6 DOF chain
        if self.model is None:
            self.model = self._build_synthetic_planar_model()

        self.data = self.model.createData()

        if self.model.nq != 9:
            raise ValueError(
                f"Expected model.nq == 9 for [base(3) + arm(6)], got {self.model.nq}"
            )

        # Create CasADi model/data to obtain symbolic FK (used inside the OCP).
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        # Cache end-effector frame index.
        ee_name = self.end_effector_frame or "ee"
        self.ee_frame_id = self.cmodel.getFrameId(ee_name)
        if self.ee_frame_id < 0:
            raise ValueError(
                f"End-effector frame '{ee_name}' not found in model."
            )

        # Build symbolic FK functions p_ee(q) and R_ee(q).
        q_sym = ca.SX.sym("q", self.cmodel.nq)

        cpin.forwardKinematics(self.cmodel, self.cdata, q_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)

        ee_placement = self.cdata.oMf[self.ee_frame_id]
        p_ee = ee_placement.translation  # position R^3
        R_ee = ee_placement.rotation     # rotation 3x3

        self.fk_pos_fun = ca.Function("fk_pos", [q_sym], [p_ee])
        self.fk_rot_fun = ca.Function("fk_rot", [q_sym], [R_ee])

    def _build_synthetic_planar_model(self) -> pin.Model:
        """
        Build a simple planar-base + 6-DOF arm model purely for MPC kinematics.

        Configuration:
            q = [x_base, y_base, φ_base, q1, ..., q6]

        - Base: 3 serial 1-DOF joints (PX, PY, RZ) for planar motion.
        - Arm: 6 revolute joints about z, links along x.
        - End-effector frame name: 'ee'
        """
        model = pin.Model()

        # Universe joint id is 0
        parent_id = 0

        # Base: x translation
        jx = pin.JointModelPX()
        jid_x = model.addJoint(parent_id, jx, pin.SE3.Identity(), "base_x")
        model.appendBodyToJoint(jid_x, pin.Inertia.Random(), pin.SE3.Identity())

        # Base: y translation
        jy = pin.JointModelPY()
        jid_y = model.addJoint(jid_x, jy, pin.SE3.Identity(), "base_y")
        model.appendBodyToJoint(jid_y, pin.Inertia.Random(), pin.SE3.Identity())

        # Base: yaw rotation
        jz = pin.JointModelRZ()
        base_joint_id = model.addJoint(jid_y, jz, pin.SE3.Identity(), "base_yaw")
        model.appendBodyToJoint(base_joint_id, pin.Inertia.Random(), pin.SE3.Identity())

        # Base frame
        model.addFrame(
            pin.Frame(
                "base_link",
                base_joint_id,
                0,
                pin.SE3.Identity(),
                pin.FrameType.BODY,
            )
        )

        # 6-DOF arm: revolute about z with links along x
        link_parent = base_joint_id
        link_offset = 0.25  # arbitrary link length

        for i in range(6):
            joint_name = f"joint{i+1}"
            link_name = f"link{i+1}"
            j = pin.JointModelRZ()
            X_pj = pin.SE3.Identity()
            # Translate along x by link_offset from previous joint
            X_pj.translation = np.array([link_offset, 0.0, 0.0])
            joint_id = model.addJoint(link_parent, j, X_pj, joint_name)

            # Attach a simple body
            model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())

            # Add frame for this link
            model.addFrame(
                pin.Frame(
                    link_name,
                    joint_id,
                    0,
                    pin.SE3.Identity(),
                    pin.FrameType.BODY,
                )
            )
            link_parent = joint_id

        # End-effector frame at the last link origin
        model.addFrame(
            pin.Frame(
                "ee",
                link_parent,
                0,
                pin.SE3.Identity(),
                pin.FrameType.OP_FRAME,
            )
        )

        if model.nq != 9:
            raise RuntimeError(
                f"Synthetic planar model has nq={model.nq}, expected 9."
            )

        print(
            f"[RobotWrapper] Built synthetic planar model with nq={model.nq}, nv={model.nv}"
        )
        return model

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    def fk_symbolic(
        self, x: ca.SX
    ) -> Tuple[ca.SX, ca.SX]:
        """
        Symbolic end-effector FK used inside the OCP.

        Args:
            x: configuration/state vector (q) as CasADi SX, shape (9,)

        Returns:
            p_ee: end-effector position (3x1 SX)
            R_ee: end-effector rotation matrix (3x3 SX)
        """
        p_ee = self.fk_pos_fun(x)
        R_ee = self.fk_rot_fun(x)
        return p_ee, R_ee

    def fk_numeric(
        self, x: ca.DM
    ) -> Tuple[ca.DM, ca.DM]:
        """
        Numeric FK using the standard Pinocchio model, useful for debugging.
        """
        q = ca.DM(x).full().flatten()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.ee_frame_id]
        return ca.DM(oMf.translation), ca.DM(oMf.rotation)


# --------------------------------------------------------------------------- #
# Utility functions: rotations, barrier, cost terms                           #
# --------------------------------------------------------------------------- #


def quat_to_rot(q: ca.SX) -> ca.SX:
    """
    Convert quaternion (w, x, y, z) to a 3x3 rotation matrix.
    Used to build the reference orientation in the tracking cost.
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Standard quaternion to rotation formula, built via concatenation
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)

    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)

    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    row0 = ca.hcat([r00, r01, r02])
    row1 = ca.hcat([r10, r11, r12])
    row2 = ca.hcat([r20, r21, r22])
    return ca.vertcat(row0, row1, row2)


def orientation_error_from_rot_matrices(R: ca.SX, R_ref: ca.SX) -> ca.SX:
    """
    SO(3) orientation error using rotation matrices.

    For small angles this is equivalent to the quaternion-based error
    (Eq. for e_ori) used in the paper:

        e_ori ≈ 0.5 * vee(R_refᵀ R - Rᵀ R_ref)

    which is proportional to the axis-angle representation of R_refᵀ R.
    """
    R_err = ca.mtimes([R_ref.T, R])
    skew = 0.5 * (R_err - R_err.T)
    # vee operator: map skew-symmetric matrix to R^3
    return ca.vertcat(skew[2, 1], skew[0, 2], skew[1, 0])


def quat_to_rot_np(q_np: "np.ndarray") -> "np.ndarray":
    """
    Quaternion (w, x, y, z) to rotation matrix using NumPy.
    Used only for visualization markers in MuJoCo.
    """
    qw, qx, qy, qz = q_np

    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)

    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)

    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])


def relaxed_log_barrier(h: ca.SX, mu: float, delta: float) -> ca.SX:
    """
    Relaxed log barrier function (scalar or elementwise on vectors).

    If h >= δ:  B(h) = -μ ln(h)
    If h <  δ:  B(h) = μ/2 * (( (h - 2δ)/δ )^2 - 1) - μ ln(δ)

    Implemented with casadi.if_else for smooth blending.
    """
    cond = h >= delta
    outside = -mu * ca.log(h)
    inside = (
        0.5 * mu * (((h - 2 * delta) / delta) ** 2 - 1.0)
        - mu * ca.log(delta)
    )
    return ca.if_else(cond, outside, inside)


# --------------------------------------------------------------------------- #
# MPC Planner: builds and solves the OCP using CasADi Opti                    #
# --------------------------------------------------------------------------- #


@dataclass
class MPCConfig:
    horizon_steps: int = 20
    dt: float = 0.1
    w_pos: float = 10.0
    w_ori: float = 5.0
    R_u: float = 0.1  # scalar factor for u^T R u (identity scaled)
    mu_barrier: float = 1e-2
    delta_barrier: float = 1e-3

    # Joint and velocity limits (per component)
    q_min: float = -3.14
    q_max: float = 3.14
    dq_min: float = -1.0
    dq_max: float = 1.0


class MPCPlanner:
    """
    Whole-body MPC planner for the planar base + 6-DOF arm.

    Builds a discrete-time optimal control problem:

        minimize sum_k (C_ee(x_k) + L_B(x_k, u_k) + u_k^T R u_k)
        subject to x_{k+1} = x_k + dt * u_k

    where C_ee encodes end-effector position + orientation tracking,
    L_B is the relaxed barrier on joint and velocity limits, and R
    is a positive definite weight on control effort.

    Implementation detail:
    ----------------------
    Instead of using `Opti` directly, we build an SX-based NLP and wrap it
    into a compiled CasADi `nlpsol` solver with IPOPT. The solver is built
    once in `__init__` and then reused at every MPC step (receding horizon).
    """

    def __init__(self, robot: RobotWrapper, config: MPCConfig) -> None:
        self.robot = robot
        self.cfg = config
        self.nx = robot.nq
        self.nu = robot.nq
        self.N = config.horizon_steps

        self._build_compiled_solver()

    def _build_compiled_solver(self) -> None:
        """
        Build SX-based NLP and create a compiled IPOPT solver.

        Decision variables (stacked into z):
          z = [X_0, ..., X_N, U_0, ..., U_{N-1}]

        Parameters:
          p = [x0, p_ref, q_ref]
        """
        N = self.N
        nx = self.nx
        nu = self.nu
        dt = self.cfg.dt

        # Sizes
        x_block_size = nx * (N + 1)
        u_block_size = nu * N
        nz = x_block_size + u_block_size

        # Decision variable and parameter
        z = ca.SX.sym("z", nz)
        p = ca.SX.sym("p", nx + 3 + 4)  # [x0 (nx), p_ref (3), q_ref (4)]

        # Slice parameters
        x0 = p[0:nx]
        p_ref = p[nx : nx + 3]
        q_ref = p[nx + 3 : nx + 7]

        # Helper lambdas to slice X_k and U_k out of z
        def X_k(k_idx: int) -> ca.SX:
            start = k_idx * nx
            return z[start : start + nx]

        def U_k(k_idx: int) -> ca.SX:
            start = x_block_size + k_idx * nu
            return z[start : start + nu]

        # Weights and limits
        w_pos = self.cfg.w_pos
        w_ori = self.cfg.w_ori
        R_u = self.cfg.R_u * ca.SX.eye(nu)

        q_min_vec = self.cfg.q_min * ca.SX.ones(nx, 1)
        q_max_vec = self.cfg.q_max * ca.SX.ones(nx, 1)
        dq_min_vec = self.cfg.dq_min * ca.SX.ones(nu, 1)
        dq_max_vec = self.cfg.dq_max * ca.SX.ones(nu, 1)

        mu_b = self.cfg.mu_barrier
        delta_b = self.cfg.delta_barrier

        # Constraint list (only equalities: initial condition + dynamics)
        g_list = []

        # Initial state constraint: X_0 - x0 = 0
        g_list.append(X_k(0) - x0)

        total_cost = 0

        for k in range(N):
            x_k = X_k(k)
            u_k = U_k(k)
            x_next = X_k(k + 1)

            # Dynamics: x_{k+1} = x_k + dt * u_k  (discretized ẋ = u)
            g_dyn = x_next - (x_k + dt * u_k)
            g_list.append(g_dyn)

            # End-effector FK: p_ee(x_k), R_ee(x_k)
            p_ee_k, R_ee_k = self.robot.fk_symbolic(x_k)

            # Tracking cost C_ee: position + orientation
            pos_err = p_ee_k - p_ref

            R_ref = quat_to_rot(q_ref)
            ori_err = orientation_error_from_rot_matrices(R_ee_k, R_ref)

            # Eq: C_ee = ||p_ee - p_ref||^2 + ||e_ori||^2 (with weighting)
            C_ee_k = w_pos * ca.dot(pos_err, pos_err) + w_ori * ca.dot(
                ori_err, ori_err
            )

            # Barrier term L_B on joint positions and velocities
            # Unified form:
            #   h = [q - q_min, q_max - q, u - dq_min, dq_max - u]
            h_q_low = x_k - q_min_vec
            h_q_high = q_max_vec - x_k
            h_dq_low = u_k - dq_min_vec
            h_dq_high = dq_max_vec - u_k
            h_vec = ca.vertcat(h_q_low, h_q_high, h_dq_low, h_dq_high)

            B_vec = relaxed_log_barrier(h_vec, mu_b, delta_b)
            L_B_k = ca.sum1(B_vec)

            # Control effort: u^T R u
            effort_k = ca.mtimes([u_k.T, R_u, u_k])

            # Eq (11) discrete-time cost: J = Σ (C_ee + L_B + uᵀ R u)
            stage_cost = C_ee_k + L_B_k + effort_k
            total_cost = total_cost + stage_cost

        # Stack constraints
        g = ca.vertcat(*g_list)

        # NLP definition
        nlp = {"x": z, "f": total_cost, "g": g, "p": p}

        # All constraints are equalities -> lbg = ubg = 0
        self.lbg = ca.DM.zeros(g.size1(), 1)
        self.ubg = ca.DM.zeros(g.size1(), 1)

        # Build compiled IPOPT solver
        solver_opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "print_time": 0,
            # Enable CasADi JIT compilation of the NLP for speed.
            # Use the 'shell' importer (available in this environment) so CasADi
            # compiles the generated C code via the system compiler.
            "jit": True,
            "compiler": "shell",
        }
        self.solver = ca.nlpsol("mpc_solver", "ipopt", nlp, solver_opts)

        # Optionally generate C code for offline compilation (one time):
        # self.solver.generate_dependencies("mpc_solver.c")

        # Store shapes for reshaping solution back to (X, U)
        self._x_block_size = x_block_size
        self._u_block_size = u_block_size

        # Simple initial guess: zero state/control
        self._z_init = ca.DM.zeros(nz, 1)

    def solve_mpc(
        self,
        x0: ca.DM,
        p_ref: ca.DM,
        q_ref: ca.DM,
        warm_start: bool = True,
    ) -> Tuple[ca.DM, ca.DM]:
        """
        Solve the MPC problem for the current state and reference pose.

        Args:
            x0: current state (9x1 DM or array-like)
            p_ref: reference end-effector position (3x1 DM or array-like)
            q_ref: reference end-effector quaternion (4x1 DM or array-like, [w, x, y, z])
            warm_start: whether to reuse the previous solution as initial guess

        Returns:
            (X_star, U_star): optimal state and control trajectories.
        """
        x0 = ca.DM(x0).reshape((self.nx, 1))
        p_ref = ca.DM(p_ref).reshape((3, 1))
        q_ref = ca.DM(q_ref).reshape((4, 1))

        P_val = ca.vertcat(x0, p_ref, q_ref)

        if warm_start:
            z0 = self._z_init
        else:
            z0 = ca.DM.zeros(self._x_block_size + self._u_block_size, 1)

        arg = {
            "x0": z0,
            "p": P_val,
            "lbg": self.lbg,
            "ubg": self.ubg,
        }

        sol = self.solver(**arg)
        z_opt = sol["x"]

        # Cache for warm-start in next MPC step
        self._z_init = z_opt

        # Reshape into X and U trajectories
        X_flat = z_opt[0 : self._x_block_size]
        U_flat = z_opt[self._x_block_size : self._x_block_size + self._u_block_size]

        X_star = ca.reshape(X_flat, self.nx, self.N + 1)
        U_star = ca.reshape(U_flat, self.nu, self.N)

        return X_star, U_star


# --------------------------------------------------------------------------- #
# MuJoCo simulation wrapper                                                   #
# --------------------------------------------------------------------------- #


class MuJoCoSim:
    """
    Simple MuJoCo wrapper to:
      - load a model
      - expose a 9-DOF state (base + arm) to match the MPC
      - apply velocity commands directly to the joints
    """

    def __init__(
        self,
        xml_path: str,
        nq_controlled: int = 9,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.nq_controlled = nq_controlled

        if self.model.nq < nq_controlled:
            raise ValueError(
                f"MuJoCo model has nq={self.model.nq}, but controller expects at least {nq_controlled} DOF."
            )

    def reset(self, x0) -> None:
        """
        Reset the MuJoCo state from a 9-D MPC state x0.

        Mapping for z1_floating_base.xml:
          qpos = [x, y, z, qw, qx, qy, qz, q1..q6]
          qvel = [ωx, ωy, ωz, vx, vy, vz, q̇1..q̇6]

        We embed:
          x_base  -> qpos[0]
          y_base  -> qpos[1]
          φ_base  -> quaternion (yaw about z)
          q1..q6  -> qpos[7:13]
        """
        x0 = ca.DM(x0).full().flatten()
        if x0.size != self.nq_controlled:
            raise ValueError(f"x0 must have size {self.nq_controlled}, got {x0.size}")

        # Base position
        self.data.qpos[0] = float(x0[0])
        self.data.qpos[1] = float(x0[1])
        # Fix base height
        self.data.qpos[2] = self.model.qpos0[2]

        # Base orientation: yaw-only quaternion
        yaw = float(x0[2])
        qw = float(ca.cos(yaw / 2))
        qz = float(ca.sin(yaw / 2))
        self.data.qpos[3] = qw
        self.data.qpos[4] = 0.0
        self.data.qpos[5] = 0.0
        self.data.qpos[6] = qz

        # Arm joints q1..q6
        self.data.qpos[7:13] = x0[3:9]

        # Zero velocities
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> ca.DM:
        """
        Return current MPC state x ∈ R^9 from MuJoCo.
        """
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()  # not strictly needed for ẋ = u model

        # Base position
        x_base = qpos[0]
        y_base = qpos[1]

        # Extract yaw from base quaternion
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        # Roll-pitch-yaw from quaternion (same convention as mujoco_viewer)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = float(ca.atan2(siny_cosp, cosy_cosp))

        # Arm joint positions
        q_arm = qpos[7:13]

        x_vec = [x_base, y_base, yaw] + list(q_arm)
        return ca.DM(x_vec)

    def apply_control(self, u, dt: float) -> None:
        """
        Apply base and joint velocities directly and advance the simulation.

        MPC command:
          u = [v_x, v_y, φ̇_base, q̇1..q̇6]

        z1_floating_base mapping (free joint):
          qvel = [ωx, ωy, ωz, vx, vy, vz, q̇1..q̇6]

        We embed:
          v_x      -> qvel[3]
          v_y      -> qvel[4]
          φ̇_base  -> qvel[2] (yaw angular velocity)
          q̇1..q̇6  -> qvel[6:12]
        """
        # Accept both CasADi DM and numpy arrays
        if isinstance(u, ca.DM):
            u_np = u.full().flatten()
        else:
            u_np = np.asarray(u, dtype=float).reshape(-1)
        if u_np.size != self.nq_controlled:
            raise ValueError(f"u must have size {self.nq_controlled}, got {u_np.size}")

        v_x = float(u_np[0])
        v_y = float(u_np[1])
        yaw_rate = float(u_np[2])
        qd_arm = u_np[3:9]

        # Base angular velocity (ωx, ωy, ωz)
        self.data.qvel[0] = 0.0
        self.data.qvel[1] = 0.0
        self.data.qvel[2] = yaw_rate

        # Base linear velocity (vx, vy, vz) in world frame
        self.data.qvel[3] = v_x
        self.data.qvel[4] = v_y
        self.data.qvel[5] = 0.0

        # Joint velocities
        self.data.qvel[6:12] = qd_arm

        # Step simulation for dt by integrating at MuJoCo's internal timestep.
        n_substeps = max(1, int(dt / self.model.opt.timestep))
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)


# --------------------------------------------------------------------------- #
# Main MPC + MuJoCo loop                                                      #
# --------------------------------------------------------------------------- #


def run_mpc_demo() -> None:
    """
    Run a simple closed-loop MPC demo with MuJoCo visualization.

    Assumptions:
      - URDF and MuJoCo XML describe the same mobile manipulator.
      - The end-effector frame name is known.
      - Base velocities [v_x, v_y, φ̇_base] and joint velocities [φ̇_1..6]
        are applied directly as joint velocities in MuJoCo.
    """

    # ------------------------------------------------------------------ #
    # Paths and configuration (adapt to your setup)                      #
    # ------------------------------------------------------------------ #
    # For this repository, we use the provided Z1 floating-base MJCF for MuJoCo.
    # There is no matching URDF here, so RobotWrapper will automatically
    # fall back to a synthetic planar base + 6-DOF arm model for FK.
    urdf_path = "robot_description/z1_floating_base.xml"  # used only to trigger fallback
    mj_xml_path = "robot_description/z1_floating_base.xml"
    ee_frame_name = "ee"  # synthetic model end-effector frame name

    robot = RobotWrapper(urdf_path=urdf_path, end_effector_frame=ee_frame_name)
    mpc_cfg = MPCConfig()
    planner = MPCPlanner(robot, mpc_cfg)

    sim = MuJoCoSim(xml_path=mj_xml_path, nq_controlled=robot.nq)

    # Initial state: all zeros by default.
    x0 = ca.DM.zeros(robot.nq, 1)
    sim.reset(x0.full().flatten())

    # Reference end-effector pose (position + quaternion).
    # Example: target position in front of the robot, identity orientation.
    p_ref = ca.DM([0.6, 0.0, 0.4])
    q_ref = ca.DM([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

    # Precompute reference pose for visualization (NumPy arrays)
    target_pos = np.asarray(p_ref.full()).reshape(-1)
    target_quat = np.asarray(q_ref.full()).reshape(-1)
    target_rot = quat_to_rot_np(target_quat)

    dt = mpc_cfg.dt
    print("Starting MPC + MuJoCo demo. Close the viewer window to stop.")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        last_time = time.time()
        iter_idx = 0
        while viewer.is_running():
            now = time.time()
            if now - last_time < dt:
                # Simple real-time pacing
                time.sleep(0.001)
                viewer.sync()
                continue
            last_time = now

            # Get current state from MuJoCo
            x_current = sim.get_state()

            # Update visualization of target pose (position + orientation)
            user_scn = getattr(viewer, "user_scn", None)
            if user_scn is not None and user_scn.maxgeom >= 2:
                # Sphere at target position
                g_sphere = user_scn.geoms[0]
                g_sphere.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g_sphere.size[:] = np.array([0.03, 0.0, 0.0])
                g_sphere.pos[:] = target_pos
                g_sphere.rgba[:] = np.array([0.1, 0.9, 0.1, 0.9])

                # Arrow showing target orientation (local +x axis)
                g_arrow = user_scn.geoms[1]
                g_arrow.type = mujoco.mjtGeom.mjGEOM_ARROW
                g_arrow.size[:] = np.array([0.01, 0.25, 0.01])
                g_arrow.pos[:] = target_pos
                g_arrow.mat[:] = target_rot
                g_arrow.rgba[:] = np.array([0.9, 0.1, 0.1, 0.9])

                user_scn.ngeom = max(user_scn.ngeom, 2)

            # Solve MPC (receding horizon) for current state
            t_solve_start = time.time()
            X_star, U_star = planner.solve_mpc(
                x0=x_current,
                p_ref=p_ref,
                q_ref=q_ref,
                warm_start=True,
            )
            t_solve_end = time.time()

            # Apply only the first control input (MPC receding horizon)
            u0 = U_star[:, 0]
            t_apply_start = time.time()
            sim.apply_control(u0, dt=dt)
            t_apply_end = time.time()

            iter_idx += 1
            solve_ms = (t_solve_end - t_solve_start) * 1000.0
            apply_ms = (t_apply_end - t_apply_start) * 1000.0
            total_ms = (t_apply_end - now) * 1000.0
            print(
                f"[MPC] iter {iter_idx:04d}: "
                f"solve = {solve_ms:.2f} ms, "
                f"apply+step = {apply_ms:.2f} ms, "
                f"total = {total_ms:.2f} ms"
            )

            viewer.sync()


if __name__ == "__main__":
    run_mpc_demo()
