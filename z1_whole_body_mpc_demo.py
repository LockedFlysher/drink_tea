"""
Z1 Whole-body MPC Demo (Floating Base + 6-DOF Arm)
==================================================

目标：
  - 按论文给定的形式，实现一个基于速度层的 Whole-body MPC：

    状态 x ∈ R^9:
        x = [x_base, y_base, φ_base, q1, q2, q3, q4, q5, q6]^T
    控制 u ∈ R^9:
        u = [v_x, v_y, φ̇_base, q̇1, q̇2, q̇3, q̇4, q̇5, q̇6]^T
    动力学:
        ẋ = u   （离散化: x_{k+1} = x_k + dt * u_k）

  - 代价:
        J = ∫ ( C_ee(x) + L_B(x,u) + u^T R u ) dt
    其中:
      - C_ee: 末端执行器位置 + 姿态跟踪代价（基于 Pinocchio FK）
      - L_B: 对关节位置和速度约束的松弛对数势函数
      - R: 控制加权矩阵

  - Z1 的 MuJoCo 模型 `robot_description/z1_floating_base.xml` 用于可视化和仿真；
    Pinocchio 使用一个等价拓扑的“平面 base + 6 关节”合成模型做符号 FK。

依赖：
  - Python + CasADi (IPOPT solver)
  - Pinocchio + pinocchio.casadi
  - MuJoCo (Python bindings)
"""

from __future__ import annotations

import math
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
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pinocchio + pinocchio.casadi 未安装。请先安装，例如：\n"
        "  pip install pin\n"
    ) from exc


# --------------------------------------------------------------------------- #
# Pinocchio Robot Wrapper: planar base (x,y,φ) + 6-DOF arm                    #
# --------------------------------------------------------------------------- #


@dataclass
class RobotWrapper:
    """
    使用 Pinocchio 构造一个 9-DOF 的 planar base + 6-DOF arm 模型：
        q = [x, y, φ, q1..q6]
    只用于在 MPC 中求符号 FK 和末端姿态。
    """

    ee_frame_name: str = "ee"

    def __post_init__(self) -> None:
        self.model = self._build_synthetic_planar_model()
        self.data = self.model.createData()

        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.ee_frame_id = self.cmodel.getFrameId(self.ee_frame_name)
        if self.ee_frame_id < 0:
            raise ValueError(f"End-effector frame '{self.ee_frame_name}' not found.")

        # 构建 CasADi 符号 FK: p_ee(q), R_ee(q)
        q_sym = ca.SX.sym("q", self.model.nq)
        cpin.forwardKinematics(self.cmodel, self.cdata, q_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        placement = self.cdata.oMf[self.ee_frame_id]
        p_ee = placement.translation
        R_ee = placement.rotation

        self.fk_pos_fun = ca.Function("fk_pos", [q_sym], [p_ee])
        self.fk_rot_fun = ca.Function("fk_rot", [q_sym], [R_ee])

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    def fk_symbolic(self, q: ca.SX) -> Tuple[ca.SX, ca.SX]:
        p_ee = self.fk_pos_fun(q)
        R_ee = self.fk_rot_fun(q)
        return p_ee, R_ee

    def _build_synthetic_planar_model(self) -> pin.Model:
        """
        构造一个简单的 planar base + 6-DOF arm 模型：
            q = [x, y, φ, q1..q6]
        base: PX -> PY -> RZ
        arm: 6x RZ，沿 x 轴串联
        """
        model = pin.Model()
        parent_id = 0

        # base_x
        jx = pin.JointModelPX()
        jid_x = model.addJoint(parent_id, jx, pin.SE3.Identity(), "base_x")
        model.appendBodyToJoint(jid_x, pin.Inertia.Random(), pin.SE3.Identity())

        # base_y
        jy = pin.JointModelPY()
        jid_y = model.addJoint(jid_x, jy, pin.SE3.Identity(), "base_y")
        model.appendBodyToJoint(jid_y, pin.Inertia.Random(), pin.SE3.Identity())

        # base_yaw (RZ)
        jz = pin.JointModelRZ()
        base_joint_id = model.addJoint(jid_y, jz, pin.SE3.Identity(), "base_yaw")
        model.appendBodyToJoint(base_joint_id, pin.Inertia.Random(), pin.SE3.Identity())

        # 机械臂 6 关节
        link_parent = base_joint_id
        link_offset = 0.25
        for i in range(6):
            joint_name = f"joint{i+1}"
            link_name = f"link{i+1}"
            j = pin.JointModelRZ()
            X_pj = pin.SE3.Identity()
            X_pj.translation = np.array([link_offset, 0.0, 0.0])
            jid = model.addJoint(link_parent, j, X_pj, joint_name)
            model.appendBodyToJoint(jid, pin.Inertia.Random(), pin.SE3.Identity())
            model.addFrame(
                pin.Frame(
                    link_name,
                    jid,
                    0,
                    pin.SE3.Identity(),
                    pin.FrameType.BODY,
                )
            )
            link_parent = jid

        # 末端执行器 frame
        model.addFrame(
            pin.Frame(
                self.ee_frame_name,
                link_parent,
                0,
                pin.SE3.Identity(),
                pin.FrameType.OP_FRAME,
            )
        )

        if model.nq != 9:
            raise RuntimeError(f"Synthetic model nq={model.nq}, expected 9.")

        return model


# --------------------------------------------------------------------------- #
# 工具函数：旋转矩阵 <-> 四元数，姿态误差，势函数                         #
# --------------------------------------------------------------------------- #


def rot_to_quat(R: ca.SX) -> ca.SX:
    """
    旋转矩阵 R -> 四元数 q = [qw, qx, qy, qz]^T

    注意：本 demo 中 MPC 代价已改用基于旋转矩阵的姿态误差，
    该函数保留仅作潜在调试用途。
    """
    qw = ca.sqrt(ca.fmax(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw + 1e-9)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw + 1e-9)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw + 1e-9)
    return ca.vertcat(qw, qx, qy, qz)


def quat_to_rot(q: ca.SX) -> ca.SX:
    """
    四元数 q = [qw,qx,qy,qz]^T -> 旋转矩阵 R。
    使用多项式形式，数值上稳定。
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
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
    基于旋转矩阵的姿态误差（SO(3)）:
        e ≈ 0.5 * vee( R_ref^T R - R^T R_ref )
    e ∈ R^3

    在小角度下与四元数误差等价，数值上更稳定。
    """
    R_err = ca.mtimes([R_ref.T, R])
    skew = 0.5 * (R_err - R_err.T)
    return ca.vertcat(skew[2, 1], skew[0, 2], skew[1, 0])


def relaxed_log_barrier(h: ca.SX, mu: float, delta: float) -> ca.SX:
    """
    松弛对数势函数:
        h >= δ:  B(h) = -μ ln(h)
        h <  δ:  B(h) = μ/2 * (((h-2δ)/δ)^2 - 1) - μ ln(δ)

    为了数值稳定，这里采用一个“平滑近似”：
        - 先将 h 截断到不小于一个正数 ε，从而避免 log(≤0)
        - 并在小于 δ 的区域施加更高的惩罚
    """
    eps = 1e-6
    h_clipped = ca.fmax(h, eps)
    # 经典 log barrier
    base_barrier = -mu * ca.log(h_clipped)

    # 对 h < delta 额外加一个二次罚，近似原文的平滑段
    penalty = ca.fmax(delta - h, 0)
    smooth_term = 0.5 * mu * (penalty / delta) ** 2

    return base_barrier + smooth_term


# --------------------------------------------------------------------------- #
# Whole-body MPC 构建 (状态 9 维, 控制 9 维)                                  #
# --------------------------------------------------------------------------- #


@dataclass
class MPCConfig:
    horizon_steps: int = 20
    dt: float = 0.1
    w_pos: float = 10.0
    w_ori: float = 5.0
    R_u: float = 0.1
    mu_barrier: float = 1e-2
    delta_barrier: float = 1e-3

    # 关节和速度上下界（这里只对 φ_base + 6 个臂关节约束，x,y 不约束）
    q_min: float = -3.14
    q_max: float = 3.14
    dq_min: float = -2.0
    dq_max: float = 2.0


class WholeBodyMPC:
    """
    Whole-body MPC:
      - 状态 x ∈ R^9: [x, y, φ, q1..q6]
      - 控制 u ∈ R^9: [v_x, v_y, φ̇, q̇1..q̇6]
      - 动力学: x_{k+1} = x_k + dt * u_k
      - 代价: Σ ( C_ee(x_k) + L_B(x_k,u_k) + u_k^T R u_k )
    """

    def __init__(self, robot: RobotWrapper, cfg: MPCConfig) -> None:
        self.robot = robot
        self.cfg = cfg
        self.nx = robot.nq  # 9
        self.nu = robot.nq  # 9
        self.N = cfg.horizon_steps

        self._build_ocp()

    def _build_ocp(self) -> None:
        N = self.N
        nx = self.nx
        nu = self.nu
        dt = self.cfg.dt

        opti = ca.Opti()
        self.opti = opti

        # 决策变量
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        self.X = X
        self.U = U

        # 参数：
        #   x0: 当前状态
        #   p_ref_flat: 3*(N+1) 维向量，每个阶段的末端位置参考
        #   q_ref_flat: 4*(N+1) 维向量，每个阶段的末端姿态参考四元数
        x0_param = opti.parameter(nx)
        p_ref_flat = opti.parameter(3 * (N + 1))
        q_ref_flat = opti.parameter(4 * (N + 1))
        self.x0_param = x0_param
        self.p_ref_flat = p_ref_flat
        self.q_ref_flat = q_ref_flat

        w_pos = self.cfg.w_pos
        w_ori = self.cfg.w_ori
        R_u = self.cfg.R_u * ca.DM.eye(nu)

        # 只对 φ_base + q1..q6 施加约束（共 7 个量），忽略 x,y
        n_joints = 7
        q_min_vec = self.cfg.q_min * ca.DM.ones(n_joints, 1)
        q_max_vec = self.cfg.q_max * ca.DM.ones(n_joints, 1)
        dq_min_vec = self.cfg.dq_min * ca.DM.ones(n_joints, 1)
        dq_max_vec = self.cfg.dq_max * ca.DM.ones(n_joints, 1)

        mu_b = self.cfg.mu_barrier
        delta_b = self.cfg.delta_barrier

        # 初始条件
        opti.subject_to(X[:, 0] == x0_param)

        total_cost = 0

        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # 离散动力学
            opti.subject_to(x_next == x_k + dt * u_k)

            # 末端 FK
            p_ee_k, R_ee_k = self.robot.fk_symbolic(x_k)

            # 从参数中取出该阶段的参考 p_ref_k, q_ref_k
            p_ref_k = p_ref_flat[3 * k : 3 * (k + 1)]
            q_ref_k = q_ref_flat[4 * k : 4 * (k + 1)]
            R_ref_k = quat_to_rot(q_ref_k)

            # 位置误差
            pos_err = p_ee_k - p_ref_k

            # 姿态误差（SO(3)）
            ori_err = orientation_error_from_rot_matrices(R_ee_k, R_ref_k)

            C_ee_k = w_pos * ca.dot(pos_err, pos_err) + w_ori * ca.dot(
                ori_err, ori_err
            )

            # 关节 + base yaw 的索引：x[2] = φ_base, x[3:9] = q1..q6
            q_joint = ca.vertcat(x_k[2], x_k[3:9])
            dq_joint = ca.vertcat(u_k[2], u_k[3:9])

            # 约束统一形式: h = [q - q_min, q_max - q, dq - dq_min, dq_max - dq]
            h_q_low = q_joint - q_min_vec
            h_q_high = q_max_vec - q_joint
            h_dq_low = dq_joint - dq_min_vec
            h_dq_high = dq_max_vec - dq_joint
            h_vec = ca.vertcat(h_q_low, h_q_high, h_dq_low, h_dq_high)

            B_vec = relaxed_log_barrier(h_vec, mu_b, delta_b)
            L_B_k = ca.sum1(B_vec)

            # 控制能量
            effort_k = ca.mtimes([u_k.T, R_u, u_k])

            total_cost += C_ee_k + L_B_k + effort_k

        opti.minimize(total_cost)

        # 可以选做简单的硬约束（这里不再额外加，势函数已起到软约束作用）
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 80,
            "print_time": 0,
        }
        opti.solver("ipopt", opts)

    def solve(
        self,
        x0: np.ndarray,
        p_ref_traj: np.ndarray,
        q_ref_traj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解一次 MPC，返回 (X*, U*)。

        Args:
            x0: 当前状态 (9,)
            p_ref_traj: 末端位置参考 (3, N+1)
            q_ref_traj: 末端姿态参考四元数 (4, N+1)
        """
        x0 = np.asarray(x0).reshape(self.nx)
        p_ref_traj = np.asarray(p_ref_traj).reshape(3, self.N + 1)
        q_ref_traj = np.asarray(q_ref_traj).reshape(4, self.N + 1)

        p_ref_flat = p_ref_traj.reshape(-1)
        q_ref_flat = q_ref_traj.reshape(-1)

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.p_ref_flat, p_ref_flat)
        self.opti.set_value(self.q_ref_flat, q_ref_flat)

        sol = self.opti.solve()

        X_star = np.array(sol.value(self.X))
        U_star = np.array(sol.value(self.U))
        return X_star, U_star


# --------------------------------------------------------------------------- #
# MuJoCo 仿真封装：使用 z1_floating_base.xml 做可视化，按 x,u 回放       #
# --------------------------------------------------------------------------- #


class Z1MuJoCoSim:
    """
    使用 MuJoCo 的 z1_floating_base.xml 作为可视化模型。
    控制层维持一个独立的 9 维状态 x = [x,y,φ,q1..q6]，每次仿真步：
      - 用 MPC 的 x 更新这个“参考状态”
      - 将其映射到 MuJoCo 的 free joint + 6 关节 qpos
      - 使用 mj_forward 更新画面
    """

    def __init__(self, xml_path: str = "robot_description/z1_floating_base.xml") -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 关闭重力，做纯 kinematic 回放
        self.model.opt.gravity[:] = np.array([0.0, 0.0, 0.0])

        # 获取 free joint 索引用于写 qpos/qvel
        j_free = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "float_base_joint"
        )
        self.qpos_base = self.model.jnt_qposadr[j_free]
        self.qvel_base = self.model.jnt_dofadr[j_free]

        # 关节 joint1..joint6 的 qpos 起始索引
        self.joint_names = [f"joint{i+1}" for i in range(6)]
        self.joint_qpos_indices = [
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.joint_names
        ]

    def reset_from_x(self, x: np.ndarray, z_base: float = 0.3) -> None:
        """
        用控制层状态 x 重置 MuJoCo 中的 base + 臂姿态。
        """
        x = np.asarray(x).reshape(9)
        x_base, y_base, phi_base = x[0], x[1], x[2]
        q_arm = x[3:9]

        # free joint: [x, y, z, qw, qx, qy, qz]
        idx = self.qpos_base
        self.data.qpos[idx + 0] = x_base
        self.data.qpos[idx + 1] = y_base
        self.data.qpos[idx + 2] = z_base

        qw = math.cos(phi_base / 2.0)
        qz = math.sin(phi_base / 2.0)
        self.data.qpos[idx + 3] = qw
        self.data.qpos[idx + 4] = 0.0
        self.data.qpos[idx + 5] = 0.0
        self.data.qpos[idx + 6] = qz

        # arm joints
        for i, q_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[q_idx] = q_arm[i]

        mujoco.mj_forward(self.model, self.data)


# --------------------------------------------------------------------------- #
# 主循环：Whole-body MPC + MuJoCo 可视化                                      #
# --------------------------------------------------------------------------- #


def run_z1_whole_body_mpc_demo() -> None:
    """
    演示：
      - 用 WholeBodyMPC 让末端执行器沿 0.6m 高度的椭圆轨迹运动；
      - base 在平面上（x,y,φ）+ arm 6 关节共同运动，实现 EE 轨迹跟踪；
      - 控制层使用 Pinocchio+CasADi，仿真和可视化使用 MuJoCo Z1 模型。
    """
    robot = RobotWrapper()
    cfg = MPCConfig()
    mpc = WholeBodyMPC(robot, cfg)

    sim = Z1MuJoCoSim()

    # 初始状态 x = [R_base, 0, 0, 0..0]
    R_base = 0.3
    x = np.zeros(9)
    x[0] = R_base
    x[1] = 0.0
    x[2] = 0.0  # φ_base

    # 用当前 x 初始化 MuJoCo 状态
    sim.reset_from_x(x, z_base=0.3)

    # 1) 使用 Pinocchio FK 计算合成模型下的 EE 初始位置（控制层坐标）
    p_ee0_pin = robot.fk_pos_fun(ca.DM(x)).full().reshape(3)

    # 2) 使用 MuJoCo 计算 Z1 实际末端（这里用 body 'link06'）的初始位置（世界坐标）
    link06_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link06")
    p_ee0_mj = sim.data.xpos[link06_id].copy()

    # 3) 记录两个坐标系之间的平移偏移，用于对齐 MPC 参考与可视化
    #    令  p_world = p_pin + offset  ⇒ offset = p_world - p_pin
    offset_ee = p_ee0_mj - p_ee0_pin

    # 椭圆的可视化中心取 MuJoCo 下的 EE 初始位置
    ee_center_xy_world = np.array([float(p_ee0_mj[0]), float(p_ee0_mj[1])])

    dt_sim = 0.02  # 仿真步长（独立于 MuJoCo 内部 dt，这里只用于 MPC）
    horizon_T = cfg.horizon_steps * cfg.dt

    print("Starting Z1 whole-body MPC demo. Close viewer to stop.")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        t0 = time.time()
        last_mpc_time = -1e9

        while viewer.is_running():
            t = time.time() - t0

            # 构造 EE 参考椭圆轨迹：
            #   - 世界坐标下：以 p_ee0_mj 为中心的椭圆
            #   - 控制层（Pinocchio）坐标：减去 offset_ee，使几何上对齐
            a = 0.2
            b = 0.1
            z_ee = 0.6
            p_ref_traj = np.zeros((3, cfg.horizon_steps + 1))
            q_ref_traj = np.zeros((4, cfg.horizon_steps + 1))

            for k in range(cfg.horizon_steps + 1):
                tk = t + k * cfg.dt
                theta_e = 2.0 * math.pi * tk / 10.0  # 10s 一圈
                x_e_world = ee_center_xy_world[0] + a * math.cos(theta_e)
                y_e_world = ee_center_xy_world[1] + b * math.sin(theta_e)
                p_world = np.array([x_e_world, y_e_world, z_ee])

                # 把期望轨迹从世界坐标映射到 Pinocchio 控制层坐标
                p_pin = p_world - offset_ee
                p_ref_traj[:, k] = p_pin
                # 姿态参考：始终朝上（单位四元数）
                q_ref_traj[:, k] = np.array([1.0, 0.0, 0.0, 0.0])

            # 每 dt_sim 调用一次 MPC
            if t - last_mpc_time >= dt_sim:
                X_star, U_star = mpc.solve(x, p_ref_traj, q_ref_traj)
                u0 = U_star[:, 0]
                # 更新内部状态 (Euler)
                x = x + cfg.dt * u0
                last_mpc_time = t

            # 用当前 x 更新 MuJoCo 姿态
            sim.reset_from_x(x, z_base=0.3)

            # 可视化参考轨迹：在 user_scn 中画 EE 椭圆 (点+箭头)
            user_scn = getattr(viewer, "user_scn", None)
            if user_scn is not None:
                user_scn.ngeom = 0
                geom_idx = 0

                # EE 椭圆离散点（世界坐标，中心在 MuJoCo EE 初始位置）
                n_ellipse = 32
                for i in range(n_ellipse):
                    theta_e = 2.0 * math.pi * i / float(n_ellipse)
                    x_e = ee_center_xy_world[0] + a * math.cos(theta_e)
                    y_e = ee_center_xy_world[1] + b * math.sin(theta_e)
                    pos_e = np.array([x_e, y_e, z_ee])

                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.006, 0.0, 0.0],
                        pos=pos_e,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 0.4, 1.0, 0.8],
                    )
                    geom_idx += 1

                    arrow_len = 0.12
                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[0.005, 0.0075, arrow_len],
                        pos=pos_e,
                        mat=np.eye(3).flatten(),
                        rgba=[1.0, 0.9, 0.1, 0.9],
                    )
                    geom_idx += 1

                user_scn.ngeom = geom_idx

            viewer.sync()


if __name__ == "__main__":
    run_z1_whole_body_mpc_demo()
