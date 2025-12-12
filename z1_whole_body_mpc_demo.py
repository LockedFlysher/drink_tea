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
    使用 Pinocchio 从 Z1 的 URDF 构造「臂」模型，
    并在 MPC 中外加一个平面 base (x,y,φ)，从而实现：
        x = [x_base, y_base, φ_base, q1..q6] ∈ R^9

    这里：
      - Pinocchio 模型只负责 6-DOF 机械臂（固定在 world 上）
      - 平面 base 的变换由我们在 FK 里显式加上 SE(3) 变换
    """

    urdf_path: str = "robot_description/z1.urdf"
    ee_frame_name: str = "link06"  # Z1 末端 link 名

    def __post_init__(self) -> None:
        # 1) 从 URDF 构造固定基座的 Z1 模型（只包含 6 个关节）
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        # 2) CasADi 模型，用于符号 FK（q_arm 维度 = model.nq）
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.ee_frame_id = self.cmodel.getFrameId(self.ee_frame_name)
        if self.ee_frame_id < 0:
            raise ValueError(
                f"End-effector frame '{self.ee_frame_name}' not found in URDF model."
            )

        # 3) 以 q_arm 为变量构造符号 FK（末端在“臂基座”坐标系下的位置和姿态）
        q_arm_sym = ca.SX.sym("q_arm", self.model.nq)
        cpin.forwardKinematics(self.cmodel, self.cdata, q_arm_sym)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        placement = self.cdata.oMf[self.ee_frame_id]
        p_ee_local = placement.translation
        R_ee_local = placement.rotation

        self.fk_arm_pos = ca.Function("fk_arm_pos", [q_arm_sym], [p_ee_local])
        self.fk_arm_rot = ca.Function("fk_arm_rot", [q_arm_sym], [R_ee_local])

        if self.model.nq != 6:
            raise RuntimeError(
                f"Expected arm nq=6 for Z1, but URDF model reports nq={self.model.nq}."
            )

    @property
    def nq_arm(self) -> int:
        """Z1 机械臂关节自由度数（应为 6）"""
        return int(self.model.nq)

    def fk_symbolic(self, q_arm: ca.SX) -> Tuple[ca.SX, ca.SX]:
        """
        固定基：输入为 6 维关节向量 q_arm，返回世界坐标的末端位姿。
        """
        if int(q_arm.shape[0]) != self.model.nq:
            raise ValueError("fk_symbolic expects 6-dof arm q")
        p_local = self.fk_arm_pos(q_arm)
        R_local = self.fk_arm_rot(q_arm)
        return p_local, R_local


# --------------------------------------------------------------------------- #
# 工具函数：旋转矩阵 <-> 四元数，姿态误差，势函数                         #
# --------------------------------------------------------------------------- #


def rot_to_quat(R: ca.SX) -> ca.SX:
    """
    旋转矩阵 R -> 四元数 q = [qw, qx, qy, qz]^T
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


# ---- 四元数工具与误差（基于“减法/相对四元数”） ----
def quat_conj(q: ca.SX) -> ca.SX:
    return ca.vertcat(q[0], -q[1], -q[2], -q[3])

def quat_mul(q1: ca.SX, q2: ca.SX) -> ca.SX:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return ca.vertcat(w, x, y, z)

def quat_normalize(q: ca.SX) -> ca.SX:
    return q / (ca.sqrt(ca.dot(q, q)) + 1e-9)

def orientation_error_from_quats(q_curr: ca.SX, q_ref: ca.SX) -> ca.SX:
    """
    基于四元数“减法”的姿态误差（左误差）：
        q_err = conj(q_ref) ⊗ q_curr  对应 R_err = R_ref^T R
        e = 2 * sign(q_err.w) * q_err.xyz  ∈ R^3
    """
    qc = quat_normalize(q_curr)
    qr = quat_normalize(q_ref)
    q_err = quat_mul(quat_conj(qr), qc)
    s = ca.if_else(q_err[0] >= 0, 1.0, -1.0)
    return 2.0 * s * ca.vertcat(q_err[1], q_err[2], q_err[3])


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
# 参考轨迹：从预生成的 NPZ 中加载并按时间采样                             #
# --------------------------------------------------------------------------- #


@dataclass
class ReferenceTrajectory:
    """
    从 npz 文件中加载预设的末端轨迹（不使用欧拉角）：
      - t_grid: (N,)     时间戳
      - p_ref: (N,3)     末端位置（世界坐标）
      - q_ref: (N,4)     末端姿态（单位四元数）

    提供 sample(t_query)：位置线性插值，姿态用四元数 SLERP。
    """

    t_grid: np.ndarray
    p_ref: np.ndarray
    q_ref: np.ndarray
    period: float

    @classmethod
    def from_npz(cls, path: str = "z1_mpc_reference_traj.npz") -> "ReferenceTrajectory":
        try:
            data = np.load(path)
        except FileNotFoundError as exc:  # pragma: no cover
            raise FileNotFoundError(
                f"Reference trajectory file '{path}' not found.\n"
                "请先在仓库根目录运行：\n"
                "  python generate_z1_mpc_reference_traj_npz.py\n"
                "生成预设的参考轨迹。"
            ) from exc

        t_grid = np.asarray(data["t"], dtype=float).ravel()
        p_ref = np.asarray(data["p_ref"], dtype=float)
        q_ref = np.asarray(data["q_ref"], dtype=float)

        if t_grid.ndim != 1:
            raise ValueError("t_grid 应为一维数组。")
        if p_ref.shape[0] != t_grid.shape[0] or q_ref.shape[0] != t_grid.shape[0]:
            raise ValueError("p_ref / q_ref 与 t_grid 的长度不一致。")
        if p_ref.shape[1] != 3 or q_ref.shape[1] != 4:
            raise ValueError("p_ref 形状应为 (N,3)，q_ref 形状应为 (N,4)。")
        if t_grid.size < 2:
            raise ValueError("参考轨迹长度过短。")

        dt = float(t_grid[1] - t_grid[0])
        period = float(dt * (t_grid.size - 1))

        return cls(
            t_grid=t_grid,
            p_ref=p_ref,
            q_ref=q_ref,
            period=period,
        )

    def sample(self, t_query: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        在时间 t_query 上插值得到 (p, q)：
          - 位置：逐分量线性插值
          - 姿态：四元数 SLERP（避免欧拉角）
        """
        t0 = float(self.t_grid[0])
        t_mod = ((float(t_query) - t0) % self.period) + t0

        px = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 0]))
        py = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 1]))
        pz = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 2]))
        p = np.array([px, py, pz], dtype=float)

        # slerp between neighboring keyframes
        idx = int(np.searchsorted(self.t_grid, t_mod, side="right"))
        i1 = min(max(idx, 1), self.t_grid.size - 1)
        i0 = i1 - 1
        t0_i = float(self.t_grid[i0])
        t1_i = float(self.t_grid[i1])
        alpha = 0.0 if t1_i == t0_i else (t_mod - t0_i) / (t1_i - t0_i)
        q0 = self.q_ref[i0].astype(float)
        q1 = self.q_ref[i1].astype(float)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        dot = min(1.0, max(-1.0, dot))
        if dot > 0.9995:
            q = q0 + alpha * (q1 - q0)
            q /= np.linalg.norm(q)
        else:
            theta_0 = math.acos(dot)
            sin_0 = math.sin(theta_0)
            s0 = math.sin((1.0 - alpha) * theta_0) / sin_0
            s1 = math.sin(alpha * theta_0) / sin_0
            q = s0 * q0 + s1 * q1
        qn = np.linalg.norm(q)
        if qn == 0.0:
            q = q0
            qn = np.linalg.norm(q)
        q = q / qn
        return p, q


# --------------------------------------------------------------------------- #
# Whole-body MPC 构建 (状态 9 维, 控制 9 维)                                  #
# --------------------------------------------------------------------------- #


@dataclass
class MPCConfig:
    horizon_steps: int = 20
    dt: float = 0.1
    # EE 跟踪权重（位置 / 姿态）
    # 位置：x/y/z 一致的权重
    w_pos: float = 600.0      # EE 位置统一权重（xyz 一致）
    w_ori:    float = 50.0    # 末端姿态（yaw 等）跟踪权重，提高 yaw 跟踪优先级
    # base (x,y) 位置跟踪权重（目前不加入代价，只保留字段备用）
    w_base: float = 0.0
    R_u: float = 1.0
    # 以下两个参数仅在使用 barrier 版本时有用，
    # 当前实现已改为“硬约束”，不再使用松弛对数势。
    mu_barrier: float = 1e-2
    delta_barrier: float = 1e-3

    # 上下界：
    #   - 关节位置：仅约束 6 个臂关节（不对 base 的 yaw 角做上下界）
    #   - 速度：约束 φ̇_base + 6 个臂关节速度（x,y 不约束）
    q_min: float = -3.14
    q_max: float = 3.14
    dq_min: float = -1.0
    dq_max: float = 1.0
    # 关节速度控制增益（用于 tau = kd*(v_des - v) + tau_g）
    # 可用标量或 6 维向量；默认标量
    kd_arm: float = 20.0

#机械臂mpc
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
        # 固定基：只优化 6 个关节
        self.nx = int(self.robot.nq_arm)  # 6
        self.nu = self.nx                 # 6
        self.N = cfg.horizon_steps

        self._build_ocp()
        # 存储上一次解用于 warm-start
        self._X_guess: np.ndarray | None = None
        self._U_guess: np.ndarray | None = None
        # 存储上一次的对偶变量（拉格朗日乘子）
        self._lam_g: np.ndarray | None = None
        self._lam_x: np.ndarray | None = None

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
        #   x0: 当前状态 (9,)
        #   p_ref: 末端位置参考 (3, N+1)
        #   q_ref: 末端姿态参考四元数 (4, N+1)
        x0_param = opti.parameter(nx)
        p_ref_param = opti.parameter(3, N + 1)
        q_ref_param = opti.parameter(4, N + 1)
        self.x0_param = x0_param
        self.p_ref_param = p_ref_param
        self.q_ref_param = q_ref_param

        w_pos = self.cfg.w_pos
        w_ori = self.cfg.w_ori
        R_u = self.cfg.R_u * ca.DM.eye(nu)

        # 固定基：6 个臂关节的位置/速度上下界
        n_pos_joints = int(self.robot.nq_arm)
        # 从 Pinocchio 读取 URDF 的位置上下界（单位：rad 或 m，随关节类型）
        q_lower_np = np.asarray(self.robot.model.lowerPositionLimit, dtype=float).reshape(-1)
        q_upper_np = np.asarray(self.robot.model.upperPositionLimit, dtype=float).reshape(-1)
        if q_lower_np.size != n_pos_joints or q_upper_np.size != n_pos_joints:
            raise RuntimeError(
                f"Pinocchio limits size mismatch: got {q_lower_np.size}/{q_upper_np.size}, expected {n_pos_joints}."
            )
        q_min_vec = ca.DM(q_lower_np).reshape((n_pos_joints, 1))
        q_max_vec = ca.DM(q_upper_np).reshape((n_pos_joints, 1))
        # 速度上下界：
        #   - base φ̇ 使用配置（仍保留速度限制）
        #   - 6 个臂关节使用 Pinocchio 的 velocityLimit
        vel_lim_np = np.asarray(self.robot.model.velocityLimit, dtype=float).reshape(-1)
        if vel_lim_np.size != n_pos_joints:
            raise RuntimeError(
                f"Pinocchio velocityLimit size mismatch: got {vel_lim_np.size}, expected {n_pos_joints}."
            )
        dq_min_vec = ca.DM((-vel_lim_np).reshape((n_pos_joints, 1)))
        dq_max_vec = ca.DM(( vel_lim_np).reshape((n_pos_joints, 1)))

        # 初始条件
        opti.subject_to(X[:, 0] == x0_param)

        total_cost = 0

        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # 离散动力学：q_{k+1} = q_k + dt * dq_k
            opti.subject_to(x_next == x_k + dt * u_k)

            # 末端 FK（固定基，直接用 q_arm）
            p_ee_k, R_ee_k = self.robot.fk_symbolic(x_k)

            # 从参数矩阵中取出该阶段的参考 p_ref_k, q_ref_k
            p_ref_k = p_ref_param[:, k]
            q_ref_k = q_ref_param[:, k]

            # 位置误差（xyz 等权）
            pos_err = p_ee_k - p_ref_k
            pos_cost = w_pos * ca.dot(pos_err, pos_err)

            # 姿态误差（四元数减法，左误差）
            q_ee_k = rot_to_quat(R_ee_k)
            ori_err = orientation_error_from_quats(q_ee_k, q_ref_k)
            ori_cost = w_ori * ca.dot(ori_err, ori_err)

            C_ee_k = pos_cost + ori_cost

            # 关节位置/速度约束（6 维）
            q_joint = x_k
            dq_joint = u_k

            # 关节位置/速度的硬约束（取代 barrier）:
            #   q_min <= q_joint <= q_max
            #   dq_min <= dq_joint <= dq_max
            opti.subject_to(opti.bounded(q_min_vec, q_joint, q_max_vec))
            opti.subject_to(opti.bounded(dq_min_vec, dq_joint, dq_max_vec))

            # 固定基：无 base 变量，无需约束 base 速度

            # 控制能量
            effort_k = ca.mtimes([u_k.T, R_u, u_k])

            total_cost += C_ee_k + effort_k

        opti.minimize(total_cost)

        # 可以选做简单的硬约束（这里不再额外加，势函数已起到软约束作用）
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 80,
            "print_time": 0,
            # 允许 warm-start
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

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.p_ref_param, p_ref_traj)
        self.opti.set_value(self.q_ref_param, q_ref_traj)

        # 初值设置（warm start）
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
        # 复用对偶变量作为初值
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
            # 回退：清空对偶，重试
            self._lam_g = None
            self._lam_x = None
            sol = self.opti.solve()

        X_star = np.array(sol.value(self.X))
        U_star = np.array(sol.value(self.U))
        # 保存作为下一次初值
        self._X_guess = X_star
        self._U_guess = U_star
        # 保存对偶变量
        try:
            lam_g_val = np.array(sol.value(self.opti.lam_g))
            lam_x_val = np.array(sol.value(self.opti.lam_x))
            self._lam_g = lam_g_val
            self._lam_x = lam_x_val
        except Exception:
            self._lam_g = None
            self._lam_x = None
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

    def __init__(self, xml_path: str = "robot_description/z1.xml") -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 使用模型中配置的重力，由 MuJoCo 自己积分动力学（固定基模型，无 free joint）

        # 关节 joint1..joint6 的 qpos 起始索引
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

    def reset_from_x(self, x: np.ndarray, z_base: float = 0.3) -> None:
        """固定基：仅设置 6 个关节位置。"""
        q_arm = np.asarray(x, dtype=float).reshape(6)

        # arm joints
        for i, q_idx in enumerate(self.joint_qpos_indices):
            self.data.qpos[q_idx] = q_arm[i]

        mujoco.mj_forward(self.model, self.data)

    def neutralize_actuators_to_q(self) -> None:
        """将 actuator ctrl 设为当前关节角，使位置伺服误差为 0，避免抵消外加力矩。"""
        # 假设模型中 actuator 名为 motor1..motor6
        for i in range(6):
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i+1}")
            if act_id >= 0:
                q_idx = self.joint_qpos_indices[i]
                self.data.ctrl[act_id] = float(self.data.qpos[q_idx])

    def get_arm_state(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (q_arm, v_arm) from MuJoCo。"""
        q = np.array([self.data.qpos[idx] for idx in self.joint_qpos_indices], dtype=float)
        v = np.array([self.data.qvel[idx] for idx in self.joint_dof_indices], dtype=float)
        return q, v

    def set_arm_torque(self, tau: np.ndarray) -> None:
        """将 6 维关节力矩写入 qfrc_applied（覆盖/保持到下一次更新）。"""
        tau = np.asarray(tau, dtype=float).reshape(6)
        # 清零旧力
        for dof_idx in self.joint_dof_indices:
            self.data.qfrc_applied[dof_idx] = 0.0
        # 写入新力矩
        for i, dof_idx in enumerate(self.joint_dof_indices):
            self.data.qfrc_applied[dof_idx] = float(tau[i])

    def step_n(self, n: int = 1) -> None:
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)


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

    # 预设的末端轨迹（位置 + yaw 姿态），从 NPZ 文件中加载
    ref_traj = ReferenceTrajectory.from_npz("z1_mpc_reference_traj.npz")

    # 初始状态（固定基）：x = q_arm ∈ R^6
    x = np.zeros(6)

    # 用当前 x 初始化 MuJoCo 状态
    sim.reset_from_x(x, z_base=0.3)

    # 计算 Pinocchio 与 MuJoCo 之间的 EE 固定偏移，用于可视化对齐
    p_ee0_pin, _ = robot.fk_symbolic(ca.DM(x))
    p_ee0_pin = np.array(p_ee0_pin.full()).reshape(3)
    link06_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link06")
    p_ee0_mj = sim.data.xpos[link06_id].copy()
    ee_vis_offset = p_ee0_mj - p_ee0_pin

    dt_sim = 0.02  # 控制更新时间（独立于 MuJoCo 内部 dt）
    horizon_T = cfg.horizon_steps * cfg.dt

    print("Starting Z1 whole-body MPC demo. Close viewer to stop.")

    # 关节位置上下界（用于将 x0 裁剪回可行域，避免 IPOPT 因初值越界而 infeasible）
    q_lower = np.asarray(robot.model.lowerPositionLimit, dtype=float).reshape(-1)
    q_upper = np.asarray(robot.model.upperPositionLimit, dtype=float).reshape(-1)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        t0 = time.time()
        last_mpc_time = -1e9

        while viewer.is_running():
            t = time.time() - t0

            # 从预设轨迹中构造 MPC 参考（位置 + 姿态）
            p_ref_traj = np.zeros((3, cfg.horizon_steps + 1))
            q_ref_traj = np.zeros((4, cfg.horizon_steps + 1))

            for k in range(cfg.horizon_steps + 1):
                tk = t + k * cfg.dt
                p_k, q_k = ref_traj.sample(tk)
                p_ref_traj[:, k] = p_k
                q_ref_traj[:, k] = q_k

            # 每 dt_sim 更新一次控制并用 MuJoCo step 积分
            if t - last_mpc_time >= dt_sim:
                # 从 MuJoCo 读取当前臂状态
                q_arm_mj, v_arm_mj = sim.get_arm_state()
                # 将当前臂状态写回 x，并裁剪到 URDF 极限
                x = np.minimum(np.maximum(q_arm_mj, q_lower), q_upper)

                # 计算一次 MPC
                X_star, U_star = mpc.solve(x, p_ref_traj, q_ref_traj)
                u0 = U_star[:, 0]

                # 为了数值稳定，在应用到系统前对速度做简单剪裁
                dq_max = 1.0       # 关节速度上限 [rad/s]

                u0_clipped = u0.copy()
                # 6 个关节速度
                u0_clipped = np.clip(u0_clipped, -dq_max, dq_max)

                u0 = u0_clipped

                # --- 速度控制 + 重力补偿：tau = kd*(v_des - v_mj) + tau_g(pin) ---
                v_des = u0.copy()
                kd = cfg.kd_arm
                if np.isscalar(kd):
                    kd_vec = np.full(6, float(kd))
                else:
                    kd_arr = np.asarray(kd, dtype=float).reshape(-1)
                    if kd_arr.size != 6:
                        raise ValueError("cfg.kd_arm must be scalar or length-6 array")
                    kd_vec = kd_arr

                # Pinocchio 重力补偿（只计算 g，不计算 M/C）
                pin.computeGeneralizedGravity(robot.model, robot.data, q_arm_mj)
                g = robot.data.g.copy()

                tau_cmd = kd_vec * (v_des - v_arm_mj) + g
                # 简单的力矩限幅（必要时可调高/去掉）
                tau_limit = np.array([60.0, 60.0, 60.0, 60.0, 40.0, 40.0], dtype=float)
                tau_cmd = np.clip(tau_cmd, -tau_limit, tau_limit)
                # 先使 actuator 位置伺服误差为 0，避免抵消外加力
                sim.neutralize_actuators_to_q()
                sim.set_arm_torque(tau_cmd)

                # 固定基：不处理 base 位姿

                # 用 MuJoCo 自身的动力学积分若干子步以覆盖一个控制周期
                mj_dt = float(sim.model.opt.timestep)
                substeps = max(1, int(round(cfg.dt / mj_dt)))
                sim.step_n(substeps)

                last_mpc_time = t

                # 调试输出（基于最新状态）
                q_arm_dbg, _ = sim.get_arm_state()
                p_ee_mpc, _ = robot.fk_symbolic(ca.DM(q_arm_dbg))
                p_ee_mpc = np.array(p_ee_mpc.full()).reshape(3)
                p_ref_now = p_ref_traj[:, 0]
                ee_err = p_ee_mpc - p_ref_now
                print(
                    f"t={t:.2f}  EE pos={p_ee_mpc}  "
                    f"ref={p_ref_now}  err={ee_err}"
                )

            # 可视化参考轨迹：在 user_scn 中画 EE 轨迹 (点+箭头)
            user_scn = getattr(viewer, "user_scn", None)
            if user_scn is not None:
                user_scn.ngeom = 0
                geom_idx = 0

                # 轨迹点：离散的蓝点 + 表示 yaw 的黄箭头
                n_points = min(64, ref_traj.p_ref.shape[0])
                indices = np.linspace(0, ref_traj.p_ref.shape[0] - 1, n_points).astype(
                    int
                )

                for idx_i in indices:
                    pos_world = ref_traj.p_ref[idx_i]
                    q_world = ref_traj.q_ref[idx_i]

                    # 将 Pinocchio 世界坐标转换到 MuJoCo 世界坐标用于可视化
                    pos_vis = pos_world + ee_vis_offset

                    # 轨迹上的离散点（蓝色小球）
                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.006, 0.0, 0.0],
                        pos=pos_vis,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 0.4, 1.0, 0.8],
                    )
                    geom_idx += 1

                    # 从四元数得到箭头：取局部 x 轴在世界坐标中的方向，投影到 XY 平面
                    Rw = np.array(quat_to_rot(ca.DM(q_world)).full()).reshape(3, 3)
                    dir3 = Rw @ np.array([1.0, 0.0, 0.0])
                    dir_xy = np.array([dir3[0], dir3[1], 0.0])
                    norm_dir = np.linalg.norm(dir_xy)
                    if norm_dir < 1e-6:
                        dir_xy = np.array([1.0, 0.0, 0.0])
                        norm_dir = 1.0
                    z_axis = dir_xy / norm_dir
                    up = np.array([0.0, 0.0, 1.0])
                    x_axis = np.cross(up, z_axis)
                    norm_x = np.linalg.norm(x_axis)
                    if norm_x < 1e-6:
                        x_axis = np.array([1.0, 0.0, 0.0])
                        norm_x = 1.0
                    x_axis /= norm_x
                    y_axis = np.cross(z_axis, x_axis)
                    R_vis = np.column_stack([x_axis, y_axis, z_axis]).astype(float)

                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[0.005, 0.0075, 0.12],
                        pos=pos_vis,
                        mat=R_vis.flatten(),
                        rgba=[1.0, 0.9, 0.1, 0.9],
                    )
                    geom_idx += 1

                # 当前（用于优化的）末端位置：用当前 MuJoCo 关节 + Pinocchio FK
                q_arm_now, _ = sim.get_arm_state()
                p_ee_now, _ = robot.fk_symbolic(ca.DM(q_arm_now))
                p_ee_now = np.array(p_ee_now.full()).reshape(3)
                pos_vis_now = p_ee_now + ee_vis_offset

                mujoco.mjv_initGeom(
                    user_scn.geoms[geom_idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.01, 0.0, 0.0],
                    pos=pos_vis_now,
                    mat=np.eye(3).flatten(),
                    rgba=[1.0, 0.0, 0.0, 1.0],  # 红点，表示“参与优化的 EE 位置”
                )
                geom_idx += 1

                user_scn.ngeom = geom_idx

            viewer.sync()


if __name__ == "__main__":
    run_z1_whole_body_mpc_demo()
