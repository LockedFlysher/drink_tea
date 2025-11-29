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

    def fk_symbolic(self, x_full: ca.SX) -> Tuple[ca.SX, ca.SX]:
        """
        在 MPC 中的 FK：
          输入  x = [x_base, y_base, φ_base, q1..q6]^T
          输出  世界坐标下的末端位置/姿态 (p_ee, R_ee)

        计算步骤：
          1) 用 q_arm = x[3:9] 调用 Z1 的 FK，获得“臂基座”系下的 p_local, R_local
          2) 平面 base 变换: T_base(x,y,φ) = Trans(x,y,0) · RotZ(φ)
          3) 世界系下末端:
                R = R_base * R_local
                p = p_base + R_base * p_local
        """
        # 拆分平面 base 与 arm 关节
        x_base = x_full[0]
        y_base = x_full[1]
        phi_base = x_full[2]
        q_arm = x_full[3 : 3 + self.model.nq]

        # 机械臂 FK（在“臂基座”坐标系下）
        p_local = self.fk_arm_pos(q_arm)
        R_local = self.fk_arm_rot(q_arm)

        # 平面 base 变换
        c = ca.cos(phi_base)
        s = ca.sin(phi_base)
        R_base = ca.vertcat(
            ca.hcat([c, -s, 0]),
            ca.hcat([s,  c, 0]),
            ca.hcat([0,  0, 1]),
        )
        p_base = ca.vertcat(x_base, y_base, 0)

        # 世界坐标下的末端位置和姿态
        p_world = p_base + ca.mtimes(R_base, p_local)
        R_world = ca.mtimes(R_base, R_local)
        return p_world, R_world
    
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 工具函数：旋转矩阵 <-> 四元数，姿态误差，势函数                         #
# --------------------------------------------------------------------------- #

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
    左误差，相当于把当前姿态 R 转换到描述 R_ref 参考姿态坐标系 下，然后比较它们的差异
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
# 参考轨迹：从预生成的 NPZ 中加载并按时间采样                             #
# --------------------------------------------------------------------------- #

@dataclass
class ReferenceTrajectory:
    """
    从 npz 文件中加载预设的末端轨迹：
      - t_grid: (N,)             时间戳
      - p_ref: (N,3)             末端位置（世界坐标）
      - q_ref: (N,4)             末端姿态（四元数）
      - psi_grid: (N,)           yaw 角（绕 z 轴）
      - theta_grid: (N,)         pitch 角（绕 y 轴）  
      - phi_grid: (N,)           roll 角（绕 x 轴）
    """
    t_grid: np.ndarray
    p_ref: np.ndarray
    q_ref: np.ndarray
    psi_grid: np.ndarray    # yaw
    theta_grid: np.ndarray  # pitch
    phi_grid: np.ndarray    # roll
    period: float

    @classmethod
    def from_npz(cls, path: str = "z1_mpc_reference_traj.npz") -> "ReferenceTrajectory":
        try:
            data = np.load(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Reference trajectory file '{path}' not found.\n"
                "请先在仓库根目录运行：\n"
                "  python generate_z1_mpc_reference_traj_npz.py\n"
                "生成预设的参考轨迹。"
            ) from exc

        # -------------------------- 时间数组 --------------------------
        if "time" not in data:
            raise KeyError("NPZ 文件中缺少 'time' 数据")
        t_grid = np.asarray(data["time"], dtype=float).ravel()

        # -------------------------- 读取 x_e y_e z_e 拼成 p_ref --------------------------
        required_xyz = ["x_e", "y_e", "z_e"]
        for k in required_xyz:
            if k not in data:
                raise KeyError(f"NPZ 缺少 {k}")

        x_e = np.asarray(data["x_e"], float).ravel()
        y_e = np.asarray(data["y_e"], float).ravel()
        z_e = np.asarray(data["z_e"], float).ravel()

        # 检查长度一致
        if not (len(x_e) == len(y_e) == len(z_e) == len(t_grid)):
            raise ValueError("x_e, y_e, z_e 与 time 长度不一致")

        # 拼成 p_ref (N,3)
        p_ref = np.column_stack([x_e, y_e, z_e])

        # -------------------------- 姿态四元数 --------------------------
        if "quaternion" not in data:
            raise KeyError("NPZ 文件中缺少四元数 'quaternion'")
        q_ref = np.asarray(data["quaternion"], float)

        if q_ref.shape[0] != t_grid.shape[0] or q_ref.shape[1] != 4:
            raise ValueError("quaternion 形状应为 (N,4) 且与 time 对齐")

        # -------------------------- 欧拉角 --------------------------
        if 'euler_angles' not in data:
            raise KeyError("NPZ 文件中缺少 'euler_angles' 数据")

        euler_angles = np.asarray(data["euler_angles"], float)

        if euler_angles.ndim != 2 or euler_angles.shape[1] != 3:
            raise ValueError("euler_angles 应为 (N,3)")

        if euler_angles.shape[0] != t_grid.shape[0]:
            raise ValueError("euler_angles 长度必须与 time一致")

        # 拆分 [psi, theta, phi]
        psi_grid   = euler_angles[:, 0].ravel()
        theta_grid = euler_angles[:, 1].ravel()
        phi_grid   = euler_angles[:, 2].ravel()

        # -------------------------- 时间参数 --------------------------
        dt = float(t_grid[1] - t_grid[0])
        period = float(dt * (t_grid.size - 1))

        return cls(
            t_grid=t_grid,
            p_ref=p_ref,       # ← 现在是 (N,3)
            q_ref=q_ref,
            psi_grid=psi_grid,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
            period=period,
        )


    def sample(self, t_query: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        在时间 t_query 上插值得到 (p, q)：
          - 对位置做逐分量线性插值
          - 对欧拉角做线性插值，再转换为四元数用于mpc采样计算
        """
        t0 = float(self.t_grid[0])
        # 周期延拓
        t_mod = ((float(t_query) - t0) % self.period) + t0

        # 位置插值
        px = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 0]))
        py = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 1]))
        pz = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 2]))
        p = np.array([px, py, pz], dtype=float)

        # 欧拉角插值（直接从预计算的网格插值）
        psi = float(np.interp(t_mod, self.t_grid, self.psi_grid))
        theta = float(np.interp(t_mod, self.t_grid, self.theta_grid))
        phi = float(np.interp(t_mod, self.t_grid, self.phi_grid))

        # 欧拉角转四元数 (Z-Y-X 旋转顺序)
        # 分别计算半角的三角函数值
        cy = math.cos(psi * 0.5)
        sy = math.sin(psi * 0.5)
        cp = math.cos(theta * 0.5) 
        sp = math.sin(theta * 0.5)
        cr = math.cos(phi * 0.5)
        sr = math.sin(phi * 0.5)

        # Z-Y-X 顺序的四元数乘法: q = q_psi * q_theta * q_phi
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy  
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        q = np.array([qw, qx, qy, qz], dtype=float)

        # ---- 归一化四元数以防数值误差 ----
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            raise ValueError("Constructed quaternion has zero norm")
        q /= q_norm

        return p, q
    
# --------------------------------------------------------------------------- #
# Whole-body MPC 构建 (状态 9 维, 控制 9 维)                                  #
# --------------------------------------------------------------------------- #

@dataclass
class MPCConfig:    
    dt: float = 0.01
    horizon_steps: int = 20

    # ---- EE tracking weights ----
    w_pos: float = 1000.0     # 位置误差权重
    w_ori: float = 100.0      # 姿态误差权重

    # ---- control weight ----
    # R_u: float = 1.0          # 输入代价 u^T R u
    R_u: ca.DM = ca.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Example values
    
    # ---- relaxed log-barrier weights ----
    mu_barrier: float = 5e-3   # 势函数权重 μ
    delta_barrier: float = 1e-4  # 松弛 δ

    # ---- joint and velocity bounds ----
    q_min: float = -3.14
    q_max: float = 3.14
    dq_min: float = -10.0
    dq_max: float = 10.0


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
        # 状态/控制维度固定为 9（[x,y,φ,q1..q6]）
        self.nx = 9
        self.nu = 9
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

        # 参数
        x0_param = opti.parameter(nx)
        p_ref_param = opti.parameter(3, N + 1)
        q_ref_param = opti.parameter(4, N + 1)
        self.x0_param = x0_param
        self.p_ref_param = p_ref_param
        self.q_ref_param = q_ref_param

        # ---- 代价权重 ----
        w_pos = self.cfg.w_pos     # 位置误差权重
        w_ori = self.cfg.w_ori        # 姿态误差权重
        # R_u = self.cfg.R_u * ca.DM.eye(nu)
        R_u = self.cfg.R_u

        mu = self.cfg.mu_barrier
        delta = self.cfg.delta_barrier

        # 关节约束（arm 6 DOF）
        n_joints = 6
        q_min_vec = self.cfg.q_min * ca.DM.ones(n_joints, 1)
        q_max_vec = self.cfg.q_max * ca.DM.ones(n_joints, 1)
        dq_min_vec = self.cfg.dq_min * ca.DM.ones(n_joints, 1)
        dq_max_vec = self.cfg.dq_max * ca.DM.ones(n_joints, 1)

        # 初始条件
        opti.subject_to(X[:, 0] == x0_param)

        total_cost = 0

        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]

            # 离散动力学
            opti.subject_to(x_next == x_k + dt * u_k)

            # ---- EE FK ----
            p_ee_k, R_ee_k = self.robot.fk_symbolic(x_k)

            p_ref_k = p_ref_param[:, k]
            q_ref_k = q_ref_param[:, k]
            R_ref_k = quat_to_rot(q_ref_k)

            # ---- 位置误差 ----
            pos_err = p_ee_k - p_ref_k
            pos_cost = w_pos * ca.dot(pos_err, pos_err)

            # ---- 姿态误差（SO(3)）----
            ori_err = orientation_error_from_rot_matrices(R_ee_k, R_ref_k)
            ori_cost = w_ori * ca.dot(ori_err, ori_err)

            # EE 总误差
            C_ee = pos_cost + ori_cost

            # ---- joint constraints (soft using relaxed barrier) ----
            q_joint = ca.vertcat(x_k[3], x_k[4:9])
            dq_joint = ca.vertcat(u_k[3], u_k[4:9])

            # q_min ≤ q_joint → barrier(h = q_joint - q_min)
            B_q_min = relaxed_log_barrier(q_joint - q_min_vec, mu, delta)
            B_q_max = relaxed_log_barrier(q_max_vec - q_joint, mu, delta)

            B_dq_min = relaxed_log_barrier(dq_joint - dq_min_vec, mu, delta)
            B_dq_max = relaxed_log_barrier(dq_max_vec - dq_joint, mu, delta)

            B_joint = ca.sum1(B_q_min + B_q_max + B_dq_min + B_dq_max)
            B_joint = ca.sum1(ca.sum2(B_joint)) # force scalar

            # ---- 输入代价 uᵀRu ----
            effort = ca.mtimes([u_k.T, R_u, u_k])

            # ---- 第 k 步总代价 ----
            total_cost += C_ee + B_joint + effort

        opti.minimize(total_cost)

        # 求解器选项
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
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

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.p_ref_param, p_ref_traj)
        self.opti.set_value(self.q_ref_param, q_ref_traj)

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

    # 预设的末端轨迹（位置 + 姿态），从 NPZ 文件中加载
    ref_traj = ReferenceTrajectory.from_npz("trajectory_with_psi_correct_20251129_225627.npz")

    # 初始状态 x = [R_base, 0, 0, 0..0]（控制层坐标）
    R_base = 0.0
    x = np.zeros(9)
    x[0] = R_base
    x[1] = 0.0
    x[2] = 0.0  # φ_base

    # 用当前 x 初始化 MuJoCo 状态
    sim.reset_from_x(x, z_base=0.0)

    # 计算 Pinocchio 与 MuJoCo 之间的 EE 固定偏移，用于可视化对齐
    p_ee0_pin, _ = robot.fk_symbolic(ca.DM(x))
    p_ee0_pin = np.array(p_ee0_pin.full()).reshape(3)
    link06_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link06")
    p_ee0_mj = sim.data.xpos[link06_id].copy()
    ee_vis_offset = p_ee0_mj - p_ee0_pin

    dt_sim = 0.02  # 控制更新时间（独立于 MuJoCo 内部 dt）
    horizon_T = cfg.horizon_steps * cfg.dt

    print("Starting Z1 whole-body MPC demo. Close viewer to stop.")

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

            # 每 dt_sim 调用一次 MPC
            if t - last_mpc_time >= dt_sim:
                X_star, U_star = mpc.solve(x, p_ref_traj, q_ref_traj)
                u0 = U_star[:, 0]

                # 为了数值稳定，在应用到系统前对速度做简单剪裁
                v_xy_max = 10.0     # base 平移速度上限 [m/s]
                v_phi_max = 10.0    # base yaw 角速度上限 [rad/s]
                dq_max = 10.0       # 关节速度上限 [rad/s]

                u0_clipped = u0.copy()
                # base 线速度
                u0_clipped[0] = float(np.clip(u0_clipped[0], -v_xy_max, v_xy_max))
                u0_clipped[1] = float(np.clip(u0_clipped[1], -v_xy_max, v_xy_max))
                # base yaw 角速度
                u0_clipped[2] = float(np.clip(u0_clipped[2], -v_phi_max, v_phi_max))
                # 6 个关节速度
                u0_clipped[3:] = np.clip(u0_clipped[3:], -dq_max, dq_max)

                u0 = u0_clipped
                # 更新内部状态 (Euler)
                x = x + cfg.dt * u0
                last_mpc_time = t

                # 调试输出：末端执行器的当前位置与参考之间的误差
                p_ee_mpc, _ = robot.fk_symbolic(ca.DM(x))
                p_ee_mpc = np.array(p_ee_mpc.full()).reshape(3)
                p_ref_now = p_ref_traj[:, 0]
                ee_err = p_ee_mpc - p_ref_now
                print(
                    f"t={t:.2f}  EE pos={p_ee_mpc}  "
                    f"ref={p_ref_now}  err={ee_err}"
                )

            # 用当前 x 更新 MuJoCo 姿态
            sim.reset_from_x(x, z_base=0.0)

            # 可视化参考轨迹：在 user_scn 中画 EE 轨迹 (点+箭头)，基于预设轨迹
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

                    # 表示 EE yaw 参考的箭头（黄色）：箭头方向沿 yaw 朝向（XY 平面）
                    arrow_len = 0.12
                    yaw_e = ref_traj.psi_grid[idx_i]
                    dir_xy = np.array(
                        [math.cos(yaw_e), math.sin(yaw_e), 0.0], dtype=float
                    )
                    norm_dir = np.linalg.norm(dir_xy)
                    if norm_dir < 1e-6:
                        dir_xy = np.array([1.0, 0.0, 0.0])
                        norm_dir = 1.0
                    z_axis = dir_xy / norm_dir  # 作为局部 z 轴（箭头方向）
                    # 构造一个正交基：先取全局 z，再叉积
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
                        size=[0.005, 0.0075, arrow_len],
                        pos=pos_vis,
                        mat=R_vis.flatten(),
                        rgba=[1.0, 0.9, 0.1, 0.9],
                    )
                    geom_idx += 1

                user_scn.ngeom = geom_idx

            viewer.sync()


if __name__ == "__main__":
    run_z1_whole_body_mpc_demo()
