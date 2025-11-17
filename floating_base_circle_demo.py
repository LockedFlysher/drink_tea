"""
Floating Base Velocity-Control Demo
===================================

This minimal example answers three questions:

1. 用一个方块代表 floating base，如何在 MuJoCo 里用“速度控制”来控制它的 X / Y / ϕ？
2. 这个 base 只有一个平面转动角度 ϕ（绕 z 轴），其他两个角度不考虑。
3. 让 base 跟踪一个与 X–Y 平面平行、高度固定为 0.3 m 的圆轨迹。

模型约定
--------
- 使用一个简易 MJCF 字符串，不依赖外部 xml 文件。
- 基座 DOF：
    q = [x, y, ϕ]  （通过 slide + slide + hinge 三个关节实现）
  身体的 z 坐标在 <body pos="0 0 0.3"> 中固定为 0.3 m。

控制方式
--------
- 直接对关节速度 q̇ = [v_x, v_y, ϕ̇] 赋值（即“速度控制”），MuJoCo 积分更新 q。
- 参考轨迹：在 z = 0.3 的平面上绕圆运动：
      x_ref(t) = R cos(ω t)
      y_ref(t) = R sin(ω t)
  其中 R = 0.3 m，ω 为轨迹角速度。
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np


def build_z1_floating_base_model() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    使用仓库中的 robot_description/z1_floating_base.xml 作为
    base + 6-DOF 机械臂的测试模型。

    - base 是一个 free joint（float_base_joint），包含 6 DOF：
        qpos[0:3] = [x, y, z]
        qpos[3:7] = quaternion [qw, qx, qy, qz]
        qvel[0:3] = [ωx, ωy, ωz]
        qvel[3:6] = [vx, vy, vz]
    - arm 是 Z1 的 6 个关节 joint1..joint6。

    本 demo 只控制 base 的平移速度 (vx, vy)，并将 yaw 固定。
    """
    model = mujoco.MjModel.from_xml_path("robot_description/z1_floating_base.xml")
    data = mujoco.MjData(model)

    # 为了简化，可关闭重力，让 base+arm 悬浮运动
    model.opt.gravity[:] = np.array([0.0, 0.0, 0.0])

    return model, data


def get_z1_base_indices(model: mujoco.MjModel) -> Tuple[int, int]:
    """
    获取 Z1 浮动基座的 free joint 起始索引。

    返回:
        qpos_base: qpos 中 free joint 起始索引（预计为 0）
        qvel_base: qvel 中 free joint 起始索引（预计为 0）
    """
    j_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "float_base_joint")
    qpos_base = model.jnt_qposadr[j_free]
    qvel_base = model.jnt_dofadr[j_free]
    return qpos_base, qvel_base


def angle_wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class BaseVelocityMPC:
    """
    简单的 2D 速度层 MPC：
        x = [x, y]^T
        u = [v_x, v_y]^T
        x_{k+1} = x_k + dt * u_k
    目标：跟踪给定的参考轨迹 x_ref_k。
    """

    def __init__(self, horizon_steps: int, dt: float, v_max: float = 1.0):
        self.N = horizon_steps
        self.dt = dt
        self.v_max = v_max

        opti = ca.Opti()
        self.opti = opti

        # 决策变量
        X = opti.variable(2, self.N + 1)
        U = opti.variable(2, self.N)
        self.X = X
        self.U = U

        # 参数：当前状态和参考轨迹
        x0_param = opti.parameter(2)
        xref_param = opti.parameter(2, self.N + 1)
        self.x0_param = x0_param
        self.xref_param = xref_param

        # 动力学约束
        opti.subject_to(X[:, 0] == x0_param)
        for k in range(self.N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k + 1]
            opti.subject_to(x_next == x_k + dt * u_k)

        # 速度约束
        opti.subject_to(opti.bounded(-v_max, U, v_max))

        # 代价函数：位置误差 + 控制能量
        Q = ca.DM.eye(2)
        R = 0.1 * ca.DM.eye(2)
        cost = 0
        for k in range(self.N + 1):
            e_k = X[:, k] - xref_param[:, k]
            cost += ca.mtimes([e_k.T, Q, e_k])
            if k < self.N:
                u_k = U[:, k]
                cost += ca.mtimes([u_k.T, R, u_k])
        opti.minimize(cost)

        # 求解器
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 50,
            "print_time": 0,
        }
        opti.solver("ipopt", opts)

        # 初始解（用于 warm-start）
        self._X_init = np.zeros((2, self.N + 1))
        self._U_init = np.zeros((2, self.N))

    def solve(self, x0: np.ndarray, xref_traj: np.ndarray) -> np.ndarray:
        """
        求解一次 MPC，返回第一个控制量 u0。

        Args:
            x0: 当前状态 (2,) 或 (2,1)
            xref_traj: 参考轨迹 (2, N+1)
        """
        x0 = np.asarray(x0).reshape(2)
        xref_traj = np.asarray(xref_traj).reshape(2, self.N + 1)

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.xref_param, xref_traj)

        # warm start
        self.opti.set_initial(self.X, self._X_init)
        self.opti.set_initial(self.U, self._U_init)

        sol = self.opti.solve()

        X_opt = np.array(sol.value(self.X))
        U_opt = np.array(sol.value(self.U))

        # 缓存 warm-start
        self._X_init = X_opt
        self._U_init = U_opt

        u0 = U_opt[:, 0]
        return u0


def run_circle_tracking_demo() -> None:
    """
    使用 robot_description/z1_floating_base.xml 作为 base+机械臂模型，
    让 Z1 浮动基座在 X/Y 平面上以速度控制方式跟踪一个圆轨迹：
        - 圆心在 (0, 0)，半径 R = 0.3 m
        - 高度固定 z = 0.3 m
        - 角速度 ω 控制绕圈速度
        - yaw 角（通过 quaternion 表示）固定为初始值，不随轨迹变化
    """
    model, data = build_z1_floating_base_model()
    qpos_base, qvel_base = get_z1_base_indices(model)

    # free joint 对应的 qpos / qvel 索引
    idx_x = qpos_base + 0
    idx_y = qpos_base + 1
    idx_z = qpos_base + 2
    idx_quat_start = qpos_base + 3  # qw

    idx_wx = qvel_base + 0
    idx_wy = qvel_base + 1
    idx_wz = qvel_base + 2
    idx_vx = qvel_base + 3
    idx_vy = qvel_base + 4
    idx_vz = qvel_base + 5

    # 轨迹参数
    R = 0.3              # 圆半径 [m]
    period = 10.0        # 一圈时间 [s]
    omega_traj = 2 * math.pi / period  # 角速度 [rad/s]

    # MPC 配置（速度层）：2D x=[x,y], u=[vx,vy]
    sim_dt = model.opt.timestep          # MuJoCo 内部时间步长
    mpc_steps = 20                       # 每隔多少个 sim 步调用一次 MPC
    mpc_dt = sim_dt * mpc_steps          # MPC 离散时间步长
    mpc_horizon = 20                     # MPC 预测步长
    v_max = 1.0                          # 速度上界 [m/s]
    mpc = BaseVelocityMPC(horizon_steps=mpc_horizon, dt=mpc_dt, v_max=v_max)
    last_u = np.zeros(2)                 # 上一次 MPC 输出的速度
    sim_step = 0

    # 初始姿态：取默认 quaternion，但把 base 放到圆上、z 提高到 0.3
    initial_quat = data.qpos[idx_quat_start:idx_quat_start + 4].copy()
    initial_z = 0.3
    data.qpos[idx_x] = R
    data.qpos[idx_y] = 0.0
    data.qpos[idx_z] = initial_z
    data.qpos[idx_quat_start:idx_quat_start + 4] = initial_quat

    mujoco.mj_forward(model, data)

    t0 = time.time()

    print("Starting Z1 floating-base circle-tracking demo (velocity control).")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0
            sim_step += 1

            # 在 user_scn 中重画一圈离散圆点（base 参考圆轨迹）
            user_scn = getattr(viewer, "user_scn", None)
            if user_scn is not None:
                user_scn.ngeom = 0
                geom_idx = 0

                # base 参考圆轨迹（红色圆点）
                n_markers = 64
                for i in range(n_markers):
                    theta_m = 2.0 * math.pi * i / float(n_markers)
                    x_m = R * math.cos(theta_m)
                    y_m = R * math.sin(theta_m)
                    pos_m = np.array([x_m, y_m, initial_z])
                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.0075, 0.0, 0.0],  # 半径缩小一半
                        pos=pos_m,
                        mat=np.eye(3).flatten(),
                        rgba=[1.0, 0.0, 0.0, 0.7],
                    )
                    geom_idx += 1

                # 末端执行器参考轨迹：在 z=0.6 的椭圆（蓝点 + 朝上的黄箭头）
                a = 0.2   # 椭圆长轴
                b = 0.1   # 椭圆短轴
                z_ee = 0.6
                n_ellipse = 32
                for i in range(n_ellipse):
                    theta_e = 2.0 * math.pi * i / float(n_ellipse)
                    x_e = a * math.cos(theta_e)
                    y_e = b * math.sin(theta_e)
                    pos_e = np.array([x_e, y_e, z_ee])

                    # 椭圆上的离散点（蓝色小球）
                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.006, 0.0, 0.0],  # 半径缩小一半
                        pos=pos_e,
                        mat=np.eye(3).flatten(),
                        rgba=[0.0, 0.4, 1.0, 0.8],
                    )
                    geom_idx += 1

                    # 始终朝上的箭头（黄色），表示执行器朝向
                    arrow_len = 0.12
                    mujoco.mjv_initGeom(
                        user_scn.geoms[geom_idx],
                        type=mujoco.mjtGeom.mjGEOM_ARROW,
                        size=[0.005, 0.0075, arrow_len],  # 箭头“更细”，长度保持不变
                        pos=pos_e,
                        mat=np.eye(3).flatten(),  # 局部 z 轴对齐全局 z 轴 ⇒ 箭头朝上
                        rgba=[1.0, 0.9, 0.1, 0.9],
                    )
                    geom_idx += 1

                user_scn.ngeom = geom_idx

            # 1) 圆轨迹参考位置和速度（X/Y 平行于地面，z 固定在 initial_z）
            theta = omega_traj * t

            x_ref = R * math.cos(theta)
            y_ref = R * math.sin(theta)

            # 参考线速度（圆的导数）
            vx_ref = -R * omega_traj * math.sin(theta)
            vy_ref = R * omega_traj * math.cos(theta)

            # 2) 读取当前 base 状态
            x = data.qpos[idx_x]
            y = data.qpos[idx_y]

            # 3) 速度层 MPC：每 mpc_steps 步求解一次
            if sim_step % mpc_steps == 0:
                # 构造 MPC 参考轨迹 x_ref[k] = 圆上的点 (k=0..N)
                t0_mpc = t
                ref_traj = np.zeros((2, mpc_horizon + 1))
                for k in range(mpc_horizon + 1):
                    tk = t0_mpc + k * mpc_dt
                    theta_k = omega_traj * tk
                    ref_traj[0, k] = R * math.cos(theta_k)
                    ref_traj[1, k] = R * math.sin(theta_k)

                x0_vec = np.array([x, y])
                last_u = mpc.solve(x0_vec, ref_traj)

            v_x_cmd, v_y_cmd = last_u[0], last_u[1]

            # 4) 直接在 qpos 上用离散积分更新 base 的 x/y（纯 kinematic 方式）
            dt = model.opt.timestep
            data.qpos[idx_x] += dt * v_x_cmd
            data.qpos[idx_y] += dt * v_y_cmd

            # 保持 z 和 quaternion 固定，yaw 不随运动改变
            data.qpos[idx_z] = initial_z
            data.qpos[idx_quat_start:idx_quat_start + 4] = initial_quat

            # 只做前向运算更新几何位置（不走动力学积分）
            mujoco.mj_forward(model, data)

            # 可选：在终端打印一些状态，方便理解
            # print(f"t={t:.2f}, x={x:.3f}, y={y:.3f}")

            viewer.sync()


if __name__ == "__main__":
    run_circle_tracking_demo()
