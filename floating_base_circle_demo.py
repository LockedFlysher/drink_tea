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

import mujoco
import mujoco.viewer
import numpy as np


def build_planar_base_model() -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    创建一个带 3 个自由度 (x, y, ϕ) 的方块 base，并直接在 MJCF 里
    用一圈红色小球离散表示目标圆轨迹。

    MJCF 结构：
        - slide joint: x
        - slide joint: y
        - hinge joint: yaw (ϕ, 绕 z 轴)
        - geom: box（作为可视化）
        - 多个 sphere geom：红色小球离散表示参考圆轨迹
    """
    R = 0.3          # 圆半径 [m]
    z = 0.3          # 圆所在高度 [m]
    n_markers = 64   # 离散点个数

    marker_geoms = []
    for i in range(n_markers):
        theta = 2.0 * math.pi * i / float(n_markers)
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        marker_geoms.append(
            f'<geom name="traj_marker_{i}" type="sphere" size="0.015 0.015 0.015" '
            f'pos="{x:.4f} {y:.4f} {z:.4f}" rgba="1 0 0 0.8"/>'
        )
    markers_xml = "\n        ".join(marker_geoms)

    mjcf = f"""
    <mujoco>
      <option timestep="0.01" gravity="0 0 0"/>
      <worldbody>
        <!-- 简单地板和光源 -->
        <light name="sun" pos="0 0 2.0" dir="0 0 -1" diffuse="1 1 1" specular="0.1 0.1 0.1"/>
        <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0"
              rgba="0.8 0.8 0.8 1.0"/>

        <!-- 参考圆轨迹的离散点（红色小球） -->
        {markers_xml}

        <!-- 浮动基座（方块），z 固定 0.3 m -->
        <body name="base" pos="0 0 0.3">
          <joint name="base_x" type="slide" axis="1 0 0" />
          <joint name="base_y" type="slide" axis="0 1 0" />
          <joint name="base_yaw" type="hinge" axis="0 0 1" />
          <geom name="base_box" type="box" size="0.05 0.05 0.05" rgba="0.2 0.4 0.8 1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(mjcf)
    data = mujoco.MjData(model)
    return model, data


def get_base_dof_indices(model: mujoco.MjModel) -> Tuple[int, int, int, int, int, int]:
    """
    获取 x, y, yaw 三个关节在 qpos 和 qvel 中的下标。
    """
    j_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")
    j_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")
    j_yaw = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")

    qx = model.jnt_qposadr[j_x]
    qy = model.jnt_qposadr[j_y]
    qphi = model.jnt_qposadr[j_yaw]

    vx = model.jnt_dofadr[j_x]
    vy = model.jnt_dofadr[j_y]
    vphi = model.jnt_dofadr[j_yaw]

    return qx, qy, qphi, vx, vy, vphi


def angle_wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def run_circle_tracking_demo() -> None:
    """
    让方块 base 以速度控制方式跟踪一个平面圆轨迹：
        - 圆心在 (0, 0)，半径 R = 0.3 m
        - 高度固定 z = 0.3 m（由 body 初始 pos 决定）
        - 角速度 ω 控制绕圈速度
        - yaw 角 ϕ 固定为 0，不随轨迹变化
    """
    model, data = build_planar_base_model()
    qx, qy, qphi, vx, vy, vphi = get_base_dof_indices(model)

    # 轨迹参数
    R = 0.3              # 圆半径 [m]
    period = 10.0        # 一圈时间 [s]
    omega_traj = 2 * math.pi / period  # 角速度 [rad/s]

    # 简单位置反馈增益（调大/调小可以观察效果）
    Kp_pos = 1.5

    # 初始状态设在圆上，方便观察
    data.qpos[qx] = R
    data.qpos[qy] = 0.0
    data.qpos[qphi] = 0.0  # yaw 固定为 0
    mujoco.mj_forward(model, data)

    t0 = time.time()

    print("Starting floating base circle-tracking demo (velocity control).")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time() - t0

            # 1) 圆轨迹参考位置和速度（X/Y 平行于地面，z 固定在 0.3）
            theta = omega_traj * t

            x_ref = R * math.cos(theta)
            y_ref = R * math.sin(theta)

            # 参考线速度（圆的导数）
            vx_ref = -R * omega_traj * math.sin(theta)
            vy_ref = R * omega_traj * math.cos(theta)

            # 2) 读取当前状态
            x = data.qpos[qx]
            y = data.qpos[qy]

            # 3) 简单的 P 控制：速度 = 参考速度 + 位置误差 * 增益
            ex = x_ref - x
            ey = y_ref - y
            v_x_cmd = vx_ref + Kp_pos * ex
            v_y_cmd = vy_ref + Kp_pos * ey

            # 4) 把命令直接写入 qvel（即“速度控制”）
            data.qvel[vx] = v_x_cmd
            data.qvel[vy] = v_y_cmd
            # yaw 角速度固定为 0，使 ϕ 始终保持为初始值 0
            data.qvel[vphi] = 0.0

            # 5) 用 MuJoCo 的时间步长积分一次
            mujoco.mj_step(model, data)

            # 可选：在终端打印一些状态，方便理解
            # print(f"t={t:.2f}, x={x:.3f}, y={y:.3f}, phi={phi:.2f}")

            viewer.sync()


if __name__ == "__main__":
    run_circle_tracking_demo()
