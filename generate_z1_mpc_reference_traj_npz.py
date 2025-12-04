from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
import numpy as np

@dataclass
class Z1ReferenceTrajectoryConfig:
    """
    配置 Z1 Whole-body MPC Demo 的参考轨迹。

    这里使用与 demo 中一致的椭圆轨迹：
      - 周期 T = 10s
      - 时间步长 dt_ref = 0.02s
      - 末端执行器在世界系中沿椭圆运动，并在 z 方向加入周期性起伏
      - 姿态仅绕世界 z 轴旋转（yaw）
    """
    dt_ref: float = 0.02
    period: float = 10.0
    ellipse_center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5]))
    a: float = 0.2  # x 方向半径
    b: float = 0.1  # y 方向半径
    z_amp: float = 0.1


def generate_reference_trajectory(cfg: Z1ReferenceTrajectoryConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成一段完整周期的参考轨迹：
      - t:    (N,)          时间戳
      - p_ref:(N,3)        末端位置
      - q_ref:(N,4)        末端姿态（四元数，yaw-only）
    """
    dt = float(cfg.dt_ref)
    period = float(cfg.period)

    num_steps = int(period / dt) + 1
    t = np.linspace(0.0, period, num_steps, dtype=float)

    p_ref = np.zeros((num_steps, 3), dtype=float)
    q_ref = np.zeros((num_steps, 4), dtype=float)

    cx, cy, cz = cfg.ellipse_center

    for i, ti in enumerate(t):
        theta_e = 2.0 * math.pi * ti / period

        x_e = cx + cfg.a * math.cos(theta_e)
        y_e = cy + cfg.b * math.sin(theta_e)
        z_e = cz + cfg.z_amp * math.sin(theta_e)
        p_ref[i] = np.array([x_e, y_e, z_e], dtype=float)

        yaw_e = theta_e
        qw = math.cos(yaw_e / 2.0)
        qz = math.sin(yaw_e / 2.0)
        q_ref[i] = np.array([qw, 0.0, 0.0, qz], dtype=float)

    return t, p_ref, q_ref


def main() -> None:
    cfg = Z1ReferenceTrajectoryConfig()
    t, p_ref, q_ref = generate_reference_trajectory(cfg)

    npz_path = "z1_mpc_reference_traj.npz"
    np.savez(npz_path, t=t, p_ref=p_ref, q_ref=q_ref)

    print(f"Saved reference trajectory to '{npz_path}'.")
    print(f"  t.shape      = {t.shape}")
    print(f"  p_ref.shape  = {p_ref.shape}")
    print(f"  q_ref.shape  = {q_ref.shape}")


if __name__ == "__main__":
    main()

