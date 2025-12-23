import numpy as np
import math


def main() -> None:
    npzx_path= "optimization_results_onlyArm.npz"
    data = np.load(npzx_path)
    t = np.asarray(data["time"], dtype=float).ravel()
    xy_path = np.asarray(data["xy_path_data"], dtype=float)
    q_ref = np.asarray(data["quaternion"], dtype=float)
    # 固定 z 值
    fixed_z = 0.3  # 或者你需要的 z 值
    # 创建 z 值的列向量，形状 (N, 1)
    z_column = np.full((xy_path.shape[0], 1), fixed_z)
    # 水平拼接成 (N, 3)
    p_ref = np.hstack([xy_path, z_column])

    print(f"p_ref is '{p_ref}'.")
    print(f"q_ref is '{q_ref}'.")

    npzc_path = "z1_mpc_reference_traj.npz"
    np.savez(npzc_path, t=t, p_ref=p_ref, q_ref=q_ref)

    print(f"Saved reference trajectory to '{npzc_path}'.")
    print(f"  t.shape      = {t.shape}")
    print(f"  p_ref.shape  = {p_ref.shape}")
    print(f"  q_ref.shape  = {q_ref.shape}")

if __name__ == "__main__":
    main()
