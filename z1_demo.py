from __future__ import annotations

import time
import os
import numpy as np
import casadi as ca
import mujoco
import mujoco.viewer

from z1mpc import RobotWrapper, MPCConfig, WholeBodyMPC, Z1MuJoCoSim, ReferenceTrajectory


def run() -> None:
    robot = RobotWrapper()
    cfg = MPCConfig()
    mpc = WholeBodyMPC(robot, cfg)
    sim = Z1MuJoCoSim()

    # Load reference trajectory from repo root
    root = os.path.abspath(os.path.dirname(__file__))
    ref_npz_path = os.path.join(root, "z1_mpc_reference_traj.npz")
    ref_traj = ReferenceTrajectory.from_npz(ref_npz_path)

    # Initialize at model keyframe 'home' for consistent starting pose
    sim.reset_keyframe("pos1")
    x, _ = sim.get_arm_state()

    # EE visualization offset between Pinocchio and MuJoCo (fallback to zero)
    try:
        p_ee0_pin, _ = robot.fk_symbolic(ca.DM(x))
        p_ee0_pin = np.array(p_ee0_pin.full()).reshape(3)
        link06_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "link06")
        p_ee0_mj = sim.data.xpos[link06_id].copy()
        ee_vis_offset = p_ee0_mj - p_ee0_pin
    except Exception:
        ee_vis_offset = np.zeros(3)

    q_lower = np.asarray(robot.model.lowerPositionLimit, dtype=float).reshape(-1)
    q_upper = np.asarray(robot.model.upperPositionLimit, dtype=float).reshape(-1)

    print("Starting Z1 MPC (6-DoF, fixed-base). Close viewer to stop.")
    print(
        f"PARAMS: dt={cfg.dt}, N={cfg.horizon_steps}, w_pos_vec="
        f"{getattr(cfg,'w_pos_vec', None)}, w_ori={cfg.w_ori}, kd={cfg.kd_arm}, R_u={cfg.R_u}"
    )

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        t0 = time.time()
        last_mpc_time = -1e9
        dt_sim = 0.02  # control update cadence
        while viewer.is_running():
            t = time.time() - t0

            # Build horizon references
            p_ref = np.zeros((3, cfg.horizon_steps + 1))
            q_ref = np.zeros((4, cfg.horizon_steps + 1))
            for k in range(cfg.horizon_steps + 1):
                tk = t + k * cfg.dt
                p_k, q_k = ref_traj.sample(tk)
                p_ref[:, k] = p_k
                q_ref[:, k] = q_k

            if t - last_mpc_time >= dt_sim:
                # State from simulator, clip to URDF bounds for OCP
                q_arm, v_arm = sim.get_arm_state()
                x = np.minimum(np.maximum(q_arm, q_lower), q_upper)

                X_star, U_star = mpc.solve(x, p_ref, q_ref)
                u0 = U_star[:, 0]
                # Clip joint velocities
                dq_max = 1.0
                u0 = np.clip(u0, -dq_max, dq_max)

                # Torque control with gravity compensation
                kd = cfg.kd_arm
                kd_vec = np.full(6, float(kd))
                g = robot.gravity(q_arm)
                tau_cmd = kd_vec * (u0 - v_arm) + g
                # Clamp torque per legacy behavior
                tau_limit = np.array([60.0, 60.0, 60.0, 60.0, 40.0, 40.0], dtype=float)
                tau_cmd = np.clip(tau_cmd, -tau_limit, tau_limit)
                sim.neutralize_actuators_to_q()
                sim.set_arm_torque(tau_cmd)

                # Integrate simulator to cover one control step
                mj_dt = float(sim.model.opt.timestep)
                sub = max(1, int(round(cfg.dt / mj_dt)))
                sim.step_n(sub)
                last_mpc_time = t

                # Debug EE error
                q_dbg, _ = sim.get_arm_state()
                p_ee, _ = robot.fk_symbolic(ca.DM(q_dbg))
                p_ee = np.array(p_ee.full()).reshape(3)
                err = p_ee - p_ref[:, 0]
                print(f"t={t:.2f} EE={p_ee} ref={p_ref[:,0]} err={err}")

            # Draw reference + current EE
            user_scn = getattr(viewer, "user_scn", None)
            if user_scn is not None:
                from z1mpc.viz import draw_reference_and_current
                q_now, _ = sim.get_arm_state()
                p_now, _ = robot.fk_symbolic(ca.DM(q_now))
                p_now = np.array(p_now.full()).reshape(3)
                draw_reference_and_current(viewer, user_scn, ref_traj.p_ref, ref_traj.q_ref, ee_vis_offset, p_now)

            viewer.sync()


if __name__ == "__main__":
    run()

