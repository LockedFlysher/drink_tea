"""Viewer helpers: draw reference points, arrows, and current EE red dot."""

from __future__ import annotations

import numpy as np
import mujoco


def draw_reference_and_current(viewer, user_scn, p_ref_all: np.ndarray, q_ref_all: np.ndarray,
                               pos_offset: np.ndarray, current_pos_world: np.ndarray) -> None:
    user_scn.ngeom = 0
    geom_idx = 0

    # Subsample reference for display
    n_points = min(64, p_ref_all.shape[0])
    indices = np.linspace(0, p_ref_all.shape[0] - 1, n_points).astype(int)

    for idx_i in indices:
        pos_world = p_ref_all[idx_i]
        q_world = q_ref_all[idx_i]

        pos_vis = pos_world + pos_offset

        mujoco.mjv_initGeom(
            user_scn.geoms[geom_idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.006, 0.0, 0.0],
            pos=pos_vis,
            mat=np.eye(3).flatten(),
            rgba=[0.0, 0.4, 1.0, 0.8],
        )
        geom_idx += 1

        # Simple arrow on XY plane using quaternion's projection
        # Build rotation matrix from quaternion (minimal; we don't need exact here)
        qw, qx, qy, qz = q_world
        # Direction of body x-axis in world (from quaternion)
        # Using standard conversion for x-axis of rotation matrix
        dir3 = np.array([
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ])
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

    # Current EE (red dot)
    pos_vis_now = current_pos_world + pos_offset
    mujoco.mjv_initGeom(
        user_scn.geoms[geom_idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.01, 0.0, 0.0],
        pos=pos_vis_now,
        mat=np.eye(3).flatten(),
        rgba=[1.0, 0.0, 0.0, 1.0],
    )
    geom_idx += 1

    user_scn.ngeom = geom_idx

