"""Utility functions: quaternions, orientation errors."""

from __future__ import annotations

import casadi as ca


def rot_to_quat(R: ca.SX) -> ca.SX:
    qw = ca.sqrt(ca.fmax(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw + 1e-9)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw + 1e-9)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw + 1e-9)
    return ca.vertcat(qw, qx, qy, qz)


def quat_to_rot(q: ca.SX) -> ca.SX:
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
    qc = quat_normalize(q_curr)
    qr = quat_normalize(q_ref)
    q_err = quat_mul(quat_conj(qr), qc)
    s = ca.if_else(q_err[0] >= 0, 1.0, -1.0)
    return 2.0 * s * ca.vertcat(q_err[1], q_err[2], q_err[3])

