"""Reference trajectory loader for EE position + orientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import math


@dataclass
class ReferenceTrajectory:
    t_grid: np.ndarray
    p_ref: np.ndarray
    q_ref: np.ndarray
    period: float

    @classmethod
    def from_npz(cls, path: str = "z1_mpc_reference_traj.npz") -> "ReferenceTrajectory":
        data = np.load(path)
        t_grid = np.asarray(data["t"], dtype=float).ravel()
        p_ref = np.asarray(data["p_ref"], dtype=float)
        q_ref = np.asarray(data["q_ref"], dtype=float)
        dt = float(t_grid[1] - t_grid[0])
        period = float(dt * (t_grid.size - 1))
        return cls(t_grid=t_grid, p_ref=p_ref, q_ref=q_ref, period=period)

    def sample(self, t_query: float) -> Tuple[np.ndarray, np.ndarray]:
        t0 = float(self.t_grid[0])
        t_mod = ((float(t_query) - t0) % self.period) + t0
        px = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 0]))
        py = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 1]))
        pz = float(np.interp(t_mod, self.t_grid, self.p_ref[:, 2]))
        p = np.array([px, py, pz], dtype=float)

        # slerp between frames (pose is constant in current generator, but keep it generic)
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

