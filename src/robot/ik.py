"""Franka DH forward kinematics and numerical inverse kinematics."""

import numpy as np

_DH_A     = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
_DH_D     = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
_DH_ALPHA = np.array([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])
_DH_OFF   = np.array([0, 0, 0, 0, 0, 0, 0, -np.pi/4])
_Q_MIN    = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_Q_MAX    = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


def _log3(R: np.ndarray) -> np.ndarray:
    c  = np.clip(np.trace(R) / 2 - 0.5, -1.0, 1.0)
    th = np.arccos(c)
    return np.zeros((3, 3)) if th == 0 else th / np.sin(th) * (R - R.T) / 2


def _vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _dh_mat(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha),  np.sin(alpha)
    return np.array([
        [ct,    -st,     0,   a    ],
        [st*ca,  ct*ca, -sa, -sa*d ],
        [st*sa,  ct*sa,  ca,  ca*d ],
        [0,      0,      0,   1    ],
    ])


def franka_fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics: joint angles (7,) → EE pose (4×4)."""
    T = np.eye(4)
    for i in range(8):
        theta = (_DH_OFF[i] + q[i]) if i < 7 else _DH_OFF[i]
        T = T @ _dh_mat(_DH_A[i], _DH_D[i], _DH_ALPHA[i], theta)
    return T


def franka_ik(T_target: np.ndarray, q_init: np.ndarray,
              max_iter: int = 400, tol: float = 5e-4,
              q7_fixed=None):
    """Damped-least-squares IK with optional J7 pinning. Returns (q, converged)."""
    eps = 1e-6; lam = 0.05; ns = 0.08
    q = np.clip(np.array(q_init, dtype=float), _Q_MIN, _Q_MAX)
    if q7_fixed is not None:
        q[6] = float(q7_fixed)
    q_mid = (_Q_MIN + _Q_MAX) / 2.0
    for _ in range(max_iter):
        T  = franka_fk(q)
        ev = T_target[:3, 3] - T[:3, 3]
        ew = _vee(_log3(T_target[:3, :3] @ T[:3, :3].T))
        err = np.concatenate([ev, ew])
        if np.linalg.norm(err) < tol:
            return q, True
        J = np.zeros((6, 7))
        for i in range(7):
            q2 = q.copy(); q2[i] += eps
            T2 = franka_fk(q2)
            J[:3, i] = (T2[:3, 3] - T[:3, 3]) / eps
            J[3:, i] = _vee(_log3(T2[:3, :3] @ T[:3, :3].T)) / eps
        J_dls = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))
        dq    = J_dls @ err + (np.eye(7) - J_dls @ J) @ (ns * (q_mid - q))
        q     = np.clip(q + 0.35 * dq, _Q_MIN, _Q_MAX)
        if q7_fixed is not None:
            q[6] = float(q7_fixed)
    return q, False
