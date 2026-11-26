"""Piper FK based on AgileX official URDF A.
Base: base_link
End-effector: link6 (= flange_link, zero fixed offset)
Input: six joint angles in radians.
"""
import numpy as np

# Official URDF A: joint origin xyz (m)
XYZ = np.array([
    [0.0, 0.0, 0.123],
    [0.0, 0.0, 0.0],
    [0.28503, 0.0, 0.0],
    [-0.02198, -0.25075, 0.0],
    [0.0, 0.0, 0.0],
    [8.8259e-05, -0.091, 0.0],
], dtype=np.float64)

# Official URDF A: joint origin rpy (rad)
RPY = np.array([
    [0.0, 0.0, 0.0],
    [1.5707963, -0.1357866, -3.1415926],
    [0.0, 0.0, -1.7938494],
    [1.5707963, 0.0, 0.0],
    [-1.5707963, 0.0, 0.0],
    [1.5707963, 0.0, 0.0],
], dtype=np.float64)

def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ], dtype=np.float64)

def _origin(i):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_matrix(RPY[i])
    T[:3, 3] = XYZ[i]
    return T

ORIGINS = [_origin(i) for i in range(6)]

def _rz_batch(theta):
    n = len(theta)
    c, s = np.cos(theta), np.sin(theta)
    T = np.zeros((n, 4, 4), dtype=np.float64)
    T[:, 0, 0] = c
    T[:, 0, 1] = -s
    T[:, 1, 0] = s
    T[:, 1, 1] = c
    T[:, 2, 2] = 1.0
    T[:, 3, 3] = 1.0
    return T

def fk_batch(q):
    """(N,6) joint angles -> (N,4,4) base_link-to-link6 poses."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f"Expected (N,6) joints, got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("Non-finite Piper joint values")

    T = np.repeat(np.eye(4)[None], len(q), axis=0)
    for i in range(6):
        T = T @ ORIGINS[i]
        T = T @ _rz_batch(q[:, i])
    return T

def fk(q):
    """Single joint vector -> 4x4 flange pose."""
    q = np.asarray(q, dtype=np.float64).reshape(1, 6)
    return fk_batch(q)[0]

def rot_log(R):
    """SO(3) logarithm: axis-angle rotation vector."""
    R = np.asarray(R, dtype=np.float64)
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(c)

    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)

    if np.pi - theta < 1e-5:
        vals, vecs = np.linalg.eigh((R + R.T) * 0.5)
        axis = vecs[:, np.argmax(vals)]
        skew = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        if np.linalg.norm(skew) > 1e-8 and np.dot(axis, skew) < 0:
            axis = -axis
        return axis * theta

    return theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])

def rot_log_batch(Rs):
    return np.stack([rot_log(R) for R in np.asarray(Rs)])

def rot6d(R):
    """Continuous 6D rotation representation for unified robot state."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]])
