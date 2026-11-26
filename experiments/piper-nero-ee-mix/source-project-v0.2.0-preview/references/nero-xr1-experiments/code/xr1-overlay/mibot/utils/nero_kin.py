"""NERO kinematics for the EF (end-effector) model: MDH forward kinematics,
geometric Jacobian and a damped-least-squares IK that stays close to a seed.

FK is a numpy copy of pyAgxArm.utiles.mdh_kinematics.fk_from_mdh with the
"nero" MDH preset from pyAgxArm.api.constants (flange frame, metres / rad);
ef_ik_roundtrip.py checks it against pyAgxArm when pyAgxArm is importable.
No robot I/O here.
"""
import numpy as np

# (d, a, alpha, theta_offset) -- pyAgxArm ROBOT_MDH_PRESET["nero"]
MDH = np.array([
    (0.138, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.5707963267948966, 3.141592653589793),
    (0.31, 0.0, 1.5707963267948966, 3.141592653589793),
    (0.0, 0.0, 1.5707963267948966, 3.141592653589793),
    (0.27001, 0.0, 1.5707963267948966, 1.5707963267948966),
    (0.0, 0.0, 1.5707963267948966, 1.5707963267948966),
    (0.0235, 0.0, 1.5707963267948966, 0.0),
])
# pyAgxArm ROBOT_JOINT_LIMIT_PRESET_RAD["nero"]
LIMITS = np.array([
    (-2.705261, 2.705261), (-1.745330, 1.745330), (-2.757621, 2.757621), (-1.012291, 2.146755),
    (-2.757621, 2.757621), (-0.733039, 0.959932), (-1.570797, 1.570797),
])


def _link(d, a, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [ca * st, ca * ct, -sa, -sa * d],
                     [sa * st, sa * ct, ca, ca * d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk(q):
    """Flange pose as a 4x4 matrix."""
    T = np.eye(4)
    for (d, a, alpha, off), qi in zip(MDH, q):
        T = T @ _link(d, a, alpha, qi + off)
    return T


def fk_jac(q):
    """(T, J) with J the 6x7 geometric Jacobian [linear; angular] in the base frame."""
    T = np.eye(4)
    z, p = [], []
    for (d, a, alpha, off), qi in zip(MDH, q):
        T = T @ _link(d, a, alpha, qi + off)
        z.append(T[:3, 2].copy())   # joint i rotates about z of frame i
        p.append(T[:3, 3].copy())   # frame-i origin lies on that axis
    pe = T[:3, 3]
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        J[:3, i] = np.cross(z[i], pe - p[i])
        J[3:, i] = z[i]
    return T, J


def rot_log(R):
    """so(3) log: rotation vector (axis * angle) of R."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:  # near pi
        w, v = np.linalg.eigh((R + R.T) / 2)
        axis = v[:, np.argmax(w)]
        return axis * th
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


ELBOW_JOINT = 2  # j3 rotates the upper arm about itself (shoulder-elbow line): it fixes the 7-DoF elbow freedom


def ik(T_target, q_seed, q_ref=None, iters=100, tol_pos=1e-5, tol_rot=1e-4,
       damping=0.02, null_gain=0.2, max_step=0.2, q3_target=None, q3_weight=1.0, tol_q3=1e-4):
    """Damped least squares IK from q_seed.
    q3_target=None: 6-D pose task; the null space pulls toward q_ref (default
      q_seed) so the redundant elbow stays where it was.
    q3_target=x: 7-D task (pose + j3 = x), so the elbow is fully specified
      (EF model with the extra elbow action dimension).
    Returns (q, ok, pos_err_m, rot_err_rad)."""
    q = np.array(q_seed, dtype=np.float64)
    q_ref = q.copy() if q_ref is None else np.asarray(q_ref, dtype=np.float64)
    m = 6 if q3_target is None else 7
    Im, I7 = np.eye(m), np.eye(len(q))
    for _ in range(iters):
        T, J = fk_jac(q)
        e = np.r_[T_target[:3, 3] - T[:3, 3], rot_log(T_target[:3, :3] @ T[:3, :3].T)]
        ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        e3 = 0.0 if q3_target is None else q3_target - q[ELBOW_JOINT]
        if ep < tol_pos and er < tol_rot and abs(e3) < tol_q3:
            return q, True, ep, er
        if q3_target is not None:
            row = np.zeros((1, len(q)))
            row[0, ELBOW_JOINT] = q3_weight
            J = np.vstack([J, row])
            e = np.r_[e, q3_weight * e3]
        JJt = J @ J.T + damping ** 2 * Im
        J_pinv = J.T @ np.linalg.solve(JJt, Im)
        dq = J_pinv @ e + (I7 - J_pinv @ J) @ (null_gain * (q_ref - q))
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q = np.clip(q + dq, LIMITS[:, 0], LIMITS[:, 1])
    T = fk(q)
    ep = np.linalg.norm(T_target[:3, 3] - T[:3, 3])
    er = np.linalg.norm(rot_log(T_target[:3, :3] @ T[:3, :3].T))
    return q, (ep < tol_pos * 10 and er < tol_rot * 10), ep, er


def rot_log_batch(Rs):
    """float64 rotation vectors for a stack of rotation matrices (XR-1's
    rotm2aa_batch works in float32 and quantizes small per-step rotations)."""
    return np.stack([rot_log(R) for R in np.asarray(Rs, dtype=np.float64)])


def rot_exp(w):
    """Rodrigues: rotation vector -> 3x3 rotation matrix (float64)."""
    w = np.asarray(w, dtype=np.float64)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx
