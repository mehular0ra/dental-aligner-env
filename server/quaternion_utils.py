"""
Quaternion utilities for SE(3) tooth pose computations.
All quaternions are stored as [qw, qx, qy, qz] (scalar-first convention).
"""
import math
import numpy as np


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 * q2. Both inputs shape (4,) [qw,qx,qy,qz]."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse = conjugate (assumes unit quaternion)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between q0 and q1.
    t=0 returns q0, t=1 returns q1.
    Both inputs must be unit quaternions.
    """
    q0 = quaternion_normalize(q0)
    q1 = quaternion_normalize(q1)

    dot = float(np.dot(q0, q1))

    # If dot is negative, negate q1 to take shorter arc
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        # Quaternions are nearly identical — use linear interpolation
        result = q0 + t * (q1 - q0)
        return quaternion_normalize(result)

    theta_0 = math.acos(dot)           # angle between q0 and q1
    theta = theta_0 * t                # angle for this interpolation
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return quaternion_normalize(s0 * q0 + s1 * q1)


def quaternion_to_angle_deg(q: np.ndarray) -> float:
    """
    Return the rotation angle (in degrees) represented by quaternion q.
    Uses the formula: angle = 2 * arccos(|qw|)
    """
    q = quaternion_normalize(q)
    qw = float(np.clip(q[0], -1.0, 1.0))
    angle_rad = 2.0 * math.acos(abs(qw))
    return math.degrees(angle_rad)


def quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Build a unit quaternion from axis (3,) and angle in radians."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    return np.array([c, axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float64)


def random_quaternion_perturbation(
    q: np.ndarray,
    max_angle_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply a random rotation of at most max_angle_deg to quaternion q.
    Returns a unit quaternion.
    """
    angle = rng.uniform(0, math.radians(max_angle_deg))
    axis = rng.standard_normal(3)
    axis /= (np.linalg.norm(axis) + 1e-12)
    delta_q = quaternion_from_axis_angle(axis, angle)
    return quaternion_normalize(quaternion_multiply(delta_q, q))
