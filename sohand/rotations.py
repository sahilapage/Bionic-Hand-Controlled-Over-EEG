"""Quaternion and rotation helpers, in MuJoCo's (w, x, y, z) convention.

Deliberately dependency-free NumPy: these are shared by the training env, the
evaluation scripts and the on-robot replay path, and the replay path must run
on a machine with neither torch nor scipy installed.
"""

import numpy as np


def qmul(q1, q2):
    """Hamilton product. Applies q2 first, then q1."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_mat(q):
    """Unit quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R):
    """3x3 rotation matrix -> unit quaternion, via the numerically stable branch.

    Picking the branch off the largest diagonal element keeps the divisor away
    from zero; the naive trace formula loses precision near a 180 deg rotation.
    """
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def q_from_vecs(v_from, v_to):
    """Shortest-arc rotation carrying `v_from` onto `v_to`."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    d = float(np.dot(v_from, v_to))
    if d > 1 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -1 + 1e-8:
        # Antiparallel: the arc is degenerate, so any perpendicular axis works.
        axis = np.cross(v_from, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, [0.0, 1.0, 0.0])
        return np.array([0.0, *(axis / np.linalg.norm(axis))])
    q = np.array([1.0 + d, *np.cross(v_from, v_to)])
    return q / np.linalg.norm(q)


def q_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    h = angle / 2.0
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def canonicalize_quat(q):
    """Force w >= 0.

    A quaternion and its negation describe the same rotation, so a raw `xquat`
    can flip sign mid-episode and hand the network a discontinuous jump across
    four observation dimensions for zero physical change. Training and
    deployment must both apply this or they see different observations.
    """
    q = np.asarray(q)
    return -q if q[0] < 0 else q


def so3_log(R):
    """Rotation matrix -> rotation vector (axis * angle), in radians."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.arccos(c))
    if ang < 1e-8:
        return np.zeros(3)
    if ang > np.pi - 1e-5:
        # Near pi the skew-symmetric part vanishes and the usual formula
        # divides by ~0; recover the axis from the symmetric part instead.
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        ax = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        n = np.linalg.norm(ax)
        return (ax / n) * ang if n > 1e-8 else np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (ang / (2.0 * np.sin(ang)))


def twist_and_swing(R, axis=(0.0, 0.0, 1.0)):
    """Swing-twist decomposition of `R` about `axis`: (twist, swing), radians.

    `twist` is how far the object has turned *about* the axis and `swing` how
    far the axis itself has been tipped away. Both come from the quaternion:
    with q = (w, v), the twist is 2*atan2(v.k, w) and the swing angle follows
    from n = hypot(w, v.k) = cos(swing / 2).

    THE TWIST IS ONLY MEANINGFUL WHILE THE SWING IS SMALL. As the swing
    approaches pi -- the object flipped over -- n goes to zero and the twist
    becomes numerically and mathematically undefined: two orientations a
    microsecond apart can report twists 180 degrees apart. Callers accumulating
    twist across a trajectory must watch the swing and discard, or at least
    flag, any stretch where it gets large. `sohand.rl.evaluate` does exactly
    that. This is not a defect in the implementation; a pure function of
    orientation cannot track rotation about a fixed axis through a flip.
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    # q and -q are the same rotation but give twists 2*pi apart, and which one
    # `mat_to_quat` returns depends on the branch it took. Pinning w >= 0 puts
    # the twist on the principal branch (-pi, pi] for every input.
    q = canonicalize_quat(mat_to_quat(np.asarray(R, dtype=np.float64)))
    proj = float(np.dot(q[1:], axis))
    n = np.hypot(q[0], proj)
    twist = 0.0 if n < 1e-9 else 2.0 * np.arctan2(proj / n, q[0] / n)
    swing = 2.0 * np.arccos(np.clip(n, 0.0, 1.0))
    return twist, swing


def twist_about(R, axis=(0.0, 0.0, 1.0)):
    """Twist angle of `R` about `axis`. See `twist_and_swing` for the caveat."""
    return twist_and_swing(R, axis)[0]


def unwrap_delta(current, previous):
    """Shortest signed step between two angles, wrapped into (-pi, pi].

    Accumulating `twist_about` across a rollout needs this, or every crossing of
    the +-pi branch cut registers as a full turn in the wrong direction.
    """
    return (current - previous + np.pi) % (2 * np.pi) - np.pi
