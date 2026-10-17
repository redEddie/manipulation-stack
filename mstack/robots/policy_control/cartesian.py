"""TCP speed surrogates for the QP and continuous nonlinear validation.

The QP freezes geometric Jacobians. Its linear constraints alone cannot certify
speed along a moving arm. Validation bounds speed between evaluated times with
an analytic acceleration bound for the same serial revolute kinematic model.
No inverse dynamics, contact model or collision geometry is implied.
"""
from itertools import product

import numpy as np

from mstack.robots.fr3_kinematics import _MDH, X_EE


# Thirteen unoriented normals: axes, edge diagonals and body diagonals.
# |D v| <= POLYTOPE_SCALE * vmax is inside the Euclidean speed ball.
DIRECTIONS = np.array([x for x in product((-1., 0., 1.), repeat=3)
                       if any(x) and next(a for a in x if a) > 0])
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]
# A vertex has components (1, sqrt(2)-1, sqrt(3)-sqrt(2)), up to signs/permutations.
POLYTOPE_SCALE = 1 / np.sqrt(1 + (np.sqrt(2)-1)**2 + (np.sqrt(3)-np.sqrt(2))**2)

# Joint i rotates all subsequent fixed translations. Rotation preserves their
# lengths. The first/second derivatives of each rotation have operator norm 1,
# so ||dp/dq_i|| <= radius[i] and ||d2p/dq_i dq_j|| <= radius[max(i,j)].
_lengths = np.linalg.norm(_MDH[1:, :2], axis=1)
RADII = np.cumsum(_lengths[::-1])[::-1] + np.linalg.norm(X_EE[:3, 3])
HESSIAN_BOUNDS = RADII[np.maximum.outer(np.arange(7), np.arange(7))]


class CartesianSpeedError(ValueError):
    """A candidate failed continuous TCP speed validation."""


def acceleration_bounds(velocity_bounds, acceleration_bounds):
    """Bounds on TCP linear/angular acceleration over joint polynomial segments."""
    v, a = velocity_bounds, acceleration_bounds
    linear = a @ RADII + np.einsum('...i,ij,...j->...', v, HESSIAN_BOUNDS, v)
    # The derivative of axis i contains only upstream revolute joints j < i.
    angular = a.sum(axis=-1) + (v.sum(axis=-1)**2 - (v*v).sum(axis=-1))/2
    return np.stack([linear, angular], axis=-1)


def validate_speed(plan, settings, kinematics=None):
    """Certify TCP norm limits throughout every interval, or reject the plan.

    At an interval midpoint, speed(t) <= speed(mid) + acceleration_bound*dt/2.
    Joint Bezier handles bound velocity and endpoint accelerations bound a for
    the entire constant-jerk segment. Bisect inconclusive intervals at most five
    times. Hitting that budget rejects; it never becomes an unchecked sample.
    """
    x, h = plan.states, plan.dt
    v0, a0, v1 = x[:-1, 1], x[:-1, 2], x[1:, 1]
    v_bound = np.max(abs(np.stack([v0, v0+h*a0/2, v1])), axis=0)
    a_bound = np.maximum(abs(a0), abs(x[1:, 2]))
    if not np.any(v_bound):
        return np.zeros(2)
    if kinematics is None:
        from mstack.robots.policy_control.kinematics import FR3Kinematics
        kinematics = FR3Kinematics()
    limits = np.array([settings.ee_linear_velocity, settings.ee_angular_velocity])
    accel = acceleration_bounds(v_bound, a_bound)
    jerk = (x[1:, 2]-a0)/h
    certified = np.zeros(2)
    k = np.arange(len(x)-1)
    left, right = np.zeros(len(k)), np.full(len(k), h)
    for depth in range(6):
        t = ((left+right)/2)[:, None]
        q = x[k, 0] + v0[k]*t + a0[k]*t*t/2 + jerk[k]*t*t*t/6
        v = v0[k] + a0[k]*t + jerk[k]*t*t/2
        twist = np.array([kinematics.velocity_jacobian(qi) @ vi for qi, vi in zip(q, v)])
        speed = np.stack([np.linalg.norm(twist[:, :3], axis=1),
                          np.linalg.norm(twist[:, 3:], axis=1)], axis=1)
        upper = speed + accel[k]*(right-left)[:, None]/2
        if not np.isfinite(upper).all() or np.any(speed > limits+settings.tolerance):
            raise CartesianSpeedError('trajectory exceeds TCP speed limits')
        passed = np.all(upper <= limits+settings.tolerance, axis=1)
        if np.any(passed):
            certified = np.maximum(certified, np.max(upper[passed], axis=0))
        if np.all(passed):
            return certified
        if depth == 5:
            raise CartesianSpeedError('TCP speed certificate subdivision budget exceeded')
        keep = ~passed
        middle = t[keep, 0]
        k = np.repeat(k[keep], 2)
        left, right = (np.column_stack([left[keep], middle]).ravel(),
                       np.column_stack([middle, right[keep]]).ravel())
