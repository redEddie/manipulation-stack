from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mstack.comm.policy_protocol import JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT
from mstack.robots.fr3_kinematics import fk, FR3_Q_MIN, FR3_Q_MAX
from mstack.robots.policy_control.cartesian import (
    DIRECTIONS, POLYTOPE_SCALE, CartesianSpeedError, acceleration_bounds, validate_speed,
)
from mstack.robots.policy_control.kinematics import FR3Kinematics
from mstack.robots.policy_control.qp import TrackerQP
from mstack.robots.policy_control.reference import Reference
from mstack.robots.policy_control.trajectory import Plan, Settings

Q = np.array([0., -.7, 0., -2., 0., 1.4, .7])


def test_geometric_tcp_velocity_matches_independent_fk_derivative():
    kin = FR3Kinematics(); rng = np.random.default_rng(43)
    eps = 1e-6
    for _ in range(30):
        q = rng.uniform(FR3_Q_MIN+.1, FR3_Q_MAX-.1)
        v, a = rng.uniform(-1.5, 1.5, 7), rng.uniform(-6, 6, 7)
        before, after = fk(q-eps*v), fk(q+eps*v)
        numeric = np.r_[(after[:3, 3]-before[:3, 3])/(2*eps),
                         Rotation.from_matrix(after[:3, :3] @ before[:3, :3].T).as_rotvec()/(2*eps)]
        np.testing.assert_allclose(kin.velocity_jacobian(q)@v, numeric, atol=1e-8)
        numeric_accel = (kin.velocity_jacobian(q+eps*v)@(v+eps*a)
                         -kin.velocity_jacobian(q-eps*v)@(v-eps*a))/(2*eps)
        actual = [np.linalg.norm(numeric_accel[:3]), np.linalg.norm(numeric_accel[3:])]
        assert np.all(actual <= acceleration_bounds(abs(v), abs(a))+1e-7)


def test_velocity_polytope_is_inside_euclidean_ball():
    normals = np.r_[DIRECTIONS, -DIRECTIONS]
    vertices = []
    for indices in combinations(range(len(normals)), 3):
        A = normals[list(indices)]
        if abs(np.linalg.det(A)) < 1e-9:
            continue
        v = np.linalg.solve(A, np.full(3, POLYTOPE_SCALE))
        if np.max(normals@v) <= POLYTOPE_SCALE+1e-12:
            vertices.append(v)
    assert len(vertices) >= 8
    np.testing.assert_allclose(np.max(np.linalg.norm(vertices, axis=1)), 1., atol=1e-12)


def test_continuous_certificate_rejects_speed_peak_between_zero_velocity_knots():
    h = .1; a = np.array([4., 0., 0., 0., 0., 0., 0.])
    states = np.array([[Q, np.zeros(7), a], [Q+a*h*h/6, np.zeros(7), -a]])
    plan = Plan(states, h)
    settings = replace(Settings(), ee_angular_velocity=.05)
    kin = FR3Kinematics()
    for x in states:
        assert np.linalg.norm((kin.velocity_jacobian(x[0])@x[1])[3:]) == 0
    with pytest.raises(CartesianSpeedError):
        validate_speed(plan, settings, kin)
    relaxed = replace(settings, ee_angular_velocity=.2)
    bound = validate_speed(plan, relaxed, kin)
    for t in np.linspace(0, h, 1001):
        q, v, _ = plan.sample(t)
        twist = kin.velocity_jacobian(q)@v
        assert np.all([np.linalg.norm(twist[:3]), np.linalg.norm(twist[3:])] <= bound+1e-10)


@pytest.mark.parametrize('kind', [JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT])
def test_both_action_kinds_obey_tight_tcp_norm_caps(kind):
    s = replace(Settings(), ee_linear_velocity=.15, ee_angular_velocity=.3)
    qp = TrackerQP(kind, s)
    if kind == JOINT_ABSOLUTE:
        target = np.tile(Q+np.array([.5, .4, .3, -.3, .5, .4, -.5]), (s.horizon, 1))
    else:
        T = fk(Q); T[:3, 3] += [.2, .1, .1]
        T[:3, :3] = Rotation.from_rotvec([.5, .5, .5]).as_matrix() @ T[:3, :3]
        target = np.tile(T, (s.horizon, 1, 1))
    plan = qp.plan(np.array([Q, np.zeros(7), np.zeros(7)]),
                   Reference(kind, target, np.zeros(s.horizon)))
    linear, angular = [], []
    for t in np.arange(0., plan.duration, .0005):
        q, v, _ = plan.sample(t)
        # FK differences also check the actual configured tool origin/frame.
        eps = 1e-6; p0, p1 = fk(q-eps*v), fk(q+eps*v)
        linear.append(np.linalg.norm(p1[:3, 3]-p0[:3, 3])/(2*eps))
        angular.append(Rotation.from_matrix(p1[:3, :3]@p0[:3, :3].T).magnitude()/(2*eps))
    assert max(linear) <= s.ee_linear_velocity+1e-7
    assert max(angular) <= s.ee_angular_velocity+1e-7
    assert max(linear) > .01  # A stationary output is not a successful tracker.


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
@pytest.mark.parametrize('field', ['ee_linear_velocity', 'ee_angular_velocity'])
def test_invalid_tcp_limits_rejected(field, value):
    with pytest.raises(ValueError):
        replace(Settings(), **{field: value})
