from dataclasses import replace

import numpy as np
import pytest

from mstack.comm.policy_protocol import JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT
from mstack.robots.fr3_kinematics import fk, FR3_Q_MIN, FR3_Q_MAX
from mstack.robots.policy_control.kinematics import FR3Kinematics
from mstack.robots.policy_control.qp import TrackerQP
from mstack.robots.policy_control.reference import Reference
from mstack.robots.policy_control.servo import CommonServo, state_at
from mstack.robots.policy_control.trajectory import Settings, HW_MAX, HW_MIN, DECEL, OFFSET

Q = np.array([0., -.7, 0., -2., 0., 1.4, .7])


def independent_check(plan, settings):
    # Dense sampling and finite-difference command checks are deliberately
    # separate from the controller's Bezier feasibility certificate.
    states = np.array([plan.sample(t) for t in np.arange(0., plan.duration+.00001, .001)])
    q, v, a = states[:, 0], states[:, 1], states[:, 2]
    assert np.min(q-settings.lower) > -1e-7
    assert np.min(settings.upper-q) > -1e-7
    assert np.max(abs(v)) <= settings.velocity+1e-7
    assert np.max(abs(a)) <= settings.acceleration+1e-7
    assert np.max(abs(np.diff(a, axis=0)/.001)) <= settings.jerk+1e-5
    upper = np.maximum(0, np.sqrt(np.maximum(0, 2*DECEL*(HW_MAX-q)))-OFFSET)
    lower = -np.maximum(0, np.sqrt(np.maximum(0, 2*DECEL*(q-HW_MIN)))-OFFSET)
    assert np.max(v-upper) <= 1e-7
    assert np.max(lower-v) <= 1e-7
    for derivative, bound in [(1, settings.velocity), (2, settings.acceleration), (3, settings.jerk)]:
        assert np.max(abs(np.diff(q, n=derivative, axis=0)/.001**derivative)) <= bound+1e-4
    np.testing.assert_allclose(states[-1, 1:], 0, atol=1e-9)


def test_pinocchio_matches_calibrated_fk_and_se3_derivative():
    kin = FR3Kinematics()
    rng = np.random.default_rng(8)
    for _ in range(12):
        q = rng.uniform(FR3_Q_MIN+.1, FR3_Q_MAX-.1)
        np.testing.assert_allclose(kin.pose(q), fk(q), atol=2e-14)
        target = fk(q+rng.normal(0, .1, 7))
        _, J = kin.residual_jacobian(q, target)
        eps = 1e-6
        numeric = np.column_stack([(kin.residual_jacobian(q+np.eye(7)[j]*eps, target)[0]
                                    -kin.residual_jacobian(q-np.eye(7)[j]*eps, target)[0])/(2*eps)
                                   for j in range(7)])
        np.testing.assert_allclose(J, numeric, atol=1e-8)


@pytest.mark.parametrize('kind', [JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT])
def test_tracker_continuity_limits_and_terminal_stop(kind):
    settings = Settings()
    qp = TrackerQP(kind, settings)
    state = np.array([Q, np.zeros(7), np.zeros(7)])
    nominal = np.tile(Q, (settings.horizon, 1))
    for i in range(30):
        if kind == JOINT_ABSOLUTE:
            target = Q+np.array([.6, .4, .3, -.3, .4, .5, -.6])*(1 if i < 15 else -1)
            target = np.tile(target, (settings.horizon, 1))
        else:
            target = fk(Q)
            target[:3, 3] += ([.03, .02, .01] if i < 15 else [-.03, -.02, -.01])
            target = np.tile(target, (settings.horizon, 1, 1))
        plan = qp.plan(state, Reference(kind, target, np.zeros(settings.horizon)), nominal, Q)
        np.testing.assert_allclose(plan.states[0], state, atol=1e-13)
        independent_check(plan, settings)
        state = plan.sample(.01)
        nominal = np.array([plan.sample((k+2)*.01)[0] for k in range(settings.horizon)])
    np.testing.assert_allclose(plan.sample(1)[1:], 0, atol=1e-12)


def test_joint_limit_and_unreachable_targets_are_soft():
    settings = Settings()
    qp = TrackerQP(JOINT_ABSOLUTE, settings)
    q = settings.upper-.01
    plan = qp.plan(np.array([q, np.zeros(7), np.zeros(7)]),
                   Reference(JOINT_ABSOLUTE, np.tile(q+10, (settings.horizon, 1)), np.zeros(settings.horizon)))
    independent_check(plan, settings)
    bad = np.array([Q, np.full(7, 3.), np.zeros(7)])
    with pytest.raises(ValueError):
        qp.plan(bad, Reference(JOINT_ABSOLUTE, np.tile(Q, (settings.horizon, 1)), np.zeros(settings.horizon)))
    ee = TrackerQP(EE_RELATIVE_WAYPOINT, settings)
    target = fk(Q); target[:3, 3] += 10
    plan = ee.plan(np.array([Q, np.zeros(7), np.zeros(7)]),
                   Reference(EE_RELATIVE_WAYPOINT, np.tile(target, (settings.horizon, 1, 1)), np.zeros(settings.horizon)))
    independent_check(plan, settings)


def test_servo_rejects_stale_discontinuous_plans_and_holds_without_reset():
    s = Settings(); servo = CommonServo(Q, 'e', s); qp = TrackerQP(settings=s)
    plan = qp.plan(np.array([Q, np.zeros(7), np.zeros(7)]),
                   Reference(JOINT_ABSOLUTE, np.tile(Q+.1, (s.horizon, 1)), np.zeros(s.horizon)))
    plan.validate(s)
    assert not servo.install(plan, 20, 1, 'old', 0)
    assert not servo.install(plan, 0, 1, 'e', 0)
    assert servo.install(plan, 20, 1, 'e', 0)
    assert not servo.install(plan, 30, 0, 'e', 0)
    assert not servo.install(plan, 30, 2, 'e', 0)  # initial state not the predicted command
    for _ in range(50):
        servo.step()
    moving = state_at(servo.plans, servo.tick, .001)
    assert np.max(abs(moving[1])) > 0
    servo.hold('test')
    np.testing.assert_allclose(state_at(servo.plans, servo.tick, .001), moving)
    assert not servo.install(plan, 60, 3, 'e', 0)
    for _ in range(400):
        state, _ = servo.step()
    assert servo.stopped
    np.testing.assert_allclose(state[1:], 0, atol=1e-12)
    corrupt = plan.states.copy(); corrupt[10, 0, 0] += .01
    with pytest.raises(ValueError):
        replace(plan, states=corrupt).validate(s)
