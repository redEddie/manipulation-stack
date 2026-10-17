"""Exercise the real driver command loop with simulated libfranka IO only."""
import threading
from types import SimpleNamespace

import numpy as np

from mstack.comm.policy_protocol import ActionSpec, JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT, PolicyInfo
from mstack.robots import franka_fr3
from mstack.robots.policy_control.qp import TrackerQP
from mstack.robots.policy_control.reference import Reference
from mstack.robots.policy_control.servo import CommonServo
from mstack.robots.policy_control.trajectory import Settings

Q = np.array([0., -.7, 0., -2., 0., 1.4, .7])


def test_real_activecontrol_loop_uses_common_servo_without_filter(monkeypatch):
    s = Settings(); servo = CommonServo(Q, 'test', s)
    qp = TrackerQP(settings=s)
    plan = qp.plan(np.array([Q, np.zeros(7), np.zeros(7)]),
                   Reference(JOINT_ABSOLUTE, np.tile(Q+.05, (s.horizon, 1)), np.zeros(s.horizon)))
    assert servo.install(plan, 20, 1, 'test', .5)
    driver = franka_fr3.FrankaFR3Robot.__new__(franka_fr3.FrankaFR3Robot)
    commands = []
    driver._lock = threading.Lock(); driver._stop = threading.Event()
    driver._dt = .001; driver._max_tick_gap = 0.; driver._late_ticks = 0; driver._success_rate = 1.
    driver._has_ft = False; driver._use_gripper = False; driver._control_error = None
    driver._policy_runtime = SimpleNamespace(step=lambda measured, now: servo.step())
    state = SimpleNamespace(q=Q, q_d=Q, dq=np.zeros(7), O_T_EE=np.eye(4).ravel(),
                            control_command_success_rate=1.)

    def write(command):
        commands.append(command.q)
        state.q = np.asarray(command.q)
        if len(commands) >= 400:
            driver._stop.set()

    control = SimpleNamespace(readOnce=lambda: (state, None), writeOnce=write)
    driver.robot = SimpleNamespace(start_joint_position_control=lambda mode: control, stop=lambda: None)
    driver._pf = SimpleNamespace(ControllerMode=SimpleNamespace(JointImpedance=1),
                                 JointPositions=lambda q: SimpleNamespace(q=q, motion_finished=False))
    monkeypatch.setattr(franka_fr3, '_RawPublisher', lambda port: SimpleNamespace(send=lambda *a: None, close=lambda: None))
    driver._control_loop()
    assert driver._control_error is None, driver._control_error
    assert len(commands) == 400
    assert driver._state_monotonic > 0
    expected = np.array([Q if k < 20 else plan.sample((k-20)*.001)[0] for k in range(400)])
    np.testing.assert_allclose(commands, expected, atol=1e-12)
    np.testing.assert_allclose(driver._qd_cmd, 0, atol=1e-12)
    np.testing.assert_allclose(driver._acc_cmd, 0, atol=1e-12)
    # Make __del__ independent of partially constructed test-only attributes.
    driver._policy_runtime = None
    driver._control_thread = driver._gripper_thread = driver._gripper_read_thread = None


def test_server_reports_explicit_spec_and_rejects_legacy_waypoint_execution():
    import pytest
    from apps.policy_server import PolicyServer, RequestError

    backend = SimpleNamespace(info=lambda: PolicyInfo('smolvla', 'fixture', 20, 32, ['agent'],
                                                       8, JOINT_ABSOLUTE, 8, 10))
    server = PolicyServer(backend)
    assert ActionSpec.from_dict(server.info.action_spec).kind == JOINT_ABSOLUTE
    spec = ActionSpec(EE_RELATIVE_WAYPOINT, .05, reference_frame='observation_ee', rotation='rotvec')
    backend.info = lambda: PolicyInfo('another-model', 'fixture', 20, 32, ['agent'], 8, 'unspecified', 7, 10)
    backend.predict = lambda obs: np.zeros((10, 7))
    server = PolicyServer(backend, action_spec=spec)
    context = {'sequence': 2, 'observation_time': 4.5, 'epoch': 'e'}
    assert server.predict({'control_context': context})['control_context'] == context
    with pytest.raises(RequestError):
        server.step({})
    with pytest.raises(RequestError):
        server.infer({})


def test_read_only_loop_updates_monotonic_observation_time():
    driver = franka_fr3.FrankaFR3Robot.__new__(franka_fr3.FrankaFR3Robot)
    driver._lock = threading.Lock(); driver._stop = threading.Event()
    driver._has_ft = False; driver._control_error = None; driver._policy_runtime = None
    state = SimpleNamespace(q=Q, dq=np.zeros(7), O_T_EE=np.eye(4).ravel(), control_command_success_rate=1.)

    def read():
        driver._stop.set()
        return state

    driver.robot = SimpleNamespace(read_once=read, stop=lambda: None)
    driver._read_only_loop()
    assert driver._control_error is None
    assert driver._state_monotonic > 0
    np.testing.assert_allclose(driver._q, Q)
    driver._control_thread = driver._gripper_thread = driver._gripper_read_thread = None
