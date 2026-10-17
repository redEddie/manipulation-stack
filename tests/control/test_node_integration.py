"""Exercise the real driver command loop with simulated libfranka IO only."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mstack.comm.policy_protocol import ActionSpec, JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT, PolicyInfo
from mstack.robots import franka_fr3
from mstack.robots.policy_control.qp import TrackerQP
from mstack.robots.policy_control.reference import Reference
from mstack.robots.policy_control.servo import CommonServo
from mstack.robots.policy_control.trajectory import Settings
from mstack.robots.fr3_kinematics import fk

Q = np.array([0., -.7, 0., -2., 0., 1.4, .7])


def test_joint_policy_checks_tcp_frame_and_passes_node_speed_caps(monkeypatch):
    from mstack.robots.policy_control import runtime as runtime_module
    captured = []

    def create_runtime(spec, q, gripper, settings):
        captured.append(settings)
        return SimpleNamespace(epoch='fixture', start=lambda: None, close=lambda: None)

    monkeypatch.setattr(runtime_module, 'PolicyRuntime', create_runtime)
    driver = franka_fr3.FrankaFR3Robot.__new__(franka_fr3.FrankaFR3Robot)
    driver._read_only = driver._use_gripper = False
    driver._control_error = driver._policy_runtime = None
    driver._lock = threading.Lock(); driver._stop = threading.Event()
    driver._q = driver._q_cmd = driver._desired_q = Q.copy()
    driver._qd_cmd = driver._acc_cmd = np.zeros(7)
    driver._state_monotonic = time.monotonic()
    driver._v_max, driver._a_max, driver._j_max = 1.5, 6., 3000.
    driver._policy_ee_linear_velocity, driver._policy_ee_angular_velocity = .2, .5
    wrong_pose = fk(Q); wrong_pose[0, 3] += .02
    driver._ee_pose = wrong_pose.ravel(order='F')
    spec = ActionSpec(JOINT_ABSOLUTE, .05).to_dict()
    with pytest.raises(RuntimeError, match='TCP frame disagrees'):
        driver.enable_policy(spec)
    assert not captured
    driver._ee_pose = fk(Q).ravel(order='F')
    assert driver.enable_policy(spec)['epoch'] == 'fixture'
    assert captured[0].ee_linear_velocity == .2
    assert captured[0].ee_angular_velocity == .5
    driver._policy_runtime = None
    driver._control_thread = driver._gripper_thread = driver._gripper_read_thread = None


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
