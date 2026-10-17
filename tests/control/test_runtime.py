import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from mstack.comm.policy_protocol import ActionSpec, JOINT_ABSOLUTE
from mstack.agents.policy_chunk_client import run_chunks
from mstack.robots.policy_control.runtime import PolicyRuntime

Q = np.array([0., -.7, 0., -2., 0., 1.4, .7])


def test_spawned_worker_lease_expiry_and_latched_hold():
    runtime = PolicyRuntime(ActionSpec(JOINT_ABSOLUTE, .05), Q)
    try:
        runtime.start()
        t0 = time.monotonic()
        assert runtime.submit(np.tile(np.r_[Q+.02, .5], (20, 1)), t0, Q, 1, runtime.epoch)
        measured = Q.copy()
        deadline = time.monotonic()+.4
        while time.monotonic() < deadline:
            state, _ = runtime.step(measured, time.monotonic())
            measured = state[0]
            runtime.status(runtime.epoch)
            time.sleep(.001)
        status = runtime.status(runtime.epoch)
        assert status['accepted'] > 0, status
        assert np.max(abs(measured-Q)) > .001
        # No heartbeats, still ticking the servo. It must finish the current
        # validated path and refuse any buffered/fresh chunk in this session.
        deadline = time.monotonic()+.9
        while time.monotonic() < deadline:
            state, _ = runtime.step(measured, time.monotonic())
            measured = state[0]
            time.sleep(.001)
        status = runtime.status(runtime.epoch)
        assert status['held'] and status['stopped'], status
        assert status['reason'] == 'client heartbeat expired'
        assert not runtime.submit(np.tile(np.r_[Q, 0], (20, 1)), time.monotonic(), Q, 2, runtime.epoch)
        np.testing.assert_allclose(state[1:], 0, atol=1e-12)
    finally:
        runtime.close()
    assert not runtime.process.is_alive()


def test_worker_exit_never_resets_command_state():
    runtime = PolicyRuntime(ActionSpec(JOINT_ABSOLUTE, .05), Q)
    try:
        runtime.start()
        runtime.process.terminate()
        runtime.process.join(2)
        for _ in range(100):
            state, _ = runtime.step(Q, time.monotonic())
            if runtime.status(runtime.epoch)['held']:
                break
            time.sleep(.005)
        assert runtime.status(runtime.epoch)['held']
        np.testing.assert_allclose(state[0], Q)
    finally:
        runtime.close()


def test_http_stall_keeps_heartbeats_and_hold_precedes_cleanup():
    events = []
    obs = {'_policy_state': {'node_time': 123., 'q': Q,
                            'commanded_v': np.zeros(7), 'commanded_a': np.zeros(7)}}
    robot = SimpleNamespace(
        get_policy_observation=lambda: dict(obs),
        enable_policy=lambda spec: {'epoch': 'e'},
        policy_status=lambda epoch: (events.append('heartbeat') or
                                     {'held': False, 'stopped': True}),
        hold=lambda: events.append('hold'),
        disable_policy=lambda epoch: events.append('disable'))
    pool = SimpleNamespace(submit=lambda fn, captured: Future())
    report = run_chunks(robot, ActionSpec(JOINT_ABSOLUTE, .05), lambda obs: None, pool, .06)
    assert report['chunks'] == 0
    assert events.count('heartbeat') >= 3
    assert events.index('hold') < events.index('disable')


def test_http_failure_holds_and_disables_without_command():
    events = []
    obs = {'_policy_state': {'node_time': 123., 'q': Q,
                            'commanded_v': np.zeros(7), 'commanded_a': np.zeros(7)}}
    robot = SimpleNamespace(
        get_policy_observation=lambda: dict(obs), enable_policy=lambda spec: {'epoch': 'e'},
        policy_status=lambda epoch: {'held': False, 'stopped': True},
        hold=lambda: events.append('hold'), disable_policy=lambda epoch: events.append('disable'))
    future = Future(); future.set_exception(TimeoutError('injected'))
    pool = SimpleNamespace(submit=lambda fn, captured: future)
    with pytest.raises(TimeoutError):
        run_chunks(robot, ActionSpec(JOINT_ABSOLUTE, .05), lambda obs: None, pool, .1)
    assert events == ['hold', 'disable']
