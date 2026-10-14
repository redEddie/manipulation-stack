"""Policy failure injection: real HTTP/backend code, simulated robot only.

Run with pytest; no checkpoint, GPU, camera or robot is used.
"""
import socket
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps"))

import fr3_policy_client as client  # noqa: E402
import policy_server as server  # noqa: E402
from mstack.comm.policy_protocol import (  # noqa: E402
    EE_DELTA, JOINT_ABSOLUTE, STATE_KEY, PolicyInfo, image_key,
)


def info():
    return PolicyInfo("fake", "-", 100, 32, ["agent", "wrist"],
                      8, JOINT_ABSOLUTE, 8, 2)


class Backend:
    def __init__(self):
        self.chunk = np.zeros((2, 8))

    def info(self):
        return info()

    def reset(self, instruction):
        return instruction

    def predict(self, obs):
        return self.chunk


@pytest.fixture
def http_policy(monkeypatch):
    backend = Backend()
    service = server.PolicyServer(backend)
    original = server.HTTPServer
    ready = threading.Event()
    captured = []

    def capture(address, handler):
        httpd = original(("127.0.0.1", 0), handler)
        captured.append(httpd)
        ready.set()
        return httpd

    monkeypatch.setattr(server, "HTTPServer", capture)
    monkeypatch.setattr(server, "REQUEST_TIMEOUT_S", 0.1, raising=False)
    thread = threading.Thread(target=service.run, args=("127.0.0.1", 0), daemon=True)
    thread.start()
    assert ready.wait(5), "HTTP server did not start"
    httpd = captured[0]
    try:
        yield backend, service, f"http://127.0.0.1:{httpd.server_port}", httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("index", [0, 7])
def test_client_rejects_nonfinite_action(value, index):
    action = np.zeros(8)
    action[index] = value
    with pytest.raises(ValueError, match="finite"):
        client.raw_to_joint(action, np.zeros(7), False)


@pytest.mark.parametrize("shape", [(7,), (9,), (1, 8)])
def test_client_rejects_wrong_action_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        client.raw_to_joint(np.zeros(shape), np.zeros(7), False)


def test_client_rejects_nonfinite_measured_pose():
    q = np.zeros(7)
    q[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        client.raw_to_joint(np.zeros(8), q, False)


def test_existing_ee_local_step_matches_server_step():
    from mstack.robots.franka_fr3 import FR3_RESET_POSES

    backend = Backend()
    backend.info = lambda: PolicyInfo("fake", "-", 20, 32, ["agent", "wrist"],
                                      8, EE_DELTA, 7, 2)
    backend.chunk = np.zeros((2, 7))
    backend.chunk[:, 6] = 0.5
    service = server.PolicyServer(backend)
    service.predict({})
    q = np.asarray(FR3_RESET_POSES[client.RESET_POSE])
    local = client.raw_to_joint(backend.chunk[0], q, True)
    remote = service.step({STATE_KEY: np.append(q, 0).tolist()})["action"]
    np.testing.assert_array_equal(local, remote)
    assert local.shape == (8,) and np.isfinite(local).all()


@pytest.mark.parametrize("chunk", [np.full((2, 8), np.nan), np.full((2, 8), np.inf),
                                   np.zeros((0, 8)), np.zeros((2, 7)), np.zeros(8)])
def test_server_rejects_bad_chunk_and_invalidates_cache(http_policy, chunk):
    backend, service, url, _ = http_policy
    assert requests.post(url + "/predict", json={}, timeout=2).status_code == 200
    backend.chunk = chunk
    response = requests.post(url + "/predict", json={}, timeout=2)
    assert response.status_code == 500
    assert service._chunk is None
    assert requests.post(url + "/step", json={}, timeout=2).status_code == 400
    backend.chunk = np.zeros((2, 8))
    assert requests.post(url + "/predict", json={}, timeout=2).status_code == 200
    assert requests.get(url + "/info", timeout=2).status_code == 200


def test_incomplete_request_does_not_block_single_threaded_server(http_policy):
    _, _, url, httpd = http_policy
    with socket.create_connection(httpd.server_address, timeout=2) as stalled:
        stalled.sendall(b"POST /predict HTTP/1.0\r\nContent-Length: 100\r\n\r\n{}")
        # Release even on the old, blocking implementation so teardown cannot hang.
        timer = threading.Timer(0.5, lambda: stalled.shutdown(socket.SHUT_WR))
        timer.start()
        try:
            response = requests.get(url + "/info", timeout=0.3)
            assert response.status_code == 200
        finally:
            timer.join(2)


def make_torch_backend():
    import torch

    backend = server.LeRobotBackend.__new__(server.LeRobotBackend)
    backend._torch = torch
    backend.device = "cpu"
    backend.instruction = "pick"
    backend.state_dim = 8
    backend.cameras = ["agent", "wrist"]
    backend.image_size = 32
    backend.fill_from_state = []
    backend.seed = 7
    backend.n_action_steps = 2
    backend.unsent = []
    batches = []

    def predict(batch):
        batches.append(batch)
        return batch[STATE_KEY][:, None, :].repeat(1, 4, 1)

    backend.policy = SimpleNamespace(predict_action_chunk=predict, reset=lambda: None)
    backend.pre = lambda batch: batch
    backend.post = lambda chunk: chunk
    return backend, batches


def test_lerobot_tensor_preparation_and_chunk_slice():
    import torch

    backend, batches = make_torch_backend()
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    obs = {STATE_KEY: np.arange(8), image_key("agent"): img, image_key("wrist"): img}
    chunk = backend.predict(obs)
    np.testing.assert_array_equal(chunk, np.tile(np.arange(8), (2, 1)))
    expected = torch.from_numpy(img).permute(2, 0, 1).float().div(255)[None]
    assert torch.equal(batches[0][image_key("agent")], expected)
    assert batches[0]["task"] == "pick"
    backend.warmup()
    assert backend.instruction == "pick"


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_lerobot_rejects_nonfinite_state_before_model(value):
    backend, batches = make_torch_backend()
    state = np.zeros(8)
    state[0] = value
    img = np.zeros((32, 32, 3), np.uint8)
    with pytest.raises(server.RequestError, match="finite"):
        backend.predict({STATE_KEY: state, image_key("agent"): img, image_key("wrist"): img})
    assert not batches


@pytest.fixture
def simulated_client(monkeypatch):
    import mstack.agents.lerobot_plugin as plugin
    from mstack.robots.franka_fr3 import FR3_RESET_POSES

    events = []
    commands = []
    q = np.asarray(FR3_RESET_POSES[client.RESET_POSE])
    observation = dict(zip(plugin.JOINT_KEYS, np.append(q, 0).tolist()))
    observation.update(agent=np.zeros((32, 32, 3), np.uint8),
                       wrist=np.zeros((32, 32, 3), np.uint8))
    robot = SimpleNamespace(connect=lambda: events.append("connect"),
                            get_observation=lambda: dict(observation),
                            send_action=lambda a: commands.append(a),
                            hold=lambda: events.append("hold"),
                            disconnect=lambda: events.append("disconnect"))
    monkeypatch.setattr(plugin, "FR3ZMQRobot", lambda cfg: robot)
    monkeypatch.setattr(client, "fetch_info", lambda url: info())
    response = SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"instruction": "pick"})
    monkeypatch.setattr(client.requests, "post", lambda *a, **kw: response)
    session = SimpleNamespace(
        post=lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None,
            json=lambda: {"actions": np.tile(np.append(q, 0.5), (2, 1)).tolist()}),
        close=lambda: events.append("session-close"))
    monkeypatch.setattr(client.requests, "Session", lambda: session)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, obs):
            future = Future()
            try:
                future.set_result(fn(obs))
            except Exception as e:
                future.set_exception(e)
            return future

        def shutdown(self, **kwargs):
            events.append("shutdown")

    monkeypatch.setattr(client, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(sys, "argv", ["fr3_policy_client", "--server", "http://unused",
                                      "--max-seconds", "0.04"])
    return robot, session, events, commands, q


def test_control_loop_keeps_valid_actions_and_holds_before_cleanup(simulated_client):
    _, _, events, commands, q = simulated_client
    client.main()
    assert commands
    for command in commands:
        np.testing.assert_array_equal(list(command.values())[:7], q)
        assert list(command.values())[7] == 0.5
    assert events.index("hold") < events.index("shutdown") < events.index("disconnect")
    assert "session-close" in events


def test_network_failure_holds_and_disconnects(simulated_client):
    _, session, events, commands, _ = simulated_client

    def fail(*a, **kw):
        raise requests.Timeout("injected network failure")

    session.post = fail
    with pytest.raises(requests.Timeout, match="injected"):
        client.main()
    assert not commands
    assert events.index("hold") < events.index("shutdown") < events.index("disconnect")


def test_cleanup_survives_hold_failure(simulated_client):
    robot, session, events, _, _ = simulated_client

    def fail(*a, **kw):
        raise requests.Timeout("predict failed")

    def failed_hold():
        events.append("hold-attempt")
        raise RuntimeError("robot node unavailable")

    session.post = fail
    robot.hold = failed_hold
    with pytest.raises(requests.Timeout, match="predict failed"):
        client.main()
    assert "hold-attempt" in events
    assert "disconnect" in events


def test_later_chunk_dimension_change_is_rejected(simulated_client):
    _, session, events, commands, q = simulated_client
    chunks = iter([np.tile(np.append(q, 0.5), (2, 1)), np.zeros((2, 7))])
    session.post = lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None,
                                    json=lambda: {"actions": next(chunks).tolist()})
    with pytest.raises(ValueError, match="shape"):
        client.main()
    assert len(commands) <= 2
    assert "hold" in events


@pytest.mark.parametrize("chunk", [np.full((2, 8), np.nan), np.full((2, 8), np.inf),
                                   np.zeros((0, 8)), np.zeros(8)])
def test_bad_bootstrap_never_reaches_robot(simulated_client, chunk):
    _, session, events, commands, _ = simulated_client
    session.post = lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None,
                                                   json=lambda: {"actions": chunk.tolist()})
    with pytest.raises(ValueError):
        client.main()
    assert not commands
    assert "hold" in events and "disconnect" in events


def test_existing_joint_and_gripper_clamps_are_preserved(simulated_client):
    _, session, _, commands, q = simulated_client
    action = np.append(q + 5, 2)
    session.post = lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None,
        json=lambda: {"actions": np.tile(action, (2, 1)).tolist()})
    client.main()
    assert commands
    for command in commands:
        np.testing.assert_allclose(list(command.values())[:7], q + client.MAX_STEP_RAD)
        assert list(command.values())[7] == 1


def test_pending_prediction_obeys_control_deadline(simulated_client, monkeypatch):
    _, _, events, _, _ = simulated_client
    future = Future()

    class DelayedPool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, obs):
            return future

        def shutdown(self, **kwargs):
            events.append("shutdown")

    monkeypatch.setattr(client, "ThreadPoolExecutor", DelayedPool)
    # Bound the pre-fix failure too: release the stuck future after the assertion budget.
    timer = threading.Timer(0.5, lambda: future.set_exception(TimeoutError("late prediction")))
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            client.main()
        elapsed = time.monotonic() - start
        assert elapsed < 0.3, f"control remained blocked after --max-seconds: {elapsed:.3f}s"
        assert events.index("hold") < events.index("shutdown")
    finally:
        timer.join(2)


def test_client_dry_run_uses_real_http(http_policy):
    _, _, url, _ = http_policy
    client.dry_run(url, "pick", n=2)


def test_legacy_info_fallback_uses_real_http(http_policy, monkeypatch):
    _, _, url, httpd = http_policy
    handler = httpd.RequestHandlerClass
    monkeypatch.setattr(handler, "do_GET", lambda self: self._send(501, {"error": "legacy"}))
    assert client.fetch_info(url) is None
    client.dry_run(url, "pick", n=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
