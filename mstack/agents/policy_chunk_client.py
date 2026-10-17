"""HTTP inference scheduling for node-local QP control (no robot SDK).

All robot/camera access stays on the calling thread; only an immutable captured
observation goes to the HTTP worker. Node timestamps are echoed, never compared
with the clock of another machine.
"""
import time
from collections import deque

import numpy as np


def run_chunks(robot, spec, predict, pool, max_seconds, inference_hz=30., exec_horizon=None):
    if not np.isfinite(inference_hz) or inference_hz <= 0:
        raise ValueError("inference_hz must be positive")
    # Home/legacy filter has finished moving. Wait for its derivatives to settle
    # before switching ownership; enable_policy repeats the check on the node.
    settle_deadline = time.monotonic()+5.
    while True:
        obs = robot.get_policy_observation()
        state = obs.get("_policy_state")
        if state is None:
            raise RuntimeError("robot node lacks QP metadata; update the node before enabling QP")
        if (np.max(abs(np.asarray(state["commanded_v"]))) < 1e-6
                and np.max(abs(np.asarray(state["commanded_a"]))) < 1e-4):
            break
        if time.monotonic() > settle_deadline:
            raise RuntimeError("robot did not settle before QP entry")
        time.sleep(.02)
    session = robot.enable_policy(spec.to_dict())
    epoch = session["epoch"]
    pending, captured, sequence, accepted, stale = None, None, 0, 0, 0
    history = deque(maxlen=128)
    deadline, next_inference = time.monotonic()+max_seconds, time.monotonic()
    try:
        while time.monotonic() < deadline:
            status = robot.policy_status(epoch)  # heartbeat continues during HTTP inference
            if status["held"]:
                raise RuntimeError(f"QP stopped: {status['reason']}; {status.get('last_error', '')}")
            if pending is not None and pending.done():
                chunk = pending.result()  # propagates failure only after the future is done
                if exec_horizon is not None:
                    chunk = chunk[:exec_horizon]
                spec.decode(chunk)
                result = robot.submit_policy_chunk(
                    actions=chunk, observation_time=captured["node_time"], anchor_q=captured["q"],
                    sequence=sequence, epoch=epoch)
                accepted += int(result["accepted"])
                stale += int(not result["accepted"])
                pending = None
            now = time.monotonic()
            if pending is None and now >= next_inference:
                obs = robot.get_policy_observation()
                captured = obs["_policy_state"]
                sequence += 1
                # Optional UMI proprio uses a 100 ms history interval, independent
                # of HTTP arrival rate. Preserve measured observations, not actions.
                previous = next((item for item in reversed(history)
                                 if item[0] <= captured["node_time"]-.1), None)
                if previous is None:
                    previous = history[0] if history else (captured["node_time"], captured["q"], obs.get("gripper.pos", 0.))
                obs["_proprio_previous"] = (previous[1], previous[2])
                history.append((captured["node_time"], np.asarray(captured["q"]).copy(), obs.get("gripper.pos", 0.)))
                obs["_control_context"] = {"sequence": sequence, "epoch": epoch,
                                            "observation_time": float(captured["node_time"]),
                                            "anchor_q": np.asarray(captured["q"]).tolist()}
                pending = pool.submit(predict, obs)
                next_inference = now+1./inference_hz
            time.sleep(.01)
        return {"chunks": accepted, "stale_chunks": stale, **robot.policy_status(epoch)}
    finally:
        # This is a latched stop; an HTTP result still in flight cannot resume it.
        robot.hold()
        stop_deadline = time.monotonic()+1.
        while time.monotonic() < stop_deadline:
            if robot.policy_status(epoch)["stopped"]:
                robot.disable_policy(epoch)
                break
            time.sleep(.01)
