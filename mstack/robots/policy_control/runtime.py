"""Node-local QP process and non-real-time mailbox supervisor.

Only CommonServo.step() runs in the robot thread. The spawned process owns
Pinocchio/PIQP, with no libfranka handle or robot socket. Missing/late/infeasible
plans leave a finite, validated trajectory ending at rest. Python's existing
ActiveControl loop still requires hardware timing qualification; this module
cannot provide a hard real-time scheduling guarantee.
"""
import multiprocessing as mp
import threading
import time
import uuid

import numpy as np

from mstack.comm.policy_protocol import ActionSpec
from mstack.robots.policy_control.servo import CommonServo, state_at
from mstack.robots.policy_control.trajectory import Settings


def _planner_process(conn, spec_dict, settings):
    try:
        from mstack.robots.policy_control.qp import TrackerQP
        from mstack.robots.policy_control.reference import TimedChunk
        spec = ActionSpec.from_dict(spec_dict)
        qp = TrackerQP(spec.kind, settings)
        conn.send({"ready": True})
        chunk = None
        while True:
            req = conn.recv()
            if req is None:
                break
            try:
                raw = req["chunk"]
                if chunk is None or raw["sequence"] != chunk.sequence:
                    chunk = TimedChunk(spec, raw["actions"], raw["observation_time"],
                                       raw["anchor_q"], raw["sequence"], raw["epoch"])
                times = req["start_time"] + np.arange(1, settings.horizon+1)*settings.knot_dt
                ref = chunk.sample(times)
                plan = qp.plan(req["state"], ref, req["nominal"], req["posture"])
                conn.send({"id": req["id"], "plan": plan, "gripper": float(chunk.sample(
                    np.array([req["start_time"]])).gripper[0])})
            except Exception as exc:
                conn.send({"id": req["id"], "error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            conn.send({"error": f"{type(exc).__name__}: {exc}"})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        conn.close()


class PolicyRuntime:
    LEASE_S = .5

    def __init__(self, spec, q, gripper=0., settings=Settings()):
        self.spec = ActionSpec.from_dict(spec) if isinstance(spec, dict) else spec
        self.spec.validate()
        self.s = settings
        self.epoch = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.servo = CommonServo(q, self.epoch, settings, gripper)
        self.posture = np.asarray(q).copy()
        self.chunk = None
        self.last_sequence = -1
        self.last_heartbeat = time.monotonic()
        self.last_state_time = time.monotonic()
        self.stop_event = threading.Event()
        self.stats = {"solved": 0, "failed": 0, "late": 0, "stale": 0, "max_plan_ms": 0.,
                      "max_roundtrip_ms": 0., "last_error": ""}
        context = mp.get_context("spawn")
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_planner_process,
                                       args=(child, self.spec.to_dict(), settings), daemon=True)
        self.process.start()
        child.close()
        if not self.conn.poll(15.):
            self.close()
            raise RuntimeError("QP worker startup timed out")
        ready = self.conn.recv()
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"QP worker startup failed: {ready}")
        self.thread = None

    def start(self):
        self.last_heartbeat = time.monotonic()
        self.thread = threading.Thread(target=self._supervise, name="policy-qp-mailbox", daemon=True)
        self.thread.start()

    def submit(self, actions, observation_time, anchor_q, sequence, epoch):
        self.spec.decode(actions)
        q = np.asarray(anchor_q, dtype=float)
        now = time.monotonic()
        if q.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(observation_time):
            raise ValueError("invalid chunk anchor/time")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("invalid sequence")
        end = observation_time+self.spec.first_action_offset+len(actions)*self.spec.action_dt
        with self.lock:
            if (self.servo.held or epoch != self.epoch or sequence <= self.last_sequence
                    or observation_time > now+.002 or end <= now
                    or (self.chunk is not None and observation_time < self.chunk["observation_time"])):
                self.stats["stale"] += 1
                return False
            self.chunk = dict(actions=np.asarray(actions).copy(), observation_time=float(observation_time),
                              anchor_q=q.copy(), sequence=sequence, epoch=epoch, end=end)
            self.last_sequence = sequence
            self.last_heartbeat = now
            return True

    def status(self, epoch):
        with self.lock:
            if epoch != self.epoch:
                raise ValueError("policy session mismatch")
            self.last_heartbeat = time.monotonic()
            return dict(epoch=self.epoch, held=self.servo.held, reason=self.servo.reason,
                        stopped=self.servo.stopped, tick=self.servo.tick,
                        accepted=self.servo.accepted, rejected=self.servo.rejected, **self.stats)

    def hold(self, reason="requested"):
        with self.lock:
            self.servo.hold(reason)
            self.chunk = None

    def step(self, measured_q, now):
        """Called once per successful robot command tick; never waits on worker IO."""
        with self.lock:
            self.last_state_time = now
            current = state_at(self.servo.plans, self.servo.tick, self.s.servo_dt)
            if now-self.last_heartbeat > self.LEASE_S and not self.servo.held:
                self.servo.hold("client heartbeat expired")
            if (not np.isfinite(measured_q).all()
                    or np.max(abs(measured_q-current[0])) > self.s.tracking_error):
                self.servo.hold("measured-command tracking error")
            return self.servo.step()

    def _supervise(self):
        pending, request_id, next_request = None, 0, time.monotonic()
        try:
            while not self.stop_event.wait(.001):
                now = time.monotonic()
                if not self.process.is_alive():
                    self.hold("QP worker exited")
                    return
                if pending is not None and self.conn.poll():
                    result = self.conn.recv()
                    with self.lock:
                        self.stats["max_roundtrip_ms"] = max(self.stats["max_roundtrip_ms"],
                                                             (now-pending["sent"])*1000)
                    if result.get("id") != pending["id"]:
                        raise RuntimeError("QP response ID mismatch")
                    if "error" in result:
                        with self.lock:
                            self.stats["failed"] += 1
                            self.stats["last_error"] = result["error"]
                    else:
                        plan = result["plan"]
                        plan.validate(self.s)  # outside the servo lock
                        with self.lock:
                            self.stats["solved"] += 1
                            self.stats["max_plan_ms"] = max(self.stats["max_plan_ms"], plan.elapsed_ms)
                            if self.servo.tick >= pending["start_tick"]:
                                self.stats["late"] += 1
                            elif (not self.servo.held and self.chunk is not None
                                  and self.chunk["sequence"] == pending["sequence"]
                                  and now < pending["end"]):
                                self.servo.install(plan, pending["start_tick"], pending["id"],
                                                   self.epoch, result["gripper"])
                            else:
                                self.stats["stale"] += 1
                    pending = None
                if pending is not None or now < next_request:
                    continue
                with self.lock:
                    chunk = self.chunk
                    tick, plans = self.servo.snapshot()
                    state_time = self.last_state_time
                    held = self.servo.held
                if held or chunk is None or now >= chunk["end"]:
                    continue
                start_time = chunk["observation_time"]+self.spec.first_action_offset
                if now < start_time:
                    continue
                knot_ticks = round(self.s.knot_dt/self.s.servo_dt)
                # Keep all replans on the same knot grid. This preserves the
                # previously feasible first Bezier handles near a velocity cap.
                start_tick = ((tick+self.s.activation_lead_ticks+knot_ticks-1)//knot_ticks)*knot_ticks
                start_time = state_time+(start_tick-tick)*self.s.servo_dt
                if start_time >= chunk["end"]:
                    continue
                state = state_at(plans, start_tick, self.s.servo_dt)
                nominal = np.array([state_at(plans, start_tick+(k+1)*knot_ticks,
                                              self.s.servo_dt)[0] for k in range(self.s.horizon)])
                request_id += 1
                req = dict(id=request_id, chunk=chunk, state=state, nominal=nominal,
                           posture=self.posture, start_time=start_time)
                self.conn.send(req)
                pending = dict(id=request_id, sequence=chunk["sequence"], start_tick=start_tick,
                               sent=now, end=chunk["end"])
                next_request += self.s.planning_period
                if next_request <= now:
                    next_request = now+self.s.planning_period  # skip missed periods; never burst
        except Exception as exc:
            with self.lock:
                self.stats["last_error"] = f"{type(exc).__name__}: {exc}"
            self.hold("QP supervisor failed")

    def close(self):
        self.stop_event.set()
        thread = getattr(self, "thread", None)
        if thread is not None:
            thread.join(timeout=1.)
        process = getattr(self, "process", None)
        if process is not None:
            if process.is_alive():
                process.terminate()  # never owns the robot; only a numerical worker
            process.join(timeout=2.)
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
