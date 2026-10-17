"""Replay recorded policy chunks through the production QP and common servo.

No robot, camera, model server, or network connection is made. Legacy dumps lack
client capture/activation timestamps: arrival alignment is an estimate, not a
reconstruction of measured robot motion. EE cases derived from joint logs are
explicitly synthetic. Run with --help for input/output paths.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.spatial.transform import Rotation

from mstack.comm.policy_protocol import ActionSpec, JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT
from mstack.robots.fr3_kinematics import fk
from mstack.robots.policy_control.qp import TrackerQP
from mstack.robots.policy_control.reference import ChunkManager
from mstack.robots.policy_control.servo import CommonServo, state_at
from mstack.robots.policy_control.trajectory import Settings, HW_MIN, HW_MAX, OFFSET, DECEL


def distribution(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist()))


def legacy_filter():
    # Execute the actual retained filter, not a second hand-maintained copy.
    source = (Path(__file__).resolve().parents[2]/'mstack/robots/franka_fr3.py').read_text()
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == 'err = target - q_cmd')
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == 'q_cmd = q_cmd + qd_cmd * dt')
    body = textwrap.dedent('\n'.join(lines[start:end+1]))
    namespace = {'np': np}
    exec('def step(self, target, q_cmd, qd_cmd, acc_prev, dt):\n'+textwrap.indent(body, '    ')
         +'\n    return np.array([q_cmd, qd_cmd, acc_prev])\n', namespace)
    return namespace['step'], SimpleNamespace(_kp=196., _kd=28., _v_max=1.5, _a_max=6., _j_max=3000.)


def read_cases(root, count):
    cases, sources = [], []
    retained = requests = excluded = 0
    for path in sorted(root.glob('*/requests.jsonl')):
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if len(rows) < 2:
            continue
        states = np.array([r['state'][:7] for r in rows])
        gaps = np.diff([r['t'] for r in rows])
        if np.max(np.ptp(states, axis=0)) < 1e-9 and np.median(gaps) < .5:
            excluded += 1
            continue
        retained += 1; requests += len(rows)
        sources.append({'session': path.parent.name, 'sha256': hashlib.sha256(data).hexdigest()})
        for i, (old, new) in enumerate(zip(rows, rows[1:]), start=1):
            offset = (new['t']-new['ms']/1000)-(old['t']-old['ms']/1000)
            old_actions, new_actions = np.asarray(old['chunk']), np.asarray(new['chunk'])
            if old_actions.shape[1] != 8 or new_actions.shape[1] != 8:
                continue
            skip = len(old_actions)-round(offset/.05)
            if not 0 <= skip < len(new_actions):
                continue
            jump = float(np.max(abs(new_actions[skip, :7]-old_actions[-1, :7])))
            cases.append(dict(session=path.parent.name, request=i, jump=jump,
                              old=old, new=new, offset=offset))
    cases.sort(key=lambda case: case['jump'], reverse=True)
    return cases[:count], dict(sessions=retained, requests=requests, excluded_synthetic_sessions=excluded,
                               eligible_boundaries=len(cases), sources=sources)


def ee_actions(row):
    anchor = fk(np.asarray(row['state'][:7]))
    out = []
    for action in row['chunk']:
        T = fk(np.asarray(action[:7]))
        p = anchor[:3, :3].T @ (T[:3, 3]-anchor[:3, 3])
        r = Rotation.from_matrix(anchor[:3, :3].T @ T[:3, :3]).as_rotvec()
        out.append(np.r_[p, r, action[-1]])
    return np.array(out)


def motion_metrics(trace, s):
    q, v, a = trace[:, 0], trace[:, 1], trace[:, 2]
    vel = np.diff(q, axis=0)/s.servo_dt
    acc = np.diff(vel, axis=0)/s.servo_dt
    jerk = np.diff(acc, axis=0)/s.servo_dt
    upper = np.maximum(0., np.sqrt(np.maximum(0, 2*DECEL*(HW_MAX-q)))-OFFSET)
    lower = -np.maximum(0., np.sqrt(np.maximum(0, 2*DECEL*(q-HW_MIN)))-OFFSET)
    return dict(peak_v=float(abs(v).max()), peak_a=float(abs(a).max()),
                command_peak_v=float(abs(vel).max()), command_peak_a=float(abs(acc).max()),
                command_peak_j=float(abs(jerk).max()), command_rms_j=float(np.sqrt(np.mean(jerk**2))),
                min_position_margin=float(np.minimum(q-s.lower, s.upper-q).min()),
                max_velocity_envelope_violation=float(max(0., np.max(v-upper), np.max(lower-v))),
                terminal_v=float(abs(v[-1]).max()), terminal_a=float(abs(a[-1]).max()))


def run_case(case, kind=JOINT_ABSOLUTE, jitter=False):
    s = Settings()
    spec = (ActionSpec(kind, .05) if kind == JOINT_ABSOLUTE else
            ActionSpec(kind, .05, reference_frame='observation_ee', rotation='rotvec'))
    qp = TrackerQP(kind, s)
    manager = ChunkManager(spec, 'replay')
    q = np.asarray(case['old']['state'][:7])
    servo = CommonServo(q, 'replay', s)
    old, new = case['old'], case['new']
    actions = [np.asarray(r['chunk']) if kind == JOINT_ABSOLUTE else ee_actions(r) for r in (old, new)]
    old_arrival = old['ms']/1000
    new_arrival = case['offset']+new['ms']/1000
    arrivals = [(old_arrival, 0., actions[0], old['state'][:7]),
                (new_arrival, case['offset'], actions[1], new['state'][:7])]
    if jitter:
        # Synthetic 20/30 Hz alternating arrivals. The action grid stays 20 Hz.
        arrivals = []
        t, sequence = .15, 0
        while t < 1.8:
            row = old if sequence % 2 == 0 else new
            action = actions[sequence % 2]
            delay = .16 if sequence % 13 == 0 else .02
            arrivals.append((t+delay, t, action, row['state'][:7]))
            t += .05 if sequence % 2 == 0 else 1/30
            sequence += 1
        arrivals = sorted([(a, b, c, d, i) for i, (a, b, c, d) in enumerate(arrivals)])
        duration = 2.5
    else:
        arrivals = [(a, b, c, d, i) for i, (a, b, c, d) in enumerate(arrivals)]
        duration = new_arrival+1.1
    steps = int(np.ceil(duration/.001))
    trace = [np.array([q, np.zeros(7), np.zeros(7)])]
    baseline = [trace[0].copy()]
    targets, plan_times, full_times, solves = [], [], [], []
    errors, delayed, late, accepted_chunks, rejected_chunks = [], 0, 0, 0, 0
    pending, event_index, plan_id = None, 0, 0
    old_step, params = legacy_filter()
    for tick in range(steps):
        now = tick*.001
        while event_index < len(arrivals) and arrivals[event_index][0] <= now:
            _, obs_time, action, anchor, seq = arrivals[event_index]
            accepted = manager.submit(action, obs_time, anchor, seq, 'replay', now)
            accepted_chunks += int(accepted); rejected_chunks += int(not accepted)
            event_index += 1
        if pending is not None and tick >= pending[0]:
            _, start, seq, source_sequence, plan, grip = pending
            if tick >= start:
                late += 1
            elif manager.current is not None and manager.current.sequence == source_sequence:
                servo.install(plan, start, seq, 'replay', grip)
            pending = None
        # Stop submitting before the end; the finite plan must stop unaided.
        stop_at = steps-350
        if tick % 10 == 0 and tick < stop_at and pending is None:
            start = tick+s.activation_lead_ticks
            times = start*.001+np.arange(1, s.horizon+1)*s.knot_dt
            begin = time.perf_counter()
            ref = manager.reference(times, now)
            if ref is not None and start*.001 < manager.current.end:
                state = state_at(servo.plans, start, .001)
                nominal = np.array([state_at(servo.plans, start+(i+1)*10, .001)[0] for i in range(s.horizon)])
                try:
                    plan = qp.plan(state, ref, nominal, q)
                    plan.validate(s)
                    full_ms = (time.perf_counter()-begin)*1000
                    plan_times.append(plan.elapsed_ms); full_times.append(full_ms); solves.append(plan.solve_ms)
                    elapsed_ticks = max(1, int(np.ceil(full_ms)))
                    if jitter and plan_id % 17 == 0:
                        elapsed_ticks += 35  # deliberate missed planning deadline
                        delayed += 1
                    plan_id += 1
                    pending = (tick+elapsed_ticks, start, plan_id, manager.current.sequence, plan,
                               float(manager.current.sample(np.array([start*.001])).gripper[0]))
                except (ValueError, RuntimeError) as exc:
                    errors.append(str(exc))
        if tick == stop_at:
            servo.hold('end of replay')
        state, _ = servo.step()
        trace.append(state)
        if kind == JOINT_ABSOLUTE:
            if manager.current is None:
                target = q
            else:
                # Original client sent one waypoint per training tick (ZOH).
                chunk = manager.current
                ix = int(np.clip(np.floor((now-chunk.start)/spec.action_dt), 0, len(chunk.targets)-1))
                target = chunk.targets[ix]
            targets.append(target)
            bs = baseline[-1]
            baseline.append(old_step(params, target if tick < stop_at else bs[0], *bs, .001))
    trace = np.array(trace)
    metrics = motion_metrics(trace, s)
    assert metrics['min_position_margin'] >= -1e-7, metrics
    assert metrics['peak_v'] <= s.velocity+1e-7 and metrics['peak_a'] <= s.acceleration+1e-7, metrics
    assert metrics['command_peak_j'] <= s.jerk+1e-3, metrics
    assert metrics['max_velocity_envelope_violation'] < 1e-7, metrics
    assert metrics['terminal_v'] < 1e-8 and metrics['terminal_a'] < 1e-8, metrics
    report = dict(session=case['session'], request=case['request'], kind=kind,
                  synthetic_variable_arrival=jitter, input_jump=case['jump'],
                  metrics=metrics, plans=len(plan_times), failed_plans=len(errors),
                  errors=sorted(set(errors)), late_plans=late, injected_delays=delayed,
                  accepted_plans=servo.accepted, accepted_chunks=accepted_chunks, rejected_chunks=rejected_chunks,
                  plan_ms=distribution(plan_times), full_update_ms=distribution(full_times), solver_ms=distribution(solves))
    if kind == JOINT_ABSOLUTE:
        baseline = np.array(baseline); targets = np.array(targets)
        report['legacy_filter'] = motion_metrics(baseline, s)
        # Compare tracking during active input, excluding the explicit stop tail.
        report['tracking_rms_rad'] = float(np.sqrt(np.mean((trace[1:stop_at, 0]-targets[:stop_at-1])**2)))
        report['legacy_tracking_rms_rad'] = float(np.sqrt(np.mean((baseline[1:stop_at, 0]-targets[:stop_at-1])**2)))
    return report, dict(qp=trace, baseline=np.array(baseline), targets=np.array(targets)), full_times, plan_times


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log-root', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--boundaries', default=12, type=int)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases, audit = read_cases(args.log_root, args.boundaries)
    if not cases:
        raise SystemExit('no eligible recorded joint-action boundaries')
    reports, timings, plan_timings, traces = [], [], [], None
    for i, case in enumerate(cases):
        report, trace, full, plan = run_case(case)
        reports.append(report); timings.extend(full); plan_timings.extend(plan)
        if i == 0:
            traces = trace
        print(f"joint {i+1}/{len(cases)} {case['session']}:{case['request']} "
              f"max={report['full_update_ms']['max']:.2f} ms failures={report['failed_plans']}", flush=True)
    ee_report, _, _, _ = run_case(cases[0], EE_RELATIVE_WAYPOINT)
    jitter_report, _, _, _ = run_case(cases[0], jitter=True)
    report = dict(audit=audit, machine=platform.uname()._asdict(), settings=vars(Settings()),
                  description='Production ActionSpec / sparse PIQP / common servo offline command replay',
                  limitations=['Server t-ms is only an estimated observation time; exact client timing is unavailable.',
                               'Commands are simulated, not measured robot motion or closed-loop VLA quality.',
                               'The EE case is derived from FK of joint logs, not a recorded EE-output model.',
                               'CPU timings are measured here, not a certified worst-case bound or robot-PC RT result.',
                               'Self collision, workspace obstacles, torque and contact dynamics are not modeled.'],
                  joint_plan_ms=distribution(plan_timings), joint_full_update_ms=distribution(timings),
                  total_joint_plans=len(timings), cases=reports,
                  synthetic_ee=ee_report, synthetic_20_30_hz_delay=jitter_report)
    (args.output/'replay_report.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output/'largest_boundary_trace.npz', **traces)
    print(json.dumps({key: report[key] for key in ('total_joint_plans','joint_plan_ms','joint_full_update_ms')}, indent=2))
    if any(r['failed_plans'] for r in reports+[ee_report, jitter_report]):
        raise SystemExit('QP feasibility failures found; review report before qualification')


if __name__ == '__main__':
    main()
