"""Audit legacy joint-policy logs and compare operational TCP speed caps offline.

These dumps contain sparse observations and predicted chunks, not dense measured
robot trajectories or successful demonstrations. Within-chunk speeds are the
midpoint geometric-Jacobian speeds of linearly interpolated joint targets.
Boundary speeds express an estimated jump over one action interval, not a
measured robot velocity. No robot, camera, network or model is accessed.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mstack.robots.policy_control.kinematics import FR3Kinematics
from mstack.robots.policy_control.trajectory import Settings
from scripts.analyze.replay_policy_qp import distribution, read_cases, run_case


def audit(root, action_dt=.05):
    kin = FR3Kinematics()
    within, boundary, observations, sources = [], [], [], []
    requests = excluded = 0

    def speed(q0, q1, dt):
        twist = kin.velocity_jacobian((q0+q1)/2) @ ((q1-q0)/dt)
        return [np.linalg.norm(twist[:3]), np.linalg.norm(twist[3:])]

    for path in sorted(root.glob('*/requests.jsonl')):
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if len(rows) < 2:
            continue
        q = np.array([r['state'][:7] for r in rows])
        times = np.array([r['t']-r['ms']/1000 for r in rows])
        if np.max(np.ptp(q, axis=0)) < 1e-9 and np.median(np.diff(times)) < .5:
            excluded += 1
            continue
        chunks = [np.asarray(r['chunk'], dtype=float) for r in rows]
        if any(a.ndim != 2 or a.shape[1] != 8 or not np.isfinite(a).all() for a in chunks):
            raise ValueError('audit expects finite legacy seven-joint-plus-gripper chunks')
        if not np.isfinite(q).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('invalid observation data or non-increasing inferred timestamps')
        sources.append(dict(session=path.parent.name, sha256=hashlib.sha256(data).hexdigest()))
        requests += len(rows)
        for chunk in chunks:
            within.extend(speed(a[:7], b[:7], action_dt) for a, b in zip(chunk, chunk[1:]))
        for i in range(1, len(rows)):
            observations.append(speed(q[i-1], q[i], times[i]-times[i-1]))
            skip = len(chunks[i-1])-round((times[i]-times[i-1])/action_dt)
            if 0 <= skip < len(chunks[i]):
                boundary.append(speed(chunks[i-1][-1, :7], chunks[i][skip, :7], action_dt))
    if not within or not boundary:
        raise ValueError('no eligible joint-policy data')
    arrays = dict(within_chunk=np.array(within), boundary_equivalent=np.array(boundary),
                  sparse_observation_estimate=np.array(observations))
    report = dict(sessions=len(sources), requests=requests, excluded_synthetic_sessions=excluded,
                  action_dt=action_dt, sources=sources)
    for name, values in arrays.items():
        report[name] = dict(samples=len(values), linear_m_s=distribution(values[:, 0]),
                            angular_rad_s=distribution(values[:, 1]))
    return report, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--compare-boundaries', type=int, default=0)
    ap.add_argument('--candidate', type=float, nargs=2, action='append', metavar=('M_S', 'RAD_S'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source, arrays = audit(args.log_root)
    np.savez_compressed(args.output/'source_speeds.npz', **arrays)
    cases, _ = read_cases(args.log_root, args.compare_boundaries)
    candidates = args.candidate or [(.2, .5), (.3, .6), (.4, .8), (.5, 1.)]
    comparisons = []
    for linear, angular in candidates:
        settings = replace(Settings(), ee_linear_velocity=linear, ee_angular_velocity=angular)
        candidate = dict(linear_m_s=linear, angular_rad_s=angular,
                         within_chunk_above_cap_fraction={
                             'linear': float(np.mean(arrays['within_chunk'][:, 0] > linear)),
                             'angular': float(np.mean(arrays['within_chunk'][:, 1] > angular))},
                         cases=[])
        timings = []
        for i, case in enumerate(cases):
            result, trace, full, _ = run_case(case, settings=settings, fixed_compute_latency=True)
            candidate['cases'].append(result); timings.extend(full)
            if i == 0:
                np.savez_compressed(args.output/f'candidate_{linear:g}_{angular:g}.npz', **trace)
            print(f'caps={linear:g}/{angular:g} case={i+1}/{len(cases)} '
                  f'failures={result["failed_plans"]} max_ms={max(full):.2f}', flush=True)
        if cases:
            candidate['full_update_ms'] = distribution(timings)
            candidate['mean_tracking_rms_rad'] = float(np.mean(
                [r['tracking_rms_rad'] for r in candidate['cases']]))
            candidate['median_best_fit_lag_ms'] = float(np.median(
                [r['best_fit_joint_lag_ms'] for r in candidate['cases']]))
        comparisons.append(candidate)
    report = dict(source=source, candidates=comparisons, limitations=[
        'Predicted targets and sparse input states; no dense measured TCP speed or demonstration labels.',
        'Legacy capture/activation times are estimated from server timestamps and inference duration.',
        'Boundary speed is a one-action-interval equivalent jump, not measured motion.',
        'Comparison fixes compute latency to one tick to isolate cap effects; CPU times are measured separately.',
        'Alignment lag is a best-fit diagnostic over 0..500 ms, not physical robot response time.',
        'No contact-safety certification, hardware qualification or learning-performance optimum.'])
    (args.output/'speed_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(source, indent=2))
    if any(r['failed_plans'] for c in comparisons for r in c['cases']):
        raise SystemExit('candidate replay failures; inspect report')


if __name__ == '__main__':
    main()
