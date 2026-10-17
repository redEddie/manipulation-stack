"""Measure the production QP process/mailbox/servo with simulated robot IO only."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mstack.comm.policy_protocol import ActionSpec, JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT
from mstack.robots.policy_control.runtime import PolicyRuntime


def percentiles(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(values, [50, 95, 99, 100]).tolist()))


def benchmark(kind, seconds):
    q = np.array([0., -.7, 0., -2., 0., 1.4, .7])
    spec = (ActionSpec(kind, .05) if kind == JOINT_ABSOLUTE else
            ActionSpec(kind, .05, reference_frame='observation_ee', rotation='rotvec'))
    runtime = PolicyRuntime(spec, q)
    try:
        runtime.start()
        timings, gaps, previous, sequence = [], [], time.monotonic(), 0
        next_tick = previous
        for tick in range(round(seconds*1000)):
            now = time.monotonic()
            if tick % 40 == 0:
                sequence += 1
                target = (np.r_[q+np.array([.03, -.02, .01, -.03, .01, .02, -.01]), 0.]
                          if kind == JOINT_ABSOLUTE else np.array([.005, .005, 0., 0., 0., 0., 0.]))
                runtime.submit(np.tile(target, (20, 1)), now, q, sequence, runtime.epoch)
                runtime.status(runtime.epoch)
            begin = time.perf_counter()
            state, _ = runtime.step(q, now)
            timings.append((time.perf_counter()-begin)*1e6)
            q = state[0]
            gaps.append((now-previous)*1000)
            previous = now
            next_tick += .001
            delay = next_tick-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # no catch-up command burst
        runtime.hold('benchmark end')
        for _ in range(350):
            state, _ = runtime.step(q, time.monotonic()); q = state[0]
            time.sleep(.001)
        status = runtime.status(runtime.epoch)
        if not status['stopped'] or status['accepted'] == 0 or status['failed']:
            raise RuntimeError(f'Runtime benchmark failed: {status}')
        return dict(kind=kind, ticks=len(timings), servo_call_us=percentiles(timings),
                    host_tick_gap_ms=percentiles(gaps[1:]), status=status)
    finally:
        runtime.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=3.)
    args = ap.parse_args()
    report = {'description': 'Synthetic no-hardware process and mailbox timing; standard Linux scheduling',
              'limitations': ['No libfranka IO, physical robot dynamics, RT priority or certified WCET.',
                              'Host tick gaps include Python and time.sleep scheduling.'],
              'results': [benchmark(kind, args.seconds) for kind in (JOINT_ABSOLUTE, EE_RELATIVE_WAYPOINT)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
