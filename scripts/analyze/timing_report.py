"""Recording timing report -- loop rate, jitter, synchronisation, long-run drift, cycle time
(analysis for the paper).

Reads what the collector records since knu-1.3.0:

* ``episode_NNN/timing/*`` in scene files (per frame, host ``time.time()``)
* ``collection_phases.jsonl`` (per phase change, next to the collection history)

and reports, per measure, n / mean / sd / p1 / p50 / p95 / p99 / max in
milliseconds:

    loop_interval      frame[i+1] - frame[i]; nominal 1000/fps (station fps)
    loop_rate_hz       1 / loop_interval
    action_to_frame    frame - action: command accepted -> observation read
    robot_state_age    frame - robot_state: age of the stored joint state
    <cam>_age          frame - <cam>_host: age of the stored image at record time
    <cam>_pipeline     <cam>_host - <cam>_device (global_time clock only):
                       device timestamp -> node arrival
    camera_skew        agentview_host - eye_in_hand_host: the two images' offset
    <cam>_drop_rate    per episode: frames lost between device and node over
                       frames the device produced, from the growth of
                       frame_no - node_seq
    <cam>_repeat_rate  per episode: share of recorded frames whose node_seq did
                       not advance -- the same image stored twice

Jitter is the standard deviation of ``loop_interval`` and its p99 - p1 spread.

Long-run drift: every episode's mean loop interval and mean ages are placed
on the time since its session started (sessions are split where the gap
between episodes exceeds --session-gap minutes), and an ordinary least-squares
slope per hour is reported with its 95% interval. A slope interval containing 0
means no measurable drift.

Cycle time (phase log): each episode end is attributed to the phases before
it -- gate, approach, recording -- and the homing and reset_wait that follow,
plus the full cycle from one gate to the next. Filters: --scene, --collector,
--since; practice sessions are excluded unless --include-practice.

Outputs to --out: ``summary.json``, ``frames.csv`` (per frame), ``episodes.csv``
(per episode), ``cycles.csv`` (per cycle), ``hist_<measure>.csv`` (1 ms bins).

Usage:
    python scripts/analyze/timing_report.py --root ~/libero_datasets/fr3-tabletop \\
        --scene S006 --out ~/teleop-franka/paper/timing
    python scripts/analyze/timing_report.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from mstack.config.station import load_station  # noqa: E402
from mstack.data.dataset_schema import TIMING_CAMERAS, TIMING_GROUP  # noqa: E402
from mstack.data.phase_log import load_phases, phase_log_path  # noqa: E402
from mstack.scene.scene_format import EPISODE_GROUP_RE, iter_scene_files  # noqa: E402

CYCLE_PHASES = ("gate", "approach", "recording", "homing", "reset_wait")


# ------------------------------------------------------------------ statistics
def describe_ms(values) -> dict:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], float)
    if v.size == 0:
        return {"n": 0}
    ms = v * 1e3
    return {"n": int(ms.size), "mean": float(ms.mean()),
            "sd": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
            "p1": float(np.percentile(ms, 1)), "p50": float(np.percentile(ms, 50)),
            "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)),
            "max": float(ms.max()), "min": float(ms.min())}


def ols_slope(x, y) -> dict:
    """Least-squares slope with a 95% interval (t quantile via normal for n>30,
    a small table below that)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    if n < 3 or np.ptp(x) == 0:
        return {"n": int(n)}
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    resid = y - (a + b * x)
    se = math.sqrt((resid ** 2).sum() / (n - 2) / sxx)
    t = _t975(n - 2)
    return {"n": int(n), "slope": float(b), "intercept": float(a),
            "ci95": [float(b - t * se), float(b + t * se)]}


def _t975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086,
             25: 2.060, 30: 2.042}
    if df >= 30:
        return 1.96 if df > 120 else 2.0
    return table.get(df) or table[max(k for k in table if k <= df)]


def histogram_ms(values, bin_ms: float = 1.0) -> list:
    ms = np.asarray([x for x in values if x is not None and math.isfinite(x)], float) * 1e3
    if ms.size == 0:
        return []
    lo = math.floor(ms.min() / bin_ms) * bin_ms
    hi = math.ceil(ms.max() / bin_ms) * bin_ms + bin_ms
    counts, edges = np.histogram(ms, bins=np.arange(lo, hi + 1e-9, bin_ms))
    return [(float(edges[i]), float(edges[i + 1]), int(c)) for i, c in enumerate(counts)]


# ------------------------------------------------------------------ frame timing
def read_episodes(root: Path, scene: str = "") -> list:
    """One dict per episode that has a timing group."""
    out = []
    for p in iter_scene_files(root):
        if scene and not p.name.startswith(f"scene_{int(scene[1:]):03d}"):
            continue
        try:
            f = h5py.File(p, "r")
        except OSError as e:
            print(f"[경고] {p.name} 읽기 실패 ({type(e).__name__}) -- 제외")
            continue
        with f:
            for name in sorted(k for k in f if EPISODE_GROUP_RE.match(k)):
                g = f[name]
                if TIMING_GROUP not in g:
                    continue
                tg = g[TIMING_GROUP]
                ep = {"file": p.name, "episode": name,
                      "uid": str(g.attrs.get("episode_uid", "")),
                      "collector": str(g.attrs.get("collector", "")),
                      "cols": {k: tg[k][()] for k in tg},
                      "attrs": {k: (v.decode() if isinstance(v, bytes) else v)
                                for k, v in tg.attrs.items()}}
                out.append(ep)
    out.sort(key=lambda e: float(e["cols"]["frame"][0]) if "frame" in e["cols"] else 0.0)
    return out


def frame_measures(ep: dict) -> dict:
    """Per-frame arrays (seconds) for one episode."""
    c = ep["cols"]
    frame = c.get("frame")
    m: dict = {}
    if frame is None or len(frame) == 0:
        return m
    m["loop_interval"] = np.diff(frame)
    if "action" in c:
        m["action_to_frame"] = frame - c["action"]
    if "robot_state" in c:
        m["robot_state_age"] = frame - c["robot_state"]
    for cam in TIMING_CAMERAS:
        if f"{cam}_host" in c:
            m[f"{cam}_age"] = frame - c[f"{cam}_host"]
        if (f"{cam}_device" in c and f"{cam}_host" in c
                and str(ep["attrs"].get(f"{cam}_domain", "")).lower() == "global_time"):
            m[f"{cam}_pipeline"] = c[f"{cam}_host"] - c[f"{cam}_device"]
    hosts = [f"{cam}_host" for cam in TIMING_CAMERAS]
    if all(h in c for h in hosts):
        m["camera_skew"] = c[hosts[0]] - c[hosts[1]]
    return m


def drop_rate(ep: dict, cam: str) -> "float | None":
    """Frames lost between device and node / frames the device produced.

    Only 20 of 30 frames per second are recorded, so a jump in the device
    counter is not a drop by itself; what the node did not receive shows up as
    the device counter running ahead of the node's."""
    c = ep["cols"]
    if f"{cam}_frame_no" not in c or f"{cam}_node_seq" not in c:
        return None
    fn, sq = c[f"{cam}_frame_no"], c[f"{cam}_node_seq"]
    produced = float(fn[-1] - fn[0]) if len(fn) > 1 else 0.0
    if produced <= 0:
        return None
    lost = float((fn[-1] - sq[-1]) - (fn[0] - sq[0]))
    return max(0.0, lost / produced)


def repeat_rate(ep: dict, cam: str) -> "float | None":
    """Share of recorded frames that stored the same node frame as the previous one."""
    c = ep["cols"]
    if f"{cam}_node_seq" not in c or len(c[f"{cam}_node_seq"]) < 2:
        return None
    d = np.diff(c[f"{cam}_node_seq"])
    return float(np.mean(d == 0))


def split_sessions(episodes: list, gap_min: float) -> list:
    """Session index per episode, splitting where episodes are far apart."""
    idx, last_end, sid = [], None, -1
    for ep in episodes:
        start = float(ep["cols"]["frame"][0])
        if last_end is None or (start - last_end) > gap_min * 60:
            sid += 1
            session_start = start
        idx.append((sid, start - session_start))
        last_end = float(ep["cols"]["frame"][-1])
    return idx


# ------------------------------------------------------------------ cycle time
def cycles_from_phases(lines: list, scene: str = "", collector: str = "",
                       since: str = "", include_practice: bool = False) -> list:
    """One row per episode end: phase durations around it, in seconds."""
    rows = []
    by_session: dict = {}
    for x in lines:
        if not include_practice and x.get("practice"):
            continue
        if scene and x.get("scene") != scene:
            continue
        if collector and x.get("collector") != collector:
            continue
        if since and str(x.get("session", "")) < since:
            continue
        by_session.setdefault((x.get("run"), x.get("session")), []).append(x)
    for (run, session), xs in by_session.items():
        xs.sort(key=lambda x: x["t"])
        phases = [x for x in xs if "phase" in x]
        ends = [x for x in xs if x.get("event") == "episode_end"]

        def dur(i):
            return phases[i + 1]["t"] - phases[i]["t"] if i + 1 < len(phases) else None

        for e in ends:
            rec = next((i for i in range(len(phases) - 1, -1, -1)
                        if phases[i]["phase"] == "recording" and phases[i]["t"] <= e["t"]),
                       None)
            if rec is None:
                continue
            row = {"run": run, "session": session, "scene": e.get("scene"),
                   "collector": e.get("collector"), "outcome": e.get("outcome"),
                   "success": e.get("success"), "frames": e.get("frames"),
                   "recording": e["t"] - phases[rec]["t"]}
            # before: nearest approach / gate going back
            for name in ("approach", "gate"):
                j = next((i for i in range(rec - 1, -1, -1) if phases[i]["phase"] == name),
                         None)
                row[name] = dur(j) if j is not None else None
            # after: the homing and reset_wait that follow, up to the next gate
            nxt_gate = next((i for i in range(rec + 1, len(phases))
                             if phases[i]["phase"] == "gate"), None)
            stop = nxt_gate if nxt_gate is not None else len(phases)
            for name in ("homing", "reset_wait"):
                row[name] = sum(dur(i) or 0.0 for i in range(rec + 1, stop)
                                if phases[i]["phase"] == name) or None
            gate_i = next((i for i in range(rec - 1, -1, -1)
                           if phases[i]["phase"] == "gate"), None)
            row["cycle"] = (phases[nxt_gate]["t"] - phases[gate_i]["t"]
                            if gate_i is not None and nxt_gate is not None else None)
            rows.append(row)
    return rows


def describe_s(values) -> dict:
    v = [x for x in values if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v, float)
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max())}


# ------------------------------------------------------------------ report
def build_report(episodes: list, cycles: list, fps: float, camera_fps: float,
                 gap_min: float) -> tuple:
    # camera_fps is reported for reference; drops come from the counters
    pooled: dict = {}
    frames_rows, ep_rows = [], []
    sessions = split_sessions(episodes, gap_min) if episodes else []
    for ep, (sid, since_s) in zip(episodes, sessions):
        m = frame_measures(ep)
        for k, v in m.items():
            pooled.setdefault(k, []).extend(np.asarray(v, float).tolist())
        n = len(ep["cols"]["frame"])
        for i in range(n):
            r = {"file": ep["file"], "episode": ep["episode"], "i": i}
            for k, v in m.items():
                j = i if k != "loop_interval" else i - 1
                r[k] = float(v[j]) if 0 <= j < len(v) else None
            frames_rows.append(r)
        er = {"file": ep["file"], "episode": ep["episode"], "uid": ep["uid"],
              "collector": ep["collector"], "session": sid,
              "minutes_since_session_start": since_s / 60.0, "frames": n}
        for k, v in m.items():
            er[f"{k}_mean_ms"] = float(np.mean(v) * 1e3) if len(v) else None
        if len(m.get("loop_interval", [])) > 1:
            er["loop_interval_sd_ms"] = float(np.std(m["loop_interval"], ddof=1) * 1e3)
        for cam in TIMING_CAMERAS:
            er[f"{cam}_drop_rate"] = drop_rate(ep, cam)
            er[f"{cam}_repeat_rate"] = repeat_rate(ep, cam)
        ep_rows.append(er)

    summary: dict = {"episodes": len(episodes), "frames": len(frames_rows),
                     "nominal_fps": fps, "nominal_interval_ms": 1000.0 / fps,
                     "camera_fps": camera_fps, "measures": {}}
    for k, v in pooled.items():
        summary["measures"][k] = describe_ms(v)
    if pooled.get("loop_interval"):
        li = np.asarray(pooled["loop_interval"], float)
        rate = 1.0 / li[li > 0]
        summary["loop_rate_hz"] = {"mean": float(rate.mean()),
                                   "p1": float(np.percentile(rate, 1)),
                                   "p50": float(np.percentile(rate, 50)),
                                   "p99": float(np.percentile(rate, 99))}
        d = summary["measures"]["loop_interval"]
        summary["jitter_ms"] = {"sd": d["sd"], "p99_minus_p1": d["p99"] - d["p1"],
                                "mean_abs_error_from_nominal":
                                    float(np.mean(np.abs(li * 1e3 - 1000.0 / fps)))}
    for cam in TIMING_CAMERAS:
        for kind in ("drop_rate", "repeat_rate"):
            rates = [r[f"{cam}_{kind}"] for r in ep_rows
                     if r.get(f"{cam}_{kind}") is not None]
            if rates:
                summary[f"{cam}_{kind}"] = describe_s(rates)
    drift = {}
    xs = [r["minutes_since_session_start"] / 60.0 for r in ep_rows]
    for key in ("loop_interval_mean_ms", "loop_interval_sd_ms", "action_to_frame_mean_ms",
                "robot_state_age_mean_ms", "agentview_age_mean_ms",
                "eye_in_hand_age_mean_ms", "camera_skew_mean_ms"):
        pts = [(x, r[key]) for x, r in zip(xs, ep_rows) if r.get(key) is not None]
        if pts:
            fit = ols_slope([p[0] for p in pts], [p[1] for p in pts])
            if "slope" in fit:
                fit["unit"] = "ms per hour"
                fit["hours_covered"] = max(p[0] for p in pts)
            drift[key] = fit
    summary["drift"] = drift
    summary["sessions"] = len({r["session"] for r in ep_rows})
    cyc = {name: describe_s([r.get(name) for r in cycles])
           for name in (*CYCLE_PHASES, "cycle")}
    cyc["episodes"] = len(cycles)
    cyc["outcomes"] = dict(Counter(str(r.get("outcome")) for r in cycles))
    summary["cycle_seconds"] = cyc
    hists = {k: histogram_ms(v) for k, v in pooled.items()}
    return summary, frames_rows, ep_rows, hists


def _write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def print_summary(s: dict) -> None:
    print(f"에피소드 {s['episodes']} · 프레임 {s['frames']} · 세션 {s.get('sessions', 0)} · "
          f"명목 {s['nominal_fps']} Hz ({s['nominal_interval_ms']:.1f} ms)")
    if "loop_rate_hz" in s:
        r, j = s["loop_rate_hz"], s["jitter_ms"]
        print(f"루프 주파수 평균 {r['mean']:.2f} Hz (p1 {r['p1']:.2f} / p99 {r['p99']:.2f}) · "
              f"지터 sd {j['sd']:.2f} ms, p99-p1 {j['p99_minus_p1']:.2f} ms")
    print(f"\n{'measure (ms)':22s} {'n':>7s} {'mean':>8s} {'sd':>7s} {'p1':>8s} "
          f"{'p50':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}")
    for k, d in s["measures"].items():
        if d.get("n"):
            print(f"{k:22s} {d['n']:7d} {d['mean']:8.2f} {d['sd']:7.2f} {d['p1']:8.2f} "
                  f"{d['p50']:8.2f} {d['p95']:8.2f} {d['p99']:8.2f} {d['max']:8.2f}")
    for cam in TIMING_CAMERAS:
        for kind, label in (("drop_rate", "드롭률"), ("repeat_rate", "같은 프레임 반복률")):
            if f"{cam}_{kind}" in s:
                d = s[f"{cam}_{kind}"]
                print(f"{cam} {label} (에피소드별) 평균 {d['mean']:.4f} · 최대 {d['max']:.4f}")
    if s["drift"]:
        print("\n장시간 열화 (세션 시작 후 시간에 대한 기울기, ms/시간, 95% 구간)")
        for k, d in s["drift"].items():
            if "slope" in d:
                print(f"  {k:28s} {d['slope']:+8.3f}  [{d['ci95'][0]:+.3f}, "
                      f"{d['ci95'][1]:+.3f}]  n={d['n']} · {d['hours_covered']:.2f} h")
    c = s["cycle_seconds"]
    if c["episodes"]:
        print(f"\n사이클 타임 (초) · 에피소드 {c['episodes']} · {c['outcomes']}")
        for name in (*CYCLE_PHASES, "cycle"):
            d = c[name]
            if d.get("n"):
                print(f"  {name:11s} n={d['n']:4d}  평균 {d['mean']:6.2f} ± {d['sd']:5.2f}  "
                      f"중앙 {d['p50']:6.2f}  p95 {d['p95']:6.2f}")


# ------------------------------------------------------------------ selftest
def _selftest() -> None:
    from mstack.data.dataset_schema import TIMING_REQUIRED

    rng = np.random.default_rng(0)
    d = Path(tempfile.mkdtemp(prefix="timing_report_"))
    path = d / "scene_006.hdf5"
    t0 = 1_789_900_000.0
    with h5py.File(path, "w") as f:
        f.create_group("metadata")
        for e in range(4):
            n = 100
            base = t0 + e * 60                     # one episode per minute
            frame = base + np.arange(n) * 0.05 + rng.normal(0, 0.002, n)
            frame.sort()
            g = f.create_group(f"episode_{e:03d}")
            g.attrs["episode_uid"] = f"EP-S006-I000-E{e:03d}"
            tg = g.create_group("timing")
            tg.create_dataset("frame", data=frame)
            tg.create_dataset("action", data=frame - 0.004)
            tg.create_dataset("robot_state", data=frame - 0.001 - 0.001 * e)  # drifts
            tg.create_dataset("agentview_host", data=frame - 0.020)
            tg.create_dataset("eye_in_hand_host", data=frame - 0.025)
            tg.create_dataset("agentview_device", data=frame - 0.020 - 0.030)
            fn = (np.arange(n) * 1.5).astype(np.int64)
            seq = fn.copy()
            fn[50:] += 3 if e == 2 else 0          # device ran 3 frames ahead: 3 lost
            seq[10] = seq[9]                       # one image recorded twice
            fn[10] = fn[9]
            tg.create_dataset("agentview_frame_no", data=fn)
            tg.create_dataset("agentview_node_seq", data=seq)
            tg.attrs["agentview_domain"] = "global_time"
            assert all(k in tg for k in TIMING_REQUIRED)
    eps = read_episodes(d)
    assert len(eps) == 4
    phases = []
    t = t0
    for e in range(3):
        for ph, dt in (("gate", 2.0), ("approach", 0.4), ("recording", 5.0),
                       ("homing", 1.2), ("reset_wait", 3.0)):
            phases.append({"run": "R", "session": "S", "scene": "S006", "collector": "c",
                           "practice": False, "t": t, "phase": ph})
            if ph == "recording":
                phases.append({"run": "R", "session": "S", "scene": "S006",
                               "collector": "c", "practice": False, "t": t + dt,
                               "event": "episode_end", "outcome": "save",
                               "success": True, "frames": 100})
            t += dt
    phases.append({"run": "R", "session": "S", "scene": "S006", "collector": "c",
                   "practice": False, "t": t, "phase": "gate"})
    phases.append({"run": "P", "session": "S", "scene": "S006", "collector": "c",
                   "practice": True, "t": t, "phase": "recording"})
    cycles = cycles_from_phases(phases)
    assert len(cycles) == 3, cycles
    for c in cycles:
        # epoch-sized floats: sub-microsecond tolerance, not 1e-9
        assert abs(c["recording"] - 5.0) < 1e-5 and abs(c["approach"] - 0.4) < 1e-5, c
        assert abs(c["homing"] - 1.2) < 1e-5 and abs(c["reset_wait"] - 3.0) < 1e-5, c
        assert abs(c["gate"] - 2.0) < 1e-5 and abs(c["cycle"] - 11.6) < 1e-5, c
    s, frames, ep_rows, hists = build_report(eps, cycles, fps=20, camera_fps=30,
                                             gap_min=10)
    m = s["measures"]
    assert abs(m["loop_interval"]["mean"] - 50.0) < 1.0, m["loop_interval"]
    # epoch-sized floats carry ~0.2 us of rounding
    assert abs(m["action_to_frame"]["mean"] - 4.0) < 1e-3
    assert abs(m["agentview_age"]["mean"] - 20.0) < 1e-3
    assert abs(m["agentview_pipeline"]["mean"] - 30.0) < 1e-3
    assert abs(m["camera_skew"]["mean"] - 5.0) < 1e-3
    assert 19.0 < s["loop_rate_hz"]["mean"] < 21.5
    assert s["sessions"] == 1
    fit = s["drift"]["robot_state_age_mean_ms"]
    assert abs(fit["slope"] - 60.0) < 1e-2, fit             # 1 ms per minute
    li = s["drift"]["loop_interval_mean_ms"]
    assert li["ci95"][0] < 0 < li["ci95"][1], li            # no drift in the loop
    rates = [r["agentview_drop_rate"] for r in ep_rows]
    assert abs(rates[2] - 3 / (148 + 3)) < 1e-9, rates
    assert all(r == 0.0 for i, r in enumerate(rates) if i != 2), rates
    reps = [r["agentview_repeat_rate"] for r in ep_rows]
    assert all(abs(x - 1 / 99) < 1e-9 for x in reps), reps
    assert s["cycle_seconds"]["episodes"] == 3
    assert hists["loop_interval"] and sum(c for *_, c in hists["loop_interval"]) == 396
    print("timing_report selftest 통과")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path.home() / "libero_datasets")
    ap.add_argument("--scene", default="", help="e.g. S006")
    ap.add_argument("--collector", default="")
    ap.add_argument("--since", default="", help="session tag lower bound, e.g. 20260918")
    ap.add_argument("--include-practice", action="store_true")
    ap.add_argument("--phase-log", type=Path, default=None)
    ap.add_argument("--session-gap", type=float, default=10.0,
                    help="minutes between episodes that start a new session")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    st = load_station()
    fps = float(st.fps)
    cams = [c.fps for c in (getattr(st, "cameras", None) or {}).values()
            if getattr(c, "fps", None)]
    camera_fps = float(cams[0]) if cams else 30.0
    episodes = read_episodes(args.root, args.scene)
    if args.collector:
        episodes = [e for e in episodes if e["collector"] == args.collector]
    lines = load_phases(args.phase_log or phase_log_path())
    cycles = cycles_from_phases(lines, args.scene, args.collector, args.since,
                                args.include_practice)
    if not episodes and not cycles:
        print("timing 이 있는 에피소드도, 단계 로그도 없습니다 "
              f"(root={args.root}, phase_log={args.phase_log or phase_log_path()})")
        return
    s, frames, ep_rows, hists = build_report(episodes, cycles, fps, camera_fps,
                                             args.session_gap)
    print_summary(s)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
        _write_csv(args.out / "frames.csv", frames)
        _write_csv(args.out / "episodes.csv", ep_rows)
        _write_csv(args.out / "cycles.csv", cycles)
        for k, h in hists.items():
            _write_csv(args.out / f"hist_{k}.csv",
                       [{"lo_ms": a, "hi_ms": b, "count": c} for a, b, c in h])
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
