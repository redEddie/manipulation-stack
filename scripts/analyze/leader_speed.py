"""리더 속도 감사 -- 놓침 임계 ``LEADER_DROP_SPEED_RAD_S`` 를 실측으로 다시 본다.

수집 워커는 리더의 **피치 관절**(J2/J4/J6) 속도를 100 ms 창으로 관절별 평균
낸 뒤 그 L2 노름이 2.1 rad/s 를 넘으면 팔을 세우고 에피소드를 폐기한다.
롤 관절은 보지 않는다 -- 중력이 롤 축에 모멘트를 못 만들어 낙하 때 오히려
정상보다 느리다. 그 2.1 은 낙하 **1건**과 정상 5개(28.9초)에서 나왔다.
표본이 얇으므로, 세션이 쌓이면 이 도구로 다시 봐야 한다.

1 kHz 원시 로그(``mstack.comm.robot_raw_logger``)의 ``q_des`` 가 곧 워커가
보낸 리더 자세라, 실제로 흘렀던 스트림을 그대로 되돌릴 수 있다. ``--replay``
는 지표를 다시 구현하지 않고 **워커가 쓰는 그 클래스**에 흘린다 -- 도구와
실물이 어긋나면 감사가 무의미하다.

읽기만 한다. 사용:

    python scripts/analyze/leader_speed.py
    python scripts/analyze/leader_speed.py --sweep      # 임계 후보별 발동 수
    python scripts/analyze/leader_speed.py --selftest

읽는 법. **정상 에피소드의 최고값이 임계에 얼마나 붙었나**를 본다. 1.4 근처에
머물면 여유가 있고, 1.8 을 넘기 시작하면 오검이 임박한 것이다. 오검은 한
번만 나도 조작자가 안전층을 꺼버리므로, 그 전에 알아야 한다.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mstack.collect.leader_guard import LeaderDropGuard  # noqa: E402
from mstack.config.constants import LEADER_DROP_SPEED_RAD_S  # noqa: E402

#: 이 단계의 구간만 본다. 다른 단계의 이동은 워커가 만든 램프라 리더가 아니다.
PHASE = "recording"

#: 이보다 짧은 구간은 창(100 ms)에 비해 표본이 안 된다.
MIN_SECONDS = 0.5


def episodes(path: str):
    """(에피소드 이름, 리더 시각, 리더 자세, 길이초) 를 내놓는다."""
    z = np.load(path)
    if "q_des" not in z.files:
        return                      # q_des 이전에 찍힌 창
    t, qdes = z["t"], z["q_des"]
    ph = json.loads(str(z["phases"]))
    for i, p in enumerate(ph):
        if p.get("phase") != PHASE:
            continue
        a = p.get("t") or float(t[0])
        b = (ph[i + 1].get("t") if i + 1 < len(ph) else None) or float(t[-1])
        m = (t >= a) & (t < b)
        if m.sum() < MIN_SECONDS * 1000:
            continue
        tt, qq = t[m], qdes[m]
        # q_des 는 1 kHz 로그 안에서 100 Hz 계단이다. 계단이 바뀌는 지점만
        # 뽑아야 실제 리더 샘플이 된다 -- 안 그러면 같은 값이 10번 들어와
        # 속도가 10분의 1 로 희석된다.
        idx = np.flatnonzero(np.any(np.diff(qq, axis=0) != 0, axis=1)) + 1
        if len(idx) < 20:
            continue
        yield Path(path).stem, tt[idx], qq[idx], float(m.sum()) / 1000.0


def replay(tt, qq, limit: float):
    """워커가 쓰는 그 클래스에 흘린다. (최고값, 첫 발동 시각) 반환."""
    g = LeaderDropGuard(limit=limit)
    trip = None
    for k in range(len(tt)):
        v = g.update(qq[k], float(tt[k]))
        if trip is None and g.tripped(v):
            trip = float(tt[k] - tt[0])
    return g.peak, trip


def report(rows, limit: float) -> int:
    if not rows:
        print("원시 로그에서 recording 구간을 못 찾았다.")
        return 1
    print(f"임계 {limit:.2f} rad/s\n")
    print(f"  {'창':<20}{'초':>6}{'최고 L2':>9}{'여유':>8}{'첫 발동':>9}")
    peaks = []
    for name, dur, peak, trip in rows:
        peaks.append(peak)
        tag = f"+{trip:.2f}s" if trip is not None else "-"
        print(f"  {name:<20}{dur:6.1f}{peak:9.2f}{limit - peak:8.2f}{tag:>9}")
    p = np.array(peaks)
    fired = sum(1 for r in rows if r[3] is not None)
    print(f"\n  에피소드 {len(rows)}개, 총 {sum(r[1] for r in rows):.1f}초")
    print(f"  최고 L2: 중앙 {np.median(p):.2f}  최대 {p.max():.2f}")
    print(f"  발동 {fired}개")

    head = limit - p.max()
    if fired:
        print("\n  발동한 에피소드가 있다. 실제로 리더를 놓친 것이면 정상 동작이다 --"
              "\n  아니라면 오검이므로 임계를 올려야 한다. 그 에피소드의 창을 열어"
              "\n  그 시각에 무슨 일이 있었는지 먼저 볼 것.")
    elif head < 0.4:
        print(f"\n  ** 여유가 {head:.2f} rad/s 뿐이다. 오검이 임박했다 --"
              f"\n  임계를 {p.max() * 1.6:.1f} 근처로 올리는 것을 검토할 것. **")
    else:
        print(f"\n  여유 {head:.2f} rad/s. 정상 쪽은 아직 넉넉하다.")
    return 0


def sweep(rows_raw, limit: float) -> None:
    """임계 후보별로 몇 개가 발동하는지. 고원 위에 있는지 절벽인지 본다."""
    print("\n임계 후보별 발동 에피소드 수")
    for cand in (1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5):
        n = sum(1 for _, tt, qq, _ in rows_raw if replay(tt, qq, cand)[1] is not None)
        mark = "  <- 지금" if abs(cand - limit) < 1e-9 else ""
        print(f"  {cand:.1f} rad/s : {n:3d} / {len(rows_raw)}{mark}")


def _selftest() -> int:
    """합성 스트림으로 되돌리기가 맞는지. 로그도 로봇도 없이 돈다."""
    hz, sec = 100.0, 2.0
    for target, should_trip in ((1.5, False), (3.0, True)):
        per = target / np.sqrt(2)
        tt = np.arange(int(sec * hz)) / hz
        qq = np.zeros((len(tt), 7))
        qq[:, 1] = per * tt
        qq[:, 3] = per * tt
        peak, trip = replay(tt, qq, LEADER_DROP_SPEED_RAD_S)
        assert abs(peak - target) < 0.02, f"{target}: 지표 {peak}"
        assert (trip is not None) == should_trip, f"{target}: 발동 {trip}"
    print("선택검사 통과 (1.5 통과 / 3.0 발동)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path.home() / "libero_gui_logs" / "robot_raw"))
    ap.add_argument("--limit", type=float, default=LEADER_DROP_SPEED_RAD_S)
    ap.add_argument("--sweep", action="store_true", help="임계 후보별 발동 수도 낸다")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    files = sorted(glob.glob(str(Path(args.dir) / "raw_*.npz")))
    if not files:
        print(f"{args.dir} 에 raw_*.npz 가 없다.")
        return 1
    print(f"{len(files)} 개 창, {args.dir}\n")

    raw, rows = [], []
    for f in files:
        for name, tt, qq, dur in episodes(f):
            raw.append((name, tt, qq, dur))
            peak, trip = replay(tt, qq, args.limit)
            rows.append((name, dur, peak, trip))
    rc = report(rows, args.limit)
    if args.sweep and raw:
        sweep(raw, args.limit)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
