"""설정점 간격 감사 — 안전 상한 ``MAX_SETPOINT_GAP_RAD`` 을 얼마까지 내릴 수 있나.

노드는 설정점(``q_des``)이 필터 출력(``q_cmd``)에서 상한보다 멀면 명령을 받지
않고 팔을 세운다. 상한은 0.9 rad 에서 시작했는데 그 근거는 "2026-09-10 실측
최대 0.532 의 1.7배" 하나뿐이다 -- **내리는 것이 전제**이고, 얼마까지 내릴 수
있는지는 추측이 아니라 이 도구가 답한다.

1 kHz 원시 로그(``mstack.comm.robot_raw_logger``)에는 ``q_des`` 와 ``q_cmd``
가 둘 다 들어 있어, 실제로 벌어졌던 간격을 단계별로 그대로 잴 수 있다.

읽기만 한다. 사용:

    python scripts/analyze/setpoint_gap.py
    python scripts/analyze/setpoint_gap.py --dir ~/libero_gui_logs/robot_raw
    python scripts/analyze/setpoint_gap.py --selftest

읽는 법. **recording 의 최대값**이 상한을 정한다 -- 다른 단계는 램프가
만들어서 작고 예측 가능하다. 여유(headroom)는 조작자가 얼마나 더 험하게 움직일
수 있느냐를 사는 값이므로, 표본이 몇 세션밖에 없다면 크게 남긴다.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mstack.robots.franka_fr3 import MAX_SETPOINT_GAP_RAD  # noqa: E402

#: 상한 후보. 실측 최대에 이 배수를 곱해 추천한다. 1.7 은 지금 쓰는 0.9 가
#: 실측 0.532 에 대해 가졌던 배수 그대로다 -- 추천이 현재 값과 같은 규칙에서
#: 나와야 "얼마나 내려가는지"가 의미를 갖는다.
RECOMMEND_FACTOR = 1.7

#: 이보다 표본이 적은 단계는 분포를 믿지 않고 표시만 한다.
MIN_TICKS = 100


def gaps_by_phase(files) -> "dict[str, list]":
    """파일들에서 단계 이름 -> 관절별 최대 간격 배열들."""
    out: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        if "q_des" not in z.files or "q_cmd" not in z.files:
            continue  # q_cmd 이전에 찍힌 창
        t, qdes, qcmd = z["t"], z["q_des"], z["q_cmd"]
        ph = json.loads(str(z["phases"]))
        if not ph:
            ph = [{"phase": "(단계 표지 없음)", "t": float(t[0])}]
        for i, p in enumerate(ph):
            t0 = p.get("t") or float(t[0])
            t1 = (ph[i + 1].get("t") if i + 1 < len(ph) else None) or float(t[-1])
            m = (t >= t0) & (t < t1)
            if not m.any():
                continue
            g = np.abs(qdes[m] - qcmd[m]).max(axis=1)
            out.setdefault(str(p.get("phase", "?")), []).append(g)
    return out


def report(by_phase: "dict[str, list]", limit: float) -> int:
    if not by_phase:
        print("원시 로그에서 q_des/q_cmd 를 가진 창을 못 찾았다.")
        return 1
    print(f"현재 상한 {limit:.2f} rad\n")
    print(f"  {'단계':<14}{'틱':>9}{'p50':>9}{'p95':>9}{'p99':>9}"
          f"{'max':>9}{'상한초과':>10}")
    rec_max = 0.0
    overall = []
    for name, chunks in sorted(by_phase.items()):
        g = np.concatenate(chunks)
        overall.append(g)
        over = int((g > limit).sum())
        thin = "  (표본 적음)" if len(g) < MIN_TICKS else ""
        print(f"  {name:<14}{len(g):>9}{np.percentile(g, 50):>9.3f}"
              f"{np.percentile(g, 95):>9.3f}{np.percentile(g, 99):>9.3f}"
              f"{g.max():>9.3f}{over:>10}{thin}")
        if name == "recording":
            rec_max = float(g.max())

    allg = np.concatenate(overall)
    print(f"\n  전체 최대 {allg.max():.3f} rad, 상한 초과 {int((allg > limit).sum())} 틱")

    # 후보 상한마다 "몇 번 세웠을까". 틱 수가 아니라 **연속 덩어리 수**를 센다
    # -- 한 번 걸리면 여러 틱 이어지므로, 틱을 세면 사건 수가 부풀려진다.
    print("\n  후보 상한별로 이 데이터에서 팔이 섰을 횟수")
    for cand in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        n_ev = 0
        for chunks in by_phase.values():
            for g in chunks:
                o = (g > cand).astype(np.int8)
                n_ev += int((np.diff(np.r_[0, o, 0]) == 1).sum())
        mark = "  <- 지금" if abs(cand - limit) < 1e-9 else ""
        print(f"    {cand:.2f} rad : {n_ev:4d} 회{mark}")

    if rec_max > 0:
        rec = np.ceil(rec_max * RECOMMEND_FACTOR * 20) / 20  # 0.05 단위
        print(f"\n  recording 실측 최대 {rec_max:.3f} rad "
              f"x{RECOMMEND_FACTOR} -> 추천 {rec:.2f} rad")
        if rec < limit:
            print(f"  현재 {limit:.2f} 에서 {rec:.2f} 로 내릴 수 있다. "
                  f"MSTACK_MAX_SETPOINT_GAP={rec:.2f} 으로 먼저 시험할 것.")
        else:
            print(f"  현재 {limit:.2f} 를 내릴 근거가 아직 없다 "
                  f"(표본이 더 필요하거나 이미 빠듯하다).")
    else:
        print("\n  recording 구간이 없어 추천을 낼 수 없다.")
    return 0


def _selftest() -> int:
    """합성 데이터로 집계가 맞는지. 로봇도 로그도 없이 돈다."""
    n = 500
    g_rec = np.full(n, 0.10)
    g_rec[10:14] = 0.55          # 덩어리 1
    g_rec[300] = 0.95            # 덩어리 2 (0.9 초과)
    by = {"recording": [g_rec], "homing": [np.full(50, 0.15)]}
    assert float(np.concatenate(by["recording"]).max()) == 0.95
    o = (g_rec > 0.5).astype(np.int8)
    assert int((np.diff(np.r_[0, o, 0]) == 1).sum()) == 2, "덩어리 세기 오류"
    o = (g_rec > 0.9).astype(np.int8)
    assert int((np.diff(np.r_[0, o, 0]) == 1).sum()) == 1
    rc = report(by, 0.9)
    assert rc == 0
    print("\n선택검사 통과")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(Path.home() / "libero_gui_logs" / "robot_raw"),
                    help="원시 로그 디렉터리")
    ap.add_argument("--limit", type=float, default=MAX_SETPOINT_GAP_RAD,
                    help="지금 쓰는 상한 (rad)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    files = sorted(glob.glob(str(Path(args.dir) / "raw_*.npz")))
    if not files:
        print(f"{args.dir} 에 raw_*.npz 가 없다.")
        return 1
    print(f"{len(files)} 개 창, {args.dir}\n")
    return report(gaps_by_phase(files), args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
