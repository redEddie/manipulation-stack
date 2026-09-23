"""homing 경로: "위로"와 "집으로"를 섞는다 (2026-09-23).

옛 경로는 "수직 10cm 리프트 -> 홈까지 직선" 두 단계였다. 실측 2,434개 종료
자세에서 그 고정 리프트가 **양쪽으로 틀렸다**: 80.3% 는 홈 높이(0.254 m)보다
낮게 끝나 10cm 를 올려도 35.6% 가 여전히 낮았고, 19.7% 는 이미 홈보다 높은데
거기에 10cm 를 더 올리고 있었다 (최고 0.618 m).

여기서 지키는 것:

1. **적응성.** 올라가는 높이를 시작 높이가 정한다. 낮게 끝나면 많이, 이미
   높으면 거의 안 올린다. 이것이 고정 리프트를 버린 이유다.
2. **수렴.** 진행도 스케줄이 없으면 Z_CLEAR 를 홈 높이 위로 올리는 순간 극한
   순환에 빠진다 -- 홈에 가까워지려면 내려가야 하고, 내려가면 높이 항이 다시
   켜진다. 실측으로 400/400 이 미수렴이었다. 스케줄이 그것을 끊는다.
3. **여유.** 수평으로 움직이는 동안의 최저 높이가 옛 경로보다 높아야 한다.
   그것이 애초에 EE 경로 homing 을 만든 이유(테이블을 쓸지 않기)다.
4. **속도 상한.** _densify 가 tick 당 이동을 cfg.home_tick_dq 아래로 묶는다.

옛 경로 대비 실측 (실제 종료 자세 200개): IK 195->197/200, homing 0.99->0.90 s,
수평 이동 중 최저 높이 최악값 0.136->0.243 m, 최고 높이 0.618->0.450 m.
"""
import sys
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
sys.argv = ["t"]

import numpy as np  # noqa: E402

from mstack.collect.worker import (  # noqa: E402
    HOME_BLEND_M,
    HOME_CLEAR_MARGIN_M,
    HOME_EE_STEP_M,
    HOME_MAX_STEPS,
    HOME_SCHED_POWER,
    CollectionWorker,
    WorkerConfig,
)
from mstack.robots import fr3_kinematics as K  # noqa: E402
from mstack.robots.franka_fr3 import FR3_RESET_POSES  # noqa: E402

RESET_Q = np.array(FR3_RESET_POSES["libero"][:7], dtype=float)
P_HOME = K.fk(RESET_Q)[:3, 3]
Z_HOME = float(P_HOME[2])

w = CollectionWorker.__new__(CollectionWorker)
w._reset_q = RESET_Q
w.cfg = WorkerConfig(task_name="t", language_instruction="l", data_root="/tmp")


def ee_path(q0):
    path = w._home_trajectory(q0)
    if path is None:
        return None
    return np.array([K.fk(q)[:3, 3] for q in path])


def _blend_only(p0, z_clear, use_schedule):
    """경로 모양만 (IK 없이). 스케줄의 효과를 따로 보려고 쓴다."""
    p = np.asarray(p0, float).copy()
    d0 = float(np.linalg.norm(P_HOME - p))
    s_done = 0.0
    for i in range(HOME_MAX_STEPS):
        d = P_HOME - p
        dist = float(np.linalg.norm(d))
        if dist < HOME_EE_STEP_M:
            return i
        s_done = max(s_done, min(max((d0 - dist) / max(d0, 1e-9), 0.0), 1.0))
        need = min(max((z_clear - p[2]) / HOME_BLEND_M, 0.0), 1.0)
        if use_schedule:
            need *= (1.0 - s_done) ** HOME_SCHED_POWER
        v = need * np.array([0.0, 0.0, 1.0]) + (1.0 - need) * (d / dist)
        n = float(np.linalg.norm(v))
        v = (d / dist) if n < 1e-9 else (v / n)
        p = p + HOME_EE_STEP_M * v
    return None                                    # 미수렴


def main() -> None:
    rng = np.random.default_rng(0)

    # ---------------------------------------------- 1. 적응성
    # 같은 xy 에서 시작 높이만 바꾸면, 올라가는 양이 그에 맞춰 줄어야 한다.
    rises, starts = [], []
    for dz in (-0.15, -0.05, 0.0, +0.10):
        q0 = RESET_Q.copy()
        T = K.fk(q0)
        T[2, 3] += dz
        q = w._ik_posture(K, T, q0, RESET_Q)
        P = ee_path(q)
        if P is None:
            continue
        z0 = float(K.fk(q)[2, 3])
        starts.append(z0)
        rises.append(float(P[:, 2].max()) - z0)
    assert len(rises) >= 3, f"경로를 거의 못 만들었다 ({len(rises)})"
    assert all(rises[i] >= rises[i + 1] - 1e-9 for i in range(len(rises) - 1)), (
        f"시작이 높아지는데 더 올라간다: 시작 {np.round(starts,3)} 상승 "
        f"{np.round(rises,3)} -- 고정 리프트를 버린 이유가 사라진다")
    assert rises[-1] < 0.02, (
        f"이미 홈보다 {starts[-1]-Z_HOME:.3f} m 높은데 {rises[-1]:.3f} m 를 "
        "더 올린다 -- 옛 고정 리프트와 같은 낭비다")
    print(f"1. 상승량이 시작 높이에 적응한다 OK "
          f"(시작 {np.round(starts,3).tolist()} -> 상승 {np.round(rises,3).tolist()})")

    # ---------------------------------------------- 2. 스케줄이 순환을 끊는다
    # 홈보다 충분히 위를 목표로 두면, 스케줄 없이는 홈 근처에서 내려가는 순간
    # 높이 항이 다시 켜져 수렴하지 않는다.
    z_high = Z_HOME + 0.06
    starts_xyz = [P_HOME + np.array([dx, dy, dz])
                  for dx, dy, dz in rng.uniform(-0.15, 0.15, (40, 3))]
    no_sched = sum(1 for p in starts_xyz if _blend_only(p, z_high, False) is None)
    with_sched = sum(1 for p in starts_xyz if _blend_only(p, z_high, True) is None)
    assert with_sched == 0, f"스케줄이 있는데도 {with_sched}/40 미수렴"
    assert no_sched > 0, (
        "스케줄 없이도 전부 수렴했다 -- 이 시험이 아무것도 안 지키고 있다. "
        f"Z_CLEAR={z_high:.3f} 이 홈({Z_HOME:.3f})보다 충분히 높은지 봐라")
    print(f"2. 진행도 스케줄이 극한 순환을 끊는다 OK "
          f"(Z_CLEAR=홈+0.06 에서 미수렴 {no_sched}/40 -> {with_sched}/40)")

    # ---------------------------------------------- 3. 여유 + 속도 상한
    qs = [np.clip(RESET_Q + rng.uniform(-0.9, 0.9, 7),
                  K.FR3_Q_MIN + 0.1, K.FR3_Q_MAX - 0.1) for _ in range(60)]
    made, worst_tick, sweeps = 0, 0.0, []
    for q0 in qs:
        path = w._home_trajectory(q0)
        if path is None:
            continue
        made += 1
        worst_tick = max(worst_tick, float(
            np.abs(np.diff(np.vstack([q0] + list(path)), axis=0)).max()))
        P = np.array([K.fk(q)[:3, 3] for q in path])
        p0 = K.fk(q0)[:3, 3]
        moved = np.linalg.norm(P[:, :2] - p0[:2], axis=1) > 0.03
        sweeps.append(float(P[moved, 2].min()) if moved.any() else float(P[:, 2].max()))
    assert made > 40, f"경로를 거의 못 만들었다 ({made}/60) -- 다른 것이 깨졌다"
    assert worst_tick <= w.cfg.home_tick_dq + 1e-9, (
        f"tick 당 {worst_tick:.4f} rad ({worst_tick * 100:.2f} rad/s) 를 명령한다")
    # 옛 2단계 경로의 최악값 0.136 m. 새 경로는 그보다 넉넉해야 한다.
    assert min(sweeps) > 0.136, (
        f"수평 이동 중 최저 높이 {min(sweeps):.3f} m -- 옛 경로(0.136)보다 "
        "나쁘면 EE 경로 homing 을 만든 이유가 사라진다")
    print(f"3. tick 당 {worst_tick:.4f} rad <= {w.cfg.home_tick_dq}, "
          f"수평 이동 중 최저 높이 {min(sweeps):.3f} m > 0.136 (옛 경로) OK "
          f"({made}/60 경로)")

    # ---------------------------------------------- 4. 홈에 도착한다
    ends = [float(np.abs(w._home_trajectory(q)[-1] - RESET_Q).max())
            for q in qs[:20] if w._home_trajectory(q) is not None]
    assert ends and max(ends) < 0.35, f"홈 잔차가 크다: {max(ends):.3f} rad"
    print(f"4. 홈 도착 잔차 최대 {max(ends):.4f} rad OK "
          f"(남는 것은 호출자의 _ramp_to 가 정리한다)")

    # ---------------------------------------------- 5. 상수의 뜻
    assert HOME_CLEAR_MARGIN_M > 0, (
        "Z_CLEAR 가 홈 높이면 수평 이동이 홈 높이에서 일어난다 -- 여유가 0 이다")
    print(f"5. Z_CLEAR = 홈 {Z_HOME:.3f} + {HOME_CLEAR_MARGIN_M} "
          f"= {Z_HOME + HOME_CLEAR_MARGIN_M:.3f} m, 게이트 폭 {HOME_BLEND_M} m OK")

    print("\nhoming 블렌딩 인수 통과")


if __name__ == "__main__":
    main()
