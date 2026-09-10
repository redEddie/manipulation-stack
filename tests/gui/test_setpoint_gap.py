"""설정점 간격 안전 가드 -- 먼 명령은 받지 않고 팔을 그 자리에 세운다.

2026-09-10 이전에는 리더를 놓치거나 난폭하게 흔들면 libfranka 가 명령을
거부하며 acceleration_discontinuity 반사로 제어 루프를 죽였다. 설계된
안전장치가 아니라 부작용이었지만 실제로 팔을 세우는 유일한 것이었고,
v/a 를 올리고(2026-08-19) v_max 테이퍼의 이산 항을 채우면서(2026-09-10)
그 경로가 사라졌다. 추종이 좋아진 만큼 필터가 **떨어진 리더를 끝까지 충실히
쫓아가게** 된 것이다. 이 가드가 그 자리를 대신한다.

이 검사가 지켜야 하는 것 중 조용히 깨지기 쉬운 것들:

  * 실측 최악(recording 0.532 rad)은 **반드시 통과**해야 한다. 여기가
    빡빡해지면 정상 수집이 툭툭 끊긴다.
  * 멈추는 방법은 ``_desired_q`` 를 필터 출력으로 래치하는 것이다. 그냥
    명령을 무시하면 직전 설정점이 남아 팔이 거기까지 계속 간다 -- 리더가
    이미 떨어진 뒤라면 그 목표가 바로 잘못된 자세다.
  * 속도 상태(``_qd_cmd``)는 건드리지 않는다. 0 으로 꽂으면 그 자체가
    거대한 저크이고, 그것이 우리가 없애려던 반사다.
  * 환경변수가 오타여도 검사가 **꺼지지 않는다**.

pylibfranka 는 생성자 안에서만 임포트되므로 로봇 없이 돈다.
"""
import os
import sys
import threading
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from mstack.robots.franka_fr3 import (  # noqa: E402
    MAX_SETPOINT_GAP_RAD,
    SETPOINT_GAP_ENV,
    FrankaFR3Robot,
    _setpoint_gap_limit,
)

#: 2026-09-10 원시 로그 1 kHz 실측 최대 (recording 단계). 정상 수집이
#: 만들어 낸 가장 큰 간격이라, 상한은 언제나 이것보다 위여야 한다.
MEASURED_WORST_RAD = 0.532


# ------------------------------------------------------------ 한계값 해석
os.environ.pop(SETPOINT_GAP_ENV, None)
assert _setpoint_gap_limit() == MAX_SETPOINT_GAP_RAD
assert _setpoint_gap_limit(0.5) == 0.5
os.environ[SETPOINT_GAP_ENV] = "0.6"
assert _setpoint_gap_limit() == 0.6
assert _setpoint_gap_limit(0.5) == 0.5, "생성자 인자가 환경변수를 이겨야 한다"
os.environ[SETPOINT_GAP_ENV] = "0"
assert _setpoint_gap_limit() == 0.0, "0 이면 검사를 끈다 (진단용)"
os.environ[SETPOINT_GAP_ENV] = "0.4rad"
assert _setpoint_gap_limit() == MAX_SETPOINT_GAP_RAD, (
    "못 읽는 값에서 검사가 꺼지면 안 된다 -- 오타 하나로 안전층이 사라진다")
os.environ.pop(SETPOINT_GAP_ENV, None)
assert MAX_SETPOINT_GAP_RAD > MEASURED_WORST_RAD, (
    f"기본 상한 {MAX_SETPOINT_GAP_RAD} 이 실측 최악 {MEASURED_WORST_RAD} 아래다")
print(f"1. 한계값 해석 (기본 {MAX_SETPOINT_GAP_RAD} rad) OK")


# ------------------------------------------------------------------ 가짜 팔
class Fake:
    """``__init__`` 을 건너뛰고 가드가 만지는 상태만 채운다.

    진짜 생성자는 FCI 에 붙으므로 여기서 못 쓴다. 메서드는 클래스에서 그대로
    빌려오므로 검사하는 것은 실제 코드다.
    """

    def __init__(self, limit=MAX_SETPOINT_GAP_RAD, gripper=True, read_only=False):
        self._lock = threading.Lock()
        self._read_only = read_only
        self._use_gripper = gripper
        self._gap_max = limit
        self._gap_peak = 0.0
        self._gap_stops = 0
        self._gap_stopped = False
        self._q_cmd = np.zeros(7)
        self._qd_cmd = np.full(7, 0.7)   # 필터 속도 상태 -- 건드리면 안 된다
        self._desired_q = np.zeros(7)
        self._gripper_target = 0.0

    command_joint_state = FrankaFR3Robot.command_joint_state
    _report_gap_peak = FrankaFR3Robot._report_gap_peak


# -------------------------------------------------- 정상 명령은 그대로 통과
r = Fake()
r.command_joint_state(np.r_[np.full(7, 0.3), 1.0])
assert np.allclose(r._desired_q, 0.3)
assert r._gripper_target == 1.0
assert r._gap_stops == 0

r._q_cmd = np.zeros(7)
r.command_joint_state(np.r_[np.zeros(6), MEASURED_WORST_RAD, 0.0])
assert np.allclose(r._desired_q[6], MEASURED_WORST_RAD), (
    "실측 최악이 걸리면 정상 수집이 끊긴다")
assert r._gap_stops == 0
print(f"2. 정상 명령 통과 (실측 최악 {MEASURED_WORST_RAD} rad 포함) OK")


# ------------------------------------------------------ 상한 초과 -> 급정거
r._q_cmd = np.full(7, 0.2)
r.command_joint_state(np.r_[np.full(7, 1.5), 1.0])
assert np.allclose(r._desired_q, 0.2), (
    f"필터 출력으로 래치돼야 한다 (명령 무시로는 부족): {r._desired_q}")
assert np.allclose(r._qd_cmd, 0.7), "속도 상태를 0 으로 꽂으면 그게 바로 저크다"
assert r._gap_stops == 1 and r._gap_stopped
assert r._gripper_target == 1.0, "그리퍼는 계속 리더를 따라야 한다"

# 걸려 있는 동안 반복해도 계속 서 있다
r.command_joint_state(np.r_[np.full(7, 1.5), 1.0])
assert np.allclose(r._desired_q, 0.2) and r._gap_stops == 2

# 리더를 상한 안으로 되가져오면 이어진다 (노드는 에피소드를 모른다 --
# 게이트를 다시 태우는 것은 워커의 일이다)
r.command_joint_state(np.r_[np.full(7, 0.5), 0.0])
assert np.allclose(r._desired_q, 0.5) and not r._gap_stopped
assert abs(r._gap_peak - 1.3) < 1e-9, r._gap_peak
print("3. 상한 초과 시 그 자리 정지 / 복귀 시 재개 OK")


# ------------------------------------------------------- 끄기 · 면제 · 변형
off = Fake(limit=0.0)
off.command_joint_state(np.r_[np.full(7, 2.5), 0.0])
assert np.allclose(off._desired_q, 2.5) and off._gap_stops == 0

ro = Fake(read_only=True)
ro.command_joint_state(np.r_[np.full(7, 2.5), 0.0])
assert np.allclose(ro._desired_q, 2.5) and ro._gap_stops == 0, (
    "read_only 는 제어 루프가 없어 _q_cmd 가 안 움직인다 -- 검사하면 영원히 걸린다")

ng = Fake(gripper=False)
ng.command_joint_state(np.full(7, 0.1))
assert np.allclose(ng._desired_q, 0.1)

tight = Fake(limit=0.4)
tight.command_joint_state(np.r_[np.zeros(6), MEASURED_WORST_RAD, 0.0])
assert np.allclose(tight._desired_q, 0.0) and tight._gap_stops == 1, (
    "상한을 내리면 실제로 더 빨리 걸려야 한다 -- 조절이 먹는지 확인")
print("4. 끄기 / read_only 면제 / 그리퍼 없음 / 상한 조절 OK")

print("\n설정점 간격 안전 가드 통과")
