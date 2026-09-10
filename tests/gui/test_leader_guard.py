"""리더 놓침 안전층 -- 판정과 급정거 배선.

2026-09-10 이전에는 리더를 놓치면 libfranka 가 명령을 거부하며
acceleration_discontinuity 반사로 제어 루프를 죽였다. 설계된 안전장치가
아니라 부작용이었지만 실제로 팔을 세우는 유일한 것이었고, v/a 를 올리고
v_max 테이퍼를 고치면서 그 경로가 사라졌다. 이 층이 그 자리를 대신한다.

여기서 못박는 것 중 조용히 깨지기 쉬운 것들:

  * **실측 정상 최대(1.51)는 통과, 실측 낙하 최고(2.84)는 발동.** 지표의
    정의를 바꾸면(예: 절대값을 나중에 취하거나 L2 대신 관절 최대를 쓰면)
    이 두 줄이 먼저 깨진다.
  * 창을 채우기 전에는 ``None`` -- "아직 모른다" 와 "0" 은 다르다.
  * 급정거는 ``robot.hold()`` 로 한다. 명령을 그냥 끊으면 설정점이 직전
    값에 남아 팔이 거기까지 계속 간다.
  * ``hold`` 가 실패해도 에피소드 폐기는 진행된다.
  * 새 GUI 상태를 만들지 않는다 -- STATE_LABELS 가 모르는 값을 받으면
    화면이 빈다. 단계 표지(PUB)로만 남긴다.

로봇도 화면도 없이 돈다.
"""
import sys
import time
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
from mstack.collect.leader_guard import LeaderDropGuard  # noqa: E402
from mstack.collect.worker import CollectionWorker  # noqa: E402
from mstack.config.constants import (  # noqa: E402
    LEADER_DROP_SPEED_RAD_S,
    LEADER_DROP_WINDOW_S,
)

#: 2026-09-10 실측을 이 코드에 흘려 얻은 값. 정상 5개(28.9초)와 낙하 2회다.
MEASURED_NORMAL_PEAK = 1.51
MEASURED_DROP_PEAKS = (2.84, 2.70)


def stream(guard, speed_l2, seconds, hz=100.0, joints=(1, 3)):
    """``speed_l2`` 의 L2 속도로 등속 이동하는 리더를 흘린다. 최고값 반환."""
    per = speed_l2 / np.sqrt(len(joints))   # 관절별 속도 -> L2 가 speed_l2
    q = np.zeros(7)
    t0 = 1000.0
    for k in range(int(seconds * hz)):
        for j in joints:
            q[j] = per * k / hz
        guard.update(q.copy(), t0 + k / hz)
    return guard.peak


# ------------------------------------------------------------- 창 채우기 전
g = LeaderDropGuard()
assert g.update(np.zeros(7), 1000.0) is None, "샘플 하나로 판정하면 안 된다"
assert g.update(np.zeros(7), 1000.01) is None
assert not g.tripped(None), "None 은 발동이 아니다"
# 창(100 ms)을 다 채우기 전까지 None
for k in range(2, 10):
    assert g.update(np.zeros(7), 1000.0 + k * 0.01) is None, f"{k} 번째에서 판정했다"
assert g.update(np.zeros(7), 1000.0 + 10 * 0.01) is not None, "창을 채우면 판정해야 한다"
print(f"1. 창({LEADER_DROP_WINDOW_S * 1000:.0f} ms) 채우기 전 판정 보류 OK")


# --------------------------------------------------- 실측값에 대한 회귀 고정
g = LeaderDropGuard()
peak = stream(g, MEASURED_NORMAL_PEAK, 2.0)
assert abs(peak - MEASURED_NORMAL_PEAK) < 0.02, f"지표가 어긋난다: {peak}"
assert not g.tripped(peak), (
    f"실측 정상 최대 {MEASURED_NORMAL_PEAK} 가 발동하면 정상 수집이 끊긴다")

for drop in MEASURED_DROP_PEAKS:
    g = LeaderDropGuard()
    peak = stream(g, drop, 2.0)
    assert g.tripped(peak), f"실측 낙하 {drop} 를 놓쳤다 (지표 {peak:.2f})"
print(f"2. 정상 {MEASURED_NORMAL_PEAK} 통과 / 낙하 {MEASURED_DROP_PEAKS} 발동 OK")

# 임계가 두 실측 사이에 있다 -- 어느 쪽으로도 붙어 있지 않은지
assert MEASURED_NORMAL_PEAK < LEADER_DROP_SPEED_RAD_S < min(MEASURED_DROP_PEAKS), (
    f"임계 {LEADER_DROP_SPEED_RAD_S} 가 실측 구간 "
    f"({MEASURED_NORMAL_PEAK}, {min(MEASURED_DROP_PEAKS)}) 밖이다")


# ----------------------------------------------------------- 한 관절 vs 여러
# 같은 L2 라도 정의가 L2 인지 관절 최대인지에 따라 결과가 갈린다.
g1 = LeaderDropGuard(); stream(g1, 2.8, 1.0, joints=(3,))
g2 = LeaderDropGuard(); stream(g2, 2.8, 1.0, joints=(1, 3, 5))
assert abs(g1.peak - g2.peak) < 0.02, (
    f"L2 는 어느 관절에 실렸는지와 무관해야 한다: {g1.peak:.2f} vs {g2.peak:.2f}")
print("3. L2 정의 (관절 분포와 무관) OK")


# ------------------------------------------------------------------- reset
g = LeaderDropGuard()
stream(g, 3.0, 1.0)
assert g.peak > LEADER_DROP_SPEED_RAD_S
g.reset()
assert g.peak == 0.0
assert g.update(np.zeros(7), 2000.0) is None, "reset 뒤에는 이력이 없어야 한다"
print("4. reset (에피소드 경계) OK")


# --------------------------------------------------- _emergency_hold 배선
class FakeRobot:
    def __init__(self, fail=False):
        self.held = 0
        self._fail = fail

    def hold(self):
        self.held += 1
        if self._fail:
            raise RuntimeError("node down")


class Sig:
    def __init__(self): self.msgs = []
    def emit(self, m): self.msgs.append(m)


class FakePub:
    def __init__(self): self.sent = []
    def publish(self, phase, **extra): self.sent.append((phase, extra))


class FakeWorker:
    _emergency_hold = CollectionWorker._emergency_hold

    def __init__(self, fail=False):
        self._robot = FakeRobot(fail)
        self.log_message = Sig()
        self._phase_pub = FakePub()
        self._obs_calls = 0

    def _get_obs(self):
        self._obs_calls += 1
        # 첫 두 번은 아직 감속 중, 그 뒤 정지
        dq = np.full(7, 1.0) if self._obs_calls <= 2 else np.zeros(7)
        return {"_joint_velocities": dq}


w = FakeWorker()
t0 = time.monotonic()
w._emergency_hold(2.9)
elapsed = time.monotonic() - t0
assert w._robot.held == 1, "hold 를 불러야 한다 -- 전송 중단만으로는 안 선다"
assert w._phase_pub.sent and w._phase_pub.sent[0][0] == "estop", w._phase_pub.sent
assert w._phase_pub.sent[0][1]["speed"] == 2.9
assert any("2.90 rad/s" in m for m in w.log_message.msgs), w.log_message.msgs
assert w._obs_calls >= 3, "팔이 실제로 설 때까지 기다려야 한다"
assert elapsed < 0.6, f"대기가 너무 길다: {elapsed:.2f}s"
print(f"5. 급정거 배선 (hold + estop 표지 + 정지 대기 {elapsed * 1000:.0f} ms) OK")

# hold 가 실패해도 폐기는 진행된다
w = FakeWorker(fail=True)
w._emergency_hold(3.1)
assert any("실패" in m for m in w.log_message.msgs), w.log_message.msgs
assert w._phase_pub.sent, "hold 실패가 단계 표지까지 막으면 안 된다"
print("6. hold 실패해도 계속 진행 OK")


# ------------------------------------------------- hold 가 경로 끝까지 있는가
from mstack.agents.lerobot_plugin import FR3ZMQRobot  # noqa: E402
from mstack.comm.zmq_core.robot_node import ZMQClientRobot  # noqa: E402
from mstack.core.robot import Robot  # noqa: E402
from mstack.robots.franka_fr3 import FrankaFR3Robot  # noqa: E402

for cls in (Robot, FrankaFR3Robot, ZMQClientRobot, FR3ZMQRobot):
    assert hasattr(cls, "hold"), f"{cls.__name__} 에 hold 가 없다"
# 기본 구현은 무동작 -- 시뮬레이터/스텁이 이 때문에 죽으면 안 된다
Robot.hold(object())
print("7. hold 경로 (core -> FR3 -> ZMQ 클라이언트 -> 플러그인) OK")

print("\n리더 놓침 안전층 통과")
