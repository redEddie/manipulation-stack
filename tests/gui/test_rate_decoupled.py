"""화면 주기·정지 판정·명령 축이 **기록 주기에 매여 있지 않은가**.

control 축을 120 Hz 로 올리려면 먼저 세 자리가 떨어져 있어야 한다. 셋 다
"기록 tick 하나 = 20 Hz 하나" 를 전제로 쓰여 있었고, 그 전제는 기록 주기를
올리는 순간 조용히 틀린다.

1. **미리보기**는 기록 tick 마다 보내고 있었다. 20 Hz 일 때는 그것이 곧
   화면 주기라 맞았지만, 120 Hz 면 초당 120장을 Qt 큐에 밀어 넣는다 --
   화면은 그보다 빠를 수 없으므로 큐가 밀리고 그 지연이 수집 루프로
   돌아온다. 부르는 자리가 램프(100 Hz)를 포함해 셋이라 tick 수가 아니라
   시각으로 끊어야 한다.

2. **정지 판정**은 "같은 프레임이 3 tick 연속이면 카메라가 멎었다" 였다.
   20 Hz 로 집어가는데 30 fps 카메라가 같은 장을 세 번 주면 정지가 맞다.
   그런데 120 Hz 면 30 fps 프레임이 네 tick 연속 같은 것이 **정상**이라,
   그대로 두면 경고가 쉬지 않고 뜬다. 자기 축으로 모으는 카메라는 애초에
   이 판정의 대상이 아니다 -- 진짜 손실은 frame_no 구멍이 본다.

3. **명령 축**은 substeps 개 중 넷이 버려지던 것을 살리려고 생겼다.
   substeps == 1 이면 버릴 것이 없어 ``command/*`` 가 ``actions`` 와 글자
   그대로 같아진다. 그때는 축을 만들지 않는다.
"""
import sys
import time
from pathlib import Path

import numpy as np

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from mstack.collect.worker import PREVIEW_HZ, CollectionWorker  # noqa: E402


class _Spy:
    """frames_ready 대신 세기만 한다 (Qt 신호를 안 띄운다)."""

    def __init__(self):
        self.n = 0

    def emit(self, *_a):
        self.n += 1


def _bare():
    """__init__ 을 거치지 않은 껍데기. 이 테스트가 보는 것은 메서드뿐이다."""
    w = CollectionWorker.__new__(CollectionWorker)
    w._preview_last = 0.0
    w._armed_capture = set()
    w.frames_ready = _Spy()
    return w


# ---------------------------------------------- 1. 미리보기는 시각으로 끊는다
w = _bare()
img = np.zeros((4, 4, 3), np.uint8)
obs = {"agent": img, "wrist": img}

t0 = time.monotonic()
calls = 0
while time.monotonic() - t0 < 1.0:
    w._emit_frames(obs)          # 120 Hz 로 부른다
    calls += 1
    time.sleep(1.0 / 120)
sent = w.frames_ready.n
assert calls > 100, calls
assert sent <= PREVIEW_HZ + 2, (
    f"1초에 {calls}번 불렀는데 {sent}장을 보냈다 -- PREVIEW_HZ({PREVIEW_HZ:g}) "
    "로 끊기지 않았다. 기록 주기를 올리면 그대로 화면 주기가 된다")
assert sent >= PREVIEW_HZ - 4, (
    f"{sent}장뿐이다 -- 너무 많이 버렸다 (목표 {PREVIEW_HZ:g})")
print(f"1. 1초에 {calls}번 불러도 {sent}장만 보낸다 (목표 {PREVIEW_HZ:g}) OK")

# 이미지가 한쪽만 있으면 안 보낸다 (예전 계약 유지)
w.frames_ready.n = 0
w._preview_last = 0.0
w._emit_frames({"agent": img, "wrist": None})
assert w.frames_ready.n == 0, "한쪽 카메라만 있는데 보냈다"
print("   한쪽만 있으면 안 보낸다 OK")

# ---------------------------------------------- 2. 정지 판정은 모으는 카메라를 뺀다
src = Path(WT) / "mstack/collect/worker.py"
body = src.read_text()
i = body.index("def _get_obs(")
guard = body[i:i + 4000]
assert "cam_key in self._armed_capture" in guard, (
    "_get_obs 의 정지 판정이 _armed_capture 를 빼지 않는다 -- 기록 주기가 "
    "카메라보다 빨라지면 정상 반복을 정지로 신고한다")
print("2. 정지 판정이 자기 축으로 모으는 카메라를 제외한다 OK")

# ---------------------------------------------- 3. substeps==1 이면 명령 축 없음
j = body.index("for k in range(substeps):")
loop = body[j:j + 3000]
k = loop.index("add_command")
before = loop[:k]
assert "if substeps > 1:" in before[-400:], (
    "add_command 가 substeps == 1 에서도 불린다 -- 그때 command/* 는 "
    "actions 와 같은 값이라 축을 두 벌 쓰게 된다")
print("3. substeps == 1 이면 command 축을 안 쓴다 OK")

print("\n주기 분리 인수 통과")
