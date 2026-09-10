"""수집 단계 표지 발행 (PUB) -- "지금 무슨 단계인가" 를 프로세스 밖으로 흘린다.

수집 워커는 이미 ``state_changed`` 시그널로 단계를 알린다
(``connecting / idle / homing / reset_wait / gate / approach / recording``).
그 시그널은 GUI 안에서만 쓰여서, 로그 파일에는 단계가 남지 않았다. 홈 복귀는
성공하면 한 줄도 안 찍으므로, 사고를 나중에 되짚을 때 "그때 무슨 단계였나" 를
**추측**할 수밖에 없었다 -- 2026-09-10 에 실제로 그 추측이 틀렸다. 조작자는
"녹화 종료 후 homing 첫 스텝" 이라고 정확히 보고 있었는데, 마지막 로그 줄만
보고 다른 단계로 읽었다.

그래서 단계를 두 곳으로 내보낸다:

* 사람이 읽는 로그 (``[단계] homing``) -- 시간과 함께 남아 되짚을 수 있다.
* PUB 소켓 -- 원시 상태 로거(:mod:`mstack.comm.robot_raw_logger`)가 이것을
  보고 파일을 끊는다.

**PUB 을 쓰는 이유.** 로봇 노드와의 REQ/REP 는 엄격한 락스텝이라, 거기에
표지를 얹으면 진단용 신호가 제어 경로의 실패를 공유하게 된다. PUB 은 구독자가
없어도, 느려도, 죽어도 블로킹하지 않는다 (HWM 을 넘으면 조용히 버린다).
진단이 수집을 멈추는 일은 없어야 한다.

이 모듈의 모든 호출은 **절대 예외를 던지지 않는다**. zmq 가 없거나 포트가
막혀 있어도 수집은 그대로 돌아야 한다.
"""

from __future__ import annotations

import json
import os
import time

#: 단계 표지 PUB 포트. 6021/6022 는 카메라 노드, 6001 은 로봇 노드가 쓴다.
DEFAULT_PHASE_PORT = 6032

#: 구독자가 붙는 주소. 발행자는 bind, 구독자는 connect.
PHASE_TOPIC = b"phase"

#: 1 이면 발행하지 않는다. 인수 테스트가 소켓을 열지 않게 하는 계약 --
#: GELLO_NO_CAMERA_NODE / GELLO_NO_ROBOT_NODE 와 같은 결이다.
NO_PHASE_BUS_ENV = "GELLO_NO_PHASE_BUS"


class PhasePublisher:
    """단계 표지를 PUB 으로 흘린다. 실패해도 조용히 무시한다.

    한 스레드(수집 워커)에서만 쓰는 것을 전제로 한다. zmq 소켓은 스레드
    안전이 아니다.
    """

    def __init__(self, port: int = DEFAULT_PHASE_PORT) -> None:
        self._sock = None
        self._ctx = None
        if os.environ.get(NO_PHASE_BUS_ENV) == "1":
            return
        try:
            import zmq

            self._ctx = zmq.Context.instance()
            sock = self._ctx.socket(zmq.PUB)
            # 구독자가 밀리면 표지를 버린다. 진단이 수집을 붙잡으면 안 된다.
            sock.setsockopt(zmq.SNDHWM, 100)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(f"tcp://127.0.0.1:{port}")
            self._sock = sock
        except Exception:  # noqa: BLE001 -- 발행 실패는 수집을 막지 않는다
            self._sock = None

    def publish(self, phase: str, **extra) -> None:
        """``phase`` 를 지금 시각과 함께 발행한다.

        시각은 ``time.time()`` -- 원시 상태 로거가 노드의 타임스탬프와 맞대야
        하는데, 둘 다 같은 호스트라 시계가 같다.
        """
        if self._sock is None:
            return
        try:
            msg = {"phase": phase, "t": time.time()}
            if extra:
                msg.update(extra)
            self._sock.send_multipart(
                [PHASE_TOPIC, json.dumps(msg).encode()], flags=1  # zmq.NOBLOCK
            )
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
            self._sock = None
