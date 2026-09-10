"""1 kHz 원시 상태 로거(:mod:`mstack.comm.robot_raw_logger`) 프로세스를 띄우는 한 곳.

로봇/카메라 노드와 같은 모양이다. 다른 점은 **이것이 없어도 수집은 그대로
돈다**는 것 -- 로거는 순수한 진단이고, 발행은 PUB 이라 구독자가 없어도
블로킹하지 않는다. 그래서 실패는 전부 조용히 넘긴다.

왜 별도 프로세스인가는 :mod:`mstack.comm.robot_raw_logger` 독스트링에 있다
(요약: 파이썬 GIL 전환 간격이 5 ms 라, 쓰기를 노드 안 스레드로 두면 1 kHz
제어 루프가 그만큼 멈춘다).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess

WT_ROOT = Path(__file__).resolve().parents[3]

#: 1 이면 로거를 띄우지 않는다. 인수 테스트는 소켓도 프로세스도 열지 않는다
#: (GELLO_NO_CAMERA_NODE / GELLO_NO_ROBOT_NODE / GELLO_NO_PHASE_BUS 와 같은 계약).
NO_LOGGER_ENV = "GELLO_NO_RAW_LOGGER"


def spawn_logger(parent=None) -> "QProcess | None":
    """원시 상태 로거를 띄운다. 억제 환경변수가 켜져 있으면 None.

    --die-with-parent: GUI 가 슬롯 예외로 즉사하면 이 프로세스가 남아 포트를
    쥔 채 다음 실행의 로거를 막는다. 커널이 대신 정리하게 한다.
    """
    if os.environ.get(NO_LOGGER_ENV) == "1":
        return None
    proc = QProcess(parent)
    proc.setProgram(sys.executable)
    proc.setArguments(["-m", "mstack.comm.robot_raw_logger", "--die-with-parent"])
    # 노드는 GUI 의 sys.path 를 물려받지 않는다 -- 저장소 루트에서 띄워야
    # `python -m mstack.comm.robot_raw_logger` 가 mstack 을 찾는다.
    proc.setWorkingDirectory(str(WT_ROOT))
    proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    proc.start()
    return proc
