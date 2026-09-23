"""Background QThread driving GELLO teleop + LIBERO-format recording for the PyQt GUI.

Ports the proven control flow from ``experiments/record_dataset.py`` (home ->
pose-gate -> approach-ramp -> record, plus robot-node death/reconnect
handling) from its blocking-loop/KeyPoller CLI shape into a
command-queue-in / Qt-signal-out worker thread, so a GUI can drive it without
freezing on robot I/O. The state machine and constants (``GATE_RAD``,
``RAMP_STEP``) are unchanged from that script; only the I/O boundary moved.
(2026-09-10: 램프 주기는 RAMP_HZ 로 옮겼다 -- 상수는 이제 rad/s 로 적고
tick 크기는 주기에서 파생한다.)

Run inside ``lerobot-venv`` (has ``mstack``, ``dynamixel-sdk``, ``pyrealsense2``,
``lerobot``). Requires ``scripts/launch/launch_nodes.py --robot fr3`` already
running in ``pylibfranka-venv``.
"""

from __future__ import annotations

import os
import queue
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np
import zmq
from PyQt6.QtCore import QThread, pyqtSignal

from mstack.data.libero_format import gzip4
from mstack.data.provenance import collector_commit
from mstack.data.dataset_schema import (
    FT_OBS_KEYS,
    ROBOT_EE_POS_QUAT,
    ROBOT_JOINT_POSITIONS,
    ROBOT_JOINT_VELOCITIES,
    ROBOT_STATE_TIME,
    TIMING_ACTION,
    TIMING_FRAME,
    TIMING_ROBOT_STATE,
    DatasetSchemaConfig,
)
from mstack.agents.lerobot_plugin import (
    JOINT_KEYS,
    FR3ZMQRobot,
    FR3ZMQRobotConfig,
    GelloFR3Teleop,
    GelloFR3TeleopConfig,
)
from mstack.data.libero_format import LiberoTaskWriter, NullTaskWriter
from mstack.config.constants import ROLL_ABORT_RAD
from mstack.collect.leader_guard import LeaderDropGuard
from mstack.robots.franka_fr3 import FR3_RESET_POSES, FR3_ROLL_JOINTS
from mstack.comm.phase_bus import PhasePublisher
from mstack.data.phase_log import append_phase
from mstack.config.constants import MATCH_GATE_RAD
from mstack.scene.scene_format import QUALITY_FAILED, QUALITY_SUCCESS, SceneMetadata, SceneWriter
from mstack.config.station import load_station

#: 자세 매칭 게이지의 초록/빨강 임계. 정본은 리더 wall 이다 -- 게이지가
#: 초록으로 보이는 것과 모터가 실제로 당기기 시작하는 것이 같은 숫자여야
#: 하기 때문이다 (issue #37A). 여기서 재정의하면 둘이 어긋난다.
GATE_RAD = MATCH_GATE_RAD
#: 게이트/정렬 루프는 50Hz 로 돌지만 게이지 갱신은 이 주기로만 보낸다
#: (_emit_gate_status 참조).
_GATE_EMIT_PERIOD_S = 1.0 / 15
#: 램프(homing·접근) 명령 주기(Hz). **기록 주기(cfg.fps)와 무관하다** --
#: 이 루프들은 기록되지 않으므로 데이터셋에 영향이 없다.
#:
#: 2026-09-10: 20 -> 100. 20 Hz 계단이 진동의 단일 원인이었다 -- 원시 로그에서
#: 세 신호(필터 출력 속도·측정 속도·토크)의 봉우리가 전부 19.5 Hz 로 명령
#: 주기와 일치했고, homing 의 J2 토크 맥동은 정지 대비 310배였다. 주기를 5배로
#: 올리면 한 tick 이동이 1/5 이 되어 오차가 확 벌어지지 않고, 남는 리플도 더
#: 높은 주파수로 옮겨가 2차 필터가 훨씬 강하게 누른다(감쇠가 주파수 제곱).
#:
#: 예산은 충분하다: ZMQ 왕복 2회가 270 µs 실측이고 리더 읽기·카메라는 캐시
#: 반환이라, 10 ms 주기에서 작업이 3% 를 넘지 않는다.
#: 값 자체는 ``WorkerConfig.ramp_hz`` (기본값은 스테이션의 ``control.ramp_hz``).
#: 상수가 아닌 이유: 모듈 수준 상수는 **import 시점**에 굳어서, 설정 화면에서
#: 바꿔도 이미 뜬 워커에는 닿지 않는다. 세션이 자기 값을 들고 다녀야
#: "저장했다고 도는 세션의 주기가 바뀌지 않는다"는 계약이 성립한다 (issue #1).

#: 텔레옵 명령 : 기록 프레임 비율. 명령 주기 = cfg.fps * TELEOP_SUBSTEPS.
#: 기본 5 이므로 20 Hz 기록에서 명령은 100 Hz 다.
#:
#: **정수배로 둔다.** 그래야 기록 프레임이 언제나 명령 틱 위에 정확히 얹히고,
#: 기록된 action 이 그 순간 실제로 나간 명령이 된다. 정수배가 아니면 둘의
#: 위상이 미끄러져 "어느 명령을 기록할 것인가" 가 규약 문제가 되고, 그 규약이
#: 틀리면 학습 라벨이 조용히 어긋난다.
#:
#: 근거는 homing 램프에서 먼저 확인했다 (2026-09-10, 20 -> 100 Hz): J2 토크의
#: 15~25 Hz 성분이 4.9배 줄고 다른 대역은 그대로였다. 조작자가 "책상 진동이
#: 아예 사라졌다" 고 보고했다. 텔레옵 구간도 같은 20 Hz 계단을 갖고 있다
#: (정지 대비 107배).
#:
#: 값은 ``WorkerConfig.teleop_substeps`` (스테이션 ``control.teleop_substeps``).
#: 설정으로 뺐다 (issue #1, 2026-09-22) -- 명령 주기는 scene/데이터세트 설정과
#: 직교하는 축이라 별도 설정 화면에 있고, 바꾸면 재시작이 필요하다.
#: **정수 강제는 ``mstack/config/station.py`` 의 ``_control_from`` 이 한다.**

#: 접근 램프 속도 (rad/s). 조작자가 리더를 목표 자세로 잡고 있고 거리가 짧다.
#:
#: 드라이버 기준 필터의 상한(1.5 rad/s, franka_fr3.py max_joint_velocity)보다
#: 크다. **그대로 둔다** (조작자 판단, 2026-09-13): 둘은 역할이 다르다 --
#: 이쪽은 명령을 만드는 램프의 목표 속도이고, 저쪽은 그 명령이 팔에 닿기 전에
#: 거르는 안전 상한이다. 필터가 깎으면 접근이 그만큼 느려질 뿐, 접근 구간은
#: 짧고 실제로 잘 동작한다. 낮추자는 제안이 두 번 나왔으므로 여기 적어 둔다
#: -- 되살리려면 새 근거(격차가 쌓여 실제로 문제가 된 기록)가 있어야 한다.
#: 값은 ``WorkerConfig.approach_speed``. tick 당 이동은 ``cfg.ramp_step``
#: (= approach_speed / ramp_hz) 로 파생된다 -- 주기를 바꿔도 **속도가 보존**되게
#: 하려고 그렇게 만들었고, 그래서 설정 화면도 속도만 노출한다.

#: 접근 완료 판정 (rad). 속도 상수와 **분리한다** -- 예전에는 RAMP_STEP 하나가
#: 스텝 크기와 수렴 임계값을 겸했는데, 주기를 올리면 스텝만 줄어야 하고 판정
#: 기준은 그대로여야 한다. 붙여 두면 주기를 바꾸는 순간 판정이 5배 엄격해진다.
#: 값은 ``WorkerConfig.approach_done_rad``.

GRIPPER_OPEN = 0.0  # GELLO/franka_fr3 convention: 0=open, 1=closed

# ---- EE 경로 homing ----
# 관절 직선 보간 homing 은 파지 직후처럼 EE 가 낮을 때 베이스가 돌면서
# 테이블 높이를 수평으로 쓸고 지나간다. 대신 EE 경로를 만들어 IK 로 푼다.
# IK 실패나 관절 점프가 크면 기존 관절 램프로 폴백 -- homing 이 안 되는
# 것보다는 예전처럼 무섭게라도 돌아가는 쪽이 낫다.
#
# 2026-09-23: "수직 10cm 리프트 -> 홈까지 직선" 두 단계를 **연속 블렌딩**으로
# 바꿨다. 실측 2,434개 종료 자세에서 고정 리프트가 양쪽으로 틀렸기 때문이다:
# 80.3% 는 홈 높이(0.254 m)보다 낮게 끝나 10cm 를 올려도 35.6% 가 여전히 낮고,
# 19.7% 는 이미 홈보다 높은데 거기에 10cm 를 더 올리고 있었다 (최대 0.618 m).
#
# 매 스텝의 진행 방향을 두 항으로 섞는다:
#
#     need = clip((Z_CLEAR - z) / HOME_BLEND_M, 0, 1) * (1 - s)**HOME_SCHED_POWER
#     dir  = normalize(need * up + (1 - need) * toward_home)
#
# 앞 항(높이 피드백)이 **리프트 양을 시작 높이에 맞춰 스스로 정한다** -- 낮게
# 끝나면 거의 수직으로 오르고, 이미 높으면 아예 안 오른다.
#
# 뒤 항(진행도 스케줄)이 없으면 Z_CLEAR 를 홈 높이 위로 못 올린다. 홈에
# 가까워질수록 내려가야 하는데 그러면 높이 항이 다시 켜져 **극한 순환**에
# 빠지기 때문이다 (실측: Z_CLEAR=홈+0.06, BLEND=0.06 에서 400/400 미수렴).
# s 를 곱하면 끝에서 강제로 0 이 되어 홈에 커밋한다 (같은 설정에서 0/400).
# s 는 **래칫**이다 -- 닫은 거리의 비율이고 줄어들지 않는다.
#
# 그래서 Z_CLEAR 를 홈보다 6cm 높게 잡을 수 있고, 그것이 실측상 가장 좋은
# 지점이다 (수평 이동 중 최저 높이 최악값 0.136 -> 0.240 m, +76%). 경로는
# 오히려 짧아진다 (0.323 -> 0.301 m, homing 0.99 -> 0.90 s). IK 성공률은
# 실제 종료 자세 200개에서 195/200 -> 197/200.
#: 홈 EE 높이보다 이만큼 위면 더 올라가지 않는다. **홈 높이 자체가 아니라
#: 그 위**여야 테이블 위를 수평으로 지날 때 여유가 생긴다.
HOME_CLEAR_MARGIN_M = 0.06
#: 높이 게이트의 무름. 이 값만큼 Z_CLEAR 아래면 완전히 "위로"가 된다.
#: 옛 HOME_LIFT_M 과 같은 0.10 이지만 뜻이 다르다 -- 올릴 **양**이 아니라
#: 게이트의 **폭**이고, 실제로 올라가는 높이는 시작 높이가 정한다.
HOME_BLEND_M = 0.10
#: 진행도 스케줄의 지수. 작을수록 일찍 홈 쪽으로 기운다.
HOME_SCHED_POWER = 0.5
HOME_EE_STEP_M = 0.010   # 웨이포인트 간 EE 이동
HOME_ROT_STEP_RAD = 0.05  # 웨이포인트 간 EE 회전
HOME_MAX_DQ = 0.35       # 연속 웨이포인트 관절 점프 상한 -- 초과 시 폴백
#: EE 경로 스텝 수 상한. 정상 경로는 최장 0.47 m = 47 스텝이라 600 이면
#: 6 m 로 한참 여유가 있다. 이 상한에 닿으면 수렴하지 않은 것이므로 폴백한다.
HOME_MAX_STEPS = 600
#: tick 당 관절 이동 상한 (rad) = HOME_SPEED / RAMP_HZ.
#:
#: 웨이포인트 하나 = tick 하나가 아니다. 위의 EE 스텝은 **직교** 속도만
#: 묶는다 -- 자코비안이 나빠지는 자세에서는 1cm 이동이 관절 0.3 rad 이 되고,
#: 그것이 한 tick 에 그대로 나가면 6.8 rad/s 를 요구하게 된다 (오프라인
#: 실측: 무작위 시작 자세 276경로 중 58%가 드라이버 상한을 넘었고 최악은
#: 0.339 rad/tick). 드라이버가 낼 수 있는 것은 max_joint_velocity=1.5 rad/s
#: 뿐이라, 그 위로 요구하면 명령이 팔보다 빨리 달아나고 그 격차가 경로 내내
#: 쌓인다 -- 명령이 끝난 뒤에도 팔은 v_max 로 계속 달리고, 그것이 조작자가
#: 본 "가끔 홈이 너무 빠르다" 와 그때의 반사다 (2026-09-06 보고).
#:
#: 그래서 웨이포인트 사이를 관절 공간에서 다시 잘라(_densify) 이 값을 넘지
#: 않게 한다. v_max 의 80% 로 두어 명령이 팔을 앞지르지 않게 한다 -- 앞지르지
#: 않으면 쌓일 격차도 없다. RAMP_STEP(APPROACH_SPEED = 2.0 rad/s)을 쓰지 않는 이유가
#: 그것이다: v_max 보다 큰 요구는 필터를 포화시키고, 포화 상태에서 제어 루프
#: 틱이 한 번 늦으면(ZMQ/GIL 간섭) 정지->재개 순간 가속도 불연속으로
#: joint_motion_generator_acceleration_discontinuity 반사가 떠 제어 루프가
#: 죽는다. 폴백 관절 램프(_ramp_to)는 목표로 clip 되어 결국 멎지만, 홈까지의
#: 긴 이동 동안 포화 구간이 계속되므로 같은 캡이 필요하다 (2026-09-07 사고).
#: (2026-09-10: 주기가 RAMP_HZ 로 바뀌어도 **속도**가 보존되도록 파생값으로
#: 바꿨다. 20 Hz 시절의 0.06 rad/tick 과 같은 1.2 rad/s 다.)
#: 값은 ``WorkerConfig.home_speed`` (rad/s -- 드라이버 v_max(1.5)의 80%).
#: tick 당 이동은 ``cfg.home_tick_dq`` (= home_speed / ramp_hz) 로 파생된다.

#: 노드 복구 재시도가 같은 이유로 계속 실패할 때 로그를 다시 찍는 주기(초).
#: 2초마다 찍으면 로그가 그것만으로 차고, 안 찍으면 멈춘 것처럼 보인다.
_RECOVERY_LOG_PERIOD_S = 30.0

#: 제어 루프가 죽었을 때 franka_fr3.get_observations 가 붙이는 접두어.
#: 이 문자열이 보이면 "노드는 살아 있는데 팔이 죽었다" -- 기다려도 낫지
#: 않으므로 안내가 달라진다 (한쪽만 바꾸면 안내가 조용히 틀려진다).
#: 미리보기 화면 주기 (Hz). **기록 주기와 무관하다** -- 사람 눈에 필요한
#: 값이고, 기록 주기는 데이터가 필요로 하는 값이다. 둘을 한 값으로 묶어 두면
#: control 축을 120 Hz 로 올리는 순간 화면도 120 Hz 가 되어 Qt 큐가 밀린다.
PREVIEW_HZ = 30.0

CONTROL_DEAD_MARK = "control loop is dead"


#: "RuntimeError: ..." 처럼 이미 타입 이름이 앞에 붙은 메시지.
_TYPED_MSG_RE = re.compile(r"^[A-Za-z_]\w*(Error|Exception|Interrupt|Exit)\s*:")


def _frame_timing(obs: dict, t_frame: float, t_action: float) -> dict:
    """Per-frame timing columns (dataset_schema.TIMING_*) from one observation.

    Keys a node did not supply are left out rather than filled -- a missing
    column says "not measured", a filled one would claim a measurement.
    """
    out = {TIMING_FRAME: t_frame, TIMING_ACTION: t_action}
    if obs.get("_state_time") is not None:
        out[TIMING_ROBOT_STATE] = obs["_state_time"]
    for cam_key, role in (("agent", "agentview"), ("wrist", "eye_in_hand")):
        st = obs.get(f"_{cam_key}_stamps") or {}
        for src, suffix in (("t_host", "host"), ("t_device", "device"),
                            ("frame_no", "frame_no"), ("seq", "node_seq"),
                            ("t_domain", "domain")):
            if st.get(src) is not None:
                out[f"{role}_{suffix}"] = st[src]
    return out


def _why(e: BaseException) -> str:
    """예외를 사람이 읽을 한 줄로. 원인 체인을 지킨다.

    반사 이름은 libfranka 의 abort 메시지 안에 있고, 그것은 노드 -> ZMQ ->
    여기로 오는 동안 문자열로만 남는다. 타입만 찍거나 자체 문구로 갈아치우면
    그 이름이 사라진다 -- 실제로 그래서 "로그에 반사 종류가 안 보였다".
    """
    text = str(e)
    if not text:
        return type(e).__name__
    # ZMQ 경계를 건너온 예외는 이미 타입 이름을 문자열로 달고 있다
    # (robot_node.py 가 {"error": "RuntimeError: ..."} 로 싣고 클라이언트가
    # 그대로 raise 한다). 여기서 또 붙이면 "RuntimeError: RuntimeError: ..."
    # 가 되어 정작 읽어야 할 뒤쪽이 밀린다.
    msg = text if _TYPED_MSG_RE.match(text) else f"{type(e).__name__}: {text}"
    cause = e.__cause__ or e.__context__
    if cause is not None and str(cause) and str(cause) not in msg:
        msg += f" -- 원인: {type(cause).__name__}: {cause}"
    return msg


def _node_down_hint(e: BaseException) -> str:
    """이유에 맞는 다음 행동. 둘은 고치는 방법이 다르다.

    * 제어 루프가 죽었다 = 노드 프로세스는 답하지만 팔이 멈췄다. 기다려도
      낫지 않는다 -- 노드를 다시 띄워야 한다 (그때 FR3 가 반사 상태를
      지우고 다시 붙는다).
    * 그 밖(ZMQ 무응답) = 프로세스가 없거나 네트워크가 끊겼다. 노드가
      돌아오면 자동으로 재연결된다.
    """
    if CONTROL_DEAD_MARK in str(e):
        return ("[NODE DOWN] 팔의 제어 루프가 멈췄습니다 (반사/오류). 기다려도 "
                "복구되지 않습니다 -- Robot 메뉴 > '노드 재시작' 을 누르세요. "
                "FR3 Desk 에 오류가 떠 있으면 먼저 지워야 합니다.")
    return ("[NODE DOWN] robot node 가 응답하지 않습니다 -- 노드가 돌아오면 "
            "자동으로 이어집니다. 안 돌아오면 '노드 재시작' 을 누르세요.")

# Fallback defaults, used only if the GUI doesn't supply a serial (e.g. a
# script driving CollectionWorker directly). The GUI itself always populates
# WorkerConfig.agent_camera_serial / wrist_camera_serial from a live device
# scan (see apps/collect_workspace.py's agent_combo/wrist_combo),
# since serials change
# whenever a camera is swapped.
_STATION = load_station()
AGENT_CAMERA_SERIAL = _STATION.camera("agent").serial
WRIST_CAMERA_SERIAL = _STATION.camera("wrist").serial


def _nice_worker() -> None:
    """압축 워커의 우선순위를 낮춘다. 20 Hz 기록 루프가 같은 기계에서 돌고
    있고, 그 루프의 50 ms 마감이 압축보다 훨씬 중요하다. repack_hdf5.py 가
    이미 쓰는 방식과 같다."""
    try:
        os.nice(10)
    except OSError:
        pass


class EpisodeSaver(QThread):
    """Owns ALL h5py-file-touching writer calls, serialized through one queue.

    h5py is not thread-safe, so once this thread starts, save_buffer /
    delete_episode / list_episodes go ONLY through here. The worker keeps
    buffer-only calls (add_frame / start_episode / discard_episode) and hands
    a detached buffer over for saving -- recording the next episode overlaps
    with compressing/writing the previous one, so homing/preview never stall
    on an episode commit.
    """

    episode_saved = pyqtSignal(str, int)      # demo_name, n_frames
    episode_list_changed = pyqtSignal(list)
    save_status = pyqtSignal(str)             # ""=idle; else short status text
    log_message = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._writer = None  # set by CollectionWorker.run() before start()
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._pool = None
        self._busy = False

    def set_writer(self, writer) -> None:
        self._writer = writer

    def pending(self) -> int:
        """아직 디스크에 닿지 않은 에피소드 수 (지금 쓰는 중인 것 포함).

        게이트가 이것을 보고 다음 녹화를 막는다 -- 버퍼가 비압축이라
        에피소드당 최대 0.74 GB 다. 실측으로 이 게이트는 거의 안 걸린다
        (가장 짧은 갭 10.3초 vs 8워커 압축 0.7초).
        """
        return self._q.qsize() + (1 if self._busy else 0)

    def _start_pool(self) -> None:
        """압축 워커 풀. **세션당 한 번만 만든다** -- 기동이 0.18초라
        에피소드마다 내면 갭을 그만큼 먹는다.

        프로세스인 이유: zlib 이 GIL 을 놓기는 하지만, 별도 프로세스면 20 Hz
        기록 루프가 도는 인터프리터와 메모리도 GC 도 공유하지 않는다.
        ``os.nice(10)`` 으로 우선순위를 낮춰 커널이 기록 루프를 먼저 깨우게
        한다 (실측: 8워커가 압축하는 동안 루프 마감 초과 최대 4.9 ms =
        50 ms 예산의 9.8%).

        코어가 적은 기계에서도 루프 몫이 남도록 ``cpu_count() - 4`` 로 묶는다.
        풀을 못 만들면 ``None`` 으로 두고 직렬로 쓴다 -- 압축을 병렬로 못 한다고
        수집을 막을 이유는 없다.
        """
        try:
            n = max(1, min(8, (os.cpu_count() or 4) - 4))
            self._pool = ProcessPoolExecutor(
                n, mp_context=get_context("spawn"), initializer=_nice_worker)
            # 워커를 실제로 띄워 둔다 -- 첫 에피소드가 기동 비용을 내지 않게.
            list(self._pool.map(gzip4, [b""] * n))
            self.log_message.emit(f"[저장] 압축 워커 {n}개 준비")
        except Exception as e:  # noqa: BLE001
            self._pool = None
            self.log_message.emit(f"[저장] 압축 워커를 못 띄워 직렬로 씁니다 ({e})")

    def _stop_pool(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def enqueue_save(self, buf, success, instruction=None, instruction_id=None) -> None:
        """instruction/instruction_id 는 scene 모드 전용 -- 에피소드가 끝난
        시점의 slot 을 워커가 캡처해서 싣는다. 저장은 백그라운드라 조작자가
        다음 slot 으로 넘어간 뒤 실행될 수 있는데, 그때 writer 의 현재 상태가
        아니라 "그 에피소드가 실제 수행한 문장"이 찍혀야 한다 (SceneWriter 가
        instruction 을 저장 시점 명시 인자로만 받는 이유와 같은 경합)."""
        self._q.put(("save", buf, success, instruction, instruction_id))

    def enqueue_delete(self, name: str) -> None:
        self._q.put(("delete", name))

    def enqueue_set_reference(self, img) -> None:
        """scene 기준 사진 후보 (h5py 는 saver 스레드 전용이라 큐 경유).
        이미 있으면(수동 촬영본 등) 건드리지 않는다."""
        self._q.put(("set_ref", img))

    def enqueue_set_success(self, name: str, success: bool) -> None:
        """Re-label an already-saved episode. Goes through the same queue as
        the save itself, so a toggle sent while that save is still running is
        applied after it rather than racing it."""
        self._q.put(("set_success", name, success))

    def finish(self) -> None:
        """Drain the queue, then exit run(). Caller must wait() afterwards."""
        self._q.put(("stop",))

    def run(self) -> None:
        self._start_pool()
        try:
            self._run_queue()
        finally:
            self._stop_pool()

    def _run_queue(self) -> None:
        while True:
            item = self._q.get()
            if item[0] == "stop":
                break
            self._busy = True
            try:
                if item[0] == "save":
                    _, buf, success, instruction, instruction_id = item
                    n = len(buf)
                    waiting = self._q.qsize()
                    self.save_status.emit(
                        f"저장 중... {n}프레임" + (f" (+대기 {waiting})" if waiting else ""))
                    t0 = time.monotonic()
                    if instruction is not None:
                        # scene 모드: SceneWriter.save_buffer 는 instruction 을
                        # 저장 시점 명시 인자로 요구한다. 라벨(success) 없는
                        # 에피소드는 여기서 규격이 거부한다(아래 except 로 감).
                        name = self._writer.save_buffer(
                            buf, success=success, pool=self._pool,
                            instruction=instruction, instruction_id=instruction_id)
                    else:
                        name = self._writer.save_buffer(buf, success=success,
                                                        pool=self._pool)
                    dt = time.monotonic() - t0
                    if name:
                        self.episode_saved.emit(name, n)
                        self.log_message.emit(f"[저장] {name} ({n} 프레임, {dt:.1f}s, 백그라운드)")
                    self.episode_list_changed.emit(self._writer.list_episodes())
                    self.save_status.emit("")
                elif item[0] == "delete":
                    name = item[1]
                    if not hasattr(self._writer, "delete_episode"):
                        # 삭제가 없는 writer (연습 모드의 NullTaskWriter 등).
                        # scene 은 SceneWriter.delete_episode(삭제 후 renumber) 가 있다.
                        self.log_message.emit(
                            f"[삭제 불가] {name}: 이 세션의 writer 는 삭제를 "
                            "지원하지 않습니다")
                        continue
                    self._writer.delete_episode(name)
                    self.log_message.emit(f"[삭제] {name}")
                    self.episode_list_changed.emit(self._writer.list_episodes())
                elif item[0] == "set_ref":
                    img = item[1]
                    if (hasattr(self._writer, "set_reference_image")
                            and not getattr(self._writer, "has_reference_image", True)):
                        self._writer.set_reference_image(img)
                        self.log_message.emit(
                            "[SCENE] 기준 사진을 첫 에피소드의 agentview 로 캡처했습니다")
                elif item[0] == "set_success":
                    _, name, success = item
                    if hasattr(self._writer, "set_episode_success"):
                        self._writer.set_episode_success(name, success)
                    else:
                        # SceneWriter: 같은 재판정이 quality_status 로 표현된다.
                        self._writer.set_quality_status(
                            name, QUALITY_SUCCESS if success else QUALITY_FAILED)
                    self.log_message.emit(
                        f"[판정] {name} -> {'성공' if success else '실패'}")
                    self.episode_list_changed.emit(self._writer.list_episodes())
            except Exception as e:  # noqa: BLE001
                self.log_message.emit(f"[저장 스레드 오류] {type(e).__name__}: {e}")
                self.save_status.emit("")
            finally:
                # 예외로 빠져나가도 반드시 내린다 -- 안 내리면 pending() 이
                # 영원히 1 이상이라 게이트가 다음 녹화를 계속 막는다.
                self._busy = False


@dataclass
class WorkerConfig:
    task_name: str
    language_instruction: str
    data_root: str
    robot_port: int = _STATION.node.port
    hostname: str = _STATION.node.host
    grip: str = "right"
    reset_pose: str = "libero"
    fps: int = _STATION.fps
    # ---- 제어 주기 (issue #1). 기본값은 스테이션의 control: 블록에서 온다.
    # 근거 주석은 이 파일 위쪽, 각 값을 쓰는 메커니즘 옆에 그대로 있다.
    # 세션이 값을 들고 다니므로, 설정을 저장해도 **도는 세션은 바뀌지 않는다**.
    ramp_hz: float = _STATION.control.ramp_hz
    teleop_substeps: int = _STATION.control.teleop_substeps
    approach_speed: float = _STATION.control.approach_speed
    approach_done_rad: float = _STATION.control.approach_done_rad
    home_speed: float = _STATION.control.home_speed
    max_episode_seconds: float = 20.0
    reset_wait_seconds: float = 10.0
    enable_wall: bool = True
    # True: teleoperate without creating any .hdf5 at all -- scene setup,
    # camera framing, letting someone try the leader. Everything else behaves
    # identically (pose gate, live view, frame counter); saving is accepted
    # and dropped. See mstack/data/libero_format.py's NullTaskWriter.
    no_dataset: bool = False
    # True: pull the leader onto the follower's reset pose at the start of
    # every episode, so each one begins from an identical joint configuration.
    # False: the operator aligns by hand, so the starting pose varies
    # episode-to-episode -- deliberate variation, not sloppiness.
    auto_match_pose: bool = True
    resume: bool = False
    # ---- scene 모드 (scene-v1) ----
    # scene_metadata(새 scene) 또는 scene_id+scene_resume(이어찍기)가 주어지면
    # LiberoTaskWriter 대신 SceneWriter 로 기록한다. 이때 task_name 은 쓰이지
    # 않고(파일명은 scene_id 에서 나온다), language_instruction 과
    # instruction_id 가 시작 slot 이 된다. slot 은 수집 중 cmd_set_slot 으로
    # 바뀔 수 있고, 에피소드에는 "기록 시작 시점의 slot" 이 찍힌다.
    scene_metadata: Optional[SceneMetadata] = None
    scene_id: Optional[str] = None
    scene_resume: bool = False
    #: 이번 세션이 기록할 스키마 버전 (마법사가 정한다). 새 scene 은
    #: SceneMetadata 에 실려 가고, 이어찍기는 SceneWriter 가 이 값으로 파일의
    #: 도장을 올린다 -- 안 그러면 새 필드 구성으로 찍으면서 도장은 옛 버전인
    #: 파일이 된다 (2026-09-06 scene_015).
    session_version: str = ""
    instruction_id: str = ""      # scene 모드 시작 slot 의 ID (예: "I000")
    collector: str = ""           # scene 모드 필수 attr -- 수집자 식별자
    #: GUI run id (collection_history.new_run_id). Tags phase-log lines so they
    #: join the history lines of the same run.
    run_id: str = ""
    agent_camera_serial: str = AGENT_CAMERA_SERIAL
    wrist_camera_serial: str = WRIST_CAMERA_SERIAL
    schema: DatasetSchemaConfig = field(default_factory=DatasetSchemaConfig)
    # 카메라별 정사각 크롭 정렬 (GUI Layout 페이지에서 조정). None 이면 기본값.
    # 에피소드마다 attrs["crop_params"] 로 찍힌다.
    crop_params: dict | None = None

    @property
    def ramp_period_s(self) -> float:
        return 1.0 / self.ramp_hz

    @property
    def ramp_step(self) -> float:
        """접근 램프의 tick 당 관절 이동 (rad). 파생값이라 주기를 바꿔도
        **속도가 보존된다** -- 그래서 설정 화면은 속도만 노출한다."""
        return self.approach_speed / self.ramp_hz

    @property
    def home_tick_dq(self) -> float:
        """homing 의 tick 당 관절 이동 (rad). 위와 같은 이유로 파생값이다."""
        return self.home_speed / self.ramp_hz

    @property
    def command_hz(self) -> float:
        """텔레옵 명령 주기. 기록 주기의 정수배라는 것이 요점이다."""
        return self.fps * self.teleop_substeps

    @property
    def scene_mode(self) -> bool:
        # no_dataset(연습) 이 scene 지정보다 우선한다 -- 연습 모드의 계약은
        # "파일을 만들지 않는다" 이고 그건 scene 에서도 그대로여야 한다.
        return not self.no_dataset and (
            self.scene_metadata is not None or self.scene_id is not None
        )


class CollectionWorker(QThread):
    state_changed = pyqtSignal(str)
    frames_ready = pyqtSignal(object, object)  # agentview_rgb, eye_in_hand_rgb (np.ndarray)
    gate_status = pyqtSignal(object, object, bool)  # leader(8,), follower(8,), all_ok
    # 리더암 벽의 상태 (idle/armed/pulling/blocked/done). 바뀔 때만 보낸다 --
    # 상태바 표시등 하나를 그리는 데 15Hz 신호가 또 필요하지는 않다.
    leader_state = pyqtSignal(str)
    pose_match_status = pyqtSignal(float, bool)  # max joint error (rad), done
    episode_progress = pyqtSignal(int, float)  # n_frames, seconds
    episode_saved = pyqtSignal(str, int)  # demo_name, n_frames
    episode_discarded = pyqtSignal(int)  # n_frames
    reset_countdown = pyqtSignal(float)  # seconds remaining
    log_message = pyqtSignal(str)
    #: (살아 있나, 왜). 이유가 함께 가야 상태표시등이 "응답 없음" 이라고만
    #: 하지 않고 무엇 때문인지 보여줄 수 있다 (2026-09-06).
    node_status = pyqtSignal(bool, str)  # True=ok, False=down
    fatal_error = pyqtSignal(str)
    connected = pyqtSignal(int, str)  # starting episode count, active .hdf5 path
    episode_list_changed = pyqtSignal(list)  # LiberoTaskWriter.list_episodes()
    session_summary = pyqtSignal(dict)  # emitted once, right before the file closes

    def __init__(self, config: WorkerConfig) -> None:
        super().__init__()
        self.cfg = config
        self._cmds: "queue.Queue[tuple]" = queue.Queue()
        self._running = True
        self._robot: Optional[FR3ZMQRobot] = None
        self._teleop: Optional[GelloFR3Teleop] = None
        self._writer: Optional[LiberoTaskWriter] = None
        self._reset_q = FR3_RESET_POSES[self.cfg.reset_pose]
        self._episode_count = 0
        #: 저장이 안 끝나 스페이스바가 막힌 횟수. 실측상 0 이어야 하고,
        #: 0 이 아니면 압축이 갭을 못 따라간다는 뜻이다 (30 Hz 판단 근거).
        self._save_gate_hits = 0
        self._last_gate_emit = 0.0
        self._last_leader_state = ""
        # 단계 표지: GUI 시그널 · 사람이 읽는 로그 · PUB 소켓 세 곳으로 나간다
        # (_set_state 참고). 소켓이 안 열려도 나머지는 그대로 동작한다.
        self._phase_pub = PhasePublisher()
        self._phase = ""
        # Phase log identity (mstack/data/phase_log.py). The session tag is
        # fixed when run() starts, so every line of one connect shares it.
        self._phase_session = ""
        # scene 모드 slot 상태. cmd_set_slot 으로 바뀌고, 에피소드에는
        # "기록 시작 시점의 slot"(_episode_slot 캡처본)이 찍힌다 -- 저장이
        # 백그라운드라 저장 시점의 현재 slot 을 읽으면 안 된다.
        self._slot_instruction = config.language_instruction
        self._slot_instruction_id = config.instruction_id
        self._episode_slot = (self._slot_instruction, self._slot_instruction_id)
        self._ref_enqueued = False  # scene 기준 사진 자동 캡처는 세션당 1회 시도
        # depth 를 켤 카메라 역할 (#17) -- 스키마 플래그에서 한 번 파생.
        sch = getattr(config, "schema", None)
        self._depth_roles = {
            role for role, flag in (
                ("agent", getattr(sch, "save_agentview_depth", False)),
                ("wrist", getattr(sch, "save_eye_in_hand_depth", False)))
            if flag}
        # GUI 스레드에서 시그널을 미리 connect할 수 있도록 여기서 생성;
        # writer 주입/start()는 run()에서 (h5py 접근 직렬화는 saver가 소유).
        self.saver = EpisodeSaver()
        # 자기 축으로 프레임을 모으고 있는 카메라 역할 (_start_capture).
        # 기록 루프가 이 카메라를 집어가는 것은 미리보기용일 뿐이라,
        # 정지 판정에서 빼야 한다 (_get_obs).
        self._armed_capture: set = set()
        #: 마지막으로 미리보기를 보낸 시각 (_emit_frames).
        self._preview_last = 0.0
        #: 다음 홈 복귀를 에피소드로 찍을 것인가 (cmd_record_reset).
        self._reset_armed = False
        # Stale-frame bookkeeping, see _get_obs.
        self._cam_last_fp: dict = {}
        self._cam_stale: dict = {}
        self._cam_stale_run: dict = {}
        self._cam_stale_max_run: dict = {}
        # depth 수집 가드용 1회 경고 플래그 (fix/depth-gate).
        self._depth_unsupported_warned = False

    # ------------------------------------------------------------------ API
    # Called from the GUI (main) thread; safe because queue.Queue is thread-safe.
    def cmd_start_teleop(self) -> None:
        self._cmds.put(("start_teleop",))

    def cmd_auto_match_pose(self) -> None:
        self._cmds.put(("auto_match_pose",))

    def cmd_save_episode(self, success: Optional[bool]) -> None:
        self._cmds.put(("save_episode", success))

    def cmd_discard_episode(self) -> None:
        self._cmds.put(("discard_episode",))

    def cmd_skip_reset_wait(self) -> None:
        self._cmds.put(("skip_reset_wait",))

    def cmd_quit(self) -> None:
        self._cmds.put(("quit",))

    def cmd_go_home(self) -> None:
        self._cmds.put(("go_home",))

    def cmd_set_episode_success(self, name: str, success: bool) -> None:
        self.saver.enqueue_set_success(name, success)

    def cmd_record_reset(self) -> None:
        """다음 홈 복귀를 **에피소드로 찍는다**. 한 번만 걸리고 소모된다.

        지금 고른 slot 이 그 에피소드에 찍히므로, 조작자는 reset slot 을
        고른 뒤 이것을 누른다. 예약제인 이유는 양 때문이다 -- 홈 복귀는 매
        에피소드마다 도는데 그것을 전부 찍으면 reset 이 데이터의 절반이 된다.
        가장 잘 된 테이크 뒤에 한 번 거는 것이 쓰임에 맞는다.
        """
        self._cmds.put(("record_reset",))

    def cmd_set_slot(self, instruction: str, instruction_id: str) -> None:
        """scene 모드: 현재 slot(수행할 instruction)을 바꾼다.

        기록 중에 도착하면 진행 중인 에피소드에는 영향이 없고 다음
        에피소드부터 적용된다 -- 에피소드에 찍히는 slot 은 기록 *시작*
        시점의 캡처본이다 (_record_episode 참고).
        """
        self._cmds.put(("set_slot", instruction, instruction_id))

    def cmd_delete_episode(self, name: str) -> None:
        self._cmds.put(("delete_episode", name))

    def current_schema(self) -> Optional[DatasetSchemaConfig]:
        """The effective schema this session's writer is currently using,
        or None if not connected yet. ``DatasetSchemaConfig`` is a plain,
        immutable-after-construction dataclass (never mutated once the
        writer is built), so reading it from the GUI thread is safe --
        unlike the writer's open h5py.File, which only this worker thread
        may touch (see _handle_delete_episode's docstring)."""
        return self._writer.schema if self._writer is not None else None

    # --------------------------------------------------------------- helpers
    def _handle_delete_episode(self, name: str) -> None:
        """File-touching ops are serialized on the saver thread (h5py is not
        thread-safe); pending saves queued before this delete commit first."""
        self.saver.enqueue_delete(name)

    def _arm_reset(self) -> None:
        self._reset_armed = True
        self.log_message.emit(
            f"[reset] 다음 홈 복귀를 {self._slot_instruction_id or '(slot 미선택)'} "
            "로 기록합니다")

    def _handle_set_slot(self, instruction: str, instruction_id: str) -> None:
        """delete_episode 처럼 상태와 무관한 인라인 커맨드 -- 모든 드레인
        지점에서 처리한다. 파일을 만지지 않으므로 워커 스레드에서 안전하다."""
        self._slot_instruction = instruction
        self._slot_instruction_id = instruction_id
        self.log_message.emit(f"[지시문] {instruction_id}: {instruction}")

    def _poll_cmd(self, block: bool = False, timeout: float = 0.0) -> Optional[tuple]:
        """Pops queued commands, servicing ``delete_episode``/``set_slot``
        inline (they don't belong to any particular state), and returns the
        newest remaining state-machine command (start_teleop/save/discard/
        skip/quit), if any.
        """
        result = None
        try:
            while True:
                cmd = self._cmds.get(block=block, timeout=timeout)
                block = False  # only the first get() honors block/timeout
                if cmd[0] == "delete_episode":
                    self._handle_delete_episode(cmd[1])
                    continue
                if cmd[0] == "set_slot":
                    self._handle_set_slot(cmd[1], cmd[2])
                    continue
                if cmd[0] == "record_reset":
                    self._arm_reset()
                    continue
                result = cmd  # last one wins if several piled up
        except queue.Empty:
            pass
        return result

    def _set_state(self, phase: str, **extra) -> None:
        """단계 전이를 한 곳에서 처리한다 -- 화면 · 로그 · PUB.

        예전에는 ``state_changed.emit`` 만 불러서 단계가 GUI 안에서만 살았다.
        홈 복귀는 성공하면 로그를 한 줄도 안 남기므로, 사고를 되짚을 때 "그때
        무슨 단계였나" 를 마지막 로그 줄로 **추측**해야 했고 2026-09-10 에
        실제로 틀렸다 (조작자는 "녹화 종료 후 homing 첫 스텝" 이라고 정확히
        보고 있었다). 이제 단계마다 시각과 함께 남는다.

        같은 단계를 연속으로 다시 알리지는 않는다 -- 반복 구간에서 로그가 같은
        줄로 차는 것을 막는다. PUB 은 매번 보낸다 (구독자가 늦게 붙어도 현재
        단계를 알 수 있어야 한다).
        """
        t = time.time()
        self.state_changed.emit(phase)
        if phase != self._phase:
            self._phase = phase
            self.log_message.emit(f"[단계] {phase}")
            self._log_phase(t, phase=phase, **extra)
        self._phase_pub.publish(phase, **extra)

    def _log_phase(self, t: float, **fields) -> None:
        """One line in the phase log (cycle-time measurement). Never raises."""
        if not self._phase_session:
            self._phase_session = time.strftime("%Y%m%dT%H%M%S")
        cfg = self.cfg
        scene = cfg.scene_id or (cfg.scene_metadata.scene_id
                                 if cfg.scene_metadata is not None else "")
        append_phase({
            "run": cfg.run_id, "session": self._phase_session,
            "collector": cfg.collector,
            "dataset": Path(cfg.data_root).name if cfg.data_root else "",
            "scene": scene or "", "practice": bool(cfg.no_dataset),
            "t": t, **fields})

    def _drain_interrupt(self, react_to_go_home: bool = True) -> Optional[str]:
        """Non-blocking: services ``delete_episode`` inline, reports whether
        ``quit`` or ``go_home`` was queued (``quit`` wins if both arrived).
        Any other command type is meaningless mid-ramp and is intentionally
        dropped (the GUI disables those buttons during ramps).

        ``react_to_go_home=False`` is for the ramp that's already heading
        home (the top-of-loop homing ramp, and the final teardown ramp) --
        a go_home click there is a no-op, not an abort, since we're already
        doing what it asked for.
        """
        quit_seen = False
        go_home_seen = False
        try:
            while True:
                cmd = self._cmds.get_nowait()
                if cmd[0] == "delete_episode":
                    self._handle_delete_episode(cmd[1])
                elif cmd[0] == "set_slot":
                    self._handle_set_slot(cmd[1], cmd[2])
                elif cmd[0] == "record_reset":
                    self._arm_reset()
                elif cmd[0] == "quit":
                    quit_seen = True
                elif cmd[0] == "go_home":
                    go_home_seen = True
        except queue.Empty:
            pass
        if quit_seen:
            return "quit"
        if go_home_seen and react_to_go_home:
            return "go_home"
        return None

    def _drain_match_interrupt(self) -> Optional[str]:
        """Like ``_drain_interrupt``, but for ``_auto_match_pose``'s loop
        specifically: a queued ``start_teleop`` there is NOT meaningless --
        the operator is allowed to start teleop mid-align (see issue #8
        follow-up), which aborts the pull rather than silently dropping the
        click. ``quit``/``go_home`` still win over a same-batch
        ``start_teleop`` (same precedence as ``_drain_interrupt``).
        """
        quit_seen = False
        go_home_seen = False
        start_seen = False
        try:
            while True:
                cmd = self._cmds.get_nowait()
                if cmd[0] == "delete_episode":
                    self._handle_delete_episode(cmd[1])
                elif cmd[0] == "set_slot":
                    self._handle_set_slot(cmd[1], cmd[2])
                elif cmd[0] == "record_reset":
                    self._arm_reset()
                elif cmd[0] == "quit":
                    quit_seen = True
                elif cmd[0] == "go_home":
                    go_home_seen = True
                elif cmd[0] == "start_teleop":
                    start_seen = True
        except queue.Empty:
            pass
        if quit_seen:
            return "quit"
        if go_home_seen:
            return "go_home"
        if start_seen:
            return "start_teleop"
        return None

    def _emit_frames(self, obs: dict) -> None:
        """미리보기 한 장. **화면 주기는 루프 주기와 끊어 둔다.**

        부르는 자리가 셋이고 주기가 제각각이다 -- 기록 루프(cfg.fps), 램프
        (ramp_hz=100), 정렬 게이지. 예전에는 부를 때마다 보냈는데, 기록
        루프가 20 Hz 이던 동안은 그것이 곧 화면 주기였다. control 축이
        120 Hz 로 올라가면 같은 코드가 초당 120장을 Qt 큐에 밀어 넣는다 --
        화면은 그보다 빠를 수 없으므로 큐가 밀리고, 그 지연이 수집 루프로
        되돌아온다.

        tick 수가 아니라 **시각**으로 끊는 이유는 부르는 주기가 자리마다
        다르기 때문이다. 이렇게 두면 램프의 100 Hz 도 같이 정리된다.
        """
        agent = obs.get("agent")
        wrist = obs.get("wrist")
        if agent is None or wrist is None:
            return
        now = time.monotonic()
        if now - self._preview_last < 1.0 / PREVIEW_HZ:
            return
        self._preview_last = now
        self.frames_ready.emit(agent, wrist)

    def _joint_vec(self, d: dict) -> np.ndarray:
        return np.array([d[k] for k in JOINT_KEYS], dtype=float)

    def _get_obs(self, with_cameras: bool = True, count_stale: bool = False) -> dict:
        """Like ``FR3ZMQRobot.get_observation()`` but also carries ``ee_pos_quat``
        and ``joint_velocities``.

        ``with_cameras=False`` returns the robot half only -- one ZMQ
        round-trip, no image payload and no staleness bookkeeping. That is
        what the alignment gauge wants (see ``_emit_gate_status``); recording
        always takes the full observation.

        The lerobot-facing ``get_observation()`` only forwards the
        ``JOINT_KEYS``-shaped dict (it feeds LeRobot's ``observation_features``
        schema, which record_dataset.py relies on and this module must not
        change). The LIBERO writer's optional fields need the Cartesian pose
        and joint velocities too, so this pulls the same raw ZMQ observation
        once and keeps all of it -- joint_velocities is cheap (the control
        loop already computes it every tick) and only gets buffered/written
        if the active DatasetSchemaConfig asks for it.
        """
        raw = self._robot._client.get_observations()
        pos = np.asarray(raw[ROBOT_JOINT_POSITIONS], dtype=float)
        out: dict = dict(zip(JOINT_KEYS, pos.tolist()))
        out["_ee_pos_quat"] = np.asarray(raw[ROBOT_EE_POS_QUAT], dtype=float)
        out["_joint_velocities"] = np.asarray(raw[ROBOT_JOINT_VELOCITIES], dtype=float)
        # 포스·토크 (hdf5 원본 전용 기록): 노드가 필드를 제공할 때만 키가 있다.
        # 없으면 add_frame 에 None 이 넘어가 그 에피소드는 해당 데이터셋을
        # 만들지 않는다 -- 0 으로 채워 "무접촉 측정"처럼 보이게 하지 않는다.
        out["_ft"] = {k: np.asarray(raw[k], dtype=float)
                      for k in FT_OBS_KEYS if raw.get(k) is not None}
        # When the 1 kHz loop read this state (host clock). Absent on a node
        # that predates it -- the timing column is then simply not written.
        if raw.get(ROBOT_STATE_TIME) is not None:
            out["_state_time"] = float(raw[ROBOT_STATE_TIME])
        if not with_cameras:
            return out
        # Stall counting belongs to the recording loop only (count_stale). The
        # 100 Hz homing/alignment ramps also read cameras for the live view, and
        # a 30 fps camera sampled at 100 Hz returns the same frame 3 ticks in a
        # row as a matter of course -- counting there logged "stall" warnings
        # after perfectly clean episodes (2026-09-17, S025: 0 repeats recorded).
        for cam_key, cam in self._robot.cameras.items():
            # max_age_ms=500 (2026-08-26 원복): 한때 2000 으로 늘렸던 것은
            # 리더 스레드가 GUI 와 GIL 을 공유하던 시절의 완화책이다 (그때
            # 실측: 단독 41ms vs GUI 안 505~550ms). 카메라 노드 분리 후에는
            # 같은 조건 실측이 최대 35ms 라 500ms 는 정상 동작에서 절대 닿지
            # 않는 순수 카메라 건강 기준이고, 낡은 프레임이 기록에 섞이기
            # 전에 빡빡하게 끊는 쪽이 데이터에 안전하다 (사용자 결정).
            if hasattr(cam, "read_latest_stamped"):
                # Same cache entry as the image: when the node got it, and the
                # device's own frame number / timestamp (recorded as timing/*).
                frame, out[f"_{cam_key}_stamps"] = cam.read_latest_stamped(
                    max_age_ms=500)
            else:
                frame = cam.read_latest(max_age_ms=500)
            # read_latest() is non-blocking by design: it hands back whatever
            # is in the buffer and only raises once that is older than
            # max_age_ms. At 20 Hz a stalled camera silently repeats the SAME
            # image for many ticks while the joint states beside it keep
            # updating -- a frozen wrist view paired with a moving arm, which
            # is exactly the temporal misalignment a policy must not be
            # trained on. Nothing upstream reports it, so count it here:
            # identical consecutive frames are tallied per camera and
            # surfaced with the episode.
            out[cam_key] = frame
            if not count_stale or cam_key in self._armed_capture:
                # 같은 프레임이 연달아 나오는 것이 **문제인 경우에만** 센다.
                # 이 판정은 기록 주기가 카메라보다 느리다는 전제 위에 있다:
                # 20 Hz 로 집어가는데 30 fps 카메라가 같은 장을 세 tick 연속
                # 주면 그건 정지다. 그런데 자기 축으로 모으는 카메라
                # (_armed_capture) 는 여기서 집어가는 것이 미리보기용일 뿐이고,
                # control 축이 120 Hz 가 되면 30 fps 프레임이 네 tick 연속
                # 같은 것이 **정상**이다 -- 그대로 두면 정지 경고가 쉬지 않고
                # 뜬다. 실제 프레임 손실은 _collect_capture 의 frame_no 구멍이
                # 보므로, 그쪽이 이 자리를 대신한다.
                continue
            # The node's receive counter identifies a frame exactly; the image
            # hash is the fallback for a camera without it.
            seq = out.get(f"_{cam_key}_stamps", {}).get("seq")
            fp = ("seq", seq) if seq is not None else hash(frame[::37, ::37].tobytes())
            if fp == self._cam_last_fp.get(cam_key):
                self._cam_stale[cam_key] = self._cam_stale.get(cam_key, 0) + 1
                run = self._cam_stale_run.get(cam_key, 0) + 1
                self._cam_stale_run[cam_key] = run
                self._cam_stale_max_run[cam_key] = max(
                    run, self._cam_stale_max_run.get(cam_key, 0)
                )
                # One repeat is a rounding artifact; a run of them is a stall.
                if run == 3:
                    self.log_message.emit(
                        f"[카메라] {cam_key} 프레임이 {run}틱 연속 동일 -- "
                        "정지(stall) 의심, 이 에피소드는 버리는 것을 고려하세요"
                    )
            else:
                self._cam_stale_run[cam_key] = 0
            self._cam_last_fp[cam_key] = fp
        # (#17) depth 는 스키마가 켠 역할만 기록하지만, 카메라 드라이버가
        # read_latest_depth 를 지원하지 않으면 그 역할을 빼고 1회 경고 후
        # 진행한다. UI 게이트와 별개로, 구버전 설정 파일이나 코드 경로 우회를
        # 막기 위한 방어 가드다 -- 어떤 경우에도 depth 때문에 세션이 죽으면
        # 안 된다. 메서드가 있어도 예외를 던지는 드라이버(lerobot read_depth 는
        # 스트림 미개시 시 RuntimeError)가 있을 수 있어 호출도 감싼다.
        unsupported: list[str] = []
        for cam_key in list(self._depth_roles):
            cam = self._robot.cameras.get(cam_key)
            if cam is None:
                continue
            if not hasattr(cam, "read_latest_depth"):
                unsupported.append(cam_key)
                self._depth_roles.discard(cam_key)
                continue
            try:
                out[f"_{cam_key}_depth"] = cam.read_latest_depth(
                    max_age_ms=500)  # color 쪽과 같은 기준 (위 주석)
            except Exception as e:  # noqa: BLE001
                unsupported.append(f"{cam_key}({type(e).__name__})")
                self._depth_roles.discard(cam_key)
        if unsupported and not self._depth_unsupported_warned:
            self._depth_unsupported_warned = True
            self.log_message.emit(
                "[경고] depth 미지원 카메라 -- "
                f"{', '.join(sorted(unsupported))} 카메라에 read_latest_depth 가 없어 "
                "이 세션은 depth 를 기록하지 않습니다"
            )
        return out

    # ------------------------------------------------------------------ ramp
    def _ramp_to(
        self, target_q: np.ndarray, timeout_s: float = 30.0,
        react_to_go_home: bool = True
    ) -> str:
        """Returns "ok", "quit", or "go_home" (go_home only possible when
        react_to_go_home). Running out of ticks without converging is
        treated like "quit" -- same abort-the-session behavior as before.

        Always commands the gripper OPEN (GRIPPER_OPEN), not whatever it
        currently is: this is the homing/reset ramp (called at the top of
        every loop iteration and during final teardown), so an episode that
        ended with the gripper closed must not leave it closed through
        reset -- it previously echoed back obs["gripper.pos"] every tick,
        which is a no-op command (send whatever it already is), so a closed
        gripper just silently stayed closed through "homing". The command
        only needs to be set once -- FrankaFR3Robot's gripper thread drives
        toward _gripper_target continuously in the background, independent
        of whether new joint commands keep arriving -- but resending it
        every tick here is harmless and keeps this loop's shape unchanged.
        """
        q_cmd = None
        # **틱 수가 아니라 초** 로 센다. 예전에는 max_ticks 였는데, 그러면
        # RAMP_HZ 를 올리는 순간 타임아웃이 같은 비율로 짧아진다 (20->100 Hz
        # 에서 30초가 6초가 된다). 주기를 바꿔도 의미가 변하지 않아야 한다.
        for _ in range(int(timeout_s * self.cfg.ramp_hz)):
            interrupt = self._drain_interrupt(react_to_go_home=react_to_go_home)
            if interrupt:
                return interrupt
            obs = self._get_obs()
            q = np.array([obs[k] for k in JOINT_KEYS[:7]])
            if np.abs(target_q - q).max() < 0.02:
                return "ok"
            # Integrate the *commanded* position instead of re-anchoring it to
            # the measured one each tick (see _advance_cmd). Capped at
            # HOME_TICK_DQ, not RAMP_STEP: this ramp covers the long homing
            # fallback/residual move, and 2.0 rad/s would keep the driver's
            # 1.5 rad/s reference filter saturated -- a late tick in
            # saturation is what fired the 2026-09-07
            # acceleration_discontinuity abort.
            if q_cmd is None:
                q_cmd = q.copy()
            q_cmd = self._advance_cmd(q_cmd, target_q, step=self.cfg.home_tick_dq)
            cmd = dict(zip(JOINT_KEYS, np.append(q_cmd, GRIPPER_OPEN).tolist()))
            self._robot.send_action(cmd)
            self._emit_frames(obs)
            time.sleep(self.cfg.ramp_period_s)
        return "quit"

    @staticmethod
    def _ik_posture(K, target: np.ndarray, q_seed: np.ndarray,
                    q_posture: np.ndarray, iters: int = 30,
                    damping: float = 1e-4, tol: float = 1e-5,
                    posture_step: float = 0.01,
                    limit_margin: float = 0.6) -> np.ndarray:
        """Task-priority IK: 1순위 EE 목표 + 널스페이스로 자세를 q_posture 로.

        fr3_kinematics.ik 와 같은 damped-LS 지만, 매 반복 널스페이스 사영
        ``N = I - J^+J`` 을 통해 2차 목표를 함께 줄인다. 사영된 이동은 1차
        근사에서 EE 를 움직이지 않으므로, EE 는 목표 궤적을 따라가면서 팔
        구성(팔꿈치)은 별도로 홈 쪽으로 풀린다 -- "지나온 궤적을 금지
        영역으로" 제약하는 것과 같은 효과를 사영이 해석적으로 보장한다.

        2차 목표는 자세 복귀 + **관절 한계 회피**다. 한계 clip 만으로는
        널 방향이 어떤 관절을 한계 근처까지 밀고 갔다가 돌아오는 여행을
        막지 못한다(실측: 손목 j7 이 여유 2.2 rad 에서 0.38 rad 까지 접근).
        여유가 limit_margin 아래로 줄면 여유에 비례해 반대 방향으로 미는
        반발 항을 널스페이스에 함께 사영해, 한계 접근을 스스로 멈추게 한다.

        posture_step 은 반복당 2차 목표 이동 상한: EE 수렴에 3~5회 걸리므로
        웨이포인트당 0.03~0.05 rad 씩, 경로 전체(30~40틱)에 걸쳐 1 rad 이상의
        꼬임도 점진적으로 풀 수 있는 예산이다.
        """
        q = np.asarray(q_seed, dtype=np.float64).copy()
        q_posture = np.asarray(q_posture, dtype=np.float64)
        eye6 = np.eye(6)
        for _ in range(iters):
            J, T = K._jacobian_analytic(q)
            ep = target[:3, 3] - T[:3, 3]
            eR = K._rot_to_axis_angle(target[:3, :3] @ T[:3, :3].T)
            e = np.concatenate([ep, eR])
            if np.linalg.norm(e) < tol:
                break
            A = J @ J.T + damping * eye6
            sol = np.linalg.solve(A, np.column_stack([e, J]))
            dq_task = J.T @ sol[:, 0]
            N = np.eye(7) - J.T @ sol[:, 1:]
            dq_post = np.clip(q_posture - q, -posture_step, posture_step)
            # 한계 반발: 여유 < limit_margin 인 관절을 여유에 비례해 안쪽으로.
            # (여유 0 에서 posture_step, margin 에서 0 -- 연속이라 떨림 없음)
            lo = q - K.FR3_Q_MIN
            hi = K.FR3_Q_MAX - q
            rep = (np.clip(limit_margin - lo, 0.0, limit_margin)
                   - np.clip(limit_margin - hi, 0.0, limit_margin))
            dq_rep = rep * (posture_step / limit_margin)
            q = np.clip(q + dq_task + N @ (dq_post + dq_rep),
                        K.FR3_Q_MIN, K.FR3_Q_MAX)
        return q

    def _home_trajectory(self, q_now: np.ndarray) -> "list[np.ndarray] | None":
        """홈 EE 포즈까지의 관절 웨이포인트. 방향을 "위로"와 "집으로" 사이에서
        연속으로 섞는다 (근거와 상수는 파일 위 HOME_* 블록).

        두 단계로 만든다. **모양**은 EE 스텝 크기로 샘플링하고, **속도**는
        마지막에 관절 공간에서 다시 잘라(_densify) tick 당 이동을
        cfg.home_tick_dq 아래로 묶는다. 이 둘을 한 번에 하려던 것이 옛
        버그였다 -- EE 스텝은 직교 속도만 묶어서, 자코비안이 나빠지는
        자세에서는 1cm 가 관절 0.3rad 이 되고 그것이 한 tick 에 그대로 나갔다.

        IK 는 직전 해를 시드로 체인하되(_ik_posture) 널스페이스로 자세를
        reset_q 쪽으로 함께 밀기 때문에, EE 가 홈에 도착할 때쯤이면 관절도
        reset_q 에 거의 수렴해 있다. 남는 잔차는 호출자의 _ramp_to(reset_q)
        가 안전망으로 정리한다.

        반환 리스트의 한 원소 = 한 tick. None = 만들 수 없음(임포트 실패,
        IK 발산, 관절 점프 초과, 경로 미수렴).
        """
        try:
            # fr3_kinematics 는 mstack.robots 에 있다. GUI/클라이언트 모두
            # experiments/ 의 스크립트로 실행되어 sys.path 에 이미 있다.
            from mstack.robots import fr3_kinematics as K
        except ImportError:
            return None
        try:
            q = np.asarray(q_now, dtype=np.float64).copy()
            reset_q = np.asarray(self._reset_q, dtype=np.float64)
            T_now = K.fk(q)
            T_home = K.fk(reset_q)

            def _solve(T_target: np.ndarray, q_seed: np.ndarray):
                q_next = self._ik_posture(K, T_target, q_seed, reset_q)
                T_got = K.fk(q_next)
                # IK 미수렴(잔차 5mm 초과)이나 큰 관절 점프는 실패로 취급
                if np.linalg.norm(T_got[:3, 3] - T_target[:3, 3]) > 0.005:
                    return None
                if np.abs(q_next - q_seed).max() > HOME_MAX_DQ:
                    return None
                return q_next

            # ---- 1) EE 위치 경로: "위로" 와 "집으로" 를 섞어 가며 한 걸음씩
            p_home = T_home[:3, 3]
            z_clear = float(p_home[2]) + HOME_CLEAR_MARGIN_M
            p = T_now[:3, 3].astype(np.float64).copy()
            d0 = float(np.linalg.norm(p_home - p))
            s_done = 0.0                      # 진행도. 래칫이라 줄어들지 않는다
            pts: list = []
            for _ in range(HOME_MAX_STEPS):
                d = p_home - p
                dist = float(np.linalg.norm(d))
                if dist < HOME_EE_STEP_M:
                    pts.append(p_home.copy())
                    break
                s_done = max(s_done, min(max((d0 - dist) / max(d0, 1e-9), 0.0), 1.0))
                need = (min(max((z_clear - p[2]) / HOME_BLEND_M, 0.0), 1.0)
                        * (1.0 - s_done) ** HOME_SCHED_POWER)
                v = need * np.array([0.0, 0.0, 1.0]) + (1.0 - need) * (d / dist)
                n = float(np.linalg.norm(v))
                v = (d / dist) if n < 1e-9 else (v / n)
                p = p + HOME_EE_STEP_M * v
                pts.append(p.copy())
            else:
                # 상한까지 홈에 못 닿았다. 폴백이 관절 램프로 데려간다.
                return None

            # ---- 2) 자세는 경로 진행도에 비례해 slerp. 회전 스텝 상한을
            # 지키려면 위치 스텝보다 촘촘해야 하는 경우가 있어 그때 다시 나눈다.
            R0 = T_now[:3, :3]
            aa = K._rot_to_axis_angle(T_home[:3, :3] @ R0.T)
            n_rot = int(np.ceil(float(np.linalg.norm(aa)) / HOME_ROT_STEP_RAD))
            if n_rot > len(pts):
                idx = np.linspace(0, len(pts) - 1, n_rot)
                pts = [pts[int(round(i))] for i in idx]

            wps: list = []
            for i, pp in enumerate(pts, 1):
                T = np.eye(4)
                T[:3, 3] = pp
                T[:3, :3] = K.axis_angle_to_rot(aa * (i / len(pts))) @ R0
                q = _solve(T, q)
                if q is None:
                    return None
                wps.append(q)
            # 여기까지가 **모양**이다. 속도는 아직 아무도 안 봤다 -- 마지막에
            # 관절 공간에서 다시 잘라 tick 당 이동을 묶는다.
            return self._densify(q_now, wps, self.cfg.home_tick_dq)
        except Exception:  # noqa: BLE001 - 어떤 실패든 폴백이 정답
            return None

    @staticmethod
    def _densify(q_start: np.ndarray, wps: "list[np.ndarray]",
                 max_dq: "float | None" = None) -> "list[np.ndarray]":
        """웨이포인트 사이를 관절 공간에서 잘라 tick 당 이동을 ``max_dq`` 아래로.

        경로의 **모양은 그대로 두고 시간만 늘린다**. 자르는 두 점은 이미
        HOME_MAX_DQ(0.35 rad) 안에 있는 이웃한 IK 해라, 그 사이의 선형
        보간이 EE 직선에서 벗어나는 양은 무시할 만하다 (곡률은 그 구간
        길이의 제곱에 비례한다). 0.35 를 넘는 점프는 IK 가 다른 가지로
        넘어간 것이고, 그때는 보간이 아니라 폴백이 맞아서 _solve 가 미리
        걸러낸다.

        시작점을 함께 받는 이유: 첫 웨이포인트로 가는 첫 tick 이 가장 큰
        점프인 경우가 실제로 있다 (텔레옵이 끝난 자세에서 리프트로).

        ``max_dq`` 는 **호출부가 넘긴다** (``cfg.home_tick_dq``). 시그니처의
        기본값으로 두지 않는 이유: 기본 인자는 import 시점에 한 번 평가되어
        그때 로드된 스테이션 값으로 굳는다. 이 메서드는 staticmethod 라
        ``self.cfg`` 도 못 본다 -- 그래서 인자로 받는 것이 유일한 길이다.
        테스트는 명시적으로 넘겨서 쓴다.
        """
        if max_dq is None:
            raise ValueError("max_dq 를 넘겨라 -- cfg.home_tick_dq 가 그 값이다")
        out: list = []
        prev = np.asarray(q_start, dtype=np.float64)
        for q in wps:
            q = np.asarray(q, dtype=np.float64)
            n = int(np.ceil(np.abs(q - prev).max() / max_dq))
            for i in range(1, max(1, n) + 1):
                out.append(prev + (q - prev) * (i / max(1, n)))
            prev = q
        return out

    def _ramp_home(self, timeout_s: float = 30.0, react_to_go_home: bool = True) -> str:
        """EE 경로(리프트 -> 직선) homing. 실패 시 기존 관절 램프로 폴백.

        반환 계약은 _ramp_to 와 동일: "ok" / "quit" / "go_home".
        """
        obs = self._get_obs()
        q_now = np.array([obs[k] for k in JOINT_KEYS[:7]])
        if np.abs(self._reset_q - q_now).max() < 0.02:
            return "ok"
        wps = self._home_trajectory(q_now)
        if wps is None:
            self.log_message.emit(
                "[HOME] EE 경로 생성 실패 -- 관절 램프로 폴백합니다")
            return self._ramp_to(self._reset_q, timeout_s=timeout_s,
                                 react_to_go_home=react_to_go_home)
        for q_cmd in wps:
            interrupt = self._drain_interrupt(react_to_go_home=react_to_go_home)
            if interrupt:
                return interrupt
            obs = self._get_obs()
            cmd = dict(zip(JOINT_KEYS, np.append(q_cmd, GRIPPER_OPEN).tolist()))
            self._robot.send_action(cmd)
            self._emit_frames(obs)
            time.sleep(self.cfg.ramp_period_s)
        # EE 는 홈 포즈에 도착. 남은 널스페이스/추종 잔차를 관절 램프로 수렴.
        return self._ramp_to(self._reset_q, timeout_s=timeout_s,
                             react_to_go_home=react_to_go_home)

    @staticmethod
    def _advance_cmd(q_cmd: np.ndarray, target_q: np.ndarray,
                     step: "float | None" = None) -> np.ndarray:
        """Move the commanded position one ``step`` toward ``target_q``.

        Both ramps used to command ``measured + clip(target - measured)``,
        re-anchoring to the encoder every tick. That looks like it asks for
        step/dt, but the follower sits behind a critically-damped reference
        filter (``franka_fr3.py``): the filter only closes part of a gap
        per tick, and re-anchoring throws away the rest instead of letting
        the target run ahead. Simulating the real filter, the arm actually
        crept at **0.23 rad/s** -- a 1 rad move took 4.4 s. Integrating the
        command instead lets the filter saturate at its own limit and the
        same move takes 1.25 s (0.80 rad/s), a 3.5x speedup with no change
        to what the driver is allowed to do.

        ``step`` is the per-tick cap (rad @ ``cfg.ramp_hz``). ``None`` means
        ``cfg.ramp_step`` (approach_speed 2.0 rad/s), which suits the short
        pre-teleop approach ramp; the long homing fallback/residual ramp
        (``_ramp_to``) passes ``cfg.home_tick_dq`` instead so the reference
        filter never sits saturated -- a late control tick in saturation is
        what fires joint_motion_generator_acceleration_discontinuity.

        The default is resolved here and not in the signature: a default
        argument is evaluated once at import, which would freeze the value
        from whichever station happened to be loaded first.

        (Same failure mode as the action-space bug: never feed a low-pass
        filter its own output back as the setpoint.)
        """
        if step is None:
            step = self.cfg.ramp_step
        return q_cmd + np.clip(target_q - q_cmd, -step, step)

    def _approach_ramp(self, timeout: float = 3600.0) -> str:
        """Blocks (emitting frames) until the follower actually reaches the
        leader's pose. Returns "ok", "quit", or "go_home".

        Used to give up after a fixed 100 ticks (5s) and proceed to
        recording regardless of whether it had actually converged -- if the
        leader had drifted since the pose gate (GATE_RAD there is a loose
        0.5 rad) or the operator kept moving, recording could start well
        before the follower caught up, which looked like "the robot is
        still slowly catching up right after recording starts, so you have
        to hold still." Waiting for a real convergence (like _pose_gate
        already does) means recording never starts out of sync; the
        `timeout` is just a safety backstop, not a normal exit path.
        """
        deadline = time.monotonic() + timeout
        last_log = 0.0
        q_cmd = None
        while True:
            interrupt = self._drain_interrupt()
            if interrupt:
                return interrupt
            obs = self._get_obs()
            act = self._teleop.get_action()
            q_rob = np.array([obs[k] for k in JOINT_KEYS[:7]])
            q_led = np.array([act[k] for k in JOINT_KEYS[:7]])
            d = q_led - q_rob
            self._emit_frames(obs)
            if np.abs(d).max() < self.cfg.approach_done_rad:
                return "ok"
            # Integrated command, not measured+step -- see _advance_cmd. The
            # target here is live (the operator may still be moving), so the
            # commanded position is re-seeded from measurement only on entry.
            if q_cmd is None:
                q_cmd = q_rob.copy()
            q_cmd = self._advance_cmd(q_cmd, q_led)
            cmd = dict(zip(JOINT_KEYS, np.append(q_cmd, act["gripper.pos"]).tolist()))
            self._robot.send_action(cmd)
            now = time.monotonic()
            if now - last_log > 2.0:
                last_log = now
                self.log_message.emit(f"[접근] 리더에 맞추는 중 (최대 차이 {np.abs(d).max():.2f} rad)")
            if now > deadline:
                self.log_message.emit(f"[접근] {timeout:.0f}s 시간 초과 -- 세션 종료")
                return "quit"
            time.sleep(self.cfg.ramp_period_s)

    # ------------------------------------------------------------------ gate
    def _emit_gate_status(self) -> tuple[np.ndarray, bool]:
        """Reads leader+follower and emits gate_status (the GUI's live
        per-joint delta bars); returns (delta, all_ok) for the caller's own
        branching. Shared by _pose_gate's main loop and _auto_match_pose's
        loop -- the delta bars keep updating live during auto-align too, not
        just manual matching (issue #8 follow-up).

        Deliberately does NOT touch the cameras (2026-09-01). It used to read
        both of them and push a full-resolution frame pair to the GUI on every
        tick, which put the gauge -- seven cheap numbers -- behind ~1.8 MB of
        image conversion in the same queued-signal path, and capped it at the
        loop's own rate. Alignment needs the joint deltas to be responsive and
        does not need video from this loop at all: the camera node publishes
        at 30 fps and the preview threads subscribe to it directly, so the
        live view is *faster* without this. Recording is untouched -- there
        obs and action must come from the same tick, and the frames the
        operator sees have to be the frames being written.
        """
        act = self._teleop.get_action()
        obs = self._get_obs(with_cameras=False)
        q_led = np.array([act[k] for k in JOINT_KEYS[:7]])
        q_rob = np.array([obs[k] for k in JOINT_KEYS[:7]])
        delta = np.abs(q_led - q_rob)
        all_ok = bool(delta.max() <= GATE_RAD)
        # 판정(delta/all_ok)은 매 틱 계산하되 화면 갱신은 줄인다. 이 시그널
        # 하나가 DeltaBar 8개를 다시 그리게 하는데, 루프가 50Hz 라 초당 400회
        # 다시 그리기가 된다. 메인 스레드가 그걸 못 따라가면 큐가 밀리고 --
        # 2026-09-04 로그에서 11초까지 밀렸다 -- 정렬이 멈춘 것처럼 보인다
        # ("[자동정렬] 시작..."이 emit 11초 뒤에 찍혔다). 15Hz 면 게이지는
        # 여전히 부드럽고 큐는 쌓이지 않는다.
        now = time.monotonic()
        if now - self._last_gate_emit >= _GATE_EMIT_PERIOD_S:
            self._last_gate_emit = now
            self.gate_status.emit(self._joint_vec(act), self._joint_vec(obs), all_ok)
            # 리더암 상태도 여기서 같이 본다. 이 루프가 정렬 구간에서 도는
            # 유일한 곳이고, 정렬 보조가 과부하로 포기하는(blocked) 것이
            # 사람이 알아야 할 유일한 "죽기 전" 신호다 -- 벽이 실제로 죽으면
            # teleop 이 같이 죽어 세션 종료로 드러난다.
            state = str(self._leader_status().get("match_state", ""))
            if state and state != self._last_leader_state:
                self._last_leader_state = state
                self.leader_state.emit(state)
        return delta, all_ok

    def _leader_status(self) -> dict:
        get = getattr(self._teleop, "leader_status", None)
        try:
            return get() if get is not None else {}
        except Exception:      # noqa: BLE001 -- 표시용이라 조회 실패로 죽지 않는다
            return {}

    def _pose_gate(self, timeout: float = 3600.0) -> str:
        """Blocks (emitting live deltas + frames) until leader matches reset_q.

        Returns "ok", "quit", or "go_home".
        """
        self._set_state("gate")
        deadline = time.monotonic() + timeout
        # 자동 정렬이 켜져 있어도 무조건 당기지 않는다: 리더가 느슨한 게이트
        # (GATE_RAD) 안으로 들어온 뒤에만 정렬한다 -- 버튼 경로와 같은 모터
        # 보호 전제. 예전에는 게이트 진입 즉시 당겼는데, 리더가 멀리 놓여
        # 있으면 전 구간을 모터로 끌고 오는 셈이었다.
        auto_pending = self.cfg.auto_match_pose
        try:
            while True:
                cmd = self._poll_cmd()
                if cmd and cmd[0] == "quit":
                    return "quit"
                if cmd and cmd[0] == "go_home":
                    return "go_home"
                delta, all_ok = self._emit_gate_status()
                # Auto-advance is disabled on purpose: matching alone never starts
                # recording, only an explicit Start Teleop click does -- so a
                # momentary match mid-motion can't silently kick things off.
                if cmd and cmd[0] == "start_teleop":
                    # 앞 에피소드가 아직 디스크에 안 닿았으면 시작하지 않는다.
                    # 버퍼가 비압축이라 에피소드당 최대 0.74 GB 이고, 겹쳐
                    # 쌓이면 메모리가 먼저 무너진다. 실측으로 이 게이트는 거의
                    # 안 걸린다 -- 가장 짧은 갭이 10.3초인데 8워커 압축은 0.7초다.
                    # 걸린다면 그것 자체가 신호이므로 로그에 남긴다.
                    waiting = self.saver.pending() if self.saver else 0
                    if waiting:
                        self._save_gate_hits += 1
                        self.log_message.emit(
                            f"[GATE] 이전 에피소드를 저장하는 중입니다 "
                            f"({waiting}개 대기). 잠시 후 다시 누르세요.")
                    elif all_ok:
                        return "ok"
                    else:
                        self.log_message.emit(
                            f"[GATE] 아직 자세가 맞지 않습니다 (최대 차이 {delta.max():.2f} rad > {GATE_RAD} rad)"
                        )
                # 자동·수동 모두 자세 오차와 무관하게 시작한다 (2026-09-01
                # 사용자 결정). 모터 보호는 wall 이 관절별로 맡는다: 정렬
                # 반경 밖 관절은 최소 전류로만 당기고, 힘도 목표에서 멀수록
                # 약한 우물 모양이다. 케이블 보호는 아래 _auto_match_pose 의
                # roll 관절 이탈 판정이 맡는다.
                run_auto = False
                if auto_pending:
                    # 켜 둔 자동 정렬은 게이트에 들어온 뒤 한 번만 발동한다.
                    # 이탈로 중단돼도 다시 걸지 않는다 -- 시작/중단을 반복하며
                    # 로그를 채우는 대신, 사람이 버튼으로 다시 요청한다.
                    auto_pending = False
                    run_auto = True
                if cmd and cmd[0] == "auto_match_pose":
                    run_auto = True
                if run_auto:
                    outcome = self._auto_match_pose()
                    if outcome in ("quit", "go_home"):
                        return outcome
                    if outcome == "start_teleop":
                        # Operator started teleop mid-align -- the pull
                        # is already released (see _auto_match_pose), so
                        # just honor it like a normal Start Teleop click.
                        return "ok"
                    # outcome == "ok": converged, and per _auto_match_pose's
                    # contract the leader is left TORQUE-HELD at the target
                    # (not released) -- the loop just keeps looping, still
                    # in "gate", holding the matched pose until the operator
                    # actually clicks Start Teleop (or quits/goes home). The
                    # `finally` below releases it whichever way this
                    # function ends up returning.
                if time.monotonic() > deadline:
                    self.log_message.emit(f"[GATE] {timeout:.0f}s 시간 초과")
                    return "quit"
                # 게이지 전용 루프라 틱이 싸다 (카메라 없음) -- 50 Hz 로
                # 돌려 손으로 자세를 맞추는 동안 바가 즉각 따라오게 한다.
                time.sleep(0.02)
        finally:
            # Single release point for every way out of the gate state:
            # Start Teleop was clicked (leader must be free again for actual
            # teleop), or we're heading home/quitting. No-op if nothing was
            # being held (e.g. the operator never used auto-match, or
            # _auto_match_pose already released it on its own timeout/abort
            # path below).
            self._teleop.cancel_pose_match()

    def _auto_match_pose(self, timeout: float = 15.0) -> str:
        """Drives the GELLO leader's own motors to pull it the rest of the
        way onto ``self._reset_q`` -- an automated version of the manual
        nudging the operator otherwise does by hand before every episode
        (see GitHub issue #8). Only reachable once the loose manual gate
        (GATE_RAD) already passed, so this only ever closes a <= GATE_RAD
        gap, never drives across the leader's full range.

        On convergence, deliberately does NOT release the hold -- the leader
        stays torque-held at the target so it can't drift again before the
        operator actually starts teleop (_pose_gate's `finally` is what
        releases it, right as the gate state is left one way or another).
        A timeout (never converged) does release here, falling back to
        manual matching in the gate loop that's still running. Starting
        teleop is also allowed mid-align (the operator doesn't have to wait
        out the full pull): a queued ``start_teleop`` aborts it, releasing
        the hold immediately -- once real teleop is about to take over,
        holding the align force serves no purpose (_approach_ramp handles
        whatever residual gap is left).

        Returns "ok" (converged-and-held, timed-out-and-released, or the
        assist isn't available so there's nothing to do -- all three just
        resume the caller's gate loop), "out_of_range" (the operator pulled
        the leader back outside GATE_RAD mid-align -- released, caller may
        re-arm), "start_teleop" (aborted-and-released, caller should proceed
        to start teleop), "quit", or "go_home" (both release before
        returning, since either leaves the gate state for good).
        """
        try:
            self._teleop.start_pose_match(self._reset_q)
        except RuntimeError as e:
            self.log_message.emit(f"[자동정렬] 사용 불가: {e}")
            return "ok"
        self.log_message.emit("[자동정렬] 시작...")
        deadline = time.monotonic() + timeout
        while True:
            interrupt = self._drain_match_interrupt()
            if interrupt:
                self._teleop.cancel_pose_match()
                if interrupt == "start_teleop":
                    self.log_message.emit("[자동정렬] 텔레옵 시작으로 중단")
                return interrupt
            delta, all_ok = self._emit_gate_status()  # delta bars live during the pull
            # 이탈 판정은 roll 관절(J1/J3/J5/J7)만 본다 (2026-09-01 사용자
            # 결정). 그쪽은 제한 없이 돌 수 있어 크게 어긋난 채로 당기면
            # 케이블을 감는 방향으로 갈 수 있으니 멈추는 것이 맞지만,
            # 굽힘 관절(J2/J4/J6)이 멀리 있는 것은 그냥 자세가 다른 것이라
            # 멈출 이유가 없다 -- 예전에는 그것 때문에 리셋 자세에서 조금만
            # 멀어도 정렬이 시작하자마자 중단됐다.
            roll_err = delta[list(FR3_ROLL_JOINTS)]
            if roll_err.max() > ROLL_ABORT_RAD:
                self._teleop.cancel_pose_match()
                worst = int(np.argmax(roll_err))
                self.log_message.emit(
                    f"[자동정렬] roll 관절 J{FR3_ROLL_JOINTS[worst] + 1} 이(가) "
                    f"범위 밖이라 정렬을 중단합니다 "
                    f"(차이 {roll_err.max():.2f} rad > {ROLL_ABORT_RAD} rad) -- "
                    "케이블이 감기지 않게 손으로 대략 맞춘 뒤 다시 시도하세요"
                )
                self.pose_match_status.emit(float(delta.max()), True)
                return "out_of_range"
            status = self._teleop.pose_match_status()
            err = status["error"]
            done = bool(status["done"])
            self.pose_match_status.emit(float(err) if err is not None else 0.0, done)
            if status.get("aborted_wrap"):
                # 어느 관절이 한 바퀴 가까이 돌아 "벽이 어느 쪽으로 밀지
                # 모르는" 지점에 들어갔다 -- wall 이 그 관절을 손에 넘기고
                # 정렬을 스스로 취소했다. 케이블을 푸는 중일 때가 대부분이라
                # 실패가 아니라 안내로 알린다 (issue #37A).
                self._teleop.cancel_pose_match()
                self.log_message.emit(
                    "[자동정렬] 리더가 한 바퀴 가까이 돌아가 그 관절을 "
                    "풀어 두었습니다 (케이블 꼬임 해제) -- 다 푸신 뒤 "
                    "자세를 되돌리고 다시 정렬하세요"
                )
                return "ok"
            if status.get("blocked"):
                # wall 이 과부하 보호로 정렬을 포기했다 (어느 조인트가 몇 초째
                # 캡에 붙어 있었다 = 걸렸거나 붙잡혀 있다). 여기서 다시
                # 요청해봐야 같은 결과라, 사람이 원인을 치우고 다시 누르는
                # 것이 유일한 재시도 경로다 (issue #37A).
                self._teleop.cancel_pose_match()
                self.log_message.emit(
                    "[자동정렬] 리더가 걸린 것 같아 과부하 방지로 중단했습니다 "
                    "-- 걸린 곳을 풀고 다시 시도하세요"
                )
                return "ok"
            if done:
                self.log_message.emit("[자동정렬] 완료 -- 텔레옵 시작 전까지 자세를 유지합니다")
                return "ok"
            if time.monotonic() > deadline:
                if not status.get("engaged"):
                    # wall 이 토크를 아예 안 켰다 = 모든 조인트가 정렬
                    # 반경 밖이다. 게이지는 리더-팔로워 차이, wall 은
                    # 리더-리셋자세 차이라 팔로워가 리셋 자세에 정확히
                    # 있지 않으면 둘이 조금 어긋날 수 있다.
                    self.log_message.emit(
                        "[자동정렬] 리더가 정렬 반경 밖이라 모터를 걸지 "
                        "않았습니다 -- 손으로 조금 더 가까이 가져다 놓고 "
                        "다시 시도하세요"
                    )
                else:
                    self.log_message.emit(
                        f"[자동정렬] {timeout:.0f}s 시간 초과 -- 수동으로 조정하세요")
                self.pose_match_status.emit(float(err) if err is not None else 0.0, True)
                self._teleop.cancel_pose_match()
                return "ok"
            time.sleep(0.02)

    # -------------------------------------------------------------- episode
    def _emergency_hold(self, speed: float, limit: float) -> None:
        """리더를 놓쳤다고 보고 팔을 세운다.

        세우는 것은 노드가 한다 (``FrankaFR3Robot.hold``) -- 필터가 지금 어디를
        내보내고 있는지는 노드만 알기 때문이다. 여기서 명령을 그냥 끊으면
        설정점이 직전 값에 남아 팔이 거기까지 계속 간다.

        ``hold`` 가 실패해도 계속 진행한다. 에피소드를 끊고 홈으로 돌리는 것이
        더 중요하고, 노드가 안 받는 상황이면 그 다음 관측에서 어차피 NODE DOWN
        으로 잡힌다 -- 여기서 예외를 올리면 그 처리만 방해한다.
        """
        try:
            self._robot.hold()
        except Exception as e:  # noqa: BLE001
            self.log_message.emit(f"[안전] 급정거 명령이 실패했습니다: {_why(e)}")
        # 단계 표지만 보낸다. state_changed 로 새 상태를 흘리지 않는 것은
        # GUI 의 상태 표(STATE_LABELS/SHORTCUT_HINTS/키 표시)가 모르는 값을
        # 받게 되기 때문이다. 이 구간은 수백 ms 뒤 homing 으로 넘어간다.
        self._phase_pub.publish("estop", speed=round(float(speed), 3))
        # 임계는 모듈 상수가 아니라 **실제로 걸린 가드의 값**을 찍는다. 다른
        # 임계로 만든 가드가 있으면 상수는 거짓말이 된다.
        self.log_message.emit(
            f"[안전] 리더 속도 {speed:.2f} rad/s (임계 {limit:.1f}) "
            "-- 팔을 세우고 에피소드를 폐기합니다. 리더를 잡고 다시 정렬하세요.")
        # 팔이 실제로 설 때까지 기다렸다가 홈 복귀로 넘긴다. _ramp_home 은
        # **측정된** 현재 자세에서 시작하는데, 아직 감속 중이면 그 자세가
        # 곧바로 낡는다. 필터는 0.25초면 서므로 0.5초면 넉넉하다.
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                dq = self._get_obs()["_joint_velocities"][:7]
            except Exception:  # noqa: BLE001 -- 노드가 죽었으면 상위가 잡는다
                return
            if float(np.abs(dq).max()) < 0.05:
                return
            time.sleep(0.02)

    def _start_capture(self, max_frames: int) -> None:
        """이 에피소드 동안 모든 카메라 프레임을 모으게 한다.

        모으지 못하는 카메라(옛 드라이버, 더미)가 있으면 그냥 안 모은다 --
        그 세션은 옛 한 행 = 한 프레임 구조로 기록된다. 카메라 한 대가
        capture 를 지원하지 않는다고 수집을 막을 이유는 없다.
        """
        armed = set()
        for role, cam in self._robot.cameras.items():
            if not hasattr(cam, "start_capture"):
                continue
            try:
                cam.start_capture(max_frames)
                armed.add(role)
            except Exception as e:  # noqa: BLE001
                self.log_message.emit(f"[수집] {role} 프레임 모으기 실패: {e}")
        # 무장한 역할만 버퍼에 안 쌓는다. 실패한 카메라는 옛 경로 그대로
        # 20 Hz 폴링본을 쌓아야 이미지가 아예 없는 에피소드가 안 나온다.
        self._writer.expect_capture(armed)
        self._armed_capture = armed

    def _collect_capture(self) -> None:
        """모은 프레임을 버퍼에 싣고 수집을 끈다. **반드시 불러야 한다** --
        안 끄면 다음 에피소드까지 계속 쌓여 메모리가 자란다.

        ``frame_no`` 에 구멍이 있으면 보고한다. 최신만 쓰던 시절에는 무해했지만
        이제는 잃어버린 프레임이고, 드레인이 막혔거나 RCVHWM(30 fps 에서
        0.4 초치)에서 버려졌다는 뜻이다.
        """
        for role, cam in self._robot.cameras.items():
            if not hasattr(cam, "stop_capture"):
                continue
            try:
                frames, dropped = cam.stop_capture()
            except Exception as e:  # noqa: BLE001
                self.log_message.emit(f"[수집] 카메라 프레임 회수 실패: {e}")
                continue
            if not frames:
                if role in getattr(self, "_armed_capture", set()):
                    # 무장했는데 한 장도 안 왔다. 그 역할은 버퍼에도 안 쌓았으므로
                    # **이 에피소드에 그 카메라 이미지가 없다.** 조용히 넘기면
                    # 나중에 파일을 열어야 알게 된다.
                    self.log_message.emit(
                        f"[수집] {role} 프레임이 하나도 안 왔습니다 -- 이 "
                        "에피소드에는 그 카메라 이미지가 없습니다. 폐기하세요.")
                continue
            self._writer.set_capture(role, frames)
            fn = [m.get("frame_no") for _t, _a, m in frames]
            if all(x is not None for x in fn):
                gaps = sum(1 for a, b in zip(fn, fn[1:]) if b - a != 1)
                if gaps:
                    self.log_message.emit(
                        f"[수집] {role} 프레임 {gaps}곳이 끊겼습니다 "
                        f"({len(frames)}장 중) -- 드레인이 밀렸거나 큐에서 버려졌습니다")
            if dropped:
                self.log_message.emit(
                    f"[수집] {role} 상한을 넘어 {dropped}장을 못 받았습니다")

    def _record_reset(self) -> str:
        """홈 복귀를 에피소드 하나로 찍고 저장한다. "ok" 또는 "quit".

        실패는 수집을 막지 않는다 -- reset 은 부가 기록이고, 여기서 멈추면
        조작자가 다음 테이크를 못 찍는다. 못 찍었으면 이유를 남기고 넘어간다.
        """
        try:
            q_now = self._joint_vec(self._get_obs(with_cameras=False))[:7]
            traj = self._home_trajectory(q_now)
        except Exception as e:  # noqa: BLE001
            self.log_message.emit(f"[reset] 궤적을 만들지 못했습니다: {e}")
            return "ok"
        if not traj:
            self.log_message.emit(
                "[reset] 홈까지의 EE 경로를 못 풀어 이번에는 안 찍습니다 "
                "(관절 램프로 그냥 돌아갑니다)")
            return "ok"
        self.log_message.emit(f"[reset] {len(traj)} tick 기록 시작")
        outcome, n = self._record_episode(home_traj=traj)
        if outcome == "quit":
            return "quit"
        if outcome != "save" or n < 2:
            self._writer.discard_episode()
            self.log_message.emit(f"[reset] 기록하지 않았습니다 ({outcome}, {n}프레임)")
            return "ok"
        instr, iid = self._episode_slot
        self.saver.enqueue_save(self._writer.detach_buffer(), True,
                                instruction=instr, instruction_id=iid)
        self._episode_count += 1
        self.log_message.emit(f"[reset] {iid} 로 {n}프레임 저장")
        return "ok"

    def _action_from_q(self, q, gripper: float) -> dict:
        """7관절 + 그리퍼를 리더가 내는 것과 **같은 모양**의 액션으로."""
        return dict(zip(JOINT_KEYS, [float(x) for x in q[:7]] + [float(gripper)]))

    def _record_episode(self, home_traj: "list | None" = None) -> tuple[str, int]:
        """Returns (outcome, n_frames); outcome is "save", "discard", "quit", or "go_home".

        ``home_traj`` 를 주면 **리더 대신 그 궤적이 팔을 몬다** -- reset 을
        하나의 에피소드로 찍는 경로다 (PlanSlot.kind == "reset").

        새 루프를 만들지 않고 이 루프를 그대로 쓰는 이유: reset 도 다른
        에피소드와 **완전히 같은 파일**이어야 한다. 같은 주기, 같은 시간축,
        같은 카메라 수집, 같은 저장 경로를 지나야 색인·트림·변환이 reset 을
        특별 취급하지 않는다. 바뀌는 것은 액션이 어디서 오는가 하나뿐이고,
        그래서 ``actions`` 에는 팔을 실제로 움직인 명령이 그대로 남는다.
        """
        self._teleop.set_teleop_mode(home_traj is None)
        try:
            self._set_state("recording")
            self._writer.start_episode()
            # 이 에피소드에 찍힐 slot 을 기록 시작 시점에 캡처한다. 이후
            # cmd_set_slot 이 와도(다음 에피소드 준비) 이 에피소드에는 무영향.
            self._episode_slot = (self._slot_instruction, self._slot_instruction_id)
            self._cam_stale = {}  # per-episode, see _get_obs
            self._cam_stale_run = {}
            self._cam_stale_max_run = {}
            self._pending_success: Optional[bool] = None
            budget = 1.0 / self.cfg.fps
            # 명령은 기록보다 TELEOP_SUBSTEPS 배 자주 나간다. 기록 프레임은
            # 각 묶음의 **마지막** 명령 틱에 얹는다 -- 그 틱에서 명령을 보내고
            # 곧바로 관측을 읽으므로, 기록되는 (action, obs) 쌍의 의미가
            # 예전과 똑같다. 나머지 틱은 명령만 보낸다.
            substeps = self.cfg.teleop_substeps
            cmd_budget = budget / substeps
            max_frames = int(self.cfg.max_episode_seconds * self.cfg.fps)
            # 카메라가 자기 주기로 주는 것을 **전부** 모은다 (knu-2.0.0).
            # 20 Hz 루프는 미리보기·제어에 최신 한 장만 계속 쓰고, 기록은
            # 이 버퍼를 쓴다 -- 30 fps 를 20 Hz 로 뽑으면 33% 가 버려진다.
            #
            # 상한은 에피소드 상한의 두 배 여유로 둔다: 카메라가 명목 fps 보다
            # 빠를 수 있고, 여기서 막히면 그만큼이 조용히 사라진다. 20초 30 fps
            # 두 대면 1.1 GB 이므로 두 배여도 메모리가 문제 되지 않는다.
            cap_max = int(self.cfg.max_episode_seconds * 2
                          * max(60.0, float(self.cfg.fps) * 3))
            self._start_capture(cap_max)
            # control 축 행의 목표 주기. 달성된 값은 기록기가 따로 잰다.
            self._writer.set_control_hz(float(self.cfg.fps))
            t_next = time.monotonic()
            n = 0
            outcome = "save"
            stop = False
            # 리더 놓침 감시. 에피소드마다 새로 만든다 -- 직전 에피소드의
            # 홈 복귀·정렬 이동이 첫 판정에 섞이면 안 된다.
            drop_guard = LeaderDropGuard()
            # reset 궤적의 커서와, 그 동안 유지할 그리퍼 상태. 집으로 가는
            # 동안 그리퍼를 여닫을 이유가 없으므로 시작 값을 그대로 쥔다.
            traj_i = 0
            home_grip = 0.0
            if home_traj is not None:
                try:
                    home_grip = float(self._get_obs(with_cameras=False)["gripper.pos"])
                except Exception:  # noqa: BLE001 -- 못 읽으면 열린 채로 간다
                    home_grip = 0.0
            for i in range(max_frames):
                for k in range(substeps):
                    # 버튼은 명령 주기로 본다 -- 기록 주기로만 보면 반응이
                    # TELEOP_SUBSTEPS 배 느려진다.
                    cmd = self._poll_cmd()
                    if cmd:
                        if cmd[0] == "discard_episode" or cmd[0] == "quit":
                            outcome = "discard" if cmd[0] == "discard_episode" else "quit"
                            stop = True
                        elif cmd[0] == "go_home":
                            outcome = "go_home"
                            stop = True
                        elif cmd[0] == "save_episode":
                            outcome = "save"
                            self._pending_success = cmd[1]
                            stop = True
                        if stop:
                            break

                    if home_traj is not None:
                        # 궤적이 끝나면 그것이 에피소드의 끝이다. 리더 낙하
                        # 감시는 걸지 않는다 -- 리더가 팔을 몰고 있지 않으므로
                        # 그 판정의 전제(명령 = 리더 자세)가 성립하지 않는다.
                        if traj_i >= len(home_traj):
                            outcome = "save"
                            stop = True
                            break
                        action = self._action_from_q(home_traj[traj_i], home_grip)
                        traj_i += 1
                    else:
                        action = self._teleop.get_action()
                        # **보내기 전에** 본다. 이 명령이 곧 떨어진 리더의 자세다.
                        speed = drop_guard.update(
                            self._joint_vec(action)[:7], time.monotonic())
                        if drop_guard.tripped(speed):
                            self._emergency_hold(speed, drop_guard.limit)
                            outcome = "discard"
                            stop = True
                            break
                    self._robot.send_action(action)
                    t_action = time.time()
                    # 보낸 명령을 명령 축에 적는다. 추가 I/O 가 없다 -- 이미
                    # 손에 있는 값이라 루프 예산이 안 변한다. 기록 루프는
                    # substeps 개 중 마지막 하나만 control 축에 남기므로,
                    # 이것이 없으면 다섯 중 넷이 사라진다.
                    #
                    # **substeps == 1 이면 적지 않는다.** 그때는 명령 하나가
                    # 곧 control 행 하나여서 ``command/*`` 가 ``actions`` 와
                    # 글자 그대로 같은 값이 된다 (실측: 지금 구조에서도 ZOH
                    # 로 맞춰 보면 최대차 0.0). 버릴 명령이 없으니 축을 따로
                    # 둘 이유도 없다 -- ``command/`` 는 스키마 필수가 아니라
                    # 안 쓰면 그냥 안 생긴다.
                    if substeps > 1:
                        self._writer.add_command(
                            t_action, self._joint_vec(action)[:7],
                            float(action["gripper.pos"]))
                    if k < substeps - 1:
                        t_next += cmd_budget
                        time.sleep(max(0.0, t_next - time.monotonic()))
                if stop:
                    break

                obs = self._get_obs(count_stale=True)
                t_frame = time.time()

                # scene 기준 사진(§6 "사진 1장 필수"): 세션 첫 기록 프레임의
                # agentview 를 자동 캡처 후보로 보낸다. 이미 있으면 saver 가 무시.
                if (self.cfg.scene_mode and not self._ref_enqueued
                        and obs.get("agent") is not None):
                    self._ref_enqueued = True
                    self.saver.enqueue_set_reference(
                        np.ascontiguousarray(obs["agent"]))

                q = self._joint_vec(obs)
                q_cmd = self._joint_vec(action)
                self._writer.add_frame(
                    agentview_rgb=obs["agent"],
                    eye_in_hand_rgb=obs["wrist"],
                    joint_positions=q[:7],
                    gripper_position=obs["gripper.pos"],
                    ee_pos_quat=obs["_ee_pos_quat"],
                    gripper_closed=action["gripper.pos"] > 0.5,
                    joint_velocities=obs["_joint_velocities"][:7],
                    timestamp=t_frame,
                    commanded_joint_positions=q_cmd[:7],
                    commanded_gripper=float(action["gripper.pos"]),
                    agentview_depth=obs.get("_agent_depth"),
                    eye_in_hand_depth=obs.get("_wrist_depth"),
                    ft=obs.get("_ft"),
                    timing=_frame_timing(obs, t_frame, t_action),
                )
                self._emit_frames(obs)
                n = i + 1
                # 마지막 substep 분만 더한다 -- 앞의 substep 들은 이미 더했다.
                t_next += cmd_budget
                self.episode_progress.emit(n, n / self.cfg.fps)
                time.sleep(max(0.0, t_next - time.monotonic()))
            else:
                # Loop ran to completion without an explicit save/discard/quit/
                # go_home -- the operator let it hit max_episode_seconds instead
                # of clicking a save button. Auto-save as a labeled failure
                # (not unlabeled) so it's obviously not a clean success.
                outcome = "save"
                self._pending_success = False
                self.log_message.emit(
                    f"[EP] 에피소드 최대 길이({self.cfg.max_episode_seconds:.0f}s) 초과 -- 실패로 자동 저장"
                )
            # Report camera stalls with the episode, while the operator can still
            # act on it: a frozen image paired with moving joint states is not
            # something the saved file makes obvious later.
            # Only a *run* of identical frames means the camera stalled. Isolated
            # repeats are the 30 fps camera being sampled at 20 Hz -- every
            # episode has one or two and they carry no information, so reporting
            # them just trains the operator to ignore this line. A percentage
            # threshold is useless here too: 1 repeat in a 3-frame episode is 33%
            # and means nothing.
            stalls = {k: v for k, v in self._cam_stale_max_run.items() if v >= 3}
            if stalls and n:
                detail = ", ".join(
                    f"{k} 최장 {v}틱({v/self.cfg.fps*1000:.0f} ms), 총 {self._cam_stale.get(k, 0)}프레임"
                    for k, v in sorted(stalls.items())
                )
                self.log_message.emit(
                    f"[카메라] 정지 감지: {detail} / 전체 {n}프레임  ← 폐기 권장"
                )
            self._log_phase(
                time.time(), event="episode_end", outcome=outcome, frames=n,
                success=(self._pending_success if outcome == "save" else None))
            self._collect_capture()
            return outcome, n
        finally:
            self._teleop.set_teleop_mode(False)

    def _read_reset_pose(self) -> "dict | None":
        """이번 세션의 리셋 자세 -- 별칭과 7관절 절대값.

        Scene 구성기도 같은 값을 채워야 해서 (로봇 없이 만드는 새 scene 이
        옛 버전으로 내려 찍히던 문제) 본체는 session_meta 로 옮겼다.
        """
        from mstack.collect.session_meta import reset_pose_from_station

        return reset_pose_from_station()

    def _read_versions(self) -> dict:
        """이 파일을 만든 소프트웨어 판번호. 못 읽은 항목은 빠진다.

        둘에서 모은다: 수집기 커밋은 이 프로세스가 직접(git), pylibfranka·FR3
        시스템 이미지는 로봇 노드가(그쪽 인터프리터에만 pylibfranka 가 있고
        로봇 주소도 거기 있다). 어느 쪽이 실패해도 나머지는 적는다 -- 부분적인
        출처가 없는 것보다 낫다.
        """
        out: dict = {}
        commit = collector_commit()
        if commit:
            out["collector_commit"] = commit
        try:
            out.update(self._robot._client.versions() or {})
        except Exception as e:  # noqa: BLE001 -- 판번호가 수집을 막지 않는다
            self.log_message.emit(f"[판번호] 로봇에서 못 읽었습니다: {e}")
        if out:
            self.log_message.emit("[판번호] " + " · ".join(
                f"{k}={v}" for k, v in out.items()))
        return out

    def _read_payload(self) -> dict:
        """로봇의 부하 모델. 못 물어보면 빈 dict.

        새 scene 은 metadata 에 실어 보내고, 이어찍기는 SceneWriter 가 도장을
        올릴 때 쓴다 -- 올라간 버전이 부하를 요구할 수 있기 때문이다.

        못 읽으면 0 을 적지 않고 비운다. 0 은 "부하가 없었다"로 읽혀 없느니만
        못하고, 비어 있으면 그 버전으로 올라가지 못해 눈에 띈다.
        """
        try:
            info = self._robot._client.payload()
        except Exception as e:  # noqa: BLE001
            self.log_message.emit(f"[부하] 로봇에서 못 읽었습니다: {e}")
            return {}
        if not info or info.get("mass") is None:
            self.log_message.emit("[부하] 이 로봇은 부하 모델을 주지 않습니다.")
            return {}
        self.log_message.emit(
            f"[부하] {float(info['mass']) * 1000:.0f} g "
            f"(무게중심 {[round(c, 4) for c in info.get('com') or []]})")
        return info

    # ------------------------------------------------------------------- run
    def run(self) -> None:  # noqa: C901 - state machine, kept in one place on purpose
        self._phase_session = time.strftime("%Y%m%dT%H%M%S")
        try:
            self._set_state("connecting")
            self._connect()
            if self.cfg.no_dataset:
                self._writer = NullTaskWriter(schema=self.cfg.schema)
            elif self.cfg.scene_mode:
                # scene 모드: 파일명·instruction 은 config 의 task_name 이 아니라
                # scene metadata 와 저장 시점 slot 에서 나온다. 소품 인벤토리
                # 검증(미등록 ID 거부)은 SceneMetadata.validate 가 한다.
                from mstack.scene.props import active_prop_ids

                # 부하 모델을 metadata 에 싣는다 (knu-1.2.0). 정적 값이라 여기서
                # 한 번만 묻고, 새 파일을 만들 때만 쓴다 -- 이어찍기면 그 파일이
                # 이미 자기가 찍힐 때의 값을 갖고 있으므로 덮어쓰면 안 된다.
                payload = self._read_payload()
                # 리셋 자세도 같이 싣는다 (knu-1.2.1). 팔이 어디서 출발했는지는
                # 궤적을 읽는 데 필요한데, 파일에 station 이름만 있으면 그
                # 시점의 설정을 알아야 자세를 복원할 수 있다 -- 설정은 바뀐다.
                reset = self._read_reset_pose()
                # 그리고 **무엇이 이 파일을 만들었는가** (knu-1.2.2): 수집기
                # 커밋과 로봇 쪽 판번호. 같은 이유다 -- 나중에 "이 데이터는
                # 어느 코드/펌웨어에서 나왔나" 를 물을 때 파일 밖에 답이 있으면
                # 그 답은 사람 기억뿐이다.
                prov = self._read_versions()
                if self.cfg.scene_metadata is not None and not self.cfg.scene_resume:
                    from mstack.collect.session_meta import apply_to_metadata

                    apply_to_metadata(self.cfg.scene_metadata, payload, reset, prov)

                self._writer = SceneWriter(
                    root=self.cfg.data_root,
                    scene_id=self.cfg.scene_id,
                    metadata=self.cfg.scene_metadata,
                    resume=self.cfg.scene_resume,
                    schema=self.cfg.schema,
                    crop_params=self.cfg.crop_params,
                    collector=self.cfg.collector,
                    known_prop_ids=active_prop_ids(),
                    session_version=self.cfg.session_version,
                    session_payload=payload,
                    session_reset=reset,
                    session_provenance=prov,
                )
                if getattr(self._writer, "version_note", ""):
                    self.log_message.emit(f"[스키마] {self._writer.version_note}")
            else:
                self._writer = LiberoTaskWriter(
                    root=self.cfg.data_root,
                    task_name=self.cfg.task_name,
                    language_instruction=self.cfg.language_instruction,
                    resume=self.cfg.resume,
                    schema=self.cfg.schema,
                    crop_params=self.cfg.crop_params,
                )
            if hasattr(self._writer, "record_session_config"):
                # legacy/연습 전용. scene 포맷은 세션 설정을 파일에 넣지 않는다
                # -- 통제 변수는 Notion §4 레지스트리가 정본 (운영 규칙 ≠ 스키마).
                self._writer.record_session_config(
                    reset_pose=self.cfg.reset_pose,
                    grip=self.cfg.grip,
                    enable_wall=self.cfg.enable_wall,
                    max_episode_seconds=self.cfg.max_episode_seconds,
                    reset_wait_seconds=self.cfg.reset_wait_seconds,
                )
        except Exception as e:  # noqa: BLE001
            # Covers both robot/camera/GELLO connect failures and writer
            # creation failing (e.g. task file exists without --resume) --
            # either way, undo whatever hardware DID connect before returning.
            #
            # zmq.Again is by far the most common one and its own message
            # ("Resource temporarily unavailable") says nothing about what
            # actually happened: the ZMQ request to the robot node timed out.
            # Name the cause and the fix instead of the errno.
            if isinstance(e, zmq.error.Again):
                self.fatal_error.emit(
                    f"연결 실패: 로봇 노드가 응답하지 않습니다 (ZMQ 타임아웃, "
                    f"{self.cfg.hostname}:{self.cfg.robot_port}).\n"
                    "'노드 시작' 버튼으로 launch_nodes.py를 띄웠는지, 이미 떠 있다면 "
                    "제어 루프가 죽지 않았는지(리플렉스 abort) 확인하세요. "
                    "'노드 재시작'이 보통 해결합니다."
                )
            else:
                self.fatal_error.emit(f"연결 실패: {type(e).__name__}: {e}")
            for cleanup in (
                getattr(self._teleop, "disconnect", None),
                getattr(self._robot, "disconnect", None),
            ):
                if cleanup is not None:
                    try:
                        cleanup()
                    except Exception:  # noqa: BLE001
                        pass
            self._set_state("idle")
            # 마지막 표지까지 나간 뒤에 닫는다.
            self._phase_pub.close()
            return

        self._episode_count = self._writer.num_episodes
        self.connected.emit(self._episode_count, str(self._writer.path))
        self.episode_list_changed.emit(self._writer.list_episodes())
        # 이 시점 이후 파일을 만지는 호출(save/delete/list)은 전부 saver 스레드로.
        self.saver.set_writer(self._writer)
        self.saver.start()

        need_reset = False
        try:
            while self._running:
                try:
                    # **예약돼 있으면 이 홈 복귀를 에피소드로 찍는다.**
                    # 직전 테이크가 끝난 자리에서 출발하므로 (물체는 놓였고
                    # 팔은 뻗어 있다) 그것이 실제로 배우게 하고 싶은 reset
                    # 조건이다. 찍고 나서도 아래 램프는 그대로 돈다 -- 이미
                    # 집 근처라 금방 끝나고, 남은 잔차를 그쪽이 정리한다.
                    if self._reset_armed:
                        self._reset_armed = False
                        r = self._record_reset()
                        if r == "quit":
                            break

                    # react_to_go_home=False: this ramp already IS "go home",
                    # so a go_home click here is a no-op, not an abort.
                    self._set_state("homing")
                    if self._ramp_home(react_to_go_home=False) != "ok":
                        break

                    if need_reset:
                        r = self._reset_wait()
                        if r == "quit":
                            break
                        if r == "go_home":
                            continue

                    g = self._pose_gate()
                    if g == "quit":
                        break
                    if g == "go_home":
                        continue

                    self._set_state("approach")
                    a = self._approach_ramp()
                    if a == "quit":
                        break
                    if a == "go_home":
                        continue

                    outcome, n = self._record_episode()
                    need_reset = True

                    if outcome == "go_home":
                        self.episode_discarded.emit(n)
                        self.log_message.emit("[EP] 홈 이동 요청으로 에피소드 폐기")
                        continue
                    if outcome == "discard":
                        self.episode_discarded.emit(n)
                        continue
                    if n < 2:
                        self.log_message.emit("[EP] 프레임이 너무 적어 저장하지 않음")
                        continue

                    # 버퍼를 떼어 백그라운드 저장으로 넘기고 즉시 홈 복귀 진행.
                    # episode_saved/episode_list_changed는 saver가 emit.
                    if self.cfg.scene_mode:
                        instr, iid = self._episode_slot
                        self.saver.enqueue_save(
                            self._writer.detach_buffer(), self._pending_success,
                            instruction=instr, instruction_id=iid)
                    else:
                        self.saver.enqueue_save(
                            self._writer.detach_buffer(), self._pending_success)
                    self._episode_count += 1

                    if outcome == "quit":
                        break
                except (zmq.ZMQError, RuntimeError) as e:
                    # Two distinct failures land here, both needing the same
                    # recovery: (a) the robot node process died/dropped off
                    # the network (zmq.ZMQError), or (b) the process is
                    # still up but FrankaFR3Robot's 1kHz control thread died
                    # on its own -- e.g. a reflex abort -- while the ZMQ
                    # server kept answering requests normally. (b) used to
                    # be invisible: get_observations() just kept returning
                    # the last position forever, so nothing here ever
                    # errored and the GUI looked "frozen" after some number
                    # of steps with no explanation. franka_fr3.py now raises
                    # once the control thread reports itself dead, which
                    # surfaces here as a RuntimeError via ZMQClientRobot.
                    # Same recovery contract as record_dataset.py either
                    # way: discard whatever episode was in flight, wait for
                    # the node to come back, then resume from home.
                    self._writer.discard_episode()
                    self.node_status.emit(False, _why(e))
                    # 이유를 그대로 싣는다. 예전에는 이 자리에서 예외를
                    # 버리고 "무응답 또는 제어 루프 다운" 이라고만 적었는데,
                    # 그 예외가 반사의 이름을 들고 있는 유일한 것이었다 --
                    # libfranka 의 abort 메시지가 여기까지 그대로 온다
                    # ("motion aborted by reflex! [joint_velocity_violation]"
                    # 같은 것). 조작자가 "로그에 반사 종류가 안 보인다"고 한
                    # 것이 이것이다 (2026-09-06).
                    self.log_message.emit(f"[NODE DOWN] {_why(e)}")
                    self.log_message.emit(_node_down_hint(e))
                    if not self._wait_node_recovery():
                        break
                    self.node_status.emit(True, "")
                    need_reset = True
        except Exception as e:  # noqa: BLE001
            # 원인 체인 유지: wall 폴트는 "joint-limit wall thread failed"
            # from <실제 원인> 으로 올라오는데, str(e) 만 보내면 서보 ID 와
            # 에러 비트(0x20 등)가 든 원인 쪽이 통째로 사라진다.
            msg = f"{type(e).__name__}: {e}"
            if e.__cause__ is not None:
                msg += f" -- 원인: {e.__cause__}"
            self.fatal_error.emit(msg)
        finally:
            try:
                self._writer.discard_episode()
                # 대기 중인 백그라운드 저장을 모두 커밋한 뒤에야 요약/close 가능
                # (saver 종료 후에는 이 스레드가 파일을 만져도 경합 없음).
                if self.saver.isRunning():
                    self.saver.finish()
                    self.saver.wait(60000)
                self._emit_session_summary()
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._ramp_home(timeout_s=10.0)
            except Exception:  # noqa: BLE001
                pass
            for cleanup in (
                getattr(self._teleop, "disconnect", None),
                getattr(self._robot, "disconnect", None),
            ):
                if cleanup is not None:
                    try:
                        cleanup()
                    except Exception:  # noqa: BLE001
                        pass
            self._set_state("idle")

    def _connect(self) -> None:
        from mstack.comm.camera_client import NodeCamera

        self._robot = FR3ZMQRobot(
            FR3ZMQRobotConfig(
                id="fr3",
                host=self.cfg.hostname,
                port=self.cfg.robot_port,
                cameras={},
            )
        )
        # 카메라는 장치를 직접 열지 않는다 (2026-08-25, 3-프로세스 분리):
        # GUI 가 띄운 카메라 노드(mstack/comm/camera_node.py)가 장치를 독점 소유하고,
        # worker 는 최신 프레임 구독자다. 이 구조가 없앤 것 세 가지 --
        # 1) GIL 기아: GUI 렌더링·기록이 리더 스레드를 굶겨 프레임 나이가
        #    500ms 를 넘던 문제 (단독 41ms vs GUI 안 505~550ms 실측),
        # 2) device busy: 미리보기<->worker 가 장치를 주고받던 12초 핸드오프,
        # 3) wedge: 세션마다 파이프라인을 여닫다 스트림이 엉키던 문제 --
        #    노드는 한 번 열고 유지하며, 죽으면 스스로 hardware_reset 한다.
        # 시리얼을 넘기는 이유: 노드가 다른 카메라 구성으로 떠 있으면
        # connect 가 즉시 ConnectionError 로 알려 준다 (조용히 엉뚱한 화면을
        # 기록하는 것보다 낫다). NodeCamera.read_latest[_depth] 는 lerobot
        # 카메라와 같은 계약이라 아래 관측 루프는 무수정이다.
        self._robot.cameras = {
            role: NodeCamera(serial)
            for role, serial in (
                ("agent", self.cfg.agent_camera_serial),
                ("wrist", self.cfg.wrist_camera_serial),
            )
        }
        self._teleop = GelloFR3Teleop(
            GelloFR3TeleopConfig(id="gello", enable_wall=self.cfg.enable_wall, grip=self.cfg.grip)
        )
        self._robot.connect()
        self._teleop.connect()

    def _reset_wait(self) -> str:
        """Returns "ok", "quit", or "go_home".

        시간이 아니라 사람이 끝낸다 -- '리셋 완료' 버튼(Enter)을 눌러야만
        다음으로 진행 (2026-08-14 사용자 결정: 자동 진행은 물체 배치가
        끝나기 전에 게이트로 넘어가는 사고를 만든다). reset_countdown 은
        남은 시간 대신 경과 시간을 싣는다. cfg.reset_wait_seconds 는 더
        이상 진행에 쓰이지 않는다.
        """
        self._set_state("reset_wait")
        t0 = time.monotonic()
        last_count = 0.0
        while True:
            cmd = self._poll_cmd()
            if cmd:
                if cmd[0] == "skip_reset_wait":
                    return "ok"
                if cmd[0] == "quit":
                    return "quit"
                if cmd[0] == "go_home":
                    return "go_home"
            # 카운트다운은 0.1초마다면 충분하다 (소수 첫째 자리까지만 보인다).
            # 오차 게이지는 그 주기에 묶여 있으면 안 된다 -- 리셋 중에도
            # 리더를 손으로 되돌려 놓는데, 10 Hz 로는 손을 눈에 띄게 늦게
            # 따라온다. 아래 루프는 게이트와 같은 50 Hz 로 돌면서 카운트다운만
            # 솎아 낸다 (2026-09-01). 틱은 싸다: 카메라를 읽지 않고 상태
            # 읽기+방출이 1~3 ms 다 (실측).
            now = time.monotonic()
            if now - last_count >= 0.1:
                last_count = now
                self.reset_countdown.emit(now - t0)
            try:
                # 리셋 중에도 오차 게이지는 살아 있어야 한다 -- 물체를
                # 되돌리는 그 시간이 화면을 가장 많이 보는 시간이다. 라이브
                # 뷰는 미리보기 스레드가 노드 속도로 직접 그린다 (기록 외
                # 단계에서는 이 루프가 카메라를 읽지 않는다, 2026-09-01).
                self._emit_gate_status()
            except Exception:  # noqa: BLE001 -- 일시적 카메라/노드 오류로
                pass           # 카운트다운을 멈추지 않는다
            time.sleep(0.02)

    def _wait_node_recovery(self) -> bool:
        """노드가 다시 살아날 때까지 2초마다 되묻는다.

        재시도가 **왜** 실패하는지를 말한다. 예전에는 예외를 통째로 삼켜서,
        화면에는 [NODE DOWN] 한 줄이 뜬 뒤 아무 일도 없었다 -- 제어 루프가
        죽은 노드는 프로세스로는 멀쩡히 살아 답하므로 이 재시도가 영영
        실패하는데, 그 사실도 이유도 보이지 않았다 (2026-09-06 보고).

        2초마다 같은 줄을 찍으면 로그가 그것만으로 찬다. 처음 한 번, 이유가
        바뀔 때, 그리고 그 뒤로는 _RECOVERY_LOG_PERIOD_S 마다 한 번만.
        """
        last_why = None
        last_log = 0.0
        while True:
            cmd = self._poll_cmd()
            if cmd and cmd[0] == "quit":
                return False
            try:
                self._robot.reconnect_node()
            except Exception as e:  # noqa: BLE001
                why = _why(e)
                now = time.monotonic()
                if why != last_why or now - last_log > _RECOVERY_LOG_PERIOD_S:
                    self.log_message.emit(f"[NODE 대기] {why}")
                    if why != last_why:
                        self.log_message.emit(_node_down_hint(e))
                    last_why, last_log = why, now
                time.sleep(2.0)
                continue
            self.log_message.emit("[NODE OK] 재연결 완료")
            return True

    def stop(self) -> None:
        self._running = False
        self.cmd_quit()

    def _emit_session_summary(self) -> None:
        # 저장 게이트가 걸린 횟수는 압축이 갭을 따라가는지의 유일한 실측이다.
        # 0 이 아니면 30 Hz 로 올리기 전에 워커 수부터 다시 본다.
        if self._save_gate_hits:
            self.log_message.emit(
                f"[저장] 저장이 안 끝나 시작이 막힌 횟수: {self._save_gate_hits}회 "
                "-- 압축이 에피소드 간격을 따라가지 못하고 있습니다")
        episodes = self._writer.list_episodes()
        n_success = sum(1 for e in episodes if e["success"] is True)
        n_fail = sum(1 for e in episodes if e["success"] is False)
        n_unlabeled = sum(1 for e in episodes if e["success"] is None)
        frames = [e["num_samples"] for e in episodes]
        self.session_summary.emit(
            {
                "path": str(self._writer.path),
                "num_episodes": len(episodes),
                "total_frames": sum(frames),
                "min_frames": min(frames) if frames else 0,
                "max_frames": max(frames) if frames else 0,
                "num_success": n_success,
                "num_fail": n_fail,
                "num_unlabeled": n_unlabeled,
            }
        )
