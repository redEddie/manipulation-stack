"""Workspace window state objects.

These hold the data that was previously scattered across WorkspaceWindow
attributes.  Phase 3-1 moves process handles and pipeline progress here;
Phase 3-4 moves session/episode state here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QProcess, QTimer


def _new_stats() -> dict:
    """수집 카운터 한 벌. 이번 task 용과 누적용이 같은 모양이라 같은 곳에서 만든다."""
    return {"saved": 0, "success": 0, "failed": 0, "discarded": 0,
            "frames": 0, "t0": time.monotonic()}


@dataclass
class ProcessRegistry:
    """QProcess handles and pipeline progress for WorkspaceWindow."""

    node_process: QProcess | None = None
    #: 로봇 노드가 "Starting robot server" 를 찍었나 = FCI 연결까지 끝났나.
    #: 그 줄은 로봇 객체를 다 만든 **뒤에** 나오므로 준비 완료의 정본이다
    #: (scripts/launch/launch_nodes.py). 노드를 띄우자마자 붙으려 하면
    #: 조용히 실패하는데, 빠른 재개는 그 사이를 기다려야 한다.
    node_ready: bool = False
    camera_node_process: QProcess | None = None
    #: 1 kHz 원시 상태 로거. 순수 진단이라 없어도 수집은 그대로 돈다
    #: (apps/workspace/shared/raw_logger_proc.py).
    raw_logger_process: QProcess | None = None
    convert_process: QProcess | None = None
    repack_process: QProcess | None = None
    upload_process: QProcess | None = None
    replay_process: QProcess | None = None
    runme_process: QProcess | None = None
    reset_protection_process: QProcess | None = None

    pipeline_proc: QProcess | None = None
    pipeline_steps: list = field(default_factory=list)
    pipeline_results: list = field(default_factory=list)
    pipeline_t0: float = 0.0
    pipeline_step_t0: float = 0.0


@dataclass
class TrimState:
    """끝 다듬기 화면의 상태 -- 지금 문 에피소드와 그 프레임·타이머·되돌리기.

    실로봇 재생의 프로세스는 여기 없다 (ProcessRegistry) -- 이 dataclass 는
    Qt 없이 시험되는 값만 담는다.
    """

    key: tuple | None = None
    n: int = 0
    n_pending: int = 0
    frames: dict = field(default_factory=lambda: {"agent": None, "wrist": None})
    loader: Any | None = None
    timer: QTimer | None = None
    series: Any | None = None
    #: 자를 양을 바꾼 이력 -- [행동취소] 가 한 걸음씩 되돌린다. 에피소드를
    #: 새로 물 때마다 비운다 (다른 에피소드의 걸음을 되돌릴 수는 없다).
    undo: list = field(default_factory=list)


@dataclass
class CameraState:
    """Camera, depth, and point-cloud state for WorkspaceWindow."""

    camera_node_spec: str = ""
    camera_node_user_stopped: bool = False
    camera_node_crashes: list = field(default_factory=list)

    last_cam_frame: dict = field(default_factory=dict)
    stream_states: dict = field(default_factory=dict)
    live_maximized: "str | None" = None

    fps_count: int = 0
    fps_value: float = 0.0
    fps_timer: Any | None = None

    depth_consumer: "str | None" = None
    depth_img: Any | None = None
    depth_cursor: "tuple | None" = None

    cloud_worker: Any | None = None
    cloud_pts: Any | None = None
    cloud_rgb: Any | None = None
    cloud_serial: str = ""


    crop_params: dict = field(default_factory=dict)
    grid_store: dict = field(default_factory=dict)
    layout_ref: dict = field(default_factory=dict)
    #: 레이아웃 점검 탭의 슬라이드쇼 플래그. Playback 탭이 있을 땐 그
    #: 상태에 끼워 있었는데, 트림과 무관한 값이라 CameraState 로 옮겼다.
    layout_playing: bool = True


@dataclass
class SessionState:
    """Session, episode, and collection-count state for WorkspaceWindow.

    Phase 3-4 deliberately moves only the scalar/list/dict fields; the
    CollectionWorker handle (``worker``) and Qt widgets stay on the window.
    ``counters`` and ``cumulative`` are separate dict instances so that task
    counters never leak into the cumulative counters.
    """

    # per-connect task counters and cumulative counters
    counters: dict = field(default_factory=_new_stats)
    cumulative: dict = field(default_factory=_new_stats)

    # analysis / episode-stat rows
    stats: list = field(default_factory=list)
    #: 마지막 스캔 뒤에 데이터가 바뀌었나. 저장·삭제·재판정·트림·폴더 변경이
    #: 올린다. 처음이 True 인 이유는 아직 아무것도 안 읽었기 때문이다.
    stats_stale: bool = True

    #: 이번 연결 세션이 시작한 벽시계 시각 (ISO). 이력 한 줄의 시작 시각이
    #: 된다 -- counters["t0"] 는 monotonic 이라 사람이 읽을 수 없다.
    started_iso: str = ""
    #: 이번 세션을 이력에 이미 남겼나. 종료 경로가 둘이라(정상 Disconnect 와
    #: 창 닫기) 없으면 같은 세션이 두 줄로 쌓인다.
    history_written: bool = False

    # scene/no-dataset session bookkeeping
    scene_session: bool = False
    no_dataset_session: bool = False
    episodes_at_connect: int = 0

    # currently active scene file and its cached episode list
    active_file_path: Path | None = None
    active_episode_cache: list | None = None

    # last-saved episode verdict + pending toggles
    last_saved_name: str | None = None
    last_saved_success: bool = True
    pending_verdict_toggle: bool = False
    pending_success: bool | None = None

    # worker state mirror (updated from worker signals, not read directly)
    current_state: str = "idle"
    gate_ok: bool = False
