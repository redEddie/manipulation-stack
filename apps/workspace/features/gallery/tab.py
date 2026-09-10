"""Gallery tab builder for WorkspaceWindow -- 큐레이션 격자.

썸네일 한 장씩 늘어놓던 목록을 **동시에 재생되는 격자**로 바꿨다 (2026-09-11).
정지 화면으로는 큐레이션이 잡으려는 것 -- 이전 명령으로 찍힌 것, 녹화를 너무
늦게 끝낸 것 -- 이 보이지 않는다. 규칙과 근거는 ``mstack.gui.clip_grid``.
"""
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mstack.gui.clip_grid import ClipGrid
from mstack.gui.i18n import tr

from apps.workspace.shared.sizing import shrinkable_combo

#: "2~3틱만 찍힌 것" 을 걸러내는 기본 문턱 (프레임). 20 Hz 기준 1초.
SHORT_FRAMES = 20


def build_gallery_tab(win) -> QWidget:
    """scene 에피소드 갤러리 (#31): 썸네일 그리드 + instruction 필터.

    더블클릭 = Playback 재생(기존 경로 재사용), 재판정 버튼 = Dataset
    페이지와 같은 코어(_relabel_episodes). 썸네일은 uid 기반 캐시라
    (에피소드 immutable) 첫 로드 이후에는 즉시 뜬다.
    """
    w = QWidget()
    col = QVBoxLayout(w)
    row = QHBoxLayout()
    win.gallery_scene_combo = QComboBox()
    shrinkable_combo(win.gallery_scene_combo)
    win.gallery_scene_combo.currentIndexChanged.connect(win.gallery_ops.refresh_gallery)
    row.addWidget(win.gallery_scene_combo, 2)
    win.gallery_filter_combo = QComboBox()
    shrinkable_combo(win.gallery_filter_combo)
    win.gallery_filter_combo.currentIndexChanged.connect(win.gallery_ops.apply_gallery_filter)
    row.addWidget(win.gallery_filter_combo, 2)
    b = QPushButton("↻")
    b.setToolTip(tr("scene 목록·썸네일 새로고침"))
    b.setMaximumWidth(32)
    b.clicked.connect(win.gallery_ops.refresh_gallery_scenes)
    row.addWidget(b)
    # 길이 필터. "2~3틱만 찍힌 것" 은 영상으로 찾을 일이 아니라 숫자로 바로
    # 걸러야 한다 -- 큐레이션 대상 셋 중 하나다 (조작자, 2026-09-10).
    win.gallery_len_combo = QComboBox()
    shrinkable_combo(win.gallery_len_combo)
    for label, key in ((tr("all lengths"), None),
                       (tr("short only"), "short"),
                       (tr("long only"), "long"),
                       (tr("failed only"), "failed")):
        win.gallery_len_combo.addItem(label, key)
    win.gallery_len_combo.currentIndexChanged.connect(
        win.gallery_ops.apply_gallery_filter)
    row.addWidget(win.gallery_len_combo, 1)
    win.gallery_relabel_btn = QPushButton(tr("선택 재판정"))
    win.gallery_relabel_btn.clicked.connect(win.dataset_ops.on_gallery_relabel)
    row.addWidget(win.gallery_relabel_btn)
    col.addLayout(row)

    # 재생 제어. 되감기가 따로 있는 이유는 격자가 **전부 끝난 뒤에만** 되감기
    # 때문이다 -- 다시 보고 싶을 때 기다리지 않아도 되게 한다.
    ctl = QHBoxLayout()
    win.gallery_play_btn = QPushButton(tr("▶ Play"))
    win.gallery_play_btn.clicked.connect(win.gallery_ops.toggle_play)
    ctl.addWidget(win.gallery_play_btn)
    b = QPushButton(tr("↺ Restart"))
    b.setToolTip(tr("모든 타일을 첫 프레임으로 되돌리고 다시 맞춰 출발합니다."))
    b.clicked.connect(win.gallery_ops.rewind)
    ctl.addWidget(b)
    ctl.addSpacing(12)
    win.gallery_cam_combo = QComboBox()
    shrinkable_combo(win.gallery_cam_combo)
    for label, key in (("agent", "agentview_rgb"),
                       ("wrist", "eye_in_hand_rgb")):
        win.gallery_cam_combo.addItem(label, key)
    win.gallery_cam_combo.currentIndexChanged.connect(win.gallery_ops.on_camera_changed)
    ctl.addWidget(win.gallery_cam_combo)
    ctl.addSpacing(12)
    win.gallery_prev_btn = QPushButton(tr("◀"))
    win.gallery_prev_btn.setMaximumWidth(36)
    win.gallery_prev_btn.clicked.connect(lambda: win.gallery_ops.step_page(-1))
    ctl.addWidget(win.gallery_prev_btn)
    win.gallery_page_spin = QSpinBox()
    win.gallery_page_spin.setRange(1, 1)
    win.gallery_page_spin.setPrefix(tr("Page "))
    win.gallery_page_spin.valueChanged.connect(win.gallery_ops.on_page_changed)
    ctl.addWidget(win.gallery_page_spin)
    win.gallery_page_total = QLabel("/ 1")
    ctl.addWidget(win.gallery_page_total)
    win.gallery_next_btn = QPushButton(tr("▶"))
    win.gallery_next_btn.setMaximumWidth(36)
    win.gallery_next_btn.clicked.connect(lambda: win.gallery_ops.step_page(1))
    ctl.addWidget(win.gallery_next_btn)
    ctl.addStretch(1)
    col.addLayout(ctl)
    win.gallery_grid = ClipGrid()
    win.gallery_grid.selection_changed.connect(win.gallery_ops.on_grid_selection)
    win.gallery_grid.activated.connect(win.gallery_ops.on_gallery_activated)
    col.addWidget(win.gallery_grid, 1)
    win.gallery_status = QLabel(tr("scene 을 선택하세요"))
    win.gallery_status.setStyleSheet("color:#888;")
    win.gallery_status.setWordWrap(True)
    col.addWidget(win.gallery_status)
    win._gallery_loader = None
    win._gallery_episodes = []
    win._gallery_shown = []
    win._gallery_selected = []
    win.gallery_ops.refresh_gallery_scenes()
    return w

