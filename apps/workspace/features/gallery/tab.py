"""Gallery tab builder for WorkspaceWindow -- 큐레이션 격자.

썸네일 한 장씩 늘어놓던 목록을 **동시에 재생되는 격자**로 바꿨다 (2026-09-11).
정지 화면으로는 큐레이션이 잡으려는 것 -- 이전 명령으로 찍힌 것, 녹화를 너무
늦게 끝낸 것 -- 이 보이지 않는다. 규칙과 근거는 ``mstack.gui.clip_grid``.

이 탭은 **재생 조작만** 둔다 (2026-09-11 조작자 결정): 무엇을 볼지 고르는
것(씬·지시문)과 무엇을 할지(판정·삭제 표시)는 전부 왼쪽 패널에 있다.
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

from apps.workspace.constants import PLAYBACK_SPEEDS
from apps.workspace.shared.sizing import shrinkable_combo


def build_gallery_tab(win) -> QWidget:
    """scene 에피소드 갤러리 (#31): 썸네일 그리드 + 재생 제어.

    더블클릭 = Trim 탭에서 크게 보기. 썸네일은 uid 기반 캐시라
    (에피소드 immutable) 첫 로드 이후에는 즉시 뜬다.
    """
    w = QWidget()
    col = QVBoxLayout(w)

    # Scene 콤보는 이 탭에 **두지 않는다** -- 왼쪽 패널(Dataset 페이지)이
    # 레이아웃에 넣어 부모를 옮긴다. 여기서는 만들기만 한다 (2026-09-11).
    win.gallery_scene_combo = QComboBox()
    shrinkable_combo(win.gallery_scene_combo)
    win.gallery_scene_combo.currentIndexChanged.connect(win.gallery_ops.refresh_gallery)

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
    win.gallery_speed_combo = QComboBox()
    shrinkable_combo(win.gallery_speed_combo)
    for label, mult in PLAYBACK_SPEEDS:
        win.gallery_speed_combo.addItem(label, mult)
    win.gallery_speed_combo.setCurrentIndex(1)          # 1x
    win.gallery_speed_combo.currentIndexChanged.connect(win.gallery_ops.on_speed_changed)
    win.gallery_speed_combo.setToolTip(tr(
        "0.5배는 접촉 순간을 한 프레임씩 볼 때, 2~3배는 긴 에피소드를 "
        "훑을 때 씁니다. 3배(60Hz)에서도 12타일이 예산의 30%로 돕니다."))
    ctl.addWidget(win.gallery_speed_combo)
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
    #: [튀는 것만 선택] 이 "저 에피소드가 보이는 자리로 가 달라" 고 적어 두는
    #: 칸. 씬 파일 로드가 비동기라 요청과 처리 사이에 한 박자가 있다
    #: (gallery/ops.go_to_episode -> on_gallery_loaded).
    win._focus_episode = None
    win.gallery_ops.refresh_gallery_scenes()
    return w
