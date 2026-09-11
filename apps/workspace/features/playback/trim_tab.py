"""Trim tab builder for WorkspaceWindow."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mstack.gui.widgets import VideoView
from mstack.gui.i18n import tr
from mstack.gui.plot_widgets import SeriesPlot


def build_trim_tab(win) -> QWidget:
    """Analysis's layout, aimed at one question: where should this take end.

    The plots are the same five as Analysis -- the tail wobble is visible
    there as clearly as anywhere -- but the right column is the episode's
    own video instead of dataset-wide statistics, because the check that
    actually matters ("did I cut the release?") is a thing you look at, not
    a number. 이 탭은 **보는 것**만 둔다: 플롯·영상·스크럽. 자를 양을
    고르고 확정하는 버튼들은 우측 패널의 Trim 상자에 있다 (2026-09-11) --
    조절·확정은 손이 선택·슬라이더에서 떠나는 오른쪽으로 뺀다. Nothing is
    written until 확정; every button before that only moves a pending count.
    """
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(4, 4, 4, 4)
    split = QSplitter(Qt.Orientation.Horizontal)

    left = QWidget()
    lcol = QVBoxLayout(left)
    lcol.setContentsMargins(0, 0, 0, 0)
    win.trim_summary = QLabel(tr("Dataset 트리나 Analysis 순위표에서 에피소드를 고르세요."))
    win.trim_summary.setWordWrap(True)
    win.trim_summary.setStyleSheet("font-weight:bold;")
    lcol.addWidget(win.trim_summary)

    grid = QGridLayout()
    win.trim_plots = {}
    for i, (title, dims) in enumerate((
        ("joint1.pos, joint2.pos", [(0, "joint1.pos"), (1, "joint2.pos")]),
        ("joint4.pos, joint5.pos", [(3, "joint4.pos"), (4, "joint5.pos")]),
        ("joint6.pos, joint7.pos", [(5, "joint6.pos"), (6, "joint7.pos")]),
        ("joint3.pos", [(2, "joint3.pos")]),
        ("gripper.pos", [(7, "gripper.pos")]),
    )):
        plot = SeriesPlot(title)
        win.trim_plots[title] = (plot, dims)
        grid.addWidget(plot, i // 2, i % 2)
    lcol.addLayout(grid, 1)
    legend = QLabel(tr("실선 observation.state   ┄ 파선 observation.commanded_state"
                       "   ┈ 점선 action     ▨ 빨간 음영 = 잘려나갈 구간"))
    legend.setStyleSheet("color:#888;")
    lcol.addWidget(legend)
    split.addWidget(left)

    right = QWidget()
    rcol = QVBoxLayout(right)
    rcol.setContentsMargins(0, 0, 0, 0)

    vids = QHBoxLayout()
    win.trim_views = {}
    for role, cap in (("agent", tr("agent")), ("wrist", tr("wrist"))):
        box = QVBoxLayout()
        v = VideoView()
        v.clear_frame(tr("에피소드를 선택하세요"))
        v.set_crop_guide(**win.cameras.crop_params[role])
        win.trim_views[role] = v
        box.addWidget(v, 1)
        lab = QLabel(cap)
        lab.setStyleSheet("color:#888;")
        lab.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        box.addWidget(lab)
        vids.addLayout(box, 1)
    rcol.addLayout(vids, 1)

    # 슬라이더는 '지금 몇 번째 프레임을 보고 있나'다. 자를 지점을 정하는
    # 것과 별개로, 잘린 뒤 마지막 프레임이 어떤 장면인지 눈으로 확인해야
    # 하기 때문에 재생/스크럽을 그대로 둔다.
    srow = QHBoxLayout()
    win.trim_play_btn = QPushButton(tr("재생"))
    win.trim_play_btn.setEnabled(False)
    win.trim_play_btn.clicked.connect(win.playback_ops.on_trim_play)
    srow.addWidget(win.trim_play_btn)
    win.trim_slider = QSlider(Qt.Orientation.Horizontal)
    win.trim_slider.setEnabled(False)
    win.trim_slider.valueChanged.connect(win.playback_ops.on_trim_scrub)
    srow.addWidget(win.trim_slider, 1)
    win.trim_pos = QLabel("-/-")
    win.trim_pos.setMinimumWidth(72)
    srow.addWidget(win.trim_pos)
    rcol.addLayout(srow)

    # 끝 다듬기 상자(단계·추천·정정·확정·경고)는 우측 패널로 옮겼다
    # (features/dataset/right_panel.py 의 Trim 상자, 2026-09-11). 이 탭에는
    # 보는 것 -- 플롯·영상·스크럽 -- 만 남긴다.
    split.addWidget(right)
    split.setSizes([640, 490])
    outer.addWidget(split)
    win.playback_ops.trim_update()
    return page

