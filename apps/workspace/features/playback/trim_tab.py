"""Trim tab builder for WorkspaceWindow."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
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
    #: 플롯 띠 하나의 높이. 5개를 다 켜도 450px 이라 영상이 밀려나지 않는다.
    PLOT_H = 92

    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(4, 4, 4, 4)
    outer.setSpacing(4)

    win.trim_summary = QLabel(tr("왼쪽에서 에피소드를 하나 고르세요."))
    win.trim_summary.setWordWrap(True)
    win.trim_summary.setStyleSheet("font-weight:bold;")
    outer.addWidget(win.trim_summary)

    # ── 영상: 맨 위, 전체 폭 ──────────────────────────────────────────
    # 좌우 분할이었던 것을 세로로 쌓았다 (2026-09-12 조작자 요청: 영상이 더
    # 크면 좋겠다). 분할이면 영상이 폭의 절반도 못 쓰는데, 시계열 플롯도
    # 가로 해상도가 필요해서 둘이 폭을 다투고 있었다. 세로로 쌓으면 **둘 다**
    # 전체 폭을 쓴다.
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
    outer.addLayout(vids, 1)          # 남는 높이는 전부 영상에게

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
    outer.addLayout(srow)

    # ── 어느 플롯을 볼지 ─────────────────────────────────────────────
    # 기본은 gripper 하나다. 끝을 어디서 자를지는 대개 **그리퍼가 언제
    # 놓았나**로 정해지고, 관절은 그것이 애매할 때만 본다. 다섯 개를 늘 켜
    # 두면 영상이 그만큼 작아지는데, 정작 봐야 할 것("놓는 순간을 잘랐나")은
    # 영상 쪽이다 (조작자, 2026-09-12).
    PANELS = (
        ("joint 1,2", [(0, "joint1.pos"), (1, "joint2.pos")]),
        ("joint 3,4", [(2, "joint3.pos"), (3, "joint4.pos")]),
        ("joint 5,6", [(4, "joint5.pos"), (5, "joint6.pos")]),
        ("joint 7", [(6, "joint7.pos")]),
        ("gripper", [(7, "gripper.pos")]),
    )
    prow = QHBoxLayout()
    prow.addWidget(QLabel(tr("플롯")))
    win.trim_plots = {}
    win.trim_plot_checks = {}
    plot_box = QVBoxLayout()
    plot_box.setSpacing(2)
    for title, dims in PANELS:
        plot = SeriesPlot(title)
        plot.setMinimumHeight(PLOT_H)
        plot.setMaximumHeight(PLOT_H)
        plot.setVisible(title == "gripper")
        win.trim_plots[title] = (plot, dims)
        plot_box.addWidget(plot)
        chk = QCheckBox(title)
        chk.setChecked(title == "gripper")
        chk.toggled.connect(
            lambda on, p=plot: (p.setVisible(on)))
        win.trim_plot_checks[title] = chk
        prow.addWidget(chk)
    prow.addStretch(1)
    outer.addLayout(prow)
    outer.addLayout(plot_box)

    legend = QLabel(tr("실선 observation.state   ┄ 파선 observation.commanded_state"
                       "   ┈ 점선 action     ▨ 빨간 음영 = 잘려나갈 구간"))
    legend.setStyleSheet("color:#888;")
    outer.addWidget(legend)

    # 끝 다듬기 상자(단계·추천·정정·확정·경고)는 우측 패널로 옮겼다
    # (features/dataset/right_panel.py 의 Trim 상자, 2026-09-11). 이 탭에는
    # 보는 것 -- 영상·스크럽·플롯 -- 만 남긴다.
    win.playback_ops.trim_update()
    return page

