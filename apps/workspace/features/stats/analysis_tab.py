"""Analysis tab builder for WorkspaceWindow."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from mstack.data.episode_stats import STILL_VEL, TASK_DEV_LIMIT
from mstack.gui.i18n import tr

from apps.workspace.shared.sizing import shrinkable_combo
from mstack.gui.plot_widgets import DistStrip, Histogram, LegendStrip, SeriesPlot


def build_analysis_tab(win) -> QWidget:
    """Center-tab analysis: the curve view plus the curation list.

    It lives in the center, next to Live/Gallery/Trim, because judging a take
    means looking at its curves and its video together -- putting the plots
    in a side panel would have made them too narrow to read.
    """
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(4, 4, 4, 4)
    split = QSplitter(Qt.Orientation.Horizontal)

    left = QWidget()
    lcol = QVBoxLayout(left)
    lcol.setContentsMargins(0, 0, 0, 0)
    # 여기 있던 **요약 텍스트 줄**(에피소드 수·프레임·그룹 수, 그리고 고른
    # 에피소드의 평균과 차이·멈춤%·길이)을 뺐다 (조작자, 2026-09-12:
    # "상단에 정보가 왜 텍스트로 표시되지? 어차피 겹치는 정보들이면 안
    # 보여줘도 좋을 것 같아"). 고른 에피소드의 값은 오른쪽 후보 목록의
    # 칸이 이미 같은 숫자를 보여주고, 범위와 개수는 상자 제목이 말한다.
    win.plot_grid = QGridLayout()
    win.series_plots = {}
    # LeRobot 뷰어와 같은 묶음: 인접 관절끼리 스케일이 비슷해 같은 축에 얹힌다.
    for i, (title, dims) in enumerate((
        ("joint1.pos, joint2.pos", [(0, "joint1.pos"), (1, "joint2.pos")]),
        ("joint4.pos, joint5.pos", [(3, "joint4.pos"), (4, "joint5.pos")]),
        ("joint6.pos, joint7.pos", [(5, "joint6.pos"), (6, "joint7.pos")]),
        ("joint3.pos", [(2, "joint3.pos")]),
        ("gripper.pos", [(7, "gripper.pos")]),
    )):
        plot = SeriesPlot(title)
        win.series_plots[title] = (plot, dims)
        win.plot_grid.addWidget(plot, i // 2, i % 2)
    lcol.addLayout(win.plot_grid, 1)
    lcol.addWidget(LegendStrip())
    split.addWidget(left)

    right = QWidget()
    rcol = QVBoxLayout(right)
    rcol.setContentsMargins(0, 0, 0, 0)

    # 차원별 σ(Δa). 막대 하나(전체 평균)가 아니라 **분포**를 그린다 -- 평균이
    # 같아도 테이크마다 두 배씩 흔들리는 차원이 있고, 큐레이션에서 묻는 것은
    # 그 흔들림이다 (조작자, 2026-09-12). 모수는 **지금 목록**이다: 큐레이션은
    # (씬 → 지시문) 안에서만 하므로, 전체 데이터셋으로 재면 작업이 다른
    # 에피소드들의 퍼짐이 섞여 비교가 흐려진다.
    win.dim_bars = DistStrip()
    # 제목이 **모수를 직접 말한다.** "지금 목록" 이라고만 적었더니 그게 지금
    # 씬인지 지금 지시문인지 알 수 없었다 (조작자, 2026-09-12). 값은
    # refresh_dim_dist 가 채운다 -- 범위를 바꾸면 제목도 따라 바뀐다.
    win.dim_box = QGroupBox(tr("차원별 σ(Δa) 분포"))
    dim_box = win.dim_box
    dim_col = QVBoxLayout(dim_box)
    dim_col.addWidget(win.dim_bars)
    dim_legend = QLabel(tr("가는 선 p10~p90   ▬ 평균±σ   ▮ 빨강 = 평균"))
    dim_legend.setStyleSheet("color:#888;")
    dim_col.addWidget(dim_legend)
    rcol.addWidget(dim_box)

    win.da_hist = Histogram(tr("에피소드 평균 |Δa| 분포"))
    rcol.addWidget(win.da_hist)

    # 제목이 **무엇의 후보인지** 말한다 (refresh_rank_list 가 채운다).
    win.filt_box = QGroupBox(tr("큐레이션 후보"))
    filt = win.filt_box
    fcol = QVBoxLayout(filt)
    # 후보 목록의 **범위**(씬 → 지시문)는 왼쪽 패널이 정본이다 (Scene 콤보 +
    # Instruction 목록). 여기에 따로 그룹 콤보를 두면 같은 축이 두 군데
    # 생긴다 (조작자, 2026-09-11).
    #
    # **정렬은 남긴다.** 상황에 따라 이상치를 찾는 수단이다 (조작자,
    # 2026-09-12): "늘어짐" 은 녹화를 늦게 끝낸 것을, "짧음" 은 2~3틱짜리를
    # 위로 끌어올린다. 다만 기본은 **에피소드 순**이다 -- 왼쪽 목록·격자와
    # 같은 순서라야 세 화면을 오갈 때 헷갈리지 않고, 기준을 바꾼 것이 눈에
    # 띈다 (예전 기본값은 '급함' 이라 처음부터 다른 순서인 줄 모르고 볼 수
    # 있었다).
    sort_row = QHBoxLayout()
    sort_row.addWidget(QLabel(tr("정렬")))
    win.rank_combo = QComboBox()
    shrinkable_combo(win.rank_combo)
    for label, key in ((tr("에피소드 순"), None),
                       (tr("급함"), "fast"),
                       (tr("늘어짐"), "slow"),
                       (tr("멈춤 많음"), "still"),
                       (tr("짧음"), "short"),
                       (tr("긺"), "long")):
        win.rank_combo.addItem(label, key)
    win.rank_combo.currentIndexChanged.connect(win.stats_ops.refresh_rank_list)
    sort_row.addWidget(win.rank_combo, 1)
    fcol.addLayout(sort_row)

    len_row = QHBoxLayout()
    len_row.addWidget(QLabel(tr("길이(초)")))
    win.len_min_spin = QSlider(Qt.Orientation.Horizontal)
    win.len_max_spin = QSlider(Qt.Orientation.Horizontal)
    for s in (win.len_min_spin, win.len_max_spin):
        s.setRange(0, 300)
        s.valueChanged.connect(win.stats_ops.refresh_rank_list)
    win.len_min_spin.setValue(0)
    win.len_max_spin.setValue(300)
    len_row.addWidget(win.len_min_spin, 1)
    len_row.addWidget(win.len_max_spin, 1)
    win.len_label = QLabel("-")
    win.len_label.setMinimumWidth(96)
    len_row.addWidget(win.len_label)
    fcol.addLayout(len_row)

    win.rank_tree = QTreeWidget()
    win.rank_tree.setColumnCount(4)
    win.rank_tree.setHeaderLabels([tr("에피소드"), tr("평균과 차이"), tr("멈춤%"),
                                    tr("길이")])
    win.rank_tree.setRootIsDecorated(False)
    win.rank_tree.setColumnWidth(0, 150)
    for c in range(1, 4):
        win.rank_tree.setColumnWidth(c, 76)
    for c, tip in enumerate((
            tr("파일 · 에피소드"),
            tr("이 에피소드의 평균 |Δa| 에서 같은 (scene·문장) 그룹 평균을 뺀 값 (rad/frame).\n"
               "+ 는 그 작업의 보통 테이크보다 급하게, - 는 느리게 움직인 것.\n"
               "±{d} 를 넘으면 빨강/파랑").format(d=TASK_DEV_LIMIT),
            tr("속도가 {v} rad/frame 미만이던 프레임 비율 — 망설임").format(v=STILL_VEL),
            tr("에피소드 길이 (초)"))):
        win.rank_tree.headerItem().setToolTip(c, tip)
    win.rank_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    win.rank_tree.itemSelectionChanged.connect(win.scene_planning.on_rank_selected)
    win.rank_tree.setMinimumHeight(220)
    fcol.addWidget(win.rank_tree, 1)
    # 판정선만 한 줄로 남긴다. 나머지 정의는 헤더 툴팁 -- 조작자가 코드를
    # 열지 않고도 "몇이면 이상한가"를 알아야 하지만, 그게 목록을 밀어내면
    # 정작 봐야 할 후보가 안 보인다.
    # 이 줄은 Statistics 패널의 '움직임 분석' 상자에 있었다. 쓰는 쪽은
    # 여기(refresh_rank_list, on_select_flagged)인데 정작 Analysis 를 볼 때는
    # 그 패널이 안 보였다 -- 몇 개 중 몇 개가 걸렸는지를 화면 밖에서 말하고
    # 있었던 셈이다 (2026-09-12).
    # 회색 줄은 **하나**다. 두 줄이던 것을 합쳤다 (조작자, 2026-09-12:
    # "회색으로 뭔가 주절주절 설명하지만 실제로 유효한 문장은 ±0.004 에 대한
    # 설명뿐이지 않나?"). 남긴 것은 판정선 한 마디와, 지금 목록에 그 밖이
    # 몇 개인지 -- 나머지(범위·표시 개수)는 상자 제목으로 올라갔다.
    hint_row = QHBoxLayout()
    win.stats_hint = QLabel(
        tr("±{d} 밖 = 급함(빨강 바탕) / 늘어짐(파랑 바탕)").format(d=TASK_DEV_LIMIT))
    win.stats_hint.setStyleSheet("color:#888;")
    win.stats_hint.setWordWrap(True)
    hint_row.addWidget(win.stats_hint, 1)
    helpb = QPushButton("?")
    helpb.setFixedWidth(24)
    helpb.setToolTip(tr("칼럼 정의와 판정 기준 (docs/curation-metrics.md)"))
    helpb.clicked.connect(win.stats_ops.on_metric_help)
    hint_row.addWidget(helpb)
    fcol.addLayout(hint_row)

    # 여기 있던 [🗑 Mark for delete] 를 뺐다 (2026-09-12). 순위표 선택이
    # 공유 선택이 된 뒤로는 왼쪽 패널의 같은 이름 버튼과 **같은 것에 같은
    # 일**을 한다 -- 삭제로 가는 문은 하나다. 남은 둘은 찾는 것(튀는 것만
    # 선택)과 보는 것(Trim 에서 재생)이다.
    btns = QHBoxLayout()
    for text, slot in ((tr("튀는 것만 선택"), win.stats_ops.on_select_flagged),
                       (tr("Trim 에서 재생"), win.trim_ops.on_rank_trim)):
        b = QPushButton(text)
        b.clicked.connect(slot)
        btns.addWidget(b)
    fcol.addLayout(btns)
    rcol.addWidget(filt, 1)
    split.addWidget(right)
    split.setSizes([700, 430])
    outer.addWidget(split)
    return page
