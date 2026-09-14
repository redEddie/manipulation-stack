"""스키마 닥터 -- 파일에 찍힌 버전이 그 내용과 맞는가.

셋 중 이것만 **파일이 스스로에 대해 하는 주장**을 다룬다. 기록 닥터는 문장과
기록을 맞대고, 진행 닥터는 목표와 개수를 맞대는데, 여기서는 데이터세트 버전과 내용을
맞댄다.

어긋나면 조용하지 않다 -- 검증이 통째로 실패한다. S015 가 그랬다: knu-1.1.0
으로 찍혀 있는데 그 버전이 요구하는 필드가 없어 40 에피소드가 전부 실패였고,
고치는 데 손으로 짠 스크립트가 필요했다. 그때 그 사실을 볼 수 있는 자리는
터미널의 check_scene_file.py 하나였다.

**버전 통일을 권하지 않는다.** 섞여 있는 것 자체는 이미 처리된다 (변환기는
여분 필드를 허용한다). 1.2.0 을 1.0.0 으로 내리면 힘·토크 필드가 파일에
그대로인 채 버전만 그 존재를 안 알리게 되어, 버전 보고 읽는 소비자가 있는
데이터를 안 읽는다. 그래서 이 화면은 분포를 보여주고 **어긋남만** 고친다.
"""
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QPushButton,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from mstack.gui.i18n import tr


def build_schema_tab(win) -> QWidget:
    w = QWidget()
    col = QVBoxLayout(w)
    col.setContentsMargins(6, 6, 6, 6)

    win.schema_title = QLabel(tr("데이터셋을 읽는 중..."))
    win.schema_title.setStyleSheet("font-weight:bold;")
    col.addWidget(win.schema_title)

    win.schema_tree = QTreeWidget()
    win.schema_tree.setHeaderLabels(
        [tr("Scene"), tr("에피소드"), tr("찍힘"), tr("내용"), tr("초기 자세"),
         tr("상태")])
    win.schema_tree.setRootIsDecorated(False)
    win.schema_tree.header().setSectionResizeMode(
        5, QHeaderView.ResizeMode.Stretch)
    win.schema_tree.setToolTip(tr(
        "'찍힘' 은 파일이 주장하는 버전, '내용' 은 실제로 만족하는 가장 높은 "
        "버전입니다. 둘이 다르면 검증이 실패합니다.\n"
        "'초기 자세' 는 적힌 리셋 자세와 **실제로 찍힌 첫 프레임**을 맞댄 "
        "것입니다 (±5도).\n줄을 누르면 무엇이 어긋났는지 오른쪽에 나옵니다."))
    # 여러 줄을 고를 수 있다 (2026-09-14 조작자): 무엇을 올릴지는 **선택**이
    # 말하고, 시스템은 "같이 올라갈 수 있는가" 만 본다.
    win.schema_tree.setSelectionMode(
        QAbstractItemView.SelectionMode.ExtendedSelection)
    win.schema_tree.itemClicked.connect(
        lambda item, _c: win.doctor.on_schema_picked(item))
    win.schema_tree.itemSelectionChanged.connect(
        lambda: win.doctor.refresh_schema_buttons())
    col.addWidget(win.schema_tree, 1)

    # 고른 줄들을 한 번에 올린다. 27개를 한 줄씩 누르는 것은 실제로 못 할
    # 일이고(조작자, 2026-09-14), 무엇을 건드릴지는 **선택이 말해야 한다** --
    # 시스템이 미리 골라 주면 누르는 사람이 무엇을 바꾸는지 모른다.
    win.schema_all_btn = QPushButton(tr("선택한 것 버전 올리기"))
    win.schema_all_btn.setToolTip(tr(
        "고른 scene 들의 빠진 값을 채우고 각자 닿는 가장 높은 버전으로 "
        "올립니다. 버전이 섞여 있어도 됩니다 -- 파일마다 따로 계산하고, 못 "
        "가는 것은 이유와 함께 건너뜁니다. 확인창이 전부 보여줍니다."))
    win.schema_all_btn.setEnabled(False)
    win.schema_all_btn.clicked.connect(lambda: win.doctor.align_selected())
    col.addWidget(win.schema_all_btn)

    win.schema_hint = QLabel("")
    win.schema_hint.setWordWrap(True)
    win.schema_hint.setStyleSheet("color:#444;")
    col.addWidget(win.schema_hint)
    return w
