"""Dataset page builder for WorkspaceWindow."""

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QPushButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from mstack.gui.i18n import tr



def build_dataset(win) -> QWidget:
    w = QWidget()
    col = QVBoxLayout(w)
    col.setContentsMargins(0, 0, 0, 0)
    # 데이터 저장 경로를 **화면에서 고치는 유일한 자리**다 (2026-09-06).
    # 전에는 이 페이지 전용 칸이 따로 있어서 수집 경로와 둘로 갈라져 있었고,
    # "어느 쪽을 고쳐야 수집이 그리로 가나"가 매번 헷갈렸다. 이제 같은
    # 위젯(win.root_edit, 창이 소유)을 여기 놓는다 -- 상태바가 늘 그 값을
    # 비추고 있으므로 어느 화면에 있어도 지금 어디에 쌓이는지는 보인다.
    dr = QHBoxLayout()
    dr.addWidget(QLabel(tr("데이터 경로")))
    win.root_edit.editingFinished.connect(win.dataset_ops.on_root_changed)
    dr.addWidget(win.root_edit, 1)
    dbrowse = QPushButton(tr("..."))
    dbrowse.setMaximumWidth(36)
    dbrowse.clicked.connect(win.dataset_ops.browse_root)
    dr.addWidget(dbrowse)
    col.addLayout(dr)
    # (씬 → 지시문) 선택은 Gallery 탭이 만든 콤보를 그대로 재사용한다. Qt 는
    # 위젯을 레이아웃에 넣으면 부모가 옮겨지므로, 결과적으로 콤보는 이
    # 왼쪽 패널에만 나타난다. build_center() 가 build_left() 보다 먼저
    # 불리므로 (collect_workspace.py) 여기 도달할 때 콤보는 이미 있다.
    if hasattr(win, "gallery_scene_combo"):
        srow = QHBoxLayout()
        srow.addWidget(QLabel(tr("Scene")))
        srow.addWidget(win.gallery_scene_combo, 1)
        b = QPushButton("↻")
        b.setMaximumWidth(32)
        b.setToolTip(tr("scene 목록·썸네일 새로고침"))
        b.clicked.connect(win.gallery_ops.refresh_gallery_scenes)
        srow.addWidget(b)
        col.addLayout(srow)
        # 지시문은 **항상 펼쳐진 목록**이다 (2026-09-11) -- 콤보는 닫힌
        # 상태로 놓으면 지금 무엇이 골라졌는지, 또 어떤 지시문이 있는지가
        # 안 보인다. 라벨은 목록 위 한 줄 (가로 배치 말고 세로).
        col.addWidget(QLabel(tr("Instruction")))
        win.instruction_list = QListWidget()
        win.instruction_list.setMaximumHeight(160)     # 4~8줄
        win.instruction_list.currentItemChanged.connect(
            win.gallery_ops.apply_gallery_filter)
        col.addWidget(win.instruction_list)
    else:
        col.addWidget(QLabel(tr("Scene 목록은 Gallery 탭을 연 뒤 나타납니다")))
    # 비활성 '에피소드 검색' 입력칸을 뺐다 (2026-09-06). 검색/필터는 아직
    # 없고, 누를 수 없는 입력칸은 트리 위에서 자리만 차지했다.
    win.dataset_tree = QTreeWidget()
    win.dataset_tree.setColumnCount(3)
    # 수집자 열이 있다 (2026-09-07 조작자 요청). 에피소드 attrs 에 늘 있던
    # 값인데 화면에 열이 없어서, "이건 누가 찍었지" 를 물으려면 파일을 열어야
    # 했다 -- 여럿이 돌아가며 찍는 데이터셋에서 그것은 자주 나오는 질문이다.
    win.dataset_tree.setHeaderLabels(
        [tr("파일 / 에피소드"), tr("프레임"), tr("결과"), tr("수집자")])
    win.dataset_tree.setColumnWidth(0, 300)
    # 큐레이션은 실패 여러 개를 한 번에 지우는 작업이다.
    win.dataset_tree.setSelectionMode(
        QAbstractItemView.SelectionMode.ExtendedSelection)
    win.dataset_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
    # 트리는 확인용으로만 남긴다 -- 큐레이션의 축은 파일이 아니라
    # (씬 → 지시문) 이고, 좁힌 목록은 위 콤보로 갤러리에서 본다.
    win.dataset_tree.setMaximumHeight(180)
    win.dataset_tree.itemSelectionChanged.connect(win.dataset_ops.on_dataset_selection)
    col.addWidget(win.dataset_tree)
    # 파일 삭제는 여기 없다. 에피소드 삭제 바로 옆에 두었더니 실제로 오클릭이
    # 났고, 한 번에 태스크 하나가 통째로 날아간다. 되돌릴 수 없는 조작은
    # 한 단계 더 들어가야 닿도록 Dataset 메뉴에만 둔다.
    #
    # 구조 확인·HDF5 트리·myHDF5·튀는 것만 선택은 메뉴에 같은 항목이 있어
    # 패널에서는 뺐다 (2026-09-11) -- 삭제로 가는 문을 하나로 모으는 것이
    # 목적이므로 이 행에는 읽기/고르기만 남긴다.
    row = QHBoxLayout()
    b = QPushButton(tr("새로고침"))
    b.setToolTip(tr("데이터 폴더를 다시 읽어 목록을 새로 그립니다."))
    b.clicked.connect(win.dataset_ops.refresh_dataset_tree)
    row.addWidget(b)
    col.addLayout(row)

    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Sunken)
    line.setStyleSheet("color:#444;")
    col.addWidget(line)

    trim_btn = QPushButton(tr("끝 다듬기 (Trim 탭에서)"))
    trim_btn.setToolTip(tr(
        "선택한 에피소드를 Trim 탭에서 엽니다.\n"
        "저장 키를 누를 때 흔들린 마지막 몇 프레임을 잘라냅니다."))
    trim_btn.clicked.connect(win.playback_ops.on_open_trim)
    col.addWidget(trim_btn)

    vrow = QHBoxLayout()
    win.verdict_ok_btn = QPushButton(tr("✓ Mark success"))
    win.verdict_ok_btn.setToolTip(tr(
        "선택한 에피소드의 판정을 성공/실패로 **정합니다**. 뒤집기가 아니라서 "
        "여러 개를 골라도 결과가 하나로 정해집니다. 되돌리려면 반대쪽을 누르세요. "
        "scene 변환은 success 만 내보냅니다."))
    win.verdict_ok_btn.clicked.connect(win.dataset_ops.on_set_verdict_success)
    vrow.addWidget(win.verdict_ok_btn)
    win.verdict_fail_btn = QPushButton(tr("✗ Mark failed"))
    win.verdict_fail_btn.setToolTip(tr(
        "선택한 에피소드의 판정을 성공/실패로 **정합니다**. 뒤집기가 아니라서 "
        "여러 개를 골라도 결과가 하나로 정해집니다. 되돌리려면 반대쪽을 누르세요. "
        "scene 변환은 success 만 내보냅니다."))
    win.verdict_fail_btn.clicked.connect(win.dataset_ops.on_set_verdict_failed)
    vrow.addWidget(win.verdict_fail_btn)
    col.addLayout(vrow)

    # 재생 중에는 이 버튼 자체가 '■ 재생 중단' 으로 바뀐다 -- 별도 중단
    # 버튼은 화면 밖으로 밀려 안 보이는 일이 있었다.
    win.replay_btn = QPushButton(tr("선택 재생 (실로봇)"))
    win.replay_btn.setToolTip(tr(
        "기록된 관절 명령을 같은 주기로 다시 보내 에피소드를 실로봇에서 "
        "재현합니다.\n로봇 노드가 켜져 있어야 하고, 로봇이 실제로 "
        "움직입니다. 주변을 비우세요.\n재생 중에는 이 버튼이 '재생 중단'"
        "이 됩니다 (중단 시 로봇은 현재 포즈 유지)."))
    win.replay_btn.clicked.connect(win.playback_ops.on_replay_selected)
    col.addWidget(win.replay_btn)

    # 삭제는 표시와 실행으로 갈라진다 (2026-09-11). 에피소드 삭제로 가는
    # 문이 넷이던 것을 하나로 모으는 것이 목적 -- 격자·트리·순위표 어디서든
    # "삭제 목록에 넣기"는 자유롭고, 실제로 지우는 것은 아래 빨간 버튼
    # 하나뿐이다. 표시는 되돌릴 수 있으니 즉시, 실행은 장바구니를 거친다.
    # 삭제 표시는 격자 선택과 한 쌍이다 (2026-09-11) -- 갤러리 탭에서
    # 옮겨 왔다. 실제 삭제는 아래 빨간 버튼 하나뿐.
    win.mark_btn = QPushButton(tr("🗑 Mark for delete"))
    win.mark_btn.setToolTip(tr(
        "선택한 에피소드를 삭제 목록에 넣습니다. 이미 들어 있으면 "
        "뺍니다. 지금 지우지는 않습니다 -- 실행은 왼쪽 패널의 "
        "삭제 실행 하나뿐입니다."))
    win.mark_btn.clicked.connect(win.gallery_ops.toggle_mark)
    col.addWidget(win.mark_btn)
    win.basket_label = QLabel("")
    win.basket_label.setStyleSheet("color:#c0392b;")
    win.basket_label.setWordWrap(True)
    col.addWidget(win.basket_label)
    brow = QHBoxLayout()
    win.basket_exec_btn = QPushButton(tr("Delete marked"))
    win.basket_exec_btn.setStyleSheet(
        "background-color:#c0392b; color:white; padding:6px;")
    win.basket_exec_btn.setToolTip(tr(
        "삭제 목록에 넣은 에피소드를 한 번에 지웁니다.\n"
        "확인창에서 무엇이 지워지는지 다시 봅니다. 되돌릴 수 없습니다.\n"
        "표시는 격자·순위표 어디서든 하고, 지우는 것은 이 버튼 하나뿐입니다."))
    win.basket_exec_btn.clicked.connect(win.dataset_ops.on_delete_marked)
    brow.addWidget(win.basket_exec_btn)
    win.basket_clear_btn = QPushButton(tr("Clear marks"))
    win.basket_clear_btn.clicked.connect(win.dataset_ops.on_clear_marks)
    brow.addWidget(win.basket_clear_btn)
    col.addLayout(brow)

    # 빈 채로 시작한다. 고정 안내문은 매번 같은 말을 차지하기만 했고, 정작
    # 알아야 할 것("N개 선택됨")은 누른 뒤에만 생긴다.
    win.dataset_hint = QLabel("")
    win.dataset_hint.setStyleSheet("color:#888;")
    win.dataset_hint.setWordWrap(True)
    col.addWidget(win.dataset_hint)
    return w

