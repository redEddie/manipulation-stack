"""Dataset 의 우측 패널 -- 고른 에피소드의 값과 그 scene 의 배치.

왼쪽 목록의 열(프레임·결과·수집자)은 패널이 좁으면 잘린다. 열을 없애자는
것이 아니라 -- 여럿을 훑을 때는 목록이 맞다 -- **고른 한 줄은 오른쪽에서
온전히 읽히게** 한다 (2026-09-07 조작자). 같은 값이 두 군데 나오지만, 두
자리가 하는 일이 다르다: 목록은 고르는 자리, 오른쪽은 읽는 자리다.

배치도는 **고른 scene 의 것**이다. 전에는 "Scene 배치 (수집 중)" 이라 세션이
없으면 늘 비어 있었는데, 큐레이션 중에 "이 에피소드가 어떤 배치였지" 를 묻는
일이 잦다 -- 그때 Doctor 나 Configure 로 건너가야 했다.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from apps.workspace.shared.collapsible import CollapsibleBox
from apps.workspace.shared.info import InfoCard, ZoneMap
from mstack.gui.i18n import tr

#: 기준 사진 가로 상한 (닥터와 같은 값).
PHOTO_W = 320


def build_dataset_right(win) -> QWidget:
    w = QWidget()
    col = QVBoxLayout(w)
    col.setContentsMargins(0, 0, 0, 0)

    box = CollapsibleBox(tr("Episode"))
    win.ds_episode_card = InfoCard()
    box.body.addWidget(win.ds_episode_card)
    col.addWidget(box)

    box = CollapsibleBox(tr("Scene"))
    win.ds_scene_card = InfoCard()
    box.body.addWidget(win.ds_scene_card)
    win.ds_scene_zones = ZoneMap()
    box.body.addWidget(win.ds_scene_zones)
    win.ds_scene_photo = QLabel(tr("기준 사진 없음"))
    win.ds_scene_photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
    win.ds_scene_photo.setStyleSheet(
        "background:#e4e4e4; color:#444; border:1px solid #949494;")
    box.body.addWidget(win.ds_scene_photo)
    col.addWidget(box)

    box = CollapsibleBox(tr("File"), open_=False)
    # 파일 단위 값은 접어 둔다 -- 에피소드를 훑는 동안에는 안 바뀌고, 펴 두면
    # 정작 바뀌는 값(위 Episode)이 화면 밖으로 밀린다.
    win.ds_file_card = InfoCard()
    box.body.addWidget(win.ds_file_card)
    col.addWidget(box)

    # Trim -- 고른 에피소드의 끝을 다듬는 조절과 확정. 중간 Trim 탭에는
    # 보는 것(플롯·영상·스크럽)만 두고, 값을 고르고 바꾸는 버튼은 여기로
    # 뺐다 (2026-09-11). 확정은 **되돌릴 수 없는** 조작(.hdf5 가 실제로
    # 바뀐다)이므로, 영상을 본 다음 슬라이더를 만지는 손에서 멀리 -- 오른쪽
    # 패널 -- 둔다. 삭제 문을 하나로 모은 것과 같은 이유다 (예전에 삭제를
    # 에피소드 선택 옆에 두었다가 오클릭 사고가 났다).
    box = CollapsibleBox(tr("Trim"))
    win.trim_count = QLabel(tr("에피소드를 고르세요"))
    win.trim_count.setStyleSheet("font-size:15px; font-weight:bold;")
    win.trim_count.setWordWrap(True)
    box.body.addWidget(win.trim_count)

    # 누른 만큼 쌓이고, + 로 되물린다. -1..-20 을 늘어놓는 대신 네 개만 두면
    # 한 자리에서 오르내릴 수 있어 "몇 번 눌렀더라"를 셀 필요가 없다.
    step_row = QHBoxLayout()
    # 라벨의 부호는 *에피소드 길이* 기준이다: "−5" 는 5프레임 짧아진다는 뜻이라
    # 자를 양(pending)은 +5 만큼 는다. 둘을 같은 부호로 두면 −5 가 되돌리기가
    # 되어 버린다.
    for label, n in ((tr("−5"), 5), (tr("−1"), 1), (tr("+1"), -1), (tr("+5"), -5)):
        b = QPushButton(label)
        b.setToolTip(
            tr("누를 때마다 {n}프레임씩 더 자릅니다 (아직 파일은 그대로)")
            .format(n=n) if n > 0 else
            tr("누를 때마다 {n}프레임씩 되돌립니다 (원본 길이 이상으로는 안 갑니다)")
            .format(n=-n))
        b.clicked.connect(lambda _=False, k=n: win.playback_ops.trim_add(k))
        step_row.addWidget(b)
    box.body.addLayout(step_row)

    act_row = QHBoxLayout()
    sug = QPushButton(tr("추천"))
    sug.setToolTip(tr("끝에서부터 속도가 그 에피소드 중앙값 아래로 떨어지는 "
                      "지점까지를 제안합니다 (최대 15프레임)"))
    sug.clicked.connect(win.playback_ops.trim_suggest)
    act_row.addWidget(sug)
    win.trim_reset_btn = QPushButton(tr("정정"))
    win.trim_reset_btn.setToolTip(tr("고른 프레임 수를 0으로 되돌립니다. "
                                      "확정 전에는 파일이 바뀌지 않습니다."))
    win.trim_reset_btn.setEnabled(False)
    win.trim_reset_btn.clicked.connect(win.playback_ops.trim_reset)
    act_row.addWidget(win.trim_reset_btn)
    win.trim_apply_btn = QPushButton(tr("확정 (파일에 적용)"))
    win.trim_apply_btn.setStyleSheet("background-color:#c0392b; color:white; padding:6px;")
    win.trim_apply_btn.setToolTip(tr("여기서부터 .hdf5 가 실제로 바뀝니다. "
                                      "되돌릴 수 없습니다."))
    # 처음엔 닫아 둔다 -- trim_update 가 채우지만, 이 페이지는 trim_update 가
    # 먼저 불리고(build_center 가 build_right 보다 앞서 Trim 탭을 짓는다) 나중에
    # 만들어지므로 초기 상태는 여기서 박는다.
    win.trim_apply_btn.setEnabled(False)
    win.trim_apply_btn.clicked.connect(win.playback_ops.trim_apply)
    act_row.addWidget(win.trim_apply_btn, 1)
    box.body.addLayout(act_row)

    win.trim_warn = QLabel("")
    win.trim_warn.setWordWrap(True)
    win.trim_warn.setStyleSheet("color:#e67e22;")
    box.body.addWidget(win.trim_warn)
    col.addWidget(box)

    col.addStretch(1)
    return w
