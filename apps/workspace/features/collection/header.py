"""수집 HUD -- 카메라 뷰 바로 위에 고정되는 지시문·수집량·상태 띠.

왜 왼쪽 패널이 아니라 여기인가 (2026-09-04, 이슈 #38/#39):
조작자는 서서 리더암을 두 손으로 잡고, 눈은 로봇에 두고 1~1.5m 떨어진
화면을 곁눈질한다. 그런데 같은 정보가 왼쪽 Collect 페이지의 "진행" 상자에
16pt 로 있었고, 그 상자는 스크롤되는 패널의 맨 아래였다 -- 세션이 시작되면
위쪽 상자들에 밀려 y=800, 높이 395 로 패널(1009px)을 넘어 아래가 잘렸다.
"수집 갯수가 어디 보이냐"는 물음이 세 번 나온 이유다.

그래서 스크롤되지 않는 자리(카메라 위)로 올리고 거리에서 읽히게 키웠다.
숫자의 정본은 그대로 CollectionOps._refresh_instruction 하나다 -- 이
모듈은 위젯만 만들고, 채우는 것은 그쪽이다.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from mstack.gui.fonts import set_bold
from mstack.gui.i18n import tr

#: 상태별 띠 배경. 기록 중만 눈에 띄게 -- 나머지는 조용히.
STATE_COLORS = {
    "recording": "#c0392b",
    "reset_wait": "#d68910",
    "gate": "#2471a3",
    "approach": "#2471a3",
    "homing": "#5d6d7e",
}
IDLE_COLOR = "#3b3b3b"


def build_collect_header(win) -> QWidget:
    bar = QFrame()
    bar.setFrameShape(QFrame.Shape.NoFrame)
    bar.setAutoFillBackground(True)
    row = QHBoxLayout(bar)
    row.setContentsMargins(14, 8, 14, 8)
    row.setSpacing(18)

    left = QVBoxLayout()
    left.setSpacing(2)
    win.hud_instruction = QLabel(tr("(수집 세션 없음)"))
    set_bold(win.hud_instruction, 20)
    win.hud_instruction.setWordWrap(True)
    win.hud_instruction.setStyleSheet("color:#fff;")
    left.addWidget(win.hud_instruction)
    win.hud_slot = QLabel("")
    win.hud_slot.setStyleSheet("color:#cfd8dc; font-size:12px;")
    left.addWidget(win.hud_slot)
    # 오래 걸리는 일이 도는 동안 그렇다고 말하는 자리. 수집과 상관없는
    # 일(큐레이션 로드·프록시 굽기·분석 스캔)도 여기에 나온다 -- 이 띠가
    # 화면에서 유일하게 스크롤되지 않는 줄이라 놓칠 수가 없다 (조작자,
    # 2026-09-12: "로딩 동안 기다리라는 표시가 있으면 좋겠어요").
    #
    # 수집 HUD 의 글자(지시문·수집량)를 빌려 쓰지 않는다. 그쪽은 정본이
    # CollectionOps._refresh_instruction 하나인데, 다른 곳에서 덮어쓰기
    # 시작하면 수집 중에 지시문이 "불러오는 중" 으로 바뀌는 사고가 난다.
    win.hud_busy = QLabel("")
    win.hud_busy.setStyleSheet("color:#f1c40f; font-size:12px;")
    win.hud_busy.setVisible(False)
    left.addWidget(win.hud_busy)
    row.addLayout(left, 1)

    # 수집량 -- 이 띠에서 가장 큰 글자. 1m 거리에서 읽혀야 한다.
    win.hud_counter = QLabel("—")
    set_bold(win.hud_counter, 30)
    win.hud_counter.setStyleSheet("color:#fff;")
    win.hud_counter.setAlignment(Qt.AlignmentFlag.AlignRight
                                 | Qt.AlignmentFlag.AlignVCenter)
    win.hud_counter.setMinimumWidth(150)
    row.addWidget(win.hud_counter)

    win.hud_state = QLabel(tr("대기"))
    set_bold(win.hud_state, 13)
    win.hud_state.setStyleSheet("color:#fff;")
    win.hud_state.setAlignment(Qt.AlignmentFlag.AlignRight
                               | Qt.AlignmentFlag.AlignVCenter)
    win.hud_state.setMinimumWidth(210)
    row.addWidget(win.hud_state)

    win.hud_bar = bar
    set_header_state(win, "idle")
    return bar


def set_header_state(win, state: str) -> None:
    """띠 배경색과 상태 문구. 색이 곧 상태다 -- 글자를 읽지 않아도 기록
    중인지 아닌지 보이게."""
    bar = getattr(win, "hud_bar", None)
    if bar is None:
        return
    bar.setStyleSheet(
        f"background-color:{STATE_COLORS.get(state, IDLE_COLOR)};")
    win.hud_state.setText(win.STATE_LABELS.get(state, state))


# set_busy 는 apps/workspace/shared/busy.py 로 옮겼다 (2026-09-12) -- 부르는
# 쪽이 수집·갤러리·분석 셋이라 한 기능의 파일에 둘 것이 아니었다. 이 머리줄이
# 만드는 라벨(win.hud_busy)을 그 함수가 찾아 쓴다.
