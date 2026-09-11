"""Stats page builder for WorkspaceWindow."""
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from mstack.gui.i18n import tr

from apps.workspace.shared.sizing import cap_rows


def build_stats(win) -> QWidget:
    """수집자 순위표 하나만 있는 화면 (조작자, 2026-09-12: "statistics 에서는
    오직 수집자 리더보드만 보이도록").

    여기 있던 것들이 어디로 갔는지:

    - **수집 현황** (저장·성공·실패·버림·프레임·경과·분당) 카운터 두 열은
      지웠다. 수집 중에 눈이 가는 자리는 카메라 위 HUD 지 이 화면이 아니고
      (그래서 2026-09-04 에 HUD 를 만들었다), 세션이 끝난 숫자는 그대로
      아래 순위표에 한 줄로 쌓인다. 세션 기록(collection_history.jsonl)은
      계속 같은 카운터에서 나오므로 숫자가 사라진 것은 아니다.
    - **최근 세션** 목록도 지웠다. 순위표가 같은 파일을 사람 단위로 접은
      것이라, 둘을 나란히 두면 같은 줄을 두 번 읽게 된다.
    - **움직임 분석** 상자의 [다시 분석] 은 툴바에 이미 있었다 (세 번째
      사본이었다). 그 상자의 상태줄(stats_hint)은 Analysis 탭으로 옮겼다 --
      쓰는 쪽이 Analysis 인데 정작 Analysis 를 볼 때는 이 패널이 안 보였다.

    지운 것은 git 에 있다.
    """
    w = QWidget()
    col = QVBoxLayout(w)
    col.setContentsMargins(0, 0, 0, 0)

    # 수집 이력 -- 세션이 끝날 때마다 한 줄씩 파일에 쌓여서, "오늘 이 속도가
    # 평소만 한가"에 답한다. 정본은 <state_dir>/collection_history.jsonl
    # (collection_history.py).
    board = QGroupBox(tr("수집 속도 순위 (이 데이터셋)"))
    bcol = QVBoxLayout(board)
    win.board_hint = QLabel(tr("(이력 없음)"))
    win.board_hint.setStyleSheet("color:#888;")
    win.board_hint.setWordWrap(True)
    bcol.addWidget(win.board_hint)
    win.board_tree = QTreeWidget()
    win.board_tree.setHeaderLabels([tr("수집자"), tr("분당"), tr("저장"),
                                    tr("성공률"), tr("시간")])
    win.board_tree.setRootIsDecorated(False)
    win.board_tree.setToolTip(tr(
        "같은 데이터셋을 찍은 세션만 셉니다 — task 가 다르면 한 에피소드에 드는 "
        "시간도 달라서, 다른 데이터셋과 한 줄에 세우면 비교가 되지 않습니다.\n"
        "'분당' 은 자리에 앉아 있던 시간 기준입니다 (리셋·재배치 포함)."))
    cap_rows(win.board_tree, 12)
    bcol.addWidget(win.board_tree, 1)
    col.addWidget(board, 1)
    return w
