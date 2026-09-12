"""오래 걸리는 일이 화면에 말하는 두 가지 -- 연결 대기와 진행 줄.

통계가 아니라 **로그·상태바의 규약**이라 StatsOps 에 있을 이유가 없었다
(kimi 구조 감사 2026-09-12). 부르는 쪽도 수집(카메라 정리 대기)과 창
(하위 프로세스 출력) 으로 서로 다르다. shared 에 두는 규칙은 set_busy 와 같다.
"""

from __future__ import annotations

from PyQt6.QtGui import QTextCursor

from mstack.gui.i18n import tr


def connect_progress(win, waited: float) -> None:
    win.statusBar().showMessage(
        tr("카메라 정리 중... {s:.0f}초 (정리되면 자동으로 연결합니다)").format(s=waited),
        1000)
    win.lights["camera"].set("busy", tr("정리 중"))


def log_progress(win, msg: str, view: str) -> None:
    """Progress that overwrites its own last line instead of stacking.

    A 1.3 GB upload prints a bar every second; appended, that buries every
    other message in the tab and makes the log useless exactly while a long
    job is running. Replacing the previous progress line keeps one live line
    and leaves the surrounding log readable.

    Deliberately not written to the log file -- the file is what gets read
    after a crash, and hundreds of superseded percentages help nobody there.
    """
    target = win._view(view)
    if win._progress_line.get(view):
        cursor = target.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.select(QTextCursor.SelectionType.LineUnderCursor)
        cursor.removeSelectedText()
        cursor.insertText(msg)
    else:
        target.appendPlainText(msg)
        win._progress_line[view] = True
