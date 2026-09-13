"""업로드 계열 대화상자가 **같은 모양**을 갖게 하는 조각들.

셋(재압축·HDF5 업로드·LeRobot 변환)은 하는 일이 같다: **어느 .hdf5 를 고를
것인가**를 정하고, 그 다음에 그 동작의 옵션을 정한다. 그런데 화면은 셋이
따로 자랐다 -- 재압축만 체크 목록이고 나머지 둘은 경로를 공백으로 이어 붙인
한 줄 입력칸이었다. 조작자가 2026-09-13 에 지적했다: *"각각 다른 팝업이 떠서
동일한 작업을 한다는 느낌을 전혀 주지 못해요. 가장 마음에 드는 인터페이스는
용량 최적화(재압축)입니다."*

그래서 그 대화상자의 표를 여기로 꺼내 셋이 함께 쓴다. 상태 열(왜 이 파일이
대상인가)만 동작마다 다르다:

* 재압축 -- "빈 공간 320 MB (12%) — 다시 필요"
* 업로드 -- "신규 — 업로드 기록 없음" / "변경 — 09/12 14:02 수정"
* 변환   -- 에피소드 수 (변환은 파일 상태와 무관하게 전부가 대상이다)

**기본 체크가 요점이다.** 어느 파일이 지금 대상인지 조작자가 기억하고 있을
필요가 없다 -- 파일 자신이 답을 갖고 있고(재압축 이력·업로드 장부), 그것이
아는 만큼만 미리 체크된다.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mstack.gui.fonts import MONO_STACK
from mstack.gui.i18n import tr


class Hdf5FileTable(QWidget):
    """체크 가능한 .hdf5 목록 + [전체 선택]/[전체 해제] + 고른 개수.

    기본 열은 **파일 · 크기**이고, 나머지는 부르는 쪽이 정한다 (``extra``).
    """

    def __init__(self, extra_headers=(), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        self._extra = list(extra_headers)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2 + len(self._extra))
        self.tree.setHeaderLabels([tr("파일"), tr("크기"), *self._extra])
        self.tree.setRootIsDecorated(False)
        self.tree.setMinimumSize(760, 240)
        self.tree.itemChanged.connect(lambda *_: self._refresh_count())
        col.addWidget(self.tree)

        row = QHBoxLayout()
        all_btn = QPushButton(tr("전체 선택"))
        all_btn.clicked.connect(lambda: self.set_all(True))
        none_btn = QPushButton(tr("전체 해제"))
        none_btn.clicked.connect(lambda: self.set_all(False))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color:#888;")
        row.addWidget(self.count_label)
        col.addLayout(row)
        self._rows: list = []

    # ------------------------------------------------------------------ 채우기
    def add_row(self, path, size_bytes: float, extra=(), checked: bool = False,
                disabled: bool = False, tint=None) -> None:
        item = QTreeWidgetItem([
            Path(path).name,
            tr("{mb:,.1f} MB").format(mb=size_bytes / 1e6),
            *[str(x) for x in extra],
        ])
        item.setToolTip(0, str(path))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Checked if checked and not disabled
                           else Qt.CheckState.Unchecked)
        if disabled:
            item.setDisabled(True)
        elif tint is not None:
            for c in range(self.tree.columnCount()):
                item.setForeground(c, tint)
        self.tree.addTopLevelItem(item)
        self._rows.append((str(path), item))
        self._refresh_count()

    def has(self, path) -> bool:
        return any(p == str(path) for p, _ in self._rows)

    def fit_columns(self) -> None:
        for c in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(c)

    # ------------------------------------------------------------------ 읽기
    def checked_paths(self) -> list:
        return [p for p, it in self._rows
                if it.checkState(0) == Qt.CheckState.Checked and not it.isDisabled()]

    def set_all(self, on: bool) -> None:
        for _p, it in self._rows:
            if not it.isDisabled():
                it.setCheckState(0, Qt.CheckState.Checked if on
                                 else Qt.CheckState.Unchecked)

    def _refresh_count(self) -> None:
        n = len(self.checked_paths())
        total = sum(1 for _p, it in self._rows if not it.isDisabled())
        self.count_label.setText(tr("{n} / {t}개 선택").format(n=n, t=total))


class RepoIdEdit(QWidget):
    """Repo ID 입력칸 -- **줄바꿈된다.**

    한 줄짜리 입력칸은 `knu-physical-ai/fr3-tabletop-...` 가 칸보다 길어지면
    앞이나 뒤가 잘려서, 정작 어디로 올리는지를 한눈에 못 본다 (조작자,
    2026-09-13). 여기서는 여러 줄로 접어 **전부 보여 준다**. 붙여넣기에 섞인
    줄바꿈·공백은 읽을 때 떼어낸다 -- 값은 언제나 한 줄이다.

    최근 값은 [최근 ▾] 메뉴로 남긴다. 편집 가능한 콤보를 쓰면 줄바꿈을 할 수
    없어서(콤보의 편집칸은 QLineEdit 고정) 둘을 갈랐다.
    """

    #: 값이 **정해졌을 때** 알린다 (칸을 떠나거나 최근에서 골랐을 때).
    #: textChanged 를 쓰지 않는 이유: 이 값이 바뀌면 듣는 쪽이 파일을 전부
    #: 다시 훑는데(업로드 장부 조회), 글자 하나마다 그러면 27개 파일을
    #: 매 키 입력마다 여는 셈이다.
    committed = pyqtSignal(str)

    #: 처음 높이(줄). 내용에 따라 이 아래위로 움직인다.
    LINES = 2
    #: 여기까지만 늘어나고, 그보다 길면 스크롤을 켠다. 끝없이 늘면 아래
    #: 항목들이 화면 밖으로 밀린다.
    MAX_LINES = 4

    def __init__(self, recents: list, placeholder: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QPlainTextEdit()
        self.edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setStyleSheet(f"font-family: {MONO_STACK};")
        self.edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # **접힌 만큼 키운다.** 두 줄로 고정해 두었더니 좁은 패널에서 세 줄로
        # 접히는 id 의 마지막 줄이 잘렸다 -- 스크롤 막대를 꺼 둔 터라 잘린
        # 것이 보이지도 않았다. "한눈에 보이도록" 이 이 칸의 존재 이유다.
        self.edit.textChanged.connect(self._fit_height)
        self.edit.resizeEvent = self._resized
        self._fit_height()
        self.edit.focusOutEvent = self._focus_out
        row.addWidget(self.edit, 1)
        self._recents = list(recents)
        self.recent_btn = QPushButton(tr("최근 ▾"))
        self.recent_btn.setEnabled(bool(self._recents))
        self.recent_btn.clicked.connect(self._show_recents)
        row.addWidget(self.recent_btn)

    def _resized(self, event) -> None:
        QPlainTextEdit.resizeEvent(self.edit, event)
        self._fit_height()          # 패널 폭이 바뀌면 접히는 줄 수도 바뀐다

    def _lines(self) -> int:
        """지금 **화면에 그려지는** 줄 수. 문단 수가 아니다 -- 줄바꿈 문자가
        없어도 폭이 모자라면 여러 줄로 그려진다."""
        doc = self.edit.document()
        n = 0
        block = doc.firstBlock()
        while block.isValid():
            layout = block.layout()
            n += max(1, layout.lineCount() if layout is not None else 1)
            block = block.next()
        return max(1, n)

    def _fit_height(self) -> None:
        try:
            fm = self.edit.fontMetrics()
            lines = self._lines()
        except RuntimeError:        # 창이 닫히는 중
            return
        shown = min(self.MAX_LINES, lines)
        self.edit.setFixedHeight(fm.lineSpacing() * shown + 12)
        self.edit.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded if lines > self.MAX_LINES
            else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def _focus_out(self, event) -> None:
        QPlainTextEdit.focusOutEvent(self.edit, event)
        # 창이 닫히는 중에도 포커스가 빠진다. 그때는 이 위젯의 C++ 쪽이 이미
        # 없어서 emit 이 RuntimeError 로 터지고, 그것이 종료 순간의 크래시
        # 대화상자가 된다 (저장소에 같은 부류의 전례가 있다 -- _proc_text 의
        # sip.isdeleted 참고). 여기서는 값만 읽어 보고 조용히 넘긴다.
        try:
            value = self.text()
        except RuntimeError:
            return
        try:
            self.committed.emit(value)
        except RuntimeError:
            pass

    def _show_recents(self) -> None:
        menu = QMenu(self)
        for value in self._recents:
            act = QAction(value, menu)
            act.triggered.connect(lambda _c=False, v=value: self.set_text(v))
            menu.addAction(act)
        menu.exec(self.recent_btn.mapToGlobal(self.recent_btn.rect().bottomLeft()))

    def text(self) -> str:
        """언제나 한 줄. 공백은 **전부 없앤다** (빈칸으로 잇지 않는다).

        repo id 에는 공백이 들어갈 수 없다. 이 칸은 줄바꿈되므로 조작자가
        접힌 값을 그대로 복사해 오면 중간에 줄바꿈이 섞여 들어오는데, 그것을
        빈칸으로 이으면 `org/ name` 같은 값이 되어 검증에서 튕긴다 -- 사람이
        보기엔 맞는 값인데.
        """
        return "".join(self.edit.toPlainText().split())

    def set_text(self, value: str) -> None:
        self.edit.setPlainText(value or "")
        self.committed.emit(self.text())
