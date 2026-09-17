"""프록시 클립 만들기 대화상자.

큐레이션 그리드는 원본 .hdf5 로는 못 돈다 -- 에피소드 하나가 480x640 카메라
2대로 280 MB 이고, gzip 청크가 19프레임 깊이라 솎아 읽기가 통째 읽기보다
14배 느리다. 그래서 미리 작은 mp4 를 구워 둔다 (``mstack.data.proxy_clip``).

이 대화상자가 하는 일은 **무엇을 구울지 고르는 것**뿐이다. 파일마다 "몇 개가
빠졌나" 를 세어 보여주고, 다른 작업이 쥐고 있는 파일은 잠근다 -- 업로드가
읽는 중인 .hdf5 를 같이 열면 그쪽을 방해한다 (2026-09-10 에 실제로 업로드가
scene_021 을 열고 있었다).

이미 있는 클립은 건너뛴다. 중단해도 구운 것은 남고, 다시 열면 남은 것만
목록에 나온다.

Each file expands into its instructions, so one instruction can be baked
without the rest of the scene (2026-09-17). The header's checkbox selects or
clears everything and shows a partial selection as [-].
"""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from mstack.data.proxy_clip import (
    DEFAULT_CRF,
    DEFAULT_SCALE,
    PROXY_DIR,
    scan_file,
)
from apps.workspace.shared import jobs
from mstack.gui.i18n import tr
from mstack.gui.workers import ProxyBuildWorker

#: 스케일을 %로 고른다. 50% 면 640x480 -> 320x240.
_SCALE_MIN, _SCALE_MAX = 20, 100

#: 실측 근거 (2026-09-10, 157프레임 320x240 카메라 1대):
#: crf 23 -> 0.075 MB, 28 -> 0.043 MB, 32 -> 0.030 MB.
_CRF_MIN, _CRF_MAX = 18, 40

#: 클립 하나의 대략 용량 (MB). scene 0~4 실측 480클립 25.4 MB 에서 나온 값으로,
#: 목록의 "예상" 칸에만 쓴다 -- 맞히는 것이 목적이 아니라 자릿수를 보여주는
#: 것이 목적이다.
_MB_PER_CLIP = 25.4 / 480

#: 클립 하나당 대략 소요 (초). 같은 실측에서 10.2분 / 480클립.
_SEC_PER_CLIP = 10.2 * 60 / 480

#: Item data roles: episode group names of an instruction row, and its clip count.
_ROLE_NAMES = Qt.ItemDataRole.UserRole + 1
_ROLE_CLIPS = Qt.ItemDataRole.UserRole + 2

_STYLE_STATE = {
    Qt.CheckState.Checked: QStyle.StateFlag.State_On,
    Qt.CheckState.Unchecked: QStyle.StateFlag.State_Off,
    Qt.CheckState.PartiallyChecked: QStyle.StateFlag.State_NoChange,
}


class CheckHeader(QHeaderView):
    """Horizontal header whose first section carries a tri-state checkbox.

    Qt has no header checkbox of its own, so the indicator is drawn with the
    platform style (State_NoChange renders as [-]) and a click inside the first
    section emits ``toggled``. The owner decides what the state is.
    """

    toggled = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._state = Qt.CheckState.Unchecked
        self.setSectionsClickable(True)

    def box_rect(self, section_rect: QRect) -> QRect:
        opt = QStyleOptionButton()
        r = self.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator, opt, self)
        return QRect(section_rect.x() + 6,
                     section_rect.y() + (section_rect.height() - r.height()) // 2,
                     r.width(), r.height())

    def label_indent(self) -> str:
        """Leading spaces that keep the section label clear of the box."""
        width = self.box_rect(QRect(0, 0, 0, 20)).width() + 12
        space = max(1, self.fontMetrics().horizontalAdvance(" "))
        return " " * (-(-width // space))

    def check_state(self) -> Qt.CheckState:
        return self._state

    def set_check_state(self, state: Qt.CheckState) -> None:
        if state != self._state:
            self._state = state
            self.viewport().update()

    def paintSection(self, painter, rect, index) -> None:  # noqa: N802 - Qt override
        painter.save()
        super().paintSection(painter, rect, index)
        painter.restore()
        if index != 0:
            return
        opt = QStyleOptionButton()
        opt.rect = self.box_rect(rect)
        opt.state = QStyle.StateFlag.State_Enabled | _STYLE_STATE[self._state]
        self.style().drawPrimitive(
            QStyle.PrimitiveElement.PE_IndicatorCheckBox, opt, painter, self)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.isEnabled() and self.logicalIndexAt(event.position().toPoint()) == 0:
            # Partial or cleared -> select all; all selected -> clear.
            self.toggled.emit(self._state != Qt.CheckState.Checked)
            return
        super().mousePressEvent(event)


class ProxyBuildDialog(QDialog):
    def __init__(self, win, paths, busy_label: str = "", proxy_dir=None) -> None:
        super().__init__(win)
        self.win = win
        # 데이터셋마다 캐시 자리가 다르다 (proxy_clip.dataset_tag).
        self.proxy_dir = proxy_dir
        self.worker = None
        self.setWindowTitle(tr("프록시 클립 만들기"))
        self.resize(720, 520)
        col = QVBoxLayout(self)

        col.addWidget(QLabel(tr(
            "큐레이션 그리드가 쓸 작은 mp4 를 미리 구워 둡니다.\n"
            "원본은 읽기만 하고, 이미 있는 클립은 건너뜁니다.")))

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.header = CheckHeader(self.tree)
        self.tree.setHeader(self.header)
        self.tree.setHeaderLabels(
            [self.header.label_indent() + tr("파일 · 지시문"), tr("만들 클립"),
             tr("예상 용량"), tr("예상 시간")])
        self.tree.setColumnWidth(0, 360)
        self.header.toggled.connect(self.set_all_checked)
        self.tree.itemChanged.connect(self._on_item_changed)
        col.addWidget(self.tree, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("해상도")))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(_SCALE_MIN, _SCALE_MAX)
        self.scale_spin.setValue(int(DEFAULT_SCALE * 100))
        self.scale_spin.setSuffix(" %")
        self.scale_spin.setToolTip(tr(
            "원본 긴 변에 대한 비율. 50% 면 640x480 이 320x240 이 됩니다.\n"
            "3x4 그리드에서 타일이 250px 안팎이라 그보다 크면 버려집니다."))
        row.addWidget(self.scale_spin)
        row.addSpacing(16)
        row.addWidget(QLabel(tr("화질 (CRF)")))
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(_CRF_MIN, _CRF_MAX)
        self.crf_spin.setValue(DEFAULT_CRF)
        self.crf_spin.setToolTip(tr(
            "낮을수록 좋고 큽니다. 실측(157프레임 320x240 카메라 1대):\n"
            "  23 -> 0.075 MB    28 -> 0.043 MB    32 -> 0.030 MB"))
        row.addWidget(self.crf_spin)
        row.addStretch(1)
        col.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setTextVisible(True)
        col.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#888;")
        self.status.setWordWrap(True)
        col.addWidget(self.status)

        btns = QDialogButtonBox()
        self.start_btn = QPushButton(tr("만들기 시작"))
        self.start_btn.clicked.connect(self.on_start)
        btns.addButton(self.start_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        self.close_btn = QPushButton(tr("닫기"))
        self.close_btn.clicked.connect(self.reject)
        btns.addButton(self.close_btn, QDialogButtonBox.ButtonRole.RejectRole)
        col.addWidget(btns)

        # 목록은 **위젯을 다 만든 뒤에** 채운다 -- _fill 이 진행바와 상태줄을
        # 건드리므로 먼저 부르면 AttributeError 다.
        self._fill(paths, busy_label)

    # ------------------------------------------------------------------ 목록
    def _fill(self, paths, busy_label: str) -> None:
        """파일마다 빠진 클립 수를 세어 넣는다. 못 여는 파일은 잠근다.

        Files with missing clips get one child per instruction; the file row
        is auto-tristate, so checking it checks its instructions and a partly
        checked file shows [-].
        """
        self.tree.blockSignals(True)
        for p in paths:
            p = Path(p)
            it = QTreeWidgetItem(self.tree)
            it.setText(0, p.name)
            it.setData(0, Qt.ItemDataRole.UserRole, str(p))
            groups: dict = {}
            try:
                sc = scan_file(p, self.proxy_dir or PROXY_DIR)
                n = len(sc["missing"])
                # 0 의 뜻이 둘이다 -- 다 만들었거나, 파일이 비었거나.
                note = "" if n else (tr("이미 있음") if sc["episodes"]
                                     else tr("에피소드 없음"))
                for name, _uid, _cam in sc["missing"]:
                    key = sc["slots"].get(name, ("", ""))
                    g = groups.setdefault(key, {"names": set(), "clips": 0})
                    g["names"].add(name)
                    g["clips"] += 1
            except Exception as e:  # noqa: BLE001
                n, note = 0, f"{type(e).__name__}"
            if busy_label and n:
                # 다른 작업이 .hdf5 를 쥐고 있을 수 있다. 어느 파일인지까지는
                # 알 수 없으므로 전부 잠근다 -- 남의 작업을 방해하는 것보다
                # 기다리는 쪽이 싸다.
                note = tr("{job} 중").format(job=busy_label)
                n = 0
            self._set_counts(it, n, note)
            if not n:
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(0, Qt.CheckState.Unchecked)
                it.setDisabled(True)
                continue
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable
                        | Qt.ItemFlag.ItemIsAutoTristate)
            for (iid, text), g in sorted(groups.items()):
                child = QTreeWidgetItem(it)
                label = f"{iid}  {text}".strip() or tr("(지시문 없음)")
                child.setText(0, label)
                child.setToolTip(0, label)
                child.setData(0, _ROLE_NAMES, sorted(g["names"]))
                child.setData(0, _ROLE_CLIPS, g["clips"])
                self._set_counts(child, g["clips"], "")
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked)
        self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.tree.blockSignals(False)
        self._refresh_selection()

    @staticmethod
    def _set_counts(it, n: int, note: str) -> None:
        it.setText(1, str(n) if n else note)
        it.setText(2, f"{n * _MB_PER_CLIP:.1f} MB" if n else "-")
        it.setText(3, f"{n * _SEC_PER_CLIP / 60:.1f} 분" if n else "-")

    def _file_items(self) -> list:
        """Top-level rows that can be baked (disabled ones are done or locked)."""
        return [self.tree.topLevelItem(i)
                for i in range(self.tree.topLevelItemCount())
                if not self.tree.topLevelItem(i).isDisabled()]

    def set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.tree.blockSignals(True)
        for it in self._file_items():
            it.setCheckState(0, state)
            for j in range(it.childCount()):
                it.child(j).setCheckState(0, state)
        self.tree.blockSignals(False)
        self._refresh_selection()

    def _on_item_changed(self, _item, column: int) -> None:
        if column == 0:
            self._refresh_selection()

    def _refresh_selection(self) -> None:
        """Header box and the status line follow the checked instructions."""
        states = [it.checkState(0) for it in self._file_items()]
        if states and all(s == Qt.CheckState.Checked for s in states):
            head = Qt.CheckState.Checked
        elif all(s == Qt.CheckState.Unchecked for s in states):
            head = Qt.CheckState.Unchecked
        else:
            head = Qt.CheckState.PartiallyChecked
        self.header.set_check_state(head)
        self.header.setEnabled(bool(states))
        total = sum(n for _p, _names, n in self._checked())
        self.bar.setRange(0, max(total, 1))
        self.status.setText(
            tr("만들 클립 {n}개, 예상 {mb:.0f} MB · {min:.0f}분").format(
                n=total, mb=total * _MB_PER_CLIP,
                min=total * _SEC_PER_CLIP / 60) if total
            else tr("고른 것이 없습니다.") if states
            else tr("만들 것이 없습니다."))
        self._total = total

    def _checked(self) -> list:
        """[(path, group names, clip count)] for every checked instruction."""
        out = []
        for it in self._file_items():
            for j in range(it.childCount()):
                c = it.child(j)
                if c.checkState(0) == Qt.CheckState.Checked:
                    out.append((it.data(0, Qt.ItemDataRole.UserRole),
                                c.data(0, _ROLE_NAMES), c.data(0, _ROLE_CLIPS)))
        return out

    def _checked_selection(self) -> dict:
        """{path: set of episode group names} to bake."""
        sel: dict = {}
        for path, names, _n in self._checked():
            sel.setdefault(path, set()).update(names)
        return sel

    # ------------------------------------------------------------------ 실행
    def on_start(self) -> None:
        if self.worker is not None:      # 돌고 있으면 이 버튼이 '중단'이다
            self.worker.stop()
            self.status.setText(tr("중단하는 중… (지금 굽는 클립까지는 끝냅니다)"))
            return
        selection = self._checked_selection()
        paths = sorted(selection)
        if not paths:
            self.status.setText(tr("고른 것이 없습니다."))
            return
        self.tree.setEnabled(False)
        self.header.setEnabled(False)
        self.scale_spin.setEnabled(False)
        self.crf_spin.setEnabled(False)
        self.start_btn.setText(tr("중단"))
        self.worker = ProxyBuildWorker(
            paths, self.scale_spin.value() / 100.0, self.crf_spin.value(),
            self.proxy_dir or PROXY_DIR, self, episodes=selection)
        self.worker.progress.connect(self.on_progress)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.done.connect(self.on_done)
        self._t0 = time.monotonic()
        jobs.start_job(self.win, tr("프록시 굽기"))
        self.worker.start()

    def on_progress(self, done: int, total: int, label: str) -> None:
        self.bar.setRange(0, max(total, 1))
        self.bar.setValue(done)
        # 남은 시간은 **실측 속도**로 낸다. 위의 예상치는 상수(_SEC_PER_CLIP)라
        # 기계와 해상도에 따라 배로 틀리는데, 몇 개만 구우면 이 기계의 진짜
        # 속도를 알 수 있다 (2026-09-13).
        left = ""
        if done >= 3 and total > done:
            per = (time.monotonic() - self._t0) / done
            left = tr("  · 남은 약 {m:.0f}분").format(m=per * (total - done) / 60)
        self.status.setText(f"{done}/{total}  {label}{left}")
        jobs.job_progress(self.win, done / total if total else None)

    def on_file_done(self, r: dict) -> None:
        name = Path(r["path"]).name
        msg = tr("[프록시] {f}: {n}개 ({mb:.1f} MB)").format(
            f=name, n=r["made"], mb=r["bytes"] / 1e6)
        if r.get("failed"):
            msg += tr("  실패 {k}개 — {e}").format(
                k=r["failed"], e=r.get("error") or "?")
        self.win.log(msg)

    def on_done(self, r: dict) -> None:
        self.worker = None
        jobs.end_job(self.win)
        self.tree.setEnabled(True)
        self.scale_spin.setEnabled(True)
        self.crf_spin.setEnabled(True)
        self.start_btn.setText(tr("만들기 시작"))
        head = tr("중단했습니다.") if r["stopped"] else tr("끝났습니다.")
        self.status.setText(tr(
            "{head} 만든 클립 {n}개 ({mb:.1f} MB), 실패 {k}개.").format(
                head=head, n=r["made"], mb=r["bytes"] / 1e6, k=r["failed"]))
        self.win.log(tr("[프록시] {head} 만든 클립 {n}개 ({mb:.1f} MB), 실패 {k}개")
                     .format(head=head, n=r["made"], mb=r["bytes"] / 1e6,
                             k=r["failed"]))
        # 남은 것을 다시 세어 목록을 갱신한다 -- 중단 후 이어 만들 때
        # "몇 개 남았나" 가 맞아야 한다.
        paths = [self.tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
                 for i in range(self.tree.topLevelItemCount())]
        self.tree.clear()
        self._fill(paths, "")

    def reject(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None
        super().reject()
