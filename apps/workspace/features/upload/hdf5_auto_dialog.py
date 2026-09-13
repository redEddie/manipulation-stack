"""재압축 + 업로드(자동)도 **고르는 화면**을 갖는다.

이 버튼만 체크 목록이 없었다. 대신 확인창 하나에 파일 이름을 글로 늘어놓고
"변경된 파일만 / 전체 강제" 체크박스 하나로 전부 아니면 전무를 골랐다 --
가운데가 없었다 (조작자, 2026-09-13: *"재압축+업로드(자동)에서는 체크목록이
없는 것 같은데요. ui가 통일된 게 맞을까요?"*).

이제 셋(재압축·업로드·변환)과 같은 표를 쓴다. 다른 점은 **한 줄이 두 가지
일을 말한다**는 것뿐이다:

* 재압축 열 -- 이 파일이 재압축이 필요한가 (파일 자신이 답한다)
* 업로드 열 -- 업로드 장부가 올리라고 하는가, 왜

체크는 "이 파일을 이번에 처리한다" 하나다. 둘 중 하나라도 해당되면 미리
체크된다. 아래 두 스위치로 **동작**을 끄고 켠다 -- 재압축만 하거나 업로드만
할 수 있다. 옛 "전체 강제 업로드" 는 따로 두지 않는다: 장부가 조용한 파일도
직접 체크하면 올라간다. 그것이 곧 강제다.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from mstack.data.hub_upload_state import upload_reason
from mstack.data.libero_format import hdf5_repack_status
from mstack.gui.dialogs.parts import Hdf5FileTable
from mstack.gui.i18n import tr


class Hdf5AutoDialog(QDialog):
    """재압축과 업로드를 한 번에 -- 무엇을, 어디까지."""

    def __init__(self, parent: QWidget, paths: list, repo_id: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("HDF5 재압축 + 업로드"))
        self.repo_id = repo_id
        layout = QVBoxLayout(self)
        head = QLabel(tr(
            "처리할 파일을 고르세요. **재압축이 필요하거나 업로드 장부가 "
            "바뀌었다고 말하는 것**이 미리 체크됩니다.\n업로드 대상: {r}")
            .format(r=repo_id))
        head.setWordWrap(True)
        layout.addWidget(head)

        self.table = Hdf5FileTable([tr("에피소드"), tr("재압축"), tr("업로드")])
        self._repack_todo = set()
        for path in paths:
            st = hdf5_repack_status(str(path))
            todo = not st["repacked"] and not st["error"]
            if todo:
                self._repack_todo.add(str(path))
            try:
                reason = upload_reason(repo_id, Path(path)) or ""
            except Exception:  # noqa: BLE001 -- 장부를 못 읽어도 고르기는 된다
                reason = ""
            self.table.add_row(
                path, st["size"],
                extra=(st["episodes"],
                       tr("필요") if todo else tr("완료"),
                       reason or tr("변경 없음")),
                checked=bool(todo or reason),
                disabled=bool(st["error"]))
        self.table.fit_columns()
        layout.addWidget(self.table)

        # 동작 스위치. 파일 선택과 **따로** 둔다 -- "이 파일들을" 과 "무엇을"
        # 은 다른 질문이라, 한 체크박스에 섞으면 둘 다 흐려진다.
        self.repack_check = QCheckBox(tr("재압축 실행 (고른 것 중 '필요' 인 것만)"))
        self.repack_check.setChecked(True)
        layout.addWidget(self.repack_check)
        self.upload_check = QCheckBox(tr("업로드 실행 (고른 것 전부)"))
        self.upload_check.setChecked(True)
        self.upload_check.setToolTip(tr(
            "장부가 '변경 없음' 이라고 한 파일도 체크했다면 올라갑니다 — "
            "그것이 강제 업로드입니다."))
        layout.addWidget(self.upload_check)

        note = QLabel(tr(
            "재압축은 원본을 검증 후 교체하고, 업로드는 Hub 에 원본 .hdf5 를 "
            "그대로 올립니다. 재압축이 파일을 바꾸므로 같은 실행에서 올리는 것이 "
            "맞습니다 — 순서는 재압축 → 업로드로 고정입니다."))
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("시작"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ 결과
    def repack_paths(self) -> list:
        """고른 것 중 **재압축이 필요한 것**만. 이미 된 파일을 다시 굽는 것은
        순수한 낭비다 (GB 당 몇 분인데 두 번째는 얻는 것이 없다)."""
        if not self.repack_check.isChecked():
            return []
        return [p for p in self.table.checked_paths() if p in self._repack_todo]

    def upload_paths(self) -> list:
        if not self.upload_check.isChecked():
            return []
        return list(self.table.checked_paths())
