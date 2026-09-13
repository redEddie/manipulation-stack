from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from mstack.data.libero_format import hdf5_repack_status
from mstack.gui.dialogs.hf_account import HfAccountDialog, hf_account
from mstack.gui.dialogs.parts import Hdf5FileTable, RepoIdEdit
from mstack.gui.widgets import Recents
from mstack.gui.fonts import MONO_STACK
from mstack.gui.i18n import tr
from mstack.gui.text_utils import repo_id_error


class LerobotConvertDialog(QDialog):
    """Collects args for scripts/convert/convert_libero_to_lerobot.py before running
    it as a subprocess (see LiberoCollectorWindow._open_lerobot_convert).
    Curation (deleting bad takes) already happened in the HDF5 workflow --
    this dialog only picks which already-curated files to convert and where
    the result goes, mirroring the script's own CLI 1:1.
    """

    def __init__(self, parent: QWidget, default_root: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("LeRobot 변환 / 업로드"))
        layout = QVBoxLayout(self)

        # 모드가 맨 위에 온다. 아래 항목 중 무엇이 보이고 무엇이 쓰이는지를
        # 이 선택이 결정하므로, 다 읽은 뒤에 고르게 두면 순서가 거꾸로다.
        # Conversion and upload are separate jobs: conversion is minutes of
        # AV1 encoding, upload is seconds. Bundling them meant "I already
        # converted, just upload it" had no answer -- re-running re-encoded
        # everything. Two explicit modes instead of one ambiguous checkbox.
        mode_box = QGroupBox(tr("실행 모드"))
        mode_col = QVBoxLayout(mode_box)
        # Deliberately NO "convert and upload in one go": an episode pushed
        # to the Hub stays there even after it is deleted locally, so the
        # local result must be reviewed before anything is uploaded. A
        # combined mode exists only to skip that review.
        self.mode_convert = QRadioButton(tr("변환만 (로컬에 만들고 결과를 확인)"))
        self.mode_push_only = QRadioButton(
            tr("업로드만 (확인 끝난 '로컬 출력 경로'를 그대로 올림 -- 재변환 없음)")
        )
        self.mode_convert.setChecked(True)
        for b in (self.mode_convert, self.mode_push_only):
            mode_col.addWidget(b)
            b.toggled.connect(self._on_mode_changed)
        layout.addWidget(mode_box)

        # 파일은 **재압축·업로드와 같은 표**로 고른다 (2026-09-13). 셋이 하는
        # 일은 "어느 .hdf5 를 고를 것인가" 로 같은데 화면만 달랐다.
        self.files_label = QLabel(tr(
            "변환할 .hdf5 파일 (이미 큐레이션 끝난 파일). 데이터 경로의 것을 "
            "모두 보여 주고, **전부 체크된 채로** 시작합니다."))
        self.files_label.setWordWrap(True)
        layout.addWidget(self.files_label)
        self.table = Hdf5FileTable([tr("에피소드"), tr("이미지 압축")])
        layout.addWidget(self.table)
        self._fill_table(default_root)

        browse_row = QHBoxLayout()
        browse_row.addStretch()
        browse_files_btn = QPushButton(tr("다른 폴더에서 추가..."))
        browse_files_btn.clicked.connect(lambda: self._browse_files(default_root))
        browse_row.addWidget(browse_files_btn)
        layout.addLayout(browse_row)

        self._recents = Recents()

        grid = QGridLayout()
        # Each row shows a filled-in example next to the field. Parts that must
        # be replaced are written as **** so a copy-paste of the example alone
        # can never be mistaken for a working value.
        ex = QLabel(tr("예)  knu-physical-ai/****"))
        ex.setStyleSheet(f"color: #888; font-family: {MONO_STACK};")
        grid.addWidget(QLabel(tr("Repo ID:")), 0, 0)
        grid.addWidget(ex, 0, 3)
        # Editable combo, not a plain edit: the previous repo IDs are right
        # there in the dropdown, so a session that appends to an existing Hub
        # dataset never depends on retyping the ID exactly.
        # 줄바꿈되는 칸 (mstack/gui/dialogs/parts.RepoIdEdit) -- 한 줄짜리
        # 입력칸은 긴 id 의 앞뒤가 잘려서 어디로 올리는지가 안 보인다.
        self.repo_id_edit = RepoIdEdit(self._recents.get("repo_id"),
                                       tr("<org>/<dataset-name> 형식"))
        self.repo_id_edit.set_text(self._recents.most_recent("repo_id"))
        grid.addWidget(self.repo_id_edit, 0, 1, 1, 2)

        ex_root = QLabel(tr("예)  ~/lerobot_upload/****"))
        ex_root.setStyleSheet(f"color: #888; font-family: {MONO_STACK};")
        grid.addWidget(QLabel(tr("로컬 출력 경로:")), 1, 0)
        grid.addWidget(ex_root, 1, 3)
        self.out_root_edit = QComboBox()
        self.out_root_edit.setEditable(True)
        self.out_root_edit.addItems(self._recents.get("lerobot_root"))
        self.out_root_edit.setCurrentText(
            self._recents.most_recent("lerobot_root", str(Path.home() / "lerobot_upload"))
        )
        grid.addWidget(self.out_root_edit, 1, 1)
        browse_root_btn = QPushButton(tr("찾아보기..."))
        browse_root_btn.clicked.connect(self._browse_root)
        grid.addWidget(browse_root_btn, 1, 2)

        grid.addWidget(QLabel(tr("FPS:")), 2, 0)
        self.fps_edit = QLineEdit("20")
        grid.addWidget(self.fps_edit, 2, 1)
        layout.addLayout(grid)

        # Options live under the mode they belong to, so a mode's irrelevant
        # knobs are not just disabled-but-visible next to the ones that matter.
        self.convert_opts = QGroupBox(tr("변환 옵션"))
        conv_col = QVBoxLayout(self.convert_opts)
        self.only_success_check = QCheckBox(tr("성공(success=True) 에피소드만 포함 (--only-success)"))
        conv_col.addWidget(self.only_success_check)

        self.resume_check = QCheckBox(
            tr(
                "처음부터 새로 만들지 않고 기존 Hub 데이터셋에 이어붙이기 (--resume) -- 다른 task를 "
                "추가할 때, 기존 데이터는 재변환/재업로드하지 않음"
            )
        )
        conv_col.addWidget(self.resume_check)
        resume_warning = QLabel(
            tr(
                "⚠ 동시에 두 명이 같은 Repo ID로 --resume 변환하지 마세요 -- 같은 파일 경로를 서로 "
                "덮어써서 조용히 데이터가 유실될 수 있습니다."
            )
        )
        resume_warning.setStyleSheet("color: #e67e22;")
        resume_warning.setWordWrap(True)
        conv_col.addWidget(resume_warning)
        layout.addWidget(self.convert_opts)

        self.upload_opts = QGroupBox(tr("업로드 옵션"))
        up_col = QVBoxLayout(self.upload_opts)

        # Which account a --push actually uploads as. This machine is shared,
        # so "whose token is cached right now" is not something to assume.
        acct_text, acct_color = hf_account()
        acct_row = QHBoxLayout()
        self.hf_account_label = QLabel(acct_text)
        self.hf_account_label.setStyleSheet(f"color: {acct_color}; font-weight: bold;")
        self.hf_account_label.setWordWrap(True)
        acct_row.addWidget(self.hf_account_label, 1)
        switch_btn = QPushButton(tr("계정 전환..."))
        switch_btn.clicked.connect(self._open_account_dialog)
        acct_row.addWidget(switch_btn)
        up_col.addLayout(acct_row)

        self.private_check = QCheckBox(tr("비공개 데이터셋으로 업로드 (--private)"))
        up_col.addWidget(self.private_check)
        layout.addWidget(self.upload_opts)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_btn = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setText(tr("변환 시작"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        # The default mode's radio is already checked, so toggled never fires
        # for it -- apply the initial visibility explicitly.
        self._on_mode_changed()


    def _open_account_dialog(self) -> None:
        dlg = HfAccountDialog(self)
        dlg.exec()
        text, color = hf_account()
        self.hf_account_label.setText(text)
        self.hf_account_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _on_mode_changed(self) -> None:
        push_only = self.mode_push_only.isChecked()
        self.convert_opts.setVisible(not push_only)
        self.upload_opts.setVisible(push_only)
        # The .hdf5 picker and FPS belong to conversion; they sit above the
        # mode box (shared layout) so hide-by-disable rather than by removal.
        # .hdf5 표와 FPS 는 변환의 것이다. 업로드만 할 때는 끄되 지우지는
        # 않는다 -- 모드를 되돌렸을 때 고른 것이 남아 있어야 한다.
        for w in (self.table, self.fps_edit):
            w.setEnabled(not push_only)
        self.files_label.setEnabled(not push_only)
        if hasattr(self, "_ok_btn"):
            self._ok_btn.setText(tr("업로드 시작") if push_only else tr("변환 시작"))
        self.adjustSize()

    def _fill_table(self, default_root: str) -> None:
        """데이터 경로의 .hdf5 를 전부 싣고 **전부 체크**한다.

        변환은 파일의 상태와 무관하다 -- 재압축처럼 "이미 했나" 를 파일이
        말해 주지도 않고(변환 결과는 다른 폴더에 있다), 보통은 데이터셋 전체를
        한 번에 만든다. 그래서 기본이 전체 선택이고, 빼고 싶은 것만 푼다.
        """
        root = Path(default_root) if default_root else None
        if root is None or not root.is_dir():
            return
        for path in sorted(root.glob("*.hdf5")):
            self._add_path(path, checked=True)
        self.table.fit_columns()

    def _add_path(self, path, checked: bool = True) -> None:
        st = hdf5_repack_status(str(path))
        self.table.add_row(
            path, st["size"],
            extra=(st["episodes"], st["compression"] or ("?" if st["error"] else "없음")),
            checked=checked, disabled=bool(st["error"]))

    def _browse_files(self, default_root: str) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("변환할 .hdf5 파일"), default_root or str(Path.home()), "HDF5 (*.hdf5)"
        )
        for p in paths:
            if not self.table.has(p):
                self._add_path(Path(p), checked=True)
        self.table.fit_columns()

    def _browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, tr("로컬 출력 경로"), self.out_root_edit.currentText()
        )
        if d:
            self.out_root_edit.setCurrentText(d)

    def build_args(self) -> "list[str] | None":
        """Returns the script's argv (sans program name), or None (with a
        warning dialog already shown) if required fields are missing."""
        push_only = self.mode_push_only.isChecked()
        paths = self.table.checked_paths()
        if not paths and not push_only:
            QMessageBox.warning(self, tr("파일 필요"), tr(".hdf5 파일을 하나 이상 선택하세요."))
            return None
        repo_id = self.repo_id_edit.text()
        err = repo_id_error(repo_id)
        if err:
            QMessageBox.warning(self, tr("Repo ID 오류"), tr(err))
            return None
        out_root = self.out_root_edit.currentText().strip()
        if not out_root:
            QMessageBox.warning(self, tr("출력 경로 필요"), tr("로컬 출력 경로를 입력하세요."))
            return None
        # Only remember values that made it past validation.
        self._recents.add("repo_id", repo_id)
        self._recents.add("lerobot_root", out_root)
        if push_only:
            return ["--repo-id", repo_id, "--root", out_root, "--push-only",
                    "--private" if self.private_check.isChecked() else "--no-private"]
        args = list(paths) + [
            "--repo-id", repo_id,
            "--root", out_root,
            "--fps", self.fps_edit.text().strip() or "20",
        ]
        if self.only_success_check.isChecked():
            args.append("--only-success")
        if self.resume_check.isChecked():
            args.append("--resume")
        return args
