from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mstack.data.hub_upload_state import upload_reason
from mstack.data.libero_format import hdf5_repack_status
from mstack.gui.dialogs.hf_account import HfAccountDialog, hf_account
from mstack.gui.dialogs.parts import Hdf5FileTable, RepoIdEdit
from mstack.gui.widgets import Recents
from mstack.gui.fonts import MONO_STACK
from mstack.gui.i18n import tr
from mstack.gui.text_utils import repo_id_error


class HdfUploadDialog(QDialog):
    """Collects args for scripts/convert/upload_to_hub.py before running it as a
    subprocess (see LiberoCollectorWindow._open_hdf5_upload). Uploads the
    RAW curated .hdf5 as-is -- the converted/LeRobot half of the dual
    upload (see ~/huggingface_upload_process.md) is the separate "LeRobot
    변환..." button's --push option.
    """

    def __init__(self, parent: QWidget, start_dir: str = "") -> None:
        """``start_dir`` is where 찾아보기 opens -- NOT a preselected file.

        It used to be a "default file" that callers filled with the data-root
        *directory*. The field then held a directory, and 'Repo 안 파일 이름'
        derived from it, so an upload went to the Hub under the folder's name
        (`libero_datasets`, 0.81 GB). Worse, that name later collided with the
        folder a multi-file upload wanted to create, and the Hub rejected the
        commit with 'Invalid file change'. Starting empty removes the whole
        class: nothing is ever uploaded under a name the operator didn't pick.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("HDF5 원본 업로드"))
        self._start_dir = start_dir
        layout = QVBoxLayout(self)

        # 파일은 **재압축과 같은 표**로 고른다 (2026-09-13). 예전에는 경로를
        # 공백으로 이어 붙인 한 줄 입력칸이라, 무엇이 골라졌는지도 그 파일이
        # 올릴 만한 것인지도 화면에 없었다.
        layout.addWidget(QLabel(tr(
            "업로드할 .hdf5 파일을 선택하세요. **업로드 장부에 견주어** 올릴 "
            "필요가 있는 것만 미리 체크됩니다.")))
        self._recents = Recents()
        self.table = Hdf5FileTable([tr("에피소드"), tr("업로드 상태")])
        layout.addWidget(self.table)
        self._fill_table(start_dir)

        browse_row = QHBoxLayout()
        browse_row.addStretch()
        browse_btn = QPushButton(tr("다른 폴더에서 추가..."))
        browse_btn.setToolTip(tr("목록에 없는 .hdf5 를 더한다 (체크된 채로 들어온다)"))
        browse_btn.clicked.connect(self._browse_file)
        browse_row.addWidget(browse_btn)
        layout.addLayout(browse_row)

        grid = QGridLayout()
        # Each row shows a filled-in example next to the field. Parts that must
        # be replaced are written as **** so a copy-paste of the example alone
        # can never be mistaken for a working value.
        ex = QLabel(tr("예)  knu-physical-ai/****"))
        ex.setStyleSheet(f"color: #888; font-family: {MONO_STACK};")
        grid.addWidget(QLabel(tr("Repo ID:")), 0, 0)
        grid.addWidget(ex, 0, 3)
        # 줄바꿈되는 칸이다 -- 한 줄짜리 입력칸은 긴 id 의 앞뒤가 잘려서
        # 어디로 올리는지를 한눈에 못 본다 (조작자, 2026-09-13).
        self.repo_id_edit = RepoIdEdit(self._recents.get("hdf5_repo_id"),
                                       tr("<org>/<dataset-name> 형식"))
        self.repo_id_edit.set_text(self._recents.most_recent("hdf5_repo_id"))
        # 값이 **정해졌을 때만** 다시 훑는다 (칸을 떠나거나 최근에서 고를 때).
        # 글자마다 훑으면 파일 27개를 키 입력마다 연다.
        self.repo_id_edit.committed.connect(self._refresh_upload_status)
        grid.addWidget(self.repo_id_edit, 0, 1, 1, 2)

        ex_name = QLabel(tr("예)  ****_demo.hdf5"))
        ex_name.setStyleSheet(f"color: #888; font-family: {MONO_STACK};")
        self.path_in_repo_label = QLabel(tr("Repo 안 파일 이름:"))
        grid.addWidget(self.path_in_repo_label, 1, 0)
        grid.addWidget(ex_name, 1, 2)
        self.path_in_repo_edit = QLineEdit()
        self.path_in_repo_edit.setPlaceholderText(
            tr("비워두면 로컬 파일 이름 그대로")
        )
        grid.addWidget(self.path_in_repo_edit, 1, 1)
        layout.addLayout(grid)

        self.file_count_label = QLabel("")
        self.file_count_label.setStyleSheet("color: #888;")
        layout.addWidget(self.file_count_label)

        self.private_check = QCheckBox(tr("비공개 데이터셋으로 업로드 (--private)"))
        layout.addWidget(self.private_check)

        self.delete_existing_check = QCheckBox(
            tr("업로드 전 Hub의 기존 파일 삭제 (다른 이름으로 올렸던 예전 파일 정리용)")
        )
        self.delete_existing_check.toggled.connect(self._on_delete_existing_toggled)
        layout.addWidget(self.delete_existing_check)

        old_name_row = QHBoxLayout()
        self.old_path_label = QLabel(tr("삭제할 기존 파일 이름:"))
        self.old_path_label.setEnabled(False)
        old_name_row.addWidget(self.old_path_label)
        self.old_path_in_repo_edit = QLineEdit()
        self.old_path_in_repo_edit.setPlaceholderText(tr("비워두면 위 'Repo 안 파일 이름'과 동일"))
        self.old_path_in_repo_edit.setEnabled(False)
        old_name_row.addWidget(self.old_path_in_repo_edit, 1)
        layout.addLayout(old_name_row)

        acct_text, acct_color = hf_account()
        acct_row = QHBoxLayout()
        self.hf_account_label = QLabel(acct_text)
        self.hf_account_label.setStyleSheet(f"color: {acct_color}; font-weight: bold;")
        self.hf_account_label.setWordWrap(True)
        acct_row.addWidget(self.hf_account_label, 1)
        switch_btn = QPushButton(tr("계정 전환..."))
        switch_btn.clicked.connect(self._open_account_dialog)
        acct_row.addWidget(switch_btn)
        layout.addLayout(acct_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("업로드 시작"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.tree.itemChanged.connect(lambda *_: self._on_files_changed())
        self._on_files_changed()


    def _open_account_dialog(self) -> None:
        dlg = HfAccountDialog(self)
        dlg.exec()
        text, color = hf_account()
        self.hf_account_label.setText(text)
        self.hf_account_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _on_delete_existing_toggled(self, on: bool) -> None:
        self.old_path_label.setEnabled(on)
        self.old_path_in_repo_edit.setEnabled(on)

    def _fill_table(self, start_dir: str) -> None:
        """데이터 경로의 .hdf5 를 훑어 표를 채운다.

        미리 체크되는 것은 **업로드 장부가 올릴 필요가 있다고 말하는 것**뿐이다
        (신규이거나 지난 업로드 뒤에 바뀐 것). 재압축 대화상자가
        hdf5_repack_status 로 같은 일을 한다 -- 조작자가 "무엇을 이미 했더라"
        를 기억할 필요가 없어야 한다.
        """
        root = Path(start_dir) if start_dir else None
        if root is None or not root.is_dir():
            return
        repo_id = ""
        try:
            repo_id = self._recents.most_recent("hdf5_repo_id")
        except Exception:  # noqa: BLE001
            pass
        for path in sorted(root.glob("*.hdf5")):
            self._add_path(path, repo_id)
        self._status_repo = repo_id
        self.table.fit_columns()

    def _add_path(self, path, repo_id: str, force_check: bool = False) -> None:
        st = hdf5_repack_status(str(path))
        reason = ""
        try:
            reason = upload_reason(repo_id, Path(path)) or "" if repo_id else ""
        except Exception:  # noqa: BLE001 -- 장부를 못 읽는 것이 선택을 막지 않는다
            reason = ""
        if not repo_id:
            status = tr("Repo ID 를 넣으면 업로드 이력을 봅니다")
        elif reason:
            status = reason
        else:
            status = tr("변경 없음 — 이미 올림")
        self.table.add_row(
            path, st["size"], extra=(st["episodes"], status),
            checked=force_check or bool(reason),
            disabled=bool(st["error"]))

    def _refresh_upload_status(self, *_args) -> None:
        """Repo ID 가 바뀌면 '이미 올렸나' 의 답이 통째로 바뀐다.

        **다른 repo 로 바뀐 때만 체크를 다시 정한다.** 같은 repo 로 다시
        들어온 것(칸을 떠났다 돌아온 것)이라면 조작자가 손으로 고른 것을
        유지한다 -- 그 손길을 지우면 "왜 내 선택이 사라지나" 가 된다.
        """
        repo_id = self.repo_id_edit.text()
        same_repo = repo_id == getattr(self, "_status_repo", None)
        checked = set(self.table.checked_paths())
        paths = [p for p, _it in self.table._rows]
        self.table.tree.clear()
        self.table._rows = []
        for p in paths:
            self._add_path(Path(p), repo_id, force_check=same_repo and p in checked)
        self._status_repo = repo_id
        self.table.fit_columns()
        self._on_files_changed()

    def _browse_file(self) -> None:
        start = self._start_dir or str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("업로드할 .hdf5 파일 (여러 개 선택 가능)"), start, "HDF5 (*.hdf5)")
        repo_id = self.repo_id_edit.text()
        for p in paths:
            if not self.table.has(p):
                self._add_path(Path(p), repo_id, force_check=True)
        self.table.fit_columns()
        self._on_files_changed()

    def _on_files_changed(self) -> None:
        """Keeps the repo-name field honest about what it will do.

        With one file it renames; with several it can only be a folder, since
        one name for many uploads would leave just the last one. Saying so here
        is cheaper than discovering it on the Hub afterwards.
        """
        files = self.table.checked_paths()
        multi = len(files) > 1
        if multi:
            self.path_in_repo_label.setText(tr("Repo 안 폴더:"))
            self.path_in_repo_edit.setPlaceholderText(
                tr("비워두면 repo 최상위. 파일 이름은 각자 그대로 유지됩니다"))
            if self.path_in_repo_edit.text().strip() in {Path(f).name for f in files}:
                self.path_in_repo_edit.clear()
            self.file_count_label.setText(
                tr("파일 {n}개 선택됨").format(n=len(files)))
        else:
            self.path_in_repo_label.setText(tr("Repo 안 파일 이름:"))
            self.path_in_repo_edit.setPlaceholderText(tr("비워두면 로컬 파일 이름 그대로"))
            if files and not self.path_in_repo_edit.text().strip():
                # 디렉터리에서 이름을 따오지 않는다 -- 그렇게 만들어진 게
                # Hub의 `libero_datasets` 파일이다.
                if Path(files[0]).is_file():
                    self.path_in_repo_edit.setText(Path(files[0]).name)
            self.file_count_label.setText("")
        # 여러 개일 때 '기존 파일 삭제'의 다른 이름 지정은 의미가 없다.
        self.old_path_label.setVisible(not multi)
        self.old_path_in_repo_edit.setVisible(not multi)

    def build_args(self) -> "list[str] | None":
        """Returns the script's argv (sans program name), or None (with a
        warning dialog already shown) if required fields are missing."""
        files = self.table.checked_paths()
        if not files:
            QMessageBox.warning(self, tr("파일 필요"), tr(".hdf5 파일을 하나 이상 선택하세요."))
            return None
        repo_id = self.repo_id_edit.text()
        err = repo_id_error(repo_id)
        if err:
            QMessageBox.warning(self, tr("Repo ID 오류"), tr(err))
            return None
        self._recents.add("hdf5_repo_id", repo_id)
        args = [*files, "--repo-id", repo_id]
        path_in_repo = self.path_in_repo_edit.text().strip()
        if path_in_repo:
            args += ["--path-in-repo", path_in_repo]
        args.append("--private" if self.private_check.isChecked() else "--no-private")
        if self.delete_existing_check.isChecked():
            args.append("--delete-existing")
            old_path_in_repo = self.old_path_in_repo_edit.text().strip()
            if old_path_in_repo and len(files) == 1:
                args += ["--old-path-in-repo", old_path_in_repo]
        return args
