"""Upload, conversion, repack, and pipeline operations for WorkspaceWindow."""

from __future__ import annotations

import json
import shutil
import sys
import time
import webbrowser
from pathlib import Path

import h5py
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtWidgets import QApplication, QCheckBox, QDialog, QMessageBox

from apps.workspace.features.upload.pipeline_dialog import PipelineDialog
from mstack.scene.dataset_sync import local_tasks, plan_sync
from mstack.data.episode_stats import hdf5_files
from apps.workspace.features.upload.hdf5_auto_dialog import Hdf5AutoDialog
from mstack.data.lerobot_local import local_lerobot_status
from mstack.config.station import load_station
from mstack.gui.dialogs import (
    HdfUploadDialog,
    HfAccountDialog,
    LerobotConvertDialog,
    RepackDialog,
    hf_account,
)
from mstack.gui.text_utils import repo_id_error
from apps.workspace.shared import jobs
from mstack.gui.i18n import tr

from apps.workspace.constants import CONVERT_SCRIPT, REPACK_SCRIPT, UPLOAD_SCRIPT


def lerobot_convert_step(paths: list, repo: str, root: str, resume: bool) -> dict:
    """LeRobot 변환 파이프라인 단계 하나.

    --fps 는 변환기 기본값에 맡기지 않고 스테이션 설정에서 명시로 넘긴다.
    이어붙이기에서 --fps 가 기존 데이터셋의 fps 와 다르면 변환기가 거부하는데,
    명시하지 않으면 그 검사가 의도한 값을 받는지 알 수 없다."""
    args = [CONVERT_SCRIPT, *paths, "--repo-id", repo, "--root", root,
            "--fps", str(load_station().fps)]
    if resume:
        args.append("--resume")
    return {"name": tr("LeRobot 변환 (이어붙이기)" if resume
                       else "LeRobot 변환 (전체 재빌드)"),
            "program": sys.executable, "args": args, "clear_root": root}


class UploadOps:
    """Upload, conversion, repack, and pipeline operations."""

    def __init__(self, win) -> None:
        self.win = win

    def repo_id_for(self, key: str) -> str:
        return self.win.repo_edits[key].text()

    def check_repo(self, key: str, what: str) -> "str | None":
        """Returns the id, or None after telling the operator what is wrong."""
        repo = self.repo_id_for(key)
        err = repo_id_error(repo)
        if err:
            QMessageBox.warning(self.win, tr("Repo ID 오류"),
                                tr("{w} 을(를) 시작할 수 없습니다.\n\n{e}").format(w=what, e=err))
            return None
        self.win._recents.add(key, repo)
        return repo

    def refresh_convert_state(self) -> None:
        """"무엇이 변환됐나" 한 줄 -- 로컬만 본다 (네트워크 없음).

        세는 것 둘: 데이터 경로의 **성공 에피소드**(변환 대상이 되는 것)와
        로컬 변환 폴더의 `meta/info.json`. 둘을 나란히 놓으면 "아직 안 된 것"
        이 뺄셈으로 나온다. Hub 과 맞는지는 여기서 말하지 않는다 -- 그것은
        네트워크가 필요하고, 이어붙이기·전체 처리가 이미 그 일을 한다.
        """
        lab = getattr(self.win, "convert_state_label", None)
        if lab is None:
            return
        data_root = self.win.root_edit.text().strip()
        root = self.win._recents.most_recent(
            "lerobot_root", str(Path.home() / "lerobot_upload"))
        st = local_lerobot_status(root)
        try:
            tasks = local_tasks(data_root)
        except Exception as e:  # noqa: BLE001
            lab.setText(tr("{r} 를 읽지 못했습니다: {e}").format(r=data_root, e=e))
            return
        usable = sum(t["episodes"] for t in tasks.values())
        if not st["exists"]:
            lab.setText(tr(
                "아직 변환한 적이 없습니다 ({o} 에 meta/info.json 없음).\n"
                "데이터셋에는 올릴 수 있는 에피소드가 {u}개 (지시문 {t}개) "
                "있습니다.").format(o=root, u=usable, t=len(tasks)))
            return
        left = usable - st["episodes"]
        lab.setText(tr(
            "변환본: 에피소드 {e}개 · 프레임 {f:,}개 · 지시문 {t}개  ({at} 변환)\n"
            "데이터셋: 성공 에피소드 {u}개 · 지시문 {ut}개\n{diff}\n{o}").format(
                e=st["episodes"], f=st["frames"], t=st["tasks"], at=st["at"],
                u=usable, ut=len(tasks), o=root,
                diff=(tr("→ 아직 변환 안 된 것 {n}개").format(n=left) if left > 0
                      else tr("→ 변환본이 데이터셋보다 {n}개 많습니다 — 지운 것이 "
                              "변환본에 남아 있을 수 있습니다").format(n=-left)
                      if left < 0 else tr("→ 개수가 같습니다"))))

    def on_myhdf5(self) -> None:
        webbrowser.open("https://myhdf5.hdfgroup.org/")
        path = self.win.dataset_ops.selected_file()
        if path is not None:
            self.win.log(tr("[myHDF5] 브라우저 창에 파일을 끌어다 놓으세요: {p}")
                     .format(p=path))

    @staticmethod
    def all_hdf5(data_root) -> list:
        """변환·업로드 대상 파일 전부: legacy 정렬 + scene 정렬.

        legacy 를 앞에 두는 순서는 plan_sync 의 길이 지문(접두 비교)과
        일치해야 하므로 dataset_sync._ordered_paths 와 같은 규칙이다.
        """
        root = Path(str(data_root))
        return ([str(p) for p in sorted(root.glob("*_demo.hdf5"))]
                + [str(p) for p in sorted(root.glob("scene_*.hdf5"))])

    def hdf5_candidates(self) -> list:
        return self.all_hdf5(self.win.root_edit.text().strip() or str(Path.home()))

    def pipeline_guard(self, what: str) -> bool:
        """Shared preconditions for every automatic button."""
        if self.win.worker is not None:
            QMessageBox.warning(self.win, tr("수집 중"),
                                tr("수집 중에는 실행할 수 없습니다. 먼저 세션을 종료하세요."))
            return False
        if self.win.procs.pipeline_steps:
            QMessageBox.information(self.win, tr("이미 실행 중"),
                                    tr("{w}이(가) 이미 진행 중입니다. 로그를 확인하세요.")
                                    .format(w=what))
            return False
        return True

    def start_pipeline(self, steps: list, tag: str) -> None:
        self.win.procs.pipeline_steps = steps
        self.win.procs.pipeline_results = []
        self.win.procs.pipeline_t0 = time.monotonic()
        self.win.bottom_tabs.setCurrentWidget(self.win.upload_view)
        self.win.log(f"[{tag}] {len(steps)}단계 시작 — "
                 + " → ".join(st["name"] for st in steps), "upload")
        for st in steps:
            if st.get("detail"):
                self.win.log(f"  · [{st['name']}] {st['detail']}", "upload")
        self.run_next_pipeline_step()

    def on_hdf5_auto(self) -> None:
        """재압축 -> 원본 HDF5 업로드."""
        if not self.pipeline_guard(tr("HDF5 자동 처리")):
            return
        data_root = self.win.root_edit.text().strip()
        paths = self.all_hdf5(data_root)
        if not paths:
            QMessageBox.warning(self.win, tr("파일 없음"),
                                tr("{r} 에 *_demo.hdf5 / scene_*.hdf5 가 없습니다.").format(r=data_root))
            return
        repo = self.check_repo("hdf5_repo_id", tr("HDF5 재압축 + 업로드"))
        if repo is None:
            return
        # 고르는 화면은 셋과 같은 표다 (2026-09-13). 예전에는 확인창 하나에
        # 파일 이름을 글로 늘어놓고 "변경분만 / 전체 강제" 체크박스 하나로
        # 전부 아니면 전무를 골랐다 -- 가운데가 없었다.
        dlg = Hdf5AutoDialog(self.win, paths, repo)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.win.log("[HDF5 자동] 취소했습니다.", "upload")
            return
        todo = dlg.repack_paths()
        picked = dlg.upload_paths()
        steps = []
        if todo:
            steps.append({"name": tr("재압축 {n}개").format(n=len(todo)),
                          "program": sys.executable,
                          "args": [REPACK_SCRIPT, *todo]})
        if picked:
            steps.append({"name": tr("HDF5 원본 업로드 ({n}개)").format(n=len(picked)),
                          "detail": ", ".join(Path(x).name for x in picked),
                          "program": sys.executable,
                          "args": [UPLOAD_SCRIPT, *picked,
                                   "--repo-id", repo, "--no-private"]})
        if not steps:
            self.win.log("[HDF5 자동] 할 일이 없습니다 — 고른 파일이 없거나 "
                         "두 동작을 모두 껐습니다.", "upload")
            return
        self.start_pipeline(steps, tr("HDF5 자동"))

    def on_lerobot_auto(self) -> None:
        """전체 재빌드 -> 교체 업로드. resume 경로는 여기에 없다.

        Curation deletes episodes from .hdf5 files that were already pushed,
        and --resume only ever appends: the deleted episodes' chunks stay on
        the Hub while the declared count drops. Rebuilding from scratch and
        pushing with --replace is the only combination that makes the Hub
        match what is actually on disk, so this button offers nothing else.
        """
        if not self.pipeline_guard(tr("LeRobot 자동 처리")):
            return
        data_root = self.win.root_edit.text().strip()
        paths = self.all_hdf5(data_root)
        if not paths:
            QMessageBox.warning(self.win, tr("파일 없음"),
                                tr("{r} 에 *_demo.hdf5 / scene_*.hdf5 가 없습니다.").format(r=data_root))
            return
        repo = self.check_repo("repo_id", tr("LeRobot 변환 + 업로드"))
        if repo is None:
            return
        root = self.win._recents.most_recent("lerobot_root", str(Path.home() / "lerobot_upload"))
        # "task 15개" 만 보여주면 이어붙이기 확인창의 "새 에피소드 25개" 와
        # 단위가 달라 개수 오류처럼 읽힌다 -- 에피소드 합계를 함께 표기한다.
        n_ep = self.count_hdf5_episodes()
        ep_txt = tr(" (에피소드 {e}개)").format(e=n_ep) if n_ep is not None else ""
        if QMessageBox.question(
                self.win, tr("LeRobot 변환 + 업로드"),
                tr("task {n}개{ep}를 처음부터 다시 변환하고, {r} 을(를) 통째로 "
                   "교체합니다.\n\n"
                   "· 로컬 변환 폴터를 비웁니다: {o}\n"
                   "· 이어붙이기(resume)를 쓰지 않으므로 큐레이션에서 지운 "
                   "에피소드가 Hub에서도 사라집니다\n"
                   "· 전체 재변환이라 시간이 걸립니다\n\n진행할까요?")
                .format(n=len(paths), ep=ep_txt, r=repo, o=root),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            self.win.log("[LeRobot 자동] 취소했습니다.", "upload")
            return
        self.win._recents.add("repo_id", repo)
        self.win._recents.add("lerobot_root", root)
        steps = [
            lerobot_convert_step(paths, repo, root, resume=False),
            {"name": tr("LeRobot 교체 업로드"), "program": sys.executable,
             "args": [CONVERT_SCRIPT, "--repo-id", repo, "--root", root,
                      "--push-only", "--replace", "--no-private"]},
        ]
        self.start_pipeline(steps, tr("LeRobot 자동"))

    def on_lerobot_resume(self) -> None:
        """이어붙이기 -- Hub과 대조해 새 에피소드만 변환·추가 업로드.

        --resume 이 안전한 조건(추가만 있음)을 plan_sync 로 먼저 검증하고,
        아니면 실행을 거부한다. 삭제/편집이 섞인 채 이어붙이면 지운 에피소드의
        청크가 Hub에 남거나(선언 개수만 줄어듦) 개수 대응이 깨진다 --
        convert_libero_to_lerobot.py 상단 docstring 2번 참고.
        """
        if not self.pipeline_guard(tr("LeRobot 이어붙이기")):
            return
        data_root = self.win.root_edit.text().strip()
        paths = self.all_hdf5(data_root)
        if not paths:
            QMessageBox.warning(self.win, tr("파일 없음"),
                                tr("{r} 에 *_demo.hdf5 / scene_*.hdf5 가 없습니다.").format(r=data_root))
            return
        repo = self.check_repo("repo_id", tr("LeRobot 이어붙이기"))
        if repo is None:
            return
        plan = plan_sync(data_root, repo)  # 네트워크 -- Hub 개수 대조
        self.win._warn_ignored_legacy(plan)
        if plan["action"] == "blocked":
            QMessageBox.warning(self.win, tr("이어붙이기 불가"),
                                tr("Hub 상태를 읽지 못했습니다: {e}\n확실하지 않은 "
                                   "채로 올리지 않습니다.").format(e=plan["error"]))
            return
        if plan["action"] == "rebuild":
            if plan["shrunk"]:
                msg = tr("이미 올라간 task에서 에피소드 {n}개가 삭제되었습니다.\n"
                         "이어붙이기는 추가만 할 수 있어 지운 에피소드가 Hub에 "
                         "남습니다.").format(n=plan["shrunk"])
            else:
                msg = tr("에피소드 이력이 Hub와 어긋난 task가 있습니다 (길이 지문 "
                         "불일치).\n이어붙이면 엉뚱한 에피소드가 붙습니다.")
            QMessageBox.warning(self.win, tr("이어붙이기 불가"),
                                msg + tr("\n'변환 + 업로드 (자동)'으로 전체 "
                                         "재빌드하세요."))
            return
        # 여기 남는 ambiguous 는 전부 개수가 Hub와 같은 task 다 (append 대상이
        # 아님 -- 대상이면서 이력이 어긋난 경우는 위 rebuild 로 빠졌다). 이번
        # 실행이 그 task 에 아무것도 추가하지 않으므로, 위험을 확인했다는
        # 체크만 받고 나머지 task 의 이어붙이기는 허용한다.
        if plan["ambiguous"] and not self.confirm_ambiguous_idle(plan["ambiguous"]):
            self.win.log("[LeRobot 이어붙이기] 취소했습니다 (이력 미확인).", "upload")
            return
        if plan["action"] == "up_to_date":
            QMessageBox.information(
                self.win, tr("이어붙일 것 없음"),
                tr("Hub이 이미 로컬과 같습니다 ({n}개 에피소드).")
                .format(n=plan["local_total"]))
            return
        root = self.win._recents.most_recent("lerobot_root",
                                         str(Path.home() / "lerobot_upload"))
        if QMessageBox.question(
                self.win, tr("LeRobot 이어붙이기"),
                tr("새 에피소드 {n}개만 변환해 이어붙입니다 "
                   "(Hub {h} → {l}).\n\n"
                   "· 로컬 변환 폴터를 비우고 Hub의 현재 상태를 기준으로 "
                   "받습니다: {o}\n"
                   "· 이미 올라간 에피소드는 다시 변환하지 않습니다\n\n진행할까요?")
                .format(n=plan["added"], h=plan["hub_total"],
                        l=plan["local_total"], o=root),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes) != QMessageBox.StandardButton.Yes:
            self.win.log("[LeRobot 이어붙이기] 취소했습니다.", "upload")
            return
        self.win._recents.add("repo_id", repo)
        self.win._recents.add("lerobot_root", root)
        steps = [
            lerobot_convert_step(paths, repo, root, resume=True),
            {"name": tr("LeRobot 추가 업로드"), "program": sys.executable,
             "args": [CONVERT_SCRIPT, "--repo-id", repo, "--root", root,
                      "--push-only", "--no-private"]},
        ]
        self.start_pipeline(steps, tr("LeRobot 이어붙이기"))

    def confirm_ambiguous_idle(self, tasks: list) -> bool:
        """이력 검증을 통과 못했지만 append 대상도 아닌 task 확인창.

        진행해도 이 task 들에는 아무것도 추가되지 않지만, 지우고 다시 찍은
        것이라면 Hub 에 옛 에피소드가 남아 있을 수 있다 -- 그 정리는 전체
        재빌드만 할 수 있다. 실수로 Yes 를 누르지 못하도록 체크박스를 켜야
        진행 버튼이 활성화된다.
        """
        box = QMessageBox(
            QMessageBox.Icon.Warning, tr("이력 확인 필요"),
            tr("다음 task는 개수는 Hub와 같지만 이력 검증(에피소드 길이 지문)을 "
               "통과하지 못했습니다:\n{t}\n\n이번 이어붙이기에서 이 task에는 "
               "아무것도 추가되지 않습니다. 다만 지우고 다시 찍은 것이라면 Hub에 "
               "옛 에피소드가 남아 있을 수 있고, 그 정리는 '변환 + 업로드 "
               "(자동)' 전체 재빌드만 할 수 있습니다.").format(
                   t="\n".join(f"· {x[:60]}" for x in tasks)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            self.win)
        yes = box.button(QMessageBox.StandardButton.Yes)
        yes.setText(tr("나머지 task만 이어붙이기 진행"))
        yes.setEnabled(False)
        ack = QCheckBox(tr("위 내용을 확인했습니다"))
        ack.toggled.connect(yes.setEnabled)
        box.setCheckBox(ack)
        return box.exec() == QMessageBox.StandardButton.Yes

    def count_hdf5_episodes(self) -> "int | None":
        """Episodes currently in the .hdf5 files. Metadata only -- no images."""
        try:
            total = 0
            for f in hdf5_files(self.win.root_edit.text().strip()):
                with h5py.File(f, "r") as h:
                    total += len(h["data"].keys())
            return total
        except Exception:  # noqa: BLE001
            return None

    def on_lerobot_reupload(self) -> None:
        """재변환 없이, 이미 만들어둔 로컬 결과로 Hub을 교체."""
        if not self.pipeline_guard(tr("LeRobot 재업로드")):
            return
        repo = self.check_repo("repo_id", tr("전체 task 다시 업로드"))
        if repo is None:
            return
        root = self.win._recents.most_recent("lerobot_root", str(Path.home() / "lerobot_upload"))
        info = Path(root) / "meta" / "info.json"
        if not info.exists():
            QMessageBox.warning(
                self.win, tr("변환 결과 없음"),
                tr("{o} 에 변환 결과가 없습니다 ({i} 없음).\n"
                   "'변환 + 업로드 (자동)' 또는 'HDF5 골라서 변환만...'을 먼저 "
                   "실행하세요.").format(o=root, i=info.name))
            return
        try:
            meta = json.loads(info.read_text())
            n_ep, n_fr = meta.get("total_episodes", "?"), meta.get("total_frames", "?")
        except Exception:  # noqa: BLE001
            n_ep = n_fr = "?"
        # 변환 결과와 현재 .hdf5 의 개수를 맞춰본다. 큐레이션으로 에피소드를
        # 지운 뒤 재변환을 잊으면, 이 버튼은 삭제 이전 결과를 그대로 Hub에
        # 올려 큐레이션을 통째로 되돌린다 -- 그리고 개수만 보고는 눈치채기
        # 어렵다. 지금 세는 값과 나란히 놓으면 그 자리에서 보인다.
        n_local = self.count_hdf5_episodes()
        stale = isinstance(n_ep, int) and n_local is not None and n_ep != n_local
        head = (tr("⚠ 변환 결과가 최신이 아닙니다 — 변환본 {e}개 vs 현재 HDF5 {l}개\n"
                   "   지금 올리면 큐레이션으로 지운 에피소드가 되살아납니다.\n"
                   "   '변환 + 업로드 (자동)'으로 다시 만드세요.\n\n").format(e=n_ep, l=n_local)
                if stale else "")
        if QMessageBox.question(
                self.win, tr("전체 task 다시 업로드"),
                head + tr("{o} 의 변환 결과를 {r} 에 통째로 올립니다.\n\n"
                          "· 변환본 에피소드 {e}개 / 프레임 {f}\n"
                          "· 현재 HDF5 에피소드 {l}개\n"
                          "· 재변환은 하지 않습니다\n"
                          "· 로컬에 없는 원격 파일은 지웁니다 (큐레이션 삭제 반영)\n\n"
                          "이 로컬 결과가 최신인지 확인하셨나요?")
                .format(o=root, r=repo, e=n_ep, f=n_fr,
                        l=n_local if n_local is not None else "?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            self.win.log("[LeRobot 재업로드] 취소했습니다.", "upload")
            return
        self.start_pipeline([
            {"name": tr("LeRobot 교체 업로드"), "program": sys.executable,
             "args": [CONVERT_SCRIPT, "--repo-id", repo, "--root", root,
                      "--push-only", "--replace", "--no-private"]}],
            tr("LeRobot 재업로드"))

    def on_pipeline(self) -> None:
        if self.win.worker is not None:
            QMessageBox.warning(self.win, tr("수집 중"),
                                tr("수집 중에는 실행할 수 없습니다. 먼저 세션을 종료하세요."))
            return
        if self.win.procs.pipeline_steps:
            QMessageBox.information(self.win, tr("이미 실행 중"),
                                    tr("전체 처리가 이미 진행 중입니다. 로그를 확인하세요."))
            return
        data_root = self.win.root_edit.text().strip()
        repo = self.check_repo("repo_id", tr("전체 처리"))
        if repo is None:
            return
        self.win.log("[전체 처리] Hub 상태를 확인하는 중...", "upload")
        self.win.bottom_tabs.setCurrentWidget(self.win.upload_view)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            plan = plan_sync(data_root, repo) if repo else {
                "action": "blocked", "error": "LeRobot Repo ID가 없습니다 (먼저 한 번 지정하세요)",
                "rows": [], "added": 0, "shrunk": 0, "ambiguous": [],
                "local_total": 0, "hub_total": 0,
                "paths": self.all_hdf5(data_root)}
        finally:
            QApplication.restoreOverrideCursor()
        dlg = PipelineDialog(
            self.win, data_root, plan, repo,
            self.repo_id_for("hdf5_repo_id"),
            self.win._recents.most_recent(
                "lerobot_root", str(Path.home() / "lerobot_upload")),
            scripts={"repack": REPACK_SCRIPT,
                     "convert": CONVERT_SCRIPT,
                     "upload": UPLOAD_SCRIPT})
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.win.log("[전체 처리] 취소했습니다.", "upload")
            return
        # 다이얼로그 안에서 바꾼 값도 검사한다. HDF5 업로드는 파이프라인의 마지막
        # 단계라, 여기서 막지 않으면 앞 단계를 다 돌고 나서야 403 으로 죽는다.
        for edit, key, what in ((dlg.lerobot_repo_edit, "repo_id", "LeRobot"),
                                (dlg.hdf5_repo_edit, "hdf5_repo_id", "HDF5")):
            if key == "hdf5_repo_id" and not dlg.hdf5_check.isChecked():
                continue
            err = repo_id_error(edit.text().strip())
            if err:
                QMessageBox.warning(self.win, tr("Repo ID 오류"),
                                    tr("{w} Repo ID: {e}").format(w=what, e=err))
                self.win.log(f"[전체 처리] 중단 — {what} Repo ID: {err}", "upload")
                return
            self.win.repo_edits[key].set_text(edit.text())
        steps = dlg.steps()
        if not steps:
            self.win.log("[전체 처리] 할 일이 없습니다.", "upload")
            return
        self.win.procs.pipeline_steps = steps
        self.win.procs.pipeline_results = []
        self.win.procs.pipeline_t0 = time.monotonic()
        self.win.log(f"[전체 처리] {len(steps)}단계 시작 — "
                 + " → ".join(s["name"] for s in steps), "upload")
        for st in steps:
            if st.get("detail"):
                self.win.log(f"  · [{st['name']}] {st['detail']}", "upload")
        self.run_next_pipeline_step()

    def run_next_pipeline_step(self) -> None:
        if not self.win.procs.pipeline_steps:
            self.finish_pipeline(True)
            return
        step = self.win.procs.pipeline_steps[0]
        if "program" not in step:
            # 정보용 단계 (예: 'HDF5 원본 업로드 — 생략') -- 프로세스 없이
            # 사유만 로그·요약에 남기고 넘어간다. 자동 선택이 파일을 하나도
            # 안 고른 날, '왜 안 올라갔는지'가 보이게 하는 장치 (2026-08-25).
            self.win.procs.pipeline_steps.pop(0)
            self.win.procs.pipeline_results.append((step["name"], 0, 0.0))
            self.win.log(f"\n[전체 처리] · {step['name']} — "
                     f"{step.get('note', '')}", "upload")
            self.run_next_pipeline_step()
            return
        if step.get("clear_root"):
            # 이어붙이기는 로컬 메타가 있으면 그걸 기준으로 삼는다. 비워야 Hub의
            # 현재 상태를 받아오고, 재빌드는 애초에 빈 폴터가 필요하다.
            root = Path(step["clear_root"])
            if root.exists():
                try:
                    shutil.rmtree(root)
                    self.win.log(f"[전체 처리] 로컬 변환 폴터를 비웠습니다: {root}", "upload")
                except OSError as e:
                    self.win.log(f"[전체 처리] 폴터를 비우지 못했습니다: {e}", "upload")
                    self.finish_pipeline(False)
                    return
        proc = QProcess(self.win)
        proc.setProgram(step["program"])
        proc.setArguments(step["args"])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        prefix = f"[{step['name']}]"
        proc.readyReadStandardOutput.connect(lambda: self.win._pipe(proc, prefix, "upload"))
        proc.finished.connect(self.on_pipeline_step_finished)
        self.win.procs.pipeline_proc = proc
        self.win.procs.pipeline_step_t0 = time.monotonic()
        # 단계마다 새 작업이다 -- 남은 시간은 단계 안에서만 뜻이 있다
        # (재압축의 진행률로 업로드가 얼마 남았는지는 알 수 없다).
        jobs.start_job(self.win, tr("전체 처리: {n}").format(n=step["name"]))
        self.win.log(f"\n[전체 처리] ▶ {step['name']} 시작", "upload")
        self.win.statusBar().showMessage(tr("전체 처리: {n}").format(n=step["name"]))
        proc.start()

    def on_pipeline_step_finished(self, code: int, _status) -> None:
        step = self.win.procs.pipeline_steps.pop(0)
        dt = time.monotonic() - self.win.procs.pipeline_step_t0
        self.win.procs.pipeline_results.append((step["name"], code, dt))
        self.win.log(f"[전체 처리] {'✔' if code == 0 else '✖'} {step['name']} "
                 f"종료 (exit={code}, {dt / 60:.1f}분)", "upload")
        self.win.procs.pipeline_proc = None
        if code != 0:
            # 뒤 단계가 앞 결과에 의존하므로(변환 -> 업로드) 잘못된 것을 올리지
            # 않는다. 아침에 로그만 볼면 어디서 멈췄는지 알 수 있게 남긴다.
            self.finish_pipeline(False)
            return
        self.run_next_pipeline_step()

    def finish_pipeline(self, ok: bool) -> None:
        jobs.end_job(self.win)
        remaining = [s["name"] for s in self.win.procs.pipeline_steps]
        self.win.procs.pipeline_steps = []
        total = time.monotonic() - self.win.procs.pipeline_t0
        lines = ["", "=" * 56,
                 tr("전체 처리 요약 — {r} (총 {m:.1f}분)").format(
                     r=tr("완료") if ok else tr("중단됨"), m=total / 60)]
        for name, code, dt in self.win.procs.pipeline_results:
            lines.append(f"  {'✔' if code == 0 else '✖'} {name:24s} "
                         f"exit={code}  {dt / 60:.1f}분")
        for name in remaining:
            lines.append(f"  · {name:24s} " + tr("실행 안 함"))
        lines.append("=" * 56)
        for ln in lines:
            self.win.log(ln, "upload")
        self.win.statusBar().showMessage(
            tr("전체 처리 완료") if ok else tr("전체 처리 중단 — 로그 확인"), 0)
        if not ok:
            self.win._alert(tr("전체 처리 중단"),
                        tr("한 단계가 실패해 이후 단계를 실행하지 않았습니다.\n"
                           "Upload 탭과 로그 파일에 자세한 내용이 있습니다."))

    def on_hf_accounts(self) -> None:
        dlg = HfAccountDialog(self.win)
        dlg.exec()
        text, color = hf_account()
        self.win.hf_label.setText(text)
        self.win.hf_label.setStyleSheet(f"color:{color}; font-weight:bold;")
        if dlg.switched_to():
            self.win.log(f"[HF] 이제 {dlg.switched_to()} 계정으로 업로드합니다.")

    def on_repack(self) -> None:
        if self.win.worker is not None:
            QMessageBox.warning(self.win, tr("수집 중"),
                                tr("수집 중에는 재압축할 수 없습니다. 먼저 세션을 종료하세요."))
            return
        paths = self.hdf5_candidates()
        if not paths:
            QMessageBox.warning(self.win, tr("파일 없음"), tr("재압축할 .hdf5 파일이 없습니다."))
            return
        dlg = RepackDialog(self.win, paths)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dlg.selected()
        if not selected:
            return
        proc = QProcess(self.win)
        proc.setProgram(sys.executable)
        proc.setArguments([REPACK_SCRIPT, *selected])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self.win._pipe(proc, "[재압축]", "upload"))
        proc.finished.connect(lambda c, _s: (self.win.log(f"[재압축] 종료 (exit={c})", "upload"),
                                             jobs.end_job(self.win),
                                             self.win.dataset_ops.refresh_dataset_tree()))
        self.win.procs.repack_process = proc
        self.win.bottom_tabs.setCurrentWidget(self.win.upload_view)
        self.win.log(f"[재압축] 시작: {len(selected)}개 파일", "upload")
        jobs.start_job(self.win, tr("재압축 {n}개").format(n=len(selected)))
        proc.start()

    def on_lerobot(self) -> None:
        dlg = LerobotConvertDialog(self.win, self.win.root_edit.text().strip())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        args = dlg.build_args()
        if args is None:
            return
        proc = QProcess(self.win)
        proc.setProgram(sys.executable)
        proc.setArguments([CONVERT_SCRIPT, *args])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self.win._pipe(proc, "[LeRobot]", "upload"))
        proc.finished.connect(lambda c, _s: (self.win.log(
            f"[LeRobot] 종료 (exit={c})" + ("" if c == 0 else " -- 실패, 위 로그를 확인하세요"),
            "upload"), jobs.end_job(self.win)))
        self.win.procs.convert_process = proc
        self.win.bottom_tabs.setCurrentWidget(self.win.upload_view)
        self.win.log(f"[LeRobot] 시작: {' '.join(args)}", "upload")
        jobs.start_job(self.win, tr("LeRobot 변환"))
        proc.start()

    def on_hdf5_upload(self) -> None:
        # 두 번째 인자는 '찾아보기'가 열릴 폴터다. 파일이 아니다.
        dlg = HdfUploadDialog(self.win, self.win.root_edit.text().strip())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        args = dlg.build_args()
        if args is None:
            return
        proc = QProcess(self.win)
        proc.setProgram(sys.executable)
        proc.setArguments([UPLOAD_SCRIPT, *args])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self.win._pipe(proc, "[HDF5 업로드]", "upload"))
        proc.finished.connect(lambda c, _s: (self.win.log(
            f"[HDF5 업로드] 종료 (exit={c})" + ("" if c == 0 else " -- 실패, 위 로그를 확인하세요"),
            "upload"), jobs.end_job(self.win)))
        self.win.procs.upload_process = proc
        self.win.bottom_tabs.setCurrentWidget(self.win.upload_view)
        self.win.log(f"[HDF5 업로드] 시작: {' '.join(args)}", "upload")
        jobs.start_job(self.win, tr("HDF5 업로드"))
        proc.start()
