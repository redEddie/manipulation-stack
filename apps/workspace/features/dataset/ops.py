"""Dataset tree, episode selection, deletion, and relabel operations."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QTreeWidgetItem

from mstack.data.dataset_schema import OBS_AGENTVIEW_RGB, normalize_schema_version
from mstack.data.libero_format import hdf5_repack_status, renumber_episodes
from mstack.gui.text_utils import repo_id_error
from mstack.gui.i18n import tr
from apps.workspace.features.dataset.right_panel import PHOTO_W
from apps.workspace.shared.info import scene_fields
from mstack.data.proxy_clip import invalidate_scene_caches
from mstack.gui.widgets.video_view import np_to_pixmap
from mstack.scene.scene_format import (
    delete_scene_episodes,
    list_scene_episodes,
    read_reference_image,
    read_scene_metadata,
)


def soft_wrap(text: str) -> str:
    """Lets a long filename wrap.

    QLabel only breaks at whitespace, and ``pick_up_the_blue_cup_..._demo.hdf5``
    has none -- so word wrap did nothing and the name sat on one clipped line.
    A zero-width space after each separator gives it legal break points without
    changing the visible characters or what a copy-paste yields... except that
    the copy would carry U+200B, so this is only ever applied to display text
    whose real value is also in the tooltip.
    """
    for sep in ("_", "-", "."):
        text = text.replace(sep, sep + "​")
    return text


class DatasetOps:
    """Dataset tree, episode selection, deletion, and relabel operations."""

    def __init__(self, win) -> None:
        self.win = win

    def on_no_dataset_toggled(self, on: bool) -> None:
        """No file is written, so the task/path fields have nothing to name."""
        self.win.task_box.setEnabled(not on)
        self.win.mode_hint.setText(tr(
            "연습 모드: 파일을 만들지 않습니다. 저장을 눌러도 버려집니다."
        ) if on else "")
        self.win.mode_hint.setStyleSheet("color:#e67e22;" if on else "color:#888;")
        for key in ("save", "savefail"):
            if key in getattr(self.win, "tb_actions", {}):
                self.win.tb_actions[key].setEnabled(not on and self.win.worker is not None)

    # -------------------------------------------------------------------- root
    def dataset_root(self) -> Path:
        """데이터 저장 경로 -- 수집도 조회도 같은 폴더다 (2026-09-06 통일).

        (빌드 순서상 아직 위젯이 없을 수 있다 -- 기본 경로로 폴백.)
        """
        edit = getattr(self.win, "root_edit", None)
        if edit is None:
            return Path.home() / "libero_datasets"
        return Path(edit.text().strip()).expanduser()

    def on_root_changed(self) -> None:
        """경로가 바뀌면 그 폴더에 딸린 것들을 전부 다시 읽는다 -- scene 목록,
        계획, 에피소드 트리, 분석. 한 곳에서 바뀌므로 한 곳에서 갱신한다."""
        self.win.stats_ops.mark_stats_stale()
        self.win.scene_ops.refresh_scene_combo()
        # 큐레이션의 Scene 콤보도 여기서 갱신한다. 안 하면 경로를 바꿔도 옛
        # 폴더의 씬 목록이 남고, 목록 뷰·격자가 전부 그 콤보에 걸려 있으므로
        # **다른 폴더를 보고 있다고 믿게 된다** (2026-09-11).
        if hasattr(self.win, "gallery_scene_combo"):
            self.win.gallery_ops.refresh_gallery_scenes()
        # 순위표도 이 경로를 따른다 -- "이 데이터셋" 의 정의가 이 경로의
        # 폴더명이라, 여기서 갱신하지 않으면 경로를 바꾼 뒤 Statistics 로
        # 갈 때까지 옛 폴더의 순위가 남는다.
        self.win.stats_ops.refresh_history()
        self.refresh_dataset_tree()

    def browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self.win, tr("데이터 저장 경로"), self.win.root_edit.text())
        if d:
            self.win.root_edit.setText(d)
            self.on_root_changed()

    # -------------------------------------------------------------------- tree
    def refresh_dataset_tree(self) -> None:
        """이름은 옛것이나 하는 일은 목록 갱신이다. 부르는 곳이 많아 이름만 남긴다."""
        self.refresh_episode_list()

    def refresh_episode_list(self) -> None:
        """왼쪽 목록 뷰 -- **(씬 → 지시문) 으로 좁혀진 에피소드 집합**을 그린다.

        예전에는 데이터 뿌리 전체를 훑어 파일마다 에피소드를 매단 2단 트리를
        그렸다. 그 방식은 이어붙이기에서 무너진다: 에피소드는 파일에 저장된
        순서대로 놓이므로 ``I000 I003 I000 I005 …`` 처럼 지시문이 섞이고,
        목록을 그대로 그리면 화면에서도 섞인다 (조작자 지적, 2026-09-11).

        그래서 파일 순서를 그리지 않는다. 위의 Scene 콤보와 Instruction 목록이
        고른 집합 -- 격자가 보여주는 바로 그 집합 -- 만 그린다. 둘은 **같은
        목록의 두 뷰**이고, 아무것도 안 골랐으면 비어 있다.

        열은 큐레이션에 필요한 것만 둔다: 번호 · 판정 · 프레임. 수집자·지시문은
        고른 뒤 우측 카드에서 읽는다 (좁은 패널에서 열을 늘리면 다 잘린다).
        """
        # build_center 가 build_left 보다 먼저 돌아서, 갤러리 탭이 만들어질 때
        # 목록 위젯은 아직 없다. 그때는 조용히 넘긴다 -- build_left 가 끝나고
        # 첫 필터 적용에서 다시 그린다.
        tree = getattr(self.win, "dataset_tree", None)
        if tree is None:
            return
        tree.clear()
        for ep in getattr(self.win, "_gallery_shown", []) or []:
            uid = ep.get("episode_uid", "")
            q = ep.get("quality_status") or (
                "-" if ep.get("success") is None
                else ("success" if ep["success"] else "failed"))
            mark = {"success": "✓", "failed": "✗"}.get(q, "·")
            item = QTreeWidgetItem([
                f"{ep.get('instruction_id', '')}-{uid.rsplit('-', 1)[-1]}",
                mark, str(ep.get("num_samples", "")),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, ep)
            item.setToolTip(0, f"{uid}\n{ep.get('instruction', '')}\n"
                               f"{ep.get('collector', '')}")
            tree.addTopLevelItem(item)
        self.show_tree_selection(getattr(self.win, "_gallery_selected", []))

    def show_tree_selection(self, episodes) -> None:
        """바깥이 정한 선택을 목록에 그린다. 신호를 내지 않는다."""
        tree = getattr(self.win, "dataset_tree", None)
        if tree is None:
            return
        names = {e.get("name") for e in (episodes or []) if e}
        tree.blockSignals(True)
        try:
            for i in range(tree.topLevelItemCount()):
                it = tree.topLevelItem(i)
                ep = it.data(0, Qt.ItemDataRole.UserRole) or {}
                it.setSelected(ep.get("name") in names)
        finally:
            tree.blockSignals(False)

    def selected_episodes(self) -> list:
        """목록 뷰에서 고른 에피소드 dict 들."""
        tree = getattr(self.win, "dataset_tree", None)
        out = []
        for it in (tree.selectedItems() if tree is not None else []):
            ep = it.data(0, Qt.ItemDataRole.UserRole)
            if ep:
                out.append(ep)
        return out

    def selected_file(self) -> Path | None:
        """지금 보고 있는 파일. **Scene 콤보가 정본이다.**

        목록 뷰가 에피소드만 담게 되면서 트리에는 파일 줄이 없다. 파일을
        고르는 곳은 콤보 하나뿐이라 여기서 읽는다 -- 두 군데서 읽으면
        "어느 파일에 대고 한 것인가" 가 갈린다.
        """
        cb = getattr(self.win, "gallery_scene_combo", None)
        p = cb.currentData() if cb is not None else None
        return Path(p) if isinstance(p, str) else None

    def busy_reason(self) -> str:
        """Anything that may currently hold an .hdf5 open, by name."""
        for proc, label in ((self.win.procs.repack_process, tr("재압축")),
                            (self.win.procs.convert_process, tr("LeRobot 변환")),
                            (self.win.procs.upload_process, tr("HDF5 업로드"))):
            if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
                return label
        return ""

    def on_dataset_selection(self) -> None:
        """목록 뷰에서 골랐다 -- 공유 선택으로 올린다 (격자도 같이 표시된다)."""
        self.win.gallery_ops.set_selection(self.selected_episodes(), source="tree")

    def on_selection_changed(self) -> None:
        """선택이 바뀌었다 (어느 뷰에서 왔든). 우측 패널과 분석을 맞춘다.

        창이 다 만들어지기 전에도 불린다 -- build_center 안의 갤러리 탭이
        첫 필터를 적용하는데, 그때 왼쪽 목록도 우측 패널도 아직 없다.
        그 시점에는 그릴 곳이 없으므로 조용히 넘긴다.
        """
        if not hasattr(self.win, "right_fields"):
            return
        eps = getattr(self.win, "_gallery_selected", []) or []
        ep = eps[0] if eps else None
        path = self.selected_file()
        self.fill_right_for(ep, path)
        self.update_dataset_panel(path)
        if self.win.session.stats and ep is not None and path is not None:
            self.win.stats_ops.refresh_rank_list()
            self.win.stats_ops.show_analysis_for(str(path), ep["name"])
        # Trim 탭도 고른 것을 문다. 하나만 골랐을 때만 무는 것은 그쪽 규칙이고,
        # 실패해도 선택 자체를 죽이지 않는다 (sync_trim_to_selection 참고).
        self.win.playback_ops.sync_trim_to_selection()


    # ------------------------------------------------------------------ select
    def on_set_verdict_success(self) -> None:
        """Set the selected episodes' verdict to success (not a toggle)."""
        self._apply_verdict(True)

    def on_set_verdict_failed(self) -> None:
        """Set the selected episodes' verdict to failed (not a toggle)."""
        self._apply_verdict(False)

    def _apply_verdict(self, success: bool) -> None:
        """Selection comes from the grid (gallery_ops.selected_keys) --
        that is the single source of truth for selection. After the
        verdict is written, redraw both the tree and the gallery."""
        by_file: dict = {}
        for path, name in self.win.gallery_ops.selected_keys():
            by_file.setdefault(Path(path), []).append(name)
        if not by_file:
            QMessageBox.information(self.win, tr("선택 필요"),
                                    tr("판정할 에피소드를 선택하세요."))
            return
        if self.set_verdict(by_file, success):
            self.refresh_dataset_tree()
            self.win.gallery_ops.refresh_gallery()

    def set_verdict(self, by_file: dict, success: bool) -> bool:
        """Shared verdict core -- SETS the given value (does not flip).

        With many episodes selected the outcome must be predictable, so
        the success argument decides regardless of the current value.
        Ownership rule is the same as deletion: when the session holds
        the file, the command goes through the saver queue
        (cmd_set_episode_success) instead of reopening the HDF5; the
        verdict is read from the active_episode_cache the saver
        maintains. States other than success/failed (bad_data etc.)
        are left untouched.
        """
        busy = self.busy_reason()
        if busy:
            QMessageBox.warning(self.win, tr("판정 불가"),
                                tr("{job}이(가) 진행 중입니다.").format(job=busy))
            return False
        new_q = "success" if success else "failed"
        set_n = skipped_state = skipped_cache = 0
        cache: dict[str, dict] = {}
        if self.win.session.active_file_path is not None and self.win.session.active_episode_cache is not None:
            cache = {e["name"]: e for e in self.win.session.active_episode_cache}
        for path, names in by_file.items():
            owned = self.win.session.active_file_path is not None and path == self.win.session.active_file_path
            try:
                if owned:
                    # Session-owned file: read from cache, write via the saver queue.
                    for name in names:
                        e = cache.get(name)
                        if e is None:
                            skipped_cache += 1
                            self.win.log(f"[판정] {path.name} / {name}: 캐시에 없어 건너뜀")
                            continue
                        q = str(e.get("quality_status", ""))
                        if "quality_status" not in e:
                            # Cache summary without quality_status: fall back to success.
                            s = e.get("success")
                            if s is True:
                                q = "success"
                            elif s is False:
                                q = "failed"
                        if q not in ("success", "failed"):
                            skipped_state += 1
                            continue
                        self.win.worker.cmd_set_episode_success(name, success)
                        set_n += 1
                else:
                    # Not owned. Callers (verdict buttons, Dataset menu) only
                    # pass scene files, so there is deliberately no legacy branch.
                    with h5py.File(path, "a") as f:
                        for name in names:
                            q = str(f[name].attrs.get("quality_status", ""))
                            if q not in ("success", "failed"):
                                skipped_state += 1
                                continue
                            f[name].attrs["quality_status"] = new_q
                            f[name].attrs["success"] = success
                            set_n += 1
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self.win, tr("판정 실패"),
                                     f"{path.name}\n{type(e).__name__}: {e}")
                return False
        parts = [f"[판정] {set_n} -> {new_q}"]
        if skipped_state:
            parts.append(f"{skipped_state}개 건너뜀 (success/failed 아님)")
        if skipped_cache:
            parts.append(f"{skipped_cache}개 건너뜀 (세션 캐시에 없음)")
        self.win.log(", ".join(parts))
        # A changed verdict changes the red (failed) marks in the rank list too.
        if set_n:
            self.win.stats_ops.mark_stats_stale()
        return True

    # ------------------------------------------------------------------ delete
    def on_delete_selected(self) -> None:
        """고른 에피소드를 삭제 목록에 넣는다 (지우지 않는다).

        선택은 **selected_keys() 하나로만** 읽는다 -- 목록 뷰와 격자가 같은
        것을 가리키므로 어느 쪽에서 골랐든 같고, 통로가 여럿이면 격자를 바꿀
        때 그 수만큼 고쳐야 한다 (gallery/ops.selected_keys 의 선언).
        """
        picks = self.win.gallery_ops.selected_keys()
        if not picks:
            QMessageBox.information(
                self.win, tr("선택 필요"),
                tr("삭제 목록에 넣을 에피소드를 선택하세요 (Ctrl/Shift로 여러 개)."))
            return
        for key in picks:
            self.win.basket.add(key)      # CurationBasket._norm 이 경로를 맞춘다
        self.refresh_basket_ui()
        self.win.gallery_grid.refresh_marks()

    def refresh_basket_ui(self) -> None:
        """삭제 목록의 개수를 화면에 반영한다.

        실행 버튼에 **개수를 박아 둔다** (``Delete 3``). 확인창을 열기 전에
        몇 개가 날아가는지 보여야 한다 -- 표시는 여러 화면에서 하고 실행은
        여기 하나라, 누르는 시점에는 무엇을 표시했는지 기억이 흐려져 있다.
        """
        if not hasattr(self.win, "basket_label"):
            return
        n = len(self.win.basket)
        # 파일은 **selected_file() 하나로만** 읽는다 (이 파일이 스스로 정한
        # 규칙인데 여기만 콤보를 직접 읽고 있었다 -- kimi 구조 감사 2026-09-12).
        path = self.selected_file()
        here = self.win.basket.count_for(path) if path else 0
        if not n:
            self.win.basket_label.setText("")
        else:
            msg = tr("삭제 목록 {n}개").format(n=n)
            if here and here != n:
                msg += tr(" (이 씬 {k}개)").format(k=here)
            self.win.basket_label.setText(msg)
        if hasattr(self.win, "basket_exec_btn"):
            self.win.basket_exec_btn.setText(
                tr("Delete {n}").format(n=n) if n else tr("Delete"))
            self.win.basket_exec_btn.setEnabled(n > 0)

    def on_clear_marks(self) -> None:
        """삭제 목록 표시를 전부 해제한다 (에피소드는 지우지 않는다)."""
        self.win.basket.clear()
        self.refresh_basket_ui()
        if hasattr(self.win, "gallery_grid"):
            self.win.gallery_grid.refresh_marks()

    def on_delete_marked(self) -> None:
        """삭제 목록에 넣은 에피소드를 한 번에 지운다 -- 삭제로 가는 유일한 문.

        삭제마다 번호가 다시 매겨지므로(파생 캐시 무효화 포함) 모아서 한 번
        지운다. 조작자가 확인창에서 취소하면 장바구니는 그대로 둔다.
        """
        by_file = self.win.basket.by_file()
        if not by_file:
            QMessageBox.information(self.win, tr("삭제 목록 비어 있음"),
                                    tr("삭제 목록에 넣은 에피소드가 없습니다. "
                                       "격자·트리·순위표에서 먼저 표시하세요."))
            return
        if self.delete_episodes(by_file):
            for path in list(by_file):
                self.win.basket.drop_file(path)
            self.refresh_dataset_tree()
            self.refresh_basket_ui()
            self.win.gallery_ops.refresh_gallery()

    def describe_delete_targets(self, by_file: dict):
        """삭제 확인창용: (행 목록, 성공 개수, Hub 안내문). 파일을 읽지 못하면
        (세션이 쥔 파일 등) 캐시로 대신하고, 그것도 없으면 이름만 나열한다."""
        rows: list = []
        n_success = 0
        tasks: set = set()
        uids: set = set()
        for path, names in by_file.items():
            eps: dict = {}
            try:
                if path.name.startswith("scene_"):
                    src = (self.win.session.active_episode_cache
                           if (self.win.session.active_file_path is not None
                               and path == self.win.session.active_file_path)
                           else list_scene_episodes(path)) or []
                    eps = {e["name"]: e for e in src}
                else:
                    with h5py.File(path, "r") as f:
                        data = f["data"]
                        for n in names:
                            if n in data:
                                g = data[n]
                                ok = g.attrs.get("success", True)
                                eps[n] = {"episode_uid": n, "instruction": "",
                                          "quality_status": "success" if ok else "failed",
                                          "num_samples": int(g.attrs.get("num_samples", 0))}
            except Exception:  # noqa: BLE001 -- 잠금 등: 이름만
                eps = {}
            for n in names:
                e = eps.get(n)
                if e is None:
                    rows.append(f"  {path.name} / {n}")
                    continue
                q = str(e.get("quality_status", "?"))
                if q == "success":
                    n_success += 1
                instr = str(e.get("instruction", ""))
                if instr:
                    tasks.add(instr)
                if path.name.startswith("scene_") and e.get("episode_uid"):
                    uids.add(str(e["episode_uid"]))
                rows.append(f"  {e.get('episode_uid', n)}  [{q}]  {e.get('num_samples', '?')}f"
                            + (f"  {instr[:40]}" if instr else ""))
        hub_note = ""
        try:
            repo = self.win.upload.repo_id_for("repo_id")
        except Exception:  # noqa: BLE001
            repo = ""
        if repo and (tasks or uids) and not repo_id_error(repo):
            # 판정 단위는 에피소드(uid)다. Hub 의 meta/episode_uids.json 사이드카에
            # 지울 uid 가 있을 때만 "올라가 있다" 고 말한다. 사이드카가 없는 repo
            # (legacy 수집분만 있는 데이터셋)는 에피소드 단위 판정이 불가능하므로
            # 문장(task) 단위 일치를 '참고' 로만 표시한다 -- 같은 문장의 legacy
            # 에피소드가 있다고 이 에피소드가 올라간 것은 아니다 (실사용 혼란).
            try:
                from mstack.scene.dataset_sync import hub_episode_uids, hub_meta

                hub_uids, err = hub_episode_uids(repo)
                if err:
                    hub_note = ""
                elif hub_uids is not None:
                    hit = sorted(uids & hub_uids)
                    if hit:
                        hub_note = tr("Hub({r})에 이 에피소드 {k}개가 이미 올라가 "
                                      "있습니다 ({u}{more}) — 다음 전체 처리에서 "
                                      "'삭제됨' 으로 잡혀 재빌드(교체)가 필요합니다.")\
                            .format(r=repo, k=len(hit), u=", ".join(hit[:3]),
                                    more=" …" if len(hit) > 3 else "")
                    else:
                        hub_note = tr("Hub({r})에는 이 에피소드가 올라가 있지 않습니다 "
                                      "(uid 대조).").format(r=repo)
                else:
                    hub, _lens, err2 = hub_meta(repo)
                    if not err2:
                        same = [t for t in tasks if hub.get(t, 0) > 0]
                        if same:
                            hub_note = tr("참고: Hub({r})에는 uid 사이드카가 없어 에피소드 "
                                          "단위 확인이 안 됩니다. 같은 문장의 task {k}개가 "
                                          "있지만(legacy 수집분일 수 있음) 이 에피소드가 "
                                          "올라갔다는 뜻은 아닙니다.").format(r=repo, k=len(same))
            except Exception:  # noqa: BLE001 -- 오프라인 등: 안내 생략
                hub_note = ""
        return rows, n_success, hub_note

    def delete_episodes(self, by_file: dict) -> bool:
        """공용 삭제 경로. Dataset 패널과 Analysis 순위표가 같은 것을 쓴다 --
        세션 소유 검사와 실행 중 작업 검사를 두 벌로 두면 반드시 갈라진다."""
        busy = self.busy_reason()
        if busy:
            QMessageBox.warning(self.win, tr("삭제 불가"),
                                tr("{job}이(가) 진행 중입니다. 끝난 뒤 삭제하세요.").format(job=busy))
            return False
        total = sum(len(v) for v in by_file.values())
        # 확인창: 무엇을 지우는지(uid·문장·판정·프레임) 목록으로 보여주고,
        # 성공분이 섞였으면 경고, Hub 에 이미 올라간 에피소드면 재빌드 안내.
        # "실패만 선택" 으로 고른 정상 경로에서는 경고가 뜨지 않는다 -- 손으로
        # 잘못 고른 성공분만 눈에 띄게 하는 것이 목적이다.
        rows, n_success, hub_note = self.describe_delete_targets(by_file)
        detail = "\n".join(rows[:30]) + ("\n  …" if len(rows) > 30 else "")
        notes = [tr("삭제 후 남은 에피소드는 번호가 다시 매겨집니다 (scene 은 지시문별 E번호·uid 도).")]
        if hub_note:
            notes.append(hub_note)
        notes.append(tr("파일 크기는 줄지 않습니다 (재압축 필요). 되돌릴 수 없습니다."))
        title = tr("에피소드 삭제")
        body = tr("에피소드 {n}개를 삭제합니다.\n\n{d}\n\n{notes}").format(
            n=total, d=detail, notes="\n".join(notes))
        if n_success:
            body = tr("⚠ 성공(success) 에피소드 {k}개가 포함되어 있습니다 — "
                      "정말 의도한 선택인지 확인하세요.\n\n").format(k=n_success) + body
            if QMessageBox.warning(
                    self.win, title, body,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return False
        elif QMessageBox.question(self.win, title, body) != QMessageBox.StandardButton.Yes:
            return False

        for path, names in by_file.items():
            owned = self.win.session.active_file_path is not None and path == self.win.session.active_file_path
            is_scene = path.name.startswith("scene_")
            if owned:
                # 세션이 파일을 쥐고 있으면 saver 스레드가 유일한 통로다. 매 삭제
                # 뒤 번호가 다시 매겨지므로 뒤에서부터 지워야 앞 이름이 안 밀린다.
                for name in sorted(names, key=lambda s: int(s.split("_")[1]), reverse=True):
                    self.win.worker.cmd_delete_episode(name)
                self.win.log(f"[삭제] {path.name}: {len(names)}개 요청 (세션 경유)")
                if is_scene:
                    # saver 가 삭제를 1건 완료할 때마다 episode_list_changed ->
                    # on_episode_list 가 카운터를 줄이며 썸네일을 지운다.
                    self.win._pending_scene_deletes += len(names)
                continue
            try:
                if is_scene:
                    delete_scene_episodes(path, names)
                    # renumber 로 uid 가 재배정되므로 해당 scene 의 파생 캐시를
                    # (프록시 클립) 전부 무효화한다. 삭제와 별도 try --
                    # 캐시 정리 실패가 "삭제 실패" 로 오표기되면 안 된다
                    # (삭제는 이미 성공했다).
                    try:
                        sid = read_scene_metadata(path).scene_id
                        c = invalidate_scene_caches(sid, self.dataset_root())
                        if c["proxies"]:
                            self.win.log(
                                f"[캐시] {path.name}: 프록시 {c['proxies']}개 무효화")
                    except Exception as e:  # noqa: BLE001
                        self.win.log(f"[캐시 정리 실패] {path.name}: {e}")
                else:
                    with h5py.File(path, "a") as f:
                        data = f["data"]
                        missing = [n for n in names if n not in data]
                        if missing:
                            raise KeyError(", ".join(missing))
                        for name in names:
                            del data[name]
                        renumber_episodes(data)
                self.win.log(f"[삭제] {path.name}: {len(names)}개 ({', '.join(sorted(names))})")
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self.win, tr("삭제 실패"), f"{path.name}\n{type(e).__name__}: {e}")
                self.win.log(f"[삭제 실패] {path.name}: {type(e).__name__}: {e}")
        self.win.collection.refresh_instruction()
        # 분석 통계는 파일에서 파생된다 -- 지운 에피소드가 순위표에 남아
        # 있으면 그 줄을 눌렀을 때 없는 것을 재생하려 든다.
        self.win.stats_ops.mark_stats_stale()
        return True

    def on_build_proxies(self) -> None:
        """프록시 클립 만들기 대화상자를 연다 (큐레이션 그리드의 전제).

        원본은 읽기만 하지만, 업로드·재압축·변환이 .hdf5 를 쥐고 있으면 목록을
        잠근다 -- 어느 파일인지까지는 알 수 없으므로 전부 잠근다. 남의 작업을
        방해하는 것보다 기다리는 쪽이 싸다 (2026-09-10 에 업로드가 scene_021 을
        열고 있었다).
        """
        from apps.workspace.features.dataset.proxy_dialog import ProxyBuildDialog

        paths = sorted(self.dataset_root().glob("**/*.hdf5"))
        if not paths:
            QMessageBox.information(
                self.win, tr("파일 없음"),
                tr("{r} 에 .hdf5 가 없습니다.").format(r=self.dataset_root()))
            return
        ProxyBuildDialog(self.win, paths, self.busy_reason(),
                         self.win.gallery_ops.proxy_dir()).exec()

    def on_delete_file(self) -> None:
        """Deletes a whole <task>_demo.hdf5. Never offered for the file a
        session is writing into -- that one is closed by ending the session."""
        path = self.selected_file()
        if path is None:
            QMessageBox.information(self.win, tr("선택 필요"), tr("삭제할 파일을 선택하세요."))
            return
        if self.win.session.active_file_path is not None and path == self.win.session.active_file_path:
            QMessageBox.warning(self.win, tr("삭제 불가"),
                                tr("지금 수집 중인 파일입니다. 먼저 세션을 종료하세요."))
            return
        busy = self.busy_reason()
        if busy:
            QMessageBox.warning(self.win, tr("삭제 불가"),
                                tr("{job}이(가) 진행 중입니다. 끝난 뒤 삭제하세요.").format(job=busy))
            return
        st = hdf5_repack_status(path)
        # 진짜 삭제한다. 오클릭 대책은 되돌리기가 아니라 닿기 어렵게 두는 것
        # (이 항목은 Dataset 메뉴에만 있다) -- 반쯤 지워진 채 디스크만 차지하는
        # 휴지통은 결국 아뮼브 비우지 않는다.
        # 이 파일의 에피소드가 Hub 에 이미 있으면 다음 전체 처리가 '삭제됨' 으로
        # 잡아 재빌드(교체)를 요구한다 -- 지금 지우는 것이 리모트에 어떤 결과를
        # 낳는지 삭제 순간에 알린다 (오프라인이면 안내 생략).
        hub_line = tr("Hub 에 올린 사본은 지금은 그대로지만, 다음 전체 처리 때 "
                      "로컬 기준으로 재빌드되어 교체됩니다.")
        if path.name.startswith("scene_"):
            try:
                names = [e["name"] for e in list_scene_episodes(path)]
                _rows, _n_ok, note = self.describe_delete_targets({path: names})
                if note:
                    hub_line = note
            except Exception:  # noqa: BLE001 -- 잠금/오프라인: 기본 안내
                pass
        confirm = QMessageBox.warning(
            self.win, tr("파일 삭제"),
            tr("{f}\n\n에피소드 {n}개, {mb:.1f} MB 를 완전히 삭제합니다.\n"
               "되돌릴 수 없습니다.\n{h}").format(
                   f=path.name, n=st["episodes"], mb=st["size"] / 1e6, h=hub_line),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink()
            self.win.log(f"[파일 삭제] {path.name} ({st['episodes']}개 에피소드, "
                     f"{st['size'] / 1e6:.1f} MB)")
        except OSError as e:
            QMessageBox.critical(self.win, tr("삭제 실패"), str(e))
            self.win.log(f"[파일 삭제 실패] {path.name}: {e}")
        self.win.stats_ops.mark_stats_stale()
        self.refresh_dataset_tree()

    def update_dataset_panel(self, path: "Path | None" = None) -> None:
        """Fills the right panel's Dataset box.

        During a session it describes what this session is writing (from the
        config, since the file's own attrs only exist after the first save).
        Otherwise it describes whichever file is selected in the tree, read
        straight off disk -- so 'what format is this old file?' is answerable
        without opening the schema dialog or connecting anything.
        """
        f = self.win.right_fields
        if self.win.worker is not None:
            cfg = self.win.worker.cfg
            name = (tr("(기록 안 함)") if self.win.session.no_dataset_session
                    else Path(str(self.win.session.active_file_path or "-")).name)
            f["ds_file"].setText(soft_wrap(name))
            f["ds_file"].setToolTip(name)
            # 연결 시점 설정이 아니라 '지금' slot 을 보여준다 -- scene 세션은
            # Disconnect 없이 slot(문장·ID)을 바꾸므로(cmd_set_slot) 설정값만
            # 보여주면 전환 뒤에도 첫 문장이 그대로 남는다 (실사용 보고).
            cur_instr = getattr(self.win.worker, "_slot_instruction", None) \
                or cfg.language_instruction or cfg.task_name
            cur_iid = getattr(self.win.worker, "_slot_instruction_id", "") or cfg.instruction_id
            task_text = f"{cur_iid}: {cur_instr}" if (self.win.session.scene_session and cur_iid) \
                else cur_instr
            f["ds_task"].setText(task_text)
            f["ds_task"].setToolTip(task_text)
            # 저장은 백그라운드라 episode_list_changed가 몇 초 늦게 온다. 그걸
            # 기다리면 방금 저장한 것이 한동안 안 세어져 "지금 몇 개째인지"를
            # 알 수 없다. 연결 시점 개수 + 이번 세션 저장 수로 즉시 계산하고,
            # 목록이 도착하면 그 값이 더 정확하므로 그쪽을 쓴다.
            listed = len(self.win.session.active_episode_cache or [])
            counted = self.win.session.episodes_at_connect + self.win.session.counters["saved"]
            total = max(listed, counted)
            f["ds_episodes"].setText(
                tr("{t}개  (이번 +{s})").format(t=total, s=self.win.session.counters["saved"]))
            f["ds_action"].setText(cfg.schema.action_space)
            f["ds_gripper"].setText(
                "0/1 (obs와 동일)" if cfg.schema.gripper_action_match_obs else "-1/+1")
            f["ds_image"].setText(f"{cfg.schema.image_size}²" if cfg.schema.image_size
                                  else tr("원본 해상도"))
            f["ds_fps"].setText(str(cfg.fps))
            # 이 세션이 찍는 버전. 파일에서 읽지 않는 이유는 기록기가 쓰는
            # 값이 정본이기 때문이다 -- 첫 저장 전에는 파일에 아직 없고,
            # 이어찍기로 승격된 파일은 다음 저장에서야 새 값이 찍힌다.
            f["ds_schema"].setText(self.win.schema_version)
            f["ds_repack"].setText("-")
            return

        if path is None or not Path(path).exists():
            for k in ("ds_file", "ds_task", "ds_episodes", "ds_schema",
                      "ds_action", "ds_gripper", "ds_image", "ds_fps",
                      "ds_repack"):
                f[k].setText("-")
            return

        path = Path(path)
        st = hdf5_repack_status(path)
        f["ds_file"].setText(soft_wrap(path.name))
        f["ds_file"].setToolTip(str(path))
        f["ds_episodes"].setText(f"{st['episodes']}  ({st['size'] / 1e6:.0f} MB)")
        f["ds_repack"].setText(
            tr("혼합 — 다시 필요") if st["mixed"]
            else (st["marker"] or (tr("완료") if st["repacked"] else tr("안 됨"))))
        task = action = gripper = image = "-"
        schema = "-"
        try:
            with h5py.File(path, "r") as h:
                if "data" in h:
                    data = h["data"]
                    info = data.attrs.get("problem_info")
                    if info:
                        try:
                            task = json.loads(json.loads(info)["language_instruction"])
                        except Exception:  # noqa: BLE001
                            task = str(info)[:60]
                    names = sorted(data.keys(), key=lambda s: int(s.split("_")[1]))
                    container = data
                    # legacy *_demo.hdf5 에는 버전 attr 이 없다. "-" 로 두면
                    # "못 읽었다"와 구별이 안 되므로 이름을 붙여 준다 --
                    # 버전 체계가 생기기 전 파일이라는 것이 사실이다.
                    schema = tr("legacy (버전 이전)")
                else:
                    # scene-v1: task 는 파일 단위 개념이 아니다 -- scene ID 로 표기
                    task = "scene " + str(h["metadata"].attrs.get("scene_id", "?"))
                    # 옛 표기(scene-v1)도 SemVer 로 풀어서 보여준다 -- 읽는
                    # 사람이 어느 시절 표기인지 몰라도 되게 (scene_format 의
                    # _read_metadata 와 같은 규칙).
                    schema = normalize_schema_version(
                        h["metadata"].attrs.get("dataset_version", "")) or "-"
                    names = sorted((k for k in h.keys() if k.startswith("episode_")),
                                   key=lambda s: int(s.split("_")[1]))
                    container = h
                if names:
                    g = container[names[0]]
                    action = str(g.attrs.get("action_space", "-"))
                    conv = str(g.attrs.get("gripper_action_convention", ""))
                    gripper = {"01": "0/1 (obs와 동일)", "pm1": "-1/+1"}.get(conv, conv or "-")
                    rgb = g.get("obs", {}).get(OBS_AGENTVIEW_RGB)
                    if rgb is not None and rgb.ndim == 4:
                        image = f"{rgb.shape[1]}×{rgb.shape[2]}"
        except Exception as e:  # noqa: BLE001
            task = f"({type(e).__name__})"
        f["ds_task"].setText(task)
        f["ds_task"].setToolTip(task)
        f["ds_schema"].setText(schema)
        f["ds_action"].setText(action)
        f["ds_gripper"].setText(gripper)
        f["ds_image"].setText(image)
        f["ds_fps"].setText("-")

    # ------------------------------------------------------- 우측 패널
    def fill_right_for(self, ep, path) -> None:
        """고른 에피소드의 값과 그 scene 의 배치를 우측에 편다.

        예전에는 트리 아이템을 받아 ``item.text(1)`` 처럼 **화면에 그려진 글자를
        도로 읽었다.** 목록의 열이 바뀌면 조용히 어긋나는 구조였고, 실제로 열을
        줄이면서 깨졌다. 이제 에피소드 dict 를 그대로 받는다 -- 목록·격자·통계가
        모두 쓰는 그 모양이라 번역 계층이 필요 없고, 화면에 없는 값(지시문 문장,
        수집자)도 파일을 다시 열지 않고 읽을 수 있다.
        """
        win = self.win
        card = getattr(win, "ds_episode_card", None)
        if card is None:
            return
        if ep is None:
            card.set_fields([(tr("에피소드"), tr("미선택"))])
            self._fill_scene_box(str(path) if path else None)
            return
        q = ep.get("quality_status") or (
            "-" if ep.get("success") is None
            else ("success" if ep["success"] else "failed"))
        fields = [
            (tr("파일"), Path(path).name if path else ""),
            (tr("에피소드"), str(ep.get("name", ""))),
            (tr("uid"), str(ep.get("episode_uid", ""))),
            (tr("프레임"), str(ep.get("num_samples", ""))),
            (tr("결과"), q),
            (tr("수집자"), str(ep.get("collector", ""))),
            (tr("지시문"), str(ep.get("instruction_id", ""))),
            (tr("문장"), str(ep.get("instruction", ""))),
            (tr("시각"), str(ep.get("timestamp", ""))),
        ]
        card.set_fields(fields)
        self._fill_scene_box(str(path) if path else None)


    def _fill_scene_box(self, path) -> None:
        """고른 scene 의 배치도와 기준 사진. 세션과 무관하다."""
        win = self.win
        card = getattr(win, "ds_scene_card", None)
        if card is None:
            return
        if not path:
            card.set_fields([(tr("Scene"), tr("미선택"))])
            win.ds_scene_zones.set_layout_spec(None)
            win.ds_scene_photo.clear()
            win.ds_scene_photo.setText(tr("기준 사진 없음"))
            return
        try:
            md = read_scene_metadata(Path(path))
        except Exception:  # noqa: BLE001 -- legacy 파일에는 scene metadata 가 없다
            card.set_fields([(tr("Scene"), tr("scene 파일이 아닙니다"))])
            win.ds_scene_zones.set_layout_spec(None)
            win.ds_scene_photo.clear()
            win.ds_scene_photo.setText(tr("기준 사진 없음"))
            return
        card.set_fields(scene_fields(md))
        win.ds_scene_zones.set_layout_spec(md.layout)
        try:
            img = read_reference_image(Path(path))
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            win.ds_scene_photo.clear()
            win.ds_scene_photo.setText(tr("기준 사진 없음"))
        else:
            win.ds_scene_photo.setText("")
            win.ds_scene_photo.setPixmap(np_to_pixmap(img).scaledToWidth(
                PHOTO_W, Qt.TransformationMode.SmoothTransformation))
