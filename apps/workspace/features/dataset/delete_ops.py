"""Deletion operations: basket display, confirmation, and execution."""

from __future__ import annotations

import h5py
from PyQt6.QtWidgets import QMessageBox

from apps.workspace.shared.caches import remap_caches_after_delete
from mstack.data.libero_format import hdf5_repack_status, renumber_episodes
from mstack.gui.i18n import tr
from apps.workspace.shared.jobs import running_job
from mstack.gui.text_utils import repo_id_error
from mstack.scene.scene_format import (
    delete_scene_episodes,
    list_scene_episodes,
)


class DeleteOps:
    """지우는 일만 -- 표시(장바구니)에서 실행·확인창까지.

    되돌릴 수 없는 조작이라 한 파일에 모아 둔다. 표시는 자유롭고 실행은
    하나이며(CurationBasket 참고), 확인창이 배치 전체를 보여주는 유일한
    검토 순간이다. 파일 통째 삭제(on_delete_file)는 에피소드 삭제와 **다른
    무게**라 메뉴에만 두지만, 확인창을 만드는 코드를 공유하므로 여기 있다.
    """

    def __init__(self, win) -> None:
        self.win = win

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
        path = self.win.dataset_ops.selected_file()
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
            self.win.dataset_ops.refresh_dataset_tree()
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
        busy = running_job(self.win)
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
                    deleted, moved = delete_scene_episodes(path, names)
                    # renumber 가 돌려준 uid 대응표로 캐시를 **맞춘다**: 지운
                    # 것의 클립만 지우고, 당겨진 것은 이름만 옮긴다. 예전에는
                    # 씬 통째를 버려 하나만 지워도 다시 구워야 했다 (2026-09-14).
                    # 삭제와 별도 try -- 캐시 정리 실패가 "삭제 실패" 로
                    # 오표기되면 안 된다 (삭제는 이미 성공했다).
                    remap_caches_after_delete(self.win, deleted, moved, path.name)
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
    def on_delete_file(self) -> None:
        """Deletes a whole <task>_demo.hdf5. Never offered for the file a
        session is writing into -- that one is closed by ending the session."""
        path = self.win.dataset_ops.selected_file()
        if path is None:
            QMessageBox.information(self.win, tr("선택 필요"), tr("삭제할 파일을 선택하세요."))
            return
        if self.win.session.active_file_path is not None and path == self.win.session.active_file_path:
            QMessageBox.warning(self.win, tr("삭제 불가"),
                                tr("지금 수집 중인 파일입니다. 먼저 세션을 종료하세요."))
            return
        busy = running_job(self.win)
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
        self.win.dataset_ops.refresh_dataset_tree()
