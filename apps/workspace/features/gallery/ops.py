"""Gallery tab: scene episode thumbnails, filter, and activation."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem

from mstack.gui.i18n import tr
from mstack.gui.workers import GalleryLoadWorker
from mstack.scene.scene_format import iter_scene_files


class GalleryOps:
    """Gallery tab: scene episode thumbnails, filter, and activation."""

    def __init__(self, win) -> None:
        self.win = win

    def refresh_gallery_scenes(self) -> None:
        combo = self.win.gallery_scene_combo
        cur = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        try:
            for p in iter_scene_files(self.win.dataset_ops.dataset_root()):
                combo.addItem(p.name, str(p))
        except Exception:  # noqa: BLE001
            pass
        idx = combo.findData(cur)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)
        self.refresh_gallery()

    def refresh_gallery(self, *_args) -> None:
        path = self.win.gallery_scene_combo.currentData()
        self.win.gallery_grid.set_episodes([])
        self.win._gallery_episodes = []
        self.win._gallery_shown = []
        self.win._gallery_selected = []
        if not path:
            self.win.gallery_status.setText(tr("표시할 scene 파일이 없습니다"))
            return
        if self.win.session.active_file_path is not None and Path(path) == self.win.session.active_file_path:
            # HDF5 잠금 -- 실패한 로드 대신 이유와 다음 행동을 말한다
            self.win.gallery_status.setText(tr(
                "수집 세션이 이 scene 파일을 사용 중입니다 — 세션을 종료하면 "
                "갤러리가 열립니다. (현황은 Collect 페이지 slot 패널에)"))
            return
        self.win.gallery_status.setText(tr("불러오는 중... (첫 로드는 썸네일 생성으로 수 초)"))
        if self.win._gallery_loader is not None:
            self.win._gallery_loader.wait()
        self.win._gallery_loader = GalleryLoadWorker(path)
        self.win._gallery_loader.loaded.connect(self.on_gallery_loaded)
        self.win._gallery_loader.failed.connect(
            lambda m: self.win.gallery_status.setText(tr("갤러리 로드 실패: {m}").format(m=m)))
        self.win._gallery_loader.start()

    def on_gallery_loaded(self, path, episodes, ref_thumb) -> None:
        if path != self.win.gallery_scene_combo.currentData():
            return  # 로드 중 scene 을 바꿨다
        self.win._gallery_episodes = episodes
        # instruction 목록 재구성 (선택 유지)
        lst = self.win.instruction_list
        cur = lst.currentItem()
        cur_iid = cur.data(Qt.ItemDataRole.UserRole) if cur else None
        counts: dict = {}
        for e in episodes:
            counts[e["instruction_id"]] = counts.get(e["instruction_id"], 0) + 1
        lst.blockSignals(True)
        try:
            lst.clear()
            first = QListWidgetItem(tr("(all instructions)"))
            first.setData(Qt.ItemDataRole.UserRole, None)
            lst.addItem(first)
            for iid, instr in sorted({(e["instruction_id"], e["instruction"])
                                      for e in episodes}):
                it = QListWidgetItem(f"{iid}  {instr[:40]}  {counts[iid]}")
                it.setData(Qt.ItemDataRole.UserRole, iid)
                lst.addItem(it)
            for row in range(lst.count()):
                it = lst.item(row)
                if it.data(Qt.ItemDataRole.UserRole) == cur_iid:
                    lst.setCurrentItem(it)
                    break
            else:
                lst.setCurrentItem(lst.item(0))
        finally:
            lst.blockSignals(False)
        self._ref_thumb = ref_thumb
        self.apply_gallery_filter()

    def apply_gallery_filter(self, *_args) -> None:
        """지시문으로 목록을 줄여 격자에 넘긴다.

        큐레이션은 **먼저 줄이고 그 다음에 본다.** 60개를 한꺼번에 훑는 것이
        아니라 (씬 → 지시문) 으로 좁혀 12칸에 담기게 만든 뒤, 그 12개를
        나란히 돌려 이상한 것을 고른다.
        """
        it = self.win.instruction_list.currentItem()
        want = it.data(Qt.ItemDataRole.UserRole) if it else None
        eps = self.win._gallery_episodes

        shown = [e for e in eps
                 if want is None or e["instruction_id"] == want]

        self.win._gallery_shown = shown
        self.win.gallery_grid.proxy_dir = self.proxy_dir()
        self.win.gallery_grid.set_episodes(
            shown, self.win.gallery_cam_combo.currentData())
        self._sync_pager()
        n_ok = sum(1 for e in eps if e["quality_status"] == "success")
        missing = sum(1 for e in shown if not self._has_clip(e))
        note = tr("  · 프록시 없음 {m}개 (Dataset 메뉴 → 프록시 클립 만들기)").format(
            m=missing) if missing else ""
        self.win.gallery_status.setText(
            tr("{s}개 표시 (전체 {n}개 · success {ok}개) — 클릭: 선택, "
               "Ctrl+클릭: 여러 개{note}").format(
                   s=len(shown), n=len(eps), ok=n_ok, note=note))

    def proxy_dir(self):
        """지금 데이터 경로의 프록시 디렉터리. 캐시 키에 데이터셋이 섞이면
        다른 데이터셋의 영상이 나온다 (proxy_clip.dataset_tag 참고)."""
        from mstack.data.proxy_clip import proxy_dir_for

        return proxy_dir_for(self.win.dataset_ops.dataset_root())

    def _has_clip(self, e) -> bool:
        from mstack.data.proxy_clip import proxy_path

        uid = e.get("episode_uid", "")
        return bool(uid) and proxy_path(uid, "agentview_rgb", self.proxy_dir()).exists()

    # ------------------------------------------------------------------ 재생
    def toggle_play(self) -> None:
        g = self.win.gallery_grid
        if g.playing:
            g.stop()
            self.win.gallery_play_btn.setText(tr("▶ Play"))
        else:
            mult = self.win.gallery_speed_combo.currentData() or 1.0
            g.start(20.0 * float(mult))
            self.win.gallery_play_btn.setText(tr("■ Stop"))

    def on_speed_changed(self, *_args) -> None:
        """배속 변경. 타이머 주기만 바꾼다 -- 프레임을 건너뛰지 않는다."""
        g = self.win.gallery_grid
        mult = self.win.gallery_speed_combo.currentData() or 1.0
        if g.playing:
            g.start(20.0 * float(mult))

    def rewind(self) -> None:
        self.win.gallery_grid.rewind_all()

    def on_camera_changed(self, *_args) -> None:
        self.win.gallery_grid.set_camera(
            self.win.gallery_cam_combo.currentData())

    # ------------------------------------------------------------------ 쪽
    def _sync_pager(self) -> None:
        g = self.win.gallery_grid
        sp = self.win.gallery_page_spin
        sp.blockSignals(True)
        sp.setRange(1, max(1, g.n_pages))
        sp.setValue(g.page + 1)
        sp.blockSignals(False)
        self.win.gallery_page_total.setText(f"/ {g.n_pages}")

    def on_page_changed(self, value: int) -> None:
        self.win.gallery_grid.set_page(value - 1)

    def step_page(self, delta: int) -> None:
        g = self.win.gallery_grid
        g.set_page(g.page + delta)
        self._sync_pager()

    # ------------------------------------------------------------------ 선택
    def on_grid_selection(self, episodes) -> None:
        self.win._gallery_selected = list(episodes)

    def toggle_mark(self) -> None:
        """선택한 에피소드를 삭제 목록에 넣거나 뺀다 (지우지는 않는다).

        전부 이미 표시되어 있으면 전부 해제하고, 아니면 전부 표시로 통일한다
        -- 토글이 제각각이면 격자에서 무엇이 표시됐는지 알 수 없다.
        """
        keys = self.selected_keys()
        if not keys:
            self.win.gallery_status.setText(tr("표시할 에피소드를 선택하세요"))
            return
        if all(k in self.win.basket for k in keys):
            for k in keys:
                self.win.basket.discard(k)
        else:
            for k in keys:
                self.win.basket.add(k)
        self.win.gallery_grid.refresh_marks()
        self.win.dataset_ops.refresh_basket_ui()

    def selected_keys(self) -> list:
        """선택을 ``(파일경로, 에피소드이름)`` 목록으로. 격자 밖(재판정·실로봇
        재생)에서 쓰는 유일한 통로다 -- 선택을 읽는 방법이 여럿이면 격자를
        바꿀 때마다 그 수만큼 고쳐야 한다."""
        path = self.win.gallery_scene_combo.currentData()
        if not path:
            return []
        return [(path, e["name"]) for e in (self.win._gallery_selected or [])]

    def on_gallery_activated(self, ep) -> None:
        """타일을 크게 보기 -- 기존 Playback 경로를 그대로 쓴다."""
        path = self.win.gallery_scene_combo.currentData()
        if path and ep:
            self.win.playback_ops.play_episode(path, ep["name"])
