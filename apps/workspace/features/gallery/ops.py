"""Gallery tab: scene episode thumbnails, filter, and activation."""

from __future__ import annotations

from pathlib import Path

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
        # instruction 필터 항목 재구성 (선택 유지)
        cur = self.win.gallery_filter_combo.currentData()
        self.win.gallery_filter_combo.blockSignals(True)
        self.win.gallery_filter_combo.clear()
        self.win.gallery_filter_combo.addItem(tr("(모든 instruction)"), None)
        for iid, instr in sorted({(e["instruction_id"], e["instruction"])
                                  for e in episodes}):
            self.win.gallery_filter_combo.addItem(f"{iid} · {instr[:44]}", iid)
        idx = self.win.gallery_filter_combo.findData(cur)
        self.win.gallery_filter_combo.setCurrentIndex(max(0, idx))
        self.win.gallery_filter_combo.blockSignals(False)
        self._ref_thumb = ref_thumb
        self.apply_gallery_filter()

    def apply_gallery_filter(self, *_args) -> None:
        """지시문·길이로 목록을 줄여 격자에 넘긴다.

        큐레이션은 **먼저 줄이고 그 다음에 본다.** 60개를 한꺼번에 훑는 것이
        아니라 (씬 → 지시문 → 길이) 로 좁혀 12칸에 담기게 만든 뒤, 그 12개를
        나란히 돌려 이상한 것을 고른다.
        """
        from apps.workspace.features.gallery.tab import SHORT_FRAMES

        want = self.win.gallery_filter_combo.currentData()
        mode = self.win.gallery_len_combo.currentData()
        eps = self.win._gallery_episodes
        lens = sorted(e.get("num_samples", 0) for e in eps)
        # "긴 것" 은 고정값이 아니라 이 씬의 분포로 정한다 -- 작업마다 정상
        # 길이가 다르므로 절대 프레임 수로 자르면 어떤 씬에서는 전부 걸린다.
        long_cut = lens[int(len(lens) * 0.8)] if lens else 0

        shown = []
        for e in eps:
            if want is not None and e["instruction_id"] != want:
                continue
            n = e.get("num_samples", 0)
            if mode == "short" and n > SHORT_FRAMES:
                continue
            if mode == "long" and n <= long_cut:
                continue
            if mode == "failed" and e.get("quality_status") == "success":
                continue
            shown.append(e)

        self.win._gallery_shown = shown
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

    @staticmethod
    def _has_clip(e) -> bool:
        from mstack.data.proxy_clip import proxy_path

        uid = e.get("episode_uid", "")
        return bool(uid) and proxy_path(uid, "agentview_rgb").exists()

    # ------------------------------------------------------------------ 재생
    def toggle_play(self) -> None:
        g = self.win.gallery_grid
        if g.playing:
            g.stop()
            self.win.gallery_play_btn.setText(tr("▶ 재생"))
        else:
            g.start()
            self.win.gallery_play_btn.setText(tr("■ 정지"))

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
