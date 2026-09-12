"""Gallery tab: scene episode clips, filter, and activation."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidgetItem

from apps.workspace.shared.busy import set_busy
from apps.workspace.shared.tabs import show_center_tab
from mstack.gui.i18n import tr
from mstack.gui.workers import GalleryLoadWorker
from mstack.scene.scene_format import iter_scene_files


class GalleryOps:
    """Gallery tab: scene episode clips, filter, and activation."""

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
            set_busy(self.win)
            self.win.gallery_status.setText(tr("표시할 scene 파일이 없습니다"))
            return
        if self.win.session.active_file_path is not None and Path(path) == self.win.session.active_file_path:
            # HDF5 잠금 -- 실패한 로드 대신 이유와 다음 행동을 말한다
            self.win.gallery_status.setText(tr(
                "수집 세션이 이 scene 파일을 사용 중입니다 — 세션을 종료하면 "
                "갤러리가 열립니다. (현황은 Collect 페이지 slot 패널에)"))
            return
        self.win.gallery_status.setText(tr("불러오는 중..."))
        set_busy(self.win, tr("{f} 에피소드 목록을 읽는 중").format(f=Path(path).name))
        if self.win._gallery_loader is not None:
            self.win._gallery_loader.wait()
        self.win._gallery_loader = GalleryLoadWorker(path)
        self.win._gallery_loader.loaded.connect(self.on_gallery_loaded)
        self.win._gallery_loader.failed.connect(
            lambda m: (set_busy(self.win),
                       self.win.gallery_status.setText(
                           tr("갤러리 로드 실패: {m}").format(m=m))))
        self.win._gallery_loader.start()

    def on_gallery_loaded(self, path, episodes, _unused=None) -> None:
        set_busy(self.win)
        if path != self.win.gallery_scene_combo.currentData():
            return  # 로드 중 scene 을 바꿨다
        self.win._gallery_episodes = episodes
        # instruction 목록 재구성 (선택 유지)
        lst = self.win.instruction_list
        cur = lst.currentItem()
        cur_iid = cur.data(0, Qt.ItemDataRole.UserRole) if cur else None
        counts: dict = {}
        for e in episodes:
            counts[e["instruction_id"]] = counts.get(e["instruction_id"], 0) + 1
        lst.blockSignals(True)
        try:
            lst.clear()
            first = QTreeWidgetItem([tr("(all instructions)"),
                                     str(len(episodes))])
            first.setData(0, Qt.ItemDataRole.UserRole, None)
            lst.addTopLevelItem(first)
            for iid, instr in sorted({(e["instruction_id"], e["instruction"])
                                      for e in episodes}):
                it = QTreeWidgetItem([f"{iid}  {instr[:44]}", str(counts[iid])])
                it.setData(0, Qt.ItemDataRole.UserRole, iid)
                it.setToolTip(0, instr)
                lst.addTopLevelItem(it)
            for row in range(lst.topLevelItemCount()):
                it = lst.topLevelItem(row)
                if it.data(0, Qt.ItemDataRole.UserRole) == cur_iid:
                    lst.setCurrentItem(it)
                    break
            else:
                lst.setCurrentItem(lst.topLevelItem(0))
        finally:
            lst.blockSignals(False)
        # 데려가기 요청이 걸려 있으면 그 지시문을 고른다. setCurrentItem 이
        # apply_gallery_filter 를 부르므로 아래를 두 번 돌지 않게 여기서 끝낸다.
        if self._focus_instruction():
            self.win.stats_ops.on_select_flagged()
            return
        self.apply_gallery_filter()

    def apply_gallery_filter(self, *_args) -> None:
        """지시문으로 목록을 줄여 격자에 넘긴다.

        큐레이션은 **먼저 줄이고 그 다음에 본다.** 60개를 한꺼번에 훑는 것이
        아니라 (씬 → 지시문) 으로 좁혀 12칸에 담기게 만든 뒤, 그 12개를
        나란히 돌려 이상한 것을 고른다.
        """
        it = self.win.instruction_list.currentItem()
        want = it.data(0, Qt.ItemDataRole.UserRole) if it else None
        eps = self.win._gallery_episodes

        shown = [e for e in eps
                 if want is None or e["instruction_id"] == want]

        self.win._gallery_shown = shown
        # Analysis 도 이 범위를 따른다. 여기서 불러야 하는 이유: 지시문 목록과
        # Scene 콤보 변경은 **둘 다 이 함수로만** 들어오고, on_selection_changed
        # 는 에피소드를 클릭했을 때만 불린다 -- 거기에만 걸면 지시문을 바꿔도
        # 순위표가 옛것을 그대로 보여준다 (kimi 조사, 2026-09-12).
        if self.win.session.stats:
            self.win.stats_ops.refresh_rank_list()
        # 좁힌 집합이 바뀌면 선택은 무효다 -- 안 보이는 것이 선택된 채로
        # 남으면 판정·표시가 화면에 없는 에피소드에 걸린다.
        self.win._gallery_selected = []
        self.win.dataset_ops.refresh_episode_list()
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
    # ------------------------------------------------------------------ 선택
    #
    # 선택은 **창이 들고 두 뷰가 같이 그린다.** 목록 뷰(왼쪽 트리)와 영상
    # 뷰(격자)가 같은 (씬·지시문) 집합을 보여주는데, 격자는 한 번에 12개만
    # 보여준다. 선택이 격자 안에 살면 쪽을 넘길 때마다 사라지고, 트리 안에
    # 살면 격자가 모른다. 어느 쪽에도 두지 않는 이유가 그것이다.
    #
    # _syncing 은 되먹임 방지다 -- 한쪽을 그리면 그 위젯이 또 신호를 내서
    # 맴돈다.

    def set_selection(self, episodes, source: str = "") -> None:
        """선택을 정하고 두 뷰에 반영한다. ``source`` 쪽은 다시 그리지 않는다."""
        if getattr(self, "_syncing", False):
            return
        self._syncing = True
        try:
            self.win._gallery_selected = [e for e in (episodes or []) if e]
            if source != "grid":
                self.win.gallery_grid.show_selection(self.win._gallery_selected)
            if source != "tree":
                self.win.dataset_ops.show_tree_selection(self.win._gallery_selected)
            self.win.dataset_ops.on_selection_changed()
        finally:
            self._syncing = False

    def on_grid_selection(self, episodes) -> None:
        self.set_selection(episodes, source="grid")

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
        path = self.win.dataset_ops.selected_file()   # 파일의 정본은 저쪽 하나
        if path is None:
            return []
        return [(str(path), e["name"]) for e in (self.win._gallery_selected or [])]

    def go_to_episode(self, path: str, demo: str) -> bool:
        """왼쪽 패널의 (씬 → 지시문) 을 그 에피소드가 보이는 자리로 옮긴다.

        Analysis 의 [튀는 것만 선택] 이 쓴다: 튄 것이 지금 목록에 없으면
        **거기로 데려간다** (조작자, 2026-09-12). 씬 파일 로드가 비동기라
        지시문 선택은 여기서 못 끝낸다 -- 원하는 에피소드를 적어 두고,
        ``on_gallery_loaded`` 가 목록을 다 만든 뒤에 집어 준다.

        콤보에 그 파일이 없으면(데이터 경로가 다른 폴더면) False.
        """
        cb = self.win.gallery_scene_combo
        idx = cb.findData(str(path))
        if idx < 0:
            return False
        self.win._focus_episode = (str(path), demo)
        if cb.currentIndex() == idx:
            self._focus_instruction()      # 같은 씬이면 지시문만 바꾼다
        else:
            cb.setCurrentIndex(idx)        # -> refresh_gallery -> on_gallery_loaded
        return True

    def _focus_instruction(self) -> bool:
        """적어 둔 에피소드의 지시문을 목록에서 고른다. 골랐으면 True."""
        want = getattr(self.win, "_focus_episode", None)
        if not want:
            return False
        path, demo = want
        if path != self.win.gallery_scene_combo.currentData():
            return False
        ep = next((e for e in self.win._gallery_episodes if e["name"] == demo), None)
        self.win._focus_episode = None
        if ep is None:
            return False
        lst = self.win.instruction_list
        for row in range(lst.topLevelItemCount()):
            it = lst.topLevelItem(row)
            if it.data(0, Qt.ItemDataRole.UserRole) == ep["instruction_id"]:
                lst.setCurrentItem(it)     # -> apply_gallery_filter
                return True
        return False

    def on_gallery_activated(self, ep) -> None:
        """타일을 더블클릭하면 **Trim 탭**에서 크게 본다.

        예전에는 Playback 탭을 열었다. 그 탭은 2026-09-12 에 없앴다 --
        한 에피소드를 크게 보는 자리는 Trim 하나로 모았다.
        """
        path = self.win.gallery_scene_combo.currentData()
        if path and ep:
            self.win.playback_ops.show_trim_for(path, ep["name"])
            show_center_tab(self.win, "trim")
