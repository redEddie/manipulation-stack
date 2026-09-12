"""Trim, replay, and HDF5 structure operations for WorkspaceWindow."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from mstack.data.episode_stats import load_series
from mstack.data.episode_trim import plan_trim, suggest_trim, trim_tail
from mstack.gui.constants import PLAYBACK_FPS
from mstack.gui.workers import EpisodeLoadWorker
from mstack.gui.i18n import tr
from apps.workspace.shared.jobs import running_job
from mstack.gui.widgets.cut_slider import CUT_COLOR
from mstack.data.libero_format import hdf5_repack_status
from mstack.data.schema_description import describe_episode
from mstack.data.proxy_clip import episode_uid_at
from mstack.scene.scene_format import count_by_slot, read_scene_metadata

from apps.workspace.constants import REPLAY_SCRIPT
from apps.workspace.shared.caches import drop_episode_caches, drop_scene_caches
from apps.workspace.shared.info import InfoCard
from apps.workspace.shared.tabs import show_center_tab


class TrimOps:
    """Trim, robot replay, and HDF5 structure inspection."""

    def __init__(self, win) -> None:
        self.win = win
        self._trim_cut_shown = False     # 영상 테두리가 지금 빨간가
        #: 재생이 잘림 지점에서 **한 번** 선다. 그 자리에서 다시 누르면 이어
        #: 봐야 하므로, 한 번 선 뒤에는 내려간다. 슬라이더를 끌거나 자를 양을
        #: 바꾸면 다시 올라간다 (그때부터는 다른 지점이다).
        self._trim_cut_stop_armed = True

    # ------------------------------------------------------------------ open
    def on_open_trim(self) -> None:
        """메뉴·버튼 진입점.

        고르는 일은 ``sync_trim_to_selection`` 과 같지만, **탭까지 옮긴다.**
        sync 쪽은 선택이 바뀔 때마다 불려서 탭을 옮기면 안 되고(보고 있던
        화면을 뺏는다), 이쪽은 사람이 "끝 다듬기 (Trim 탭에서)" 를 직접
        누른 것이다. 2026-09-12 까지 여기서 탭을 안 옮겨서, 버튼을 눌러도
        아무 일도 안 일어나는 것처럼 보였다 (조작자 보고).
        """
        self.sync_trim_to_selection()
        if len(self.win.gallery_ops.selected_keys()) == 1:
            show_center_tab(self.win, "trim")

    def sync_trim_to_selection(self) -> None:
        """선택이 바뀌면 Trim 탭이 그 에피소드를 문다.

        하나만 골랐을 때만 문다 -- 트림은 한 개짜리 조작이고, 여러 개를 고른
        채로 아무거나 물면 어느 것을 자르는지 알 수 없다. 0개거나 2개 이상이면
        지금 것을 그대로 두고, 화면에 '하나만 고르세요' 를 띄운다.
        """
        picks = self.win.gallery_ops.selected_keys()
        if len(picks) != 1:
            warn = getattr(self.win, "trim_warn", None)
            if warn is not None:
                warn.setText(tr("하나만 고르세요"))
            return
        try:
            self.show_trim_for(*picks[0])
        except Exception as e:  # noqa: BLE001 -- 트림 준비 실패가 선택을 죽이면 안 된다
            self.win.log(tr("[트림] 열기 실패: {e}").format(e=e))

    def show_trim_for(self, path: str, demo: str) -> None:
        """Dataset 트리와 Analysis 순위표가 공유하는 트림 진입점."""
        if not path or not demo:
            return
        if self.win.session.active_file_path is not None and Path(path) == self.win.session.active_file_path:
            self.win.trim_summary.setText(tr("수집 중인 파일은 편집할 수 없습니다."))
            return
        self.win.trim.key = (path, demo)
        self.win.trim.n_pending = 0
        self.win.trim.undo.clear()   # 다른 에피소드의 걸음은 못 되돌린다
        try:
            series = load_series(path, demo)
        except Exception as e:  # noqa: BLE001
            self.win.trim_summary.setText(tr("불러오기 실패: {e}").format(e=e))
            return
        self.win.trim.series = series
        self.win.trim.n = int(series["n"])
        for plot, dims in self.win.trim_plots.values():
            plot.set_data(series, dims)
        self.win.trim.frames = {"agent": None, "wrist": None}
        self._trim_cut_shown = False
        self.win.trim_play_btn.setText(tr("재생"))
        for v in self.win.trim_views.values():
            v.setStyleSheet("border:2px solid transparent;")
            v.clear_frame(tr("영상 불러오는 중..."))
        if self.win.trim.loader is not None:
            self.win.trim.loader.wait()
        self.win.trim.loader = EpisodeLoadWorker(path, demo)
        self.win.trim.loader.loaded.connect(self.on_trim_loaded)
        self.win.trim.loader.failed.connect(
            lambda m: [v.clear_frame(tr("영상 없음")) for v in self.win.trim_views.values()])
        self.win.trim.loader.start()
        self.trim_update()

    def on_trim_loaded(self, path, demo, agent, wrist) -> None:
        if self.win.trim.key != (path, demo):
            return
        self.win.trim.frames = {"agent": agent, "wrist": wrist}
        self.trim_update()
        self.trim_seek(self.trim_keep() - 1)

    def trim_pending(self) -> int:
        return self.win.trim.n_pending

    def trim_keep(self) -> int:
        return max(0, self.win.trim.n - self.trim_pending())

    def _trim_push(self) -> None:
        """지금 자를 양을 되돌리기 더미에 쌓는다. 값을 바꾸기 **전에** 부른다."""
        self.win.trim.undo.append(self.win.trim.n_pending)
        del self.win.trim.undo[:-50]          # 더미가 무한정 자라지 않게

    def trim_undo(self) -> None:
        """행동취소 -- 마지막 한 걸음만 되돌린다.

        [원래대로] 와 다르다: 원래대로는 0 으로 되돌리고, 이쪽은 −5 를 한 번
        잘못 눌렀을 때 그 한 번만 무른다 (조작자, 2026-09-12: "자를 구간을
        설정했다가 취소하고 싶을 수 있잖아? 행동취소 버튼을 만들자"). 확정
        전이라 어느 쪽도 파일은 건드리지 않는다.
        """
        if self.win.trim.key is None or not self.win.trim.undo:
            return
        self.win.trim.n_pending = self.win.trim.undo.pop()
        self.trim_update()
        self.trim_seek(self.trim_keep() - 1)

    def trim_add(self, n: int) -> None:
        """+/- 를 누른 만큼 옮긴다. 0 아래로는 못 간다 -- 원본보다 길어질 수 없다."""
        if self.win.trim.key is None:
            return
        self._trim_push()
        self.win.trim.n_pending = max(0, self.win.trim.n_pending + n)
        self.trim_update()
        self.trim_seek(self.trim_keep() - 1)

    def trim_reset(self) -> None:
        """원래대로 -- 고른 것을 통째로 0으로. 한 걸음씩 무르는 것은
        [행동취소](trim_undo)가 한다."""
        if self.win.trim.key is None:
            return
        self._trim_push()
        self.win.trim.n_pending = 0
        self.trim_update()
        self.trim_seek(self.trim_keep() - 1)

    def trim_suggest(self) -> None:
        if self.win.trim.key is None:
            return
        n = suggest_trim(*self.win.trim.key)
        self._trim_push()
        self.win.trim.n_pending = n
        self.win.log(f"[트림] 추천 {n}프레임" + ("" if n else " (이미 조용하게 끝납니다)"))
        self.trim_update()
        self.trim_seek(self.trim_keep() - 1)

    def trim_seek(self, i: int) -> None:
        n = self.win.trim.n
        if n <= 0:
            return
        i = max(0, min(n - 1, i))
        self.win.trim_slider.blockSignals(True)
        self.win.trim_slider.setRange(0, n - 1)
        self.win.trim_slider.setValue(i)
        self.win.trim_slider.blockSignals(False)
        self.trim_show_frame(i)

    def trim_show_frame(self, i: int) -> None:
        keep = self.trim_keep()
        for role, v in self.win.trim_views.items():
            arr = self.win.trim.frames.get(role)
            if arr is None or len(arr) == 0:
                continue
            v.set_frame(arr[min(i, len(arr) - 1)])
        # 예전에는 여기서 마지막 남는 프레임에 " ← 잘린 뒤 마지막" 을 붙였다.
        # 뺐다 -- 그 프레임에 정확히 서 있을 때만 나오는 글자라 무슨 뜻인지
        # 알 수 없었고 (조작자, 2026-09-12: "의미하는 바를 전혀 알 수 없어"),
        # 같은 것을 재생바의 빨간 선이 늘 보여 준다.
        mark = tr("  (잘려나갈 구간)") if i >= keep else ""
        self.win.trim_pos.setText(f"{i + 1}/{self.win.trim.n}{mark}")
        # 지금 보는 프레임이 사라질 것이면 영상 테두리가 빨개진다. 테두리는
        # 늘 2px 자리를 차지하고 색만 바뀌므로 (transparent <-> 빨강) 영상
        # 크기는 흔들리지 않는다.
        cut_now = bool(self.trim_pending()) and i >= keep
        if cut_now != self._trim_cut_shown:
            self._trim_cut_shown = cut_now
            color = CUT_COLOR if cut_now else "transparent"
            for v in self.win.trim_views.values():
                v.setStyleSheet(f"border:2px solid {color};")
        for plot, _ in self.win.trim_plots.values():
            plot.set_cursor(i)

    def on_trim_scrub(self, i: int) -> None:
        self._trim_cut_stop_armed = True     # 사람이 끌었으면 다시 선다
        self.trim_show_frame(i)

    def on_trim_play(self) -> None:
        """에피소드를 처음부터 끝까지 튼다. 누르면 멈춘다.

        2026-09-12 까지는 ``keep - 40`` 에서 ``keep`` 까지, 곧 새 끝 근처
        2초만 훑었다. 확정 직전에 "여기가 새 끝이 맞나"를 보기에는 그게
        맞지만, 조작자가 이 탭에서 실제로 하는 일은 그보다 앞이다 --
        **이전 명령으로 찍힌 것인지, 녹화를 언제부터 늦게 끝냈는지**를
        보려면 통째로 봐야 한다 ("재생이 전체 재생이 안 되요").
        잘릴 구간은 플롯의 빨간 음영과 위치 표시로 그대로 보인다.
        """
        if self.win.trim.key is None:
            return
        t = self.win.trim.timer
        if t is not None and t.isActive():   # 누르면 멈춘다
            t.stop()
            self.win.trim_play_btn.setText(tr("재생"))
            return
        # 끝에 서 있으면 처음으로 되감고 튼다 -- 안 그러면 한 프레임 만에 선다.
        if self.win.trim_slider.value() >= self.win.trim.n - 1:
            self.trim_seek(0)
        # 잘림 지점 **앞**에서 시작하는 재생은 거기서 다시 선다. 몇 번을
        # 돌려 보든 매번 서야 한다 (조작자, 2026-09-12: "여러번 보더라도
        # 계속 멈춰주면 좋겠어"). 그 자리에 서 있는 채로 누른 것만이
        # "이어보기" 라서, 그때만 안 선다.
        if self.win.trim_slider.value() < self.trim_keep() - 1:
            self._trim_cut_stop_armed = True
        if self.win.trim.timer is None:
            self.win.trim.timer = QTimer(self.win)
            self.win.trim.timer.timeout.connect(self.trim_tick)
        self.apply_trim_speed()
        self.win.trim.timer.start()
        self.win.trim_play_btn.setText(tr("정지"))

    def trim_speed(self) -> float:
        combo = getattr(self.win, "trim_speed_combo", None)
        data = combo.currentData() if combo is not None else None
        return float(data) if data else 1.0

    def apply_trim_speed(self) -> None:
        """배속은 타이머 주기로 낸다 -- 3배(60Hz)까지는 프레임을 건너뛸 필요가
        없어, 빠르게 훑을 때도 놓치는 프레임이 없다 (Playback 탭과 같은 규약)."""
        t = self.win.trim.timer
        if t is not None:
            t.setInterval(max(5, int(round(1000 / PLAYBACK_FPS / self.trim_speed()))))

    def on_trim_speed_changed(self) -> None:
        self.apply_trim_speed()

    def trim_tick(self) -> None:
        # 끝은 **원본 길이**다. trim_keep() 에서 멈추면 자를 양을 바꿀 때마다
        # 재생이 끝나는 지점이 달라져서, 같은 에피소드를 두 번 틀면 다른
        # 길이로 보인다.
        i = self.win.trim_slider.value() + 1
        if i >= self.win.trim.n:
            self.win.trim.timer.stop()
            self.win.trim_play_btn.setText(tr("재생"))
            return
        # 다만 **잘릴 자리에서 한 번 선다.** 통째로 흘려보내면 새 끝이 어느
        # 프레임인지 재생만으로는 알 수 없다 (조작자, 2026-09-12: "빨간
        # 구분선 이후까지 재생되어서 정확히 어디까지가 에피소드인지 파악이
        # 안 돼"). 한 번 더 누르면 잘려나갈 구간까지 이어서 본다 -- 전체를
        # 볼 수 있어야 한다는 요구도 그대로다.
        keep = self.trim_keep()
        if self.trim_pending() and i == keep and self._trim_cut_stop_armed:
            self._trim_cut_stop_armed = False
            self.win.trim.timer.stop()
            self.win.trim_play_btn.setText(tr("잘린 뒤 이어보기"))
            return
        self.trim_seek(i)

    def trim_update(self) -> None:
        """Recomputes every label, guard and shading from the pending count."""
        self._trim_cut_stop_armed = True     # 자를 양이 바뀌면 설 자리도 바뀐다
        # 우측 패널 위젯은 build_center(Trim 탭) 뒤의 build_right 에서 만들어진다
        # -- Trim 탭을 짓는 도중에 이 메서드가 불리므로 없을 수 있고, 그때는
        # 가운데 패널 쪽(플롯·슬라이더)만 갱신한다.
        reset_btn = getattr(self.win, "trim_reset_btn", None)
        count = getattr(self.win, "trim_count", None)
        apply_btn = getattr(self.win, "trim_apply_btn", None)
        warn = getattr(self.win, "trim_warn", None)
        has = self.win.trim.key is not None
        self.win.trim_play_btn.setEnabled(has and self.win.trim.frames.get("agent") is not None)
        self.win.trim_slider.setEnabled(has)
        if reset_btn is not None:
            reset_btn.setEnabled(bool(self.win.trim.n_pending))
        undo_btn = getattr(self.win, "trim_undo_btn", None)
        if undo_btn is not None:
            undo_btn.setEnabled(has and bool(self.win.trim.undo))
        if not has:
            if count is not None:
                count.setText(tr("에피소드를 고르세요"))
            if apply_btn is not None:
                apply_btn.setEnabled(False)
            if warn is not None:
                warn.setText("")
            for plot, _ in self.win.trim_plots.values():
                plot.set_cut(None)
            self.win.trim_slider.set_cut(None)
            return
        path, demo = self.win.trim.key
        n_trim, keep = self.trim_pending(), self.trim_keep()
        plan = plan_trim(path, [demo], max(n_trim, 1))[0]
        self.win.trim_summary.setText(
            tr("{d} · {n}프레임 ({s:.1f}s) · 마지막 그리퍼 동작 −{g}프레임").format(
                d=demo, n=self.win.trim.n, s=self.win.trim.n / 20.0,
                g=plan.gripper_tail if plan.gripper_tail is not None else "?"))
        if count is not None:
            count.setText(
                tr("{a} → {b} 프레임   (−{n})").format(a=self.win.trim.n, b=keep, n=n_trim)
                if n_trim else tr("{a} 프레임 — 자를 구간 없음").format(a=self.win.trim.n))
        for plot, _ in self.win.trim_plots.values():
            plot.set_cut(keep if n_trim else None)
        # 재생바에도 같은 선을 긋는다. 자를 양을 바꿀 때 **눈에 보이는 것**이
        # 이것뿐이다 -- 위치 라벨은 그 프레임에 서 있을 때만 말해 준다.
        self.win.trim_slider.set_cut(keep - 1 if n_trim else None)
        blocked = plan_trim(path, [demo], n_trim)[0].blocked if n_trim else None
        if apply_btn is not None:
            apply_btn.setEnabled(bool(n_trim) and not blocked)
        if warn is not None:
            if blocked:
                warn.setText(tr("⚠ {b}").format(b=blocked))
            elif plan.already:
                warn.setText(tr("이미 다듬은 이력: {a}").format(a=plan.already))
            else:
                warn.setText("")

    def trim_apply(self) -> None:
        if self.win.trim.key is None or not self.trim_pending():
            return
        path, demo = self.win.trim.key
        n_trim, keep = self.trim_pending(), self.trim_keep()
        if QMessageBox.question(
                self.win, tr("끝 다듬기 확정"),
                tr("{f}\n{d}\n\n{a} → {b} 프레임 (뒤에서 {n}개 삭제)\n\n"
                   "되돌릴 수 없습니다. 진행할까요?").format(
                       f=Path(path).name, d=demo, a=self.win.trim.n, b=keep, n=n_trim),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            new_n = trim_tail(path, demo, n_trim)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self.win, tr("다듬기 실패"), f"{type(e).__name__}: {e}")
            self.win.log(f"[트림 실패] {Path(path).name} {demo}: {type(e).__name__}: {e}")
            return
        self.win.log(f"[트림] {Path(path).name} {demo}: {self.win.trim.n} → {new_n}프레임 "
                 f"(−{n_trim})")
        # 프록시 클립은 잘린 꼬리를 아직 갖고 있다 -- 지우지 않으면 큐레이션
        # 그리드가 **이미 없는 프레임을 계속 보여준다.** 썸네일은 첫 프레임이라
        # 트림에 안 변하므로 여기서는 건드리지 않는다 (둘의 수명이 다르다).
        # uid 도 번호도 안 바뀌었으니 scene 통째가 아니라 이 에피소드만.
        drop_episode_caches(self.win, lambda: episode_uid_at(path, demo))
        self.win.dataset_ops.refresh_dataset_tree()
        self.win.stats_ops.refresh_analysis(force=True)
        self.show_trim_for(path, demo)

    # -------------------------------------------------------------- analysis
    def on_rank_trim(self) -> None:
        """순위표에서 고른 것을 **Trim 탭**에서 본다.

        Playback 탭으로 보내던 것을 옮겼다 (조작자, 2026-09-12). 순위표에서
        확인하고 싶은 것은 "이 테이크를 어떻게 할까"이고, 그 자리에서 바로
        할 수 있는 일(끝 다듬기·판정)은 Trim 쪽에 있다. 행을 고르는 것만으로
        이미 ``show_trim_for`` 가 물려 있으므로(공유 선택 → on_selection_changed
        → sync_trim_to_selection), 여기서 하는 일은 **탭을 옮기는 것**이다 --
        탭 전환을 선택에 묶으면 곡선만 보려던 사람의 화면을 뺏는다. 선택은
        순위표가 아니라 공유 선택에서 읽는다 (통로 하나, 2026-09-12).
        """
        picks = self.win.gallery_ops.selected_keys()
        if not picks:
            return
        if self.win.trim.key != picks[0]:
            self.show_trim_for(*picks[0])
        show_center_tab(self.win, "trim")

    # ---------------------------------------------------------------- replay
    def replay_running(self) -> bool:
        return (self.win.procs.replay_process is not None and
                self.win.procs.replay_process.state() != QProcess.ProcessState.NotRunning)

    def on_replay_selected(self) -> None:
        """실로봇 재생 -- 공유 선택에서 하나만."""
        if self.replay_running():
            self.on_replay_stop()
            return
        picks = self.win.gallery_ops.selected_keys()
        if len(picks) != 1:
            QMessageBox.information(
                self.win, tr("선택 필요"),
                tr("실로봇 재생은 에피소드 하나만 선택하세요."))
            return
        self.replay_on_robot(*picks[0])
    def replay_on_robot(self, path: str, demo: str) -> None:
        """Dataset 트리와 Gallery 가 공유하는 실로봇 재생 진입점.

        replay_episode.py 를 --yes 로 하위 프로세스 실행한다 (램프·틱당
        클램프 같은 안전장치는 스크립트 쪽에 있다). 로봇을 쥐는 것은 결국
        로봇 노드 하나이므로, 여기서는 GUI 세션과의 충돌만 막는다.
        """
        if self.win.worker is not None:
            QMessageBox.warning(self.win, tr("재생 불가"),
                                tr("수집 세션 중에는 실로봇 재생을 할 수 "
                                   "없습니다. 먼저 세션을 종료하세요."))
            return
        busy = running_job(self.win)
        if busy:
            QMessageBox.warning(self.win, tr("재생 불가"),
                                tr("{w} 이(가) 파일을 사용 중입니다. 끝난 뒤 "
                                   "다시 시도하세요.").format(w=busy))
            return
        if self.win.procs.replay_process is not None and \
                self.win.procs.replay_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self.win, tr("이미 재생 중"),
                                    tr("이전 재생이 끝나기를 기다리거나 '재생 "
                                       "중단'을 누르세요."))
            return
        speed, ok = QInputDialog.getDouble(
            self.win, tr("실로봇 재생"),
            tr("재생 배속 (0.1~1.0, 첫 재생은 0.5 권장)"),
            0.5, 0.1, 1.0, 1)
        if not ok:
            return
        ans = QMessageBox.warning(
            self.win, tr("로봇이 움직입니다"),
            tr("{d} ({f}) 을(를) {s}배속으로 실로봇에서 재현합니다.\n\n"
               "· 로봇 노드가 켜져 있어야 합니다\n"
               "· 로봇이 시작 포즈로 이동한 뒤 바로 재생됩니다\n"
               "· 주변 공간을 비우고, 비상정지를 준비하세요\n\n"
               "시작할까요?").format(d=demo, f=Path(path).name, s=speed),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if ans != QMessageBox.StandardButton.Yes:
            return
        proc = QProcess(self.win)
        proc.setProgram(sys.executable)
        proc.setArguments([REPLAY_SCRIPT, path, demo,
                           "--speed", f"{speed:g}", "--yes"])
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(
            lambda: self.win._pipe(proc, "[실로봇 재생]", "log"))
        proc.finished.connect(self.on_replay_finished)
        self.win.procs.replay_process = proc
        self.win.log(f"[실로봇 재생] ▶ {Path(path).name} / {demo} ({speed:g}x)")
        proc.start()
        self.set_replay_ui(True)

    def on_replay_stop(self) -> None:
        """재생 하위 프로세스를 끊는다. 로봇 노드의 레퍼런스 필터가 현재
        포즈를 유지하므로(Ctrl-C 와 동일) 팔이 낙하하지는 않는다."""
        proc = self.win.procs.replay_process
        if proc is None or proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.win.log(tr("[실로봇 재생] 중단 요청 — 현재 포즈에서 정지합니다"))
        proc.terminate()
        if not proc.waitForFinished(2000):
            proc.kill()

    def set_replay_ui(self, running: bool) -> None:
        """재생/중단 토글.

        버튼은 Dataset 패널 하나뿐이다. 큐레이션 격자에서는 뺐다 (2026-09-11)
        -- 거기는 타일을 빠르게 눌러 고르는 화면이라, 로봇이 실제로 움직이는
        동작을 선택 바로 옆에 두면 안 된다. "삭제를 에피소드 선택 옆에 두었다가
        오클릭이 났다" 와 같은 종류의 인접성이다.

        ``sip.isdeleted`` 로 먼저 막는다. 종료 중에 QProcess.finished 가 큐에
        남아 뒤늦게 도착하면 창은 이미 C++ 쪽이 지워져 있고, 그때는
        ``getattr(win, ..., None)`` 조차 AttributeError 가 아니라
        RuntimeError 를 던져 기본값으로 넘어가지 않는다 (실제로 코어 덤프까지
        갔다). 창 쪽 규약과 같다 -- collect_workspace.py 의 같은 주석 참고.
        """
        from PyQt6 import sip

        if self.win is None or sip.isdeleted(self.win):
            return
        for b, idle_text in ((getattr(self.win, "replay_btn", None),
                              tr("선택 재생 (실로봇)")),):
            if b is None or sip.isdeleted(b):
                continue
            b.setText(tr("■ 재생 중단") if running else idle_text)
            b.setStyleSheet(
                "background-color:#c0392b; color:white;" if running else "")

    def on_replay_finished(self, code: int, _status) -> None:
        self.win.procs.replay_process = None
        self.set_replay_ui(False)
        self.win.log(tr("[실로봇 재생] {r} (exit={c})").format(
            r=tr("완료") if code == 0 else tr("중단/실패 — 로그 확인"), c=code))

    # ---------------------------------------------------------------- session
    def on_episode_list(self, episodes) -> None:
        prev_n = len(self.win.session.active_episode_cache) if self.win.session.active_episode_cache else None
        self.win.session.active_episode_cache = episodes
        if self.win._pending_scene_deletes > 0 and self.win.session.scene_session:
            # 목록이 줄어든 emit 만 삭제 완료로 센다 -- 사이에 낀 저장/재판정
            # emit(개수 불변·증가)이 카운터를 잘못 소진하지 않게. 삭제 1걧마다
            # renumber 로 uid 가 재배정되므로 매번 통째로 무효화한다.
            if prev_n is not None and len(episodes) < prev_n:
                self.win._pending_scene_deletes = max(
                    0, self.win._pending_scene_deletes - (prev_n - len(episodes)))
                # 파일은 saver 가 잠그고 있다 -- 다시 열지 않고 세션 설정에서
                # scene_id 를 얻는다 (session_scene_id).
                drop_scene_caches(self.win, self.win.scene_ops.session_scene_id)
        self.win.dataset_ops.refresh_dataset_tree()
        if self.win.session.scene_session:
            # 저장/재판정마다 saver 가 새 목록을 보내온다 -- slot 카운트 갱신
            self.win.scene_planning.refresh_instruction_list()
            self.win.scene_planning.refresh_start_instruction()  # Configure 쪽 표시도 동기화
            self.win.collection.refresh_instruction()

    # -------------------------------------------------------------- hdf5 view
    def on_show_structure(self) -> None:
        path = self.win.dataset_ops.selected_file()
        if path is None:
            QMessageBox.information(self.win, tr("선택 필요"), tr("파일을 선택하세요."))
            return
        if path.name.startswith("scene_"):
            # scene 파일은 legacy 구조 검사 대신 표준 뷰(격자 지도 + slot 현황).
            card = InfoCard()
            card.setMinimumWidth(360)
            try:
                md = read_scene_metadata(path)
                counts = count_by_slot(path)
                card.set_scene(md, counts=counts or None, extra=[(
                    tr("정밀 검사"),
                    tr("python scripts/check/check_scene_file.py {p}").format(p=path))])
            except Exception as e:  # noqa: BLE001
                card.setText(f"{path.name}\n읽기 실패: {type(e).__name__}: {e}")
            self.win._alert(tr("Scene 구조"), "", icon=QMessageBox.Icon.Information,
                            content=card)
            return
        st = hdf5_repack_status(path)
        lines = [f"{path.name}",
                 f"  에피소드 {st['episodes']}개, {st['size'] / 1e6:.1f} MB",
                 f"  이미지 압축: {st['compression']} (혼합={st['mixed']})",
                 f"  재압축 이력: {st['marker'] or '-'}"]
        try:
            with h5py.File(path, "r") as f:
                names = sorted(f["data"], key=lambda s: int(s.split("_")[1]))
                if names:
                    lines.append("  " + describe_episode(f["data"][names[0]]).replace("\n", "\n  "))
        except Exception as e:  # noqa: BLE001
            lines.append(f"  (구조 읽기 실패: {e})")
        self.win.log("\n".join(lines), view="validation")
        self.win.bottom_tabs.setCurrentWidget(self.win.validation_view)
