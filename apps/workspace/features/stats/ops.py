"""Statistics and analysis operations for WorkspaceWindow."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QTextCursor
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QTreeWidgetItem,
    QVBoxLayout,
)

from apps.workspace.shared.busy import set_busy
from mstack.data.episode_stats import (
    TASK_DEV_LIMIT,
    hdf5_files,
    load_series,
    scan_dataset,
    summarize,
)
from mstack.data.collection_history import (
    SessionRecord,
    append_session,
    history_path,
    leaderboard,
    load_sessions,
    now_iso,
)
from mstack.gui.i18n import tr
from mstack.scene.scene_format import iter_scene_files

#: 저장 경로 여유가 이보다 적으면 상태바가 주황(주의) / 빨강(위험)이 된다.
#: 에피소드 하나가 60MB 안팎이라 10GB 면 한 세션이 안 들어간다.
DISK_WARN_GB = 50.0
DISK_CRITICAL_GB = 10.0

#: 데이터가 바뀐 뒤 자동 재분석까지 기다리는 시간. 삭제 한 번이 신호 여러
#: 개를 내므로 그것들을 한 번으로 뭉치는 것이 목적이다.
ANALYSIS_DEBOUNCE_MS = 2000


def _bold_row(item) -> None:
    """트리 한 줄 전체를 굵게. 항목은 위젯이 아니라 칸마다 글꼴을 들고 있어
    (fonts.set_bold 처럼) 위젯 하나를 고칠 수가 없다."""
    font = item.font(0)
    font.setBold(True)
    for c in range(item.columnCount()):
        item.setFont(c, font)


def _episode_sort_key(demo: str):
    """에피소드 번호 순 정렬 키 ("episode_012" -> 12).

    숫자를 못 뽾으면 숫자 있는 것들 뒤에 문자열 순으로 둔다."""
    m = re.search(r"(\d+)", demo)
    if m:
        return (0, int(m.group(1)), "")
    return (1, 0, demo)


class StatsOps:
    """Session counters and dataset analysis for the workspace."""

    def __init__(self, win) -> None:
        self.win = win
        #: [튀는 것만 선택] 이 방금 데려간 곳. 같은 곳으로 두 번 데려가지
        #: 않기 위한 빗장이다 -- 옮긴 목록에도 그것이 안 보이면(길이 필터에
        #: 가렸거나) 다시 옮기려 들어 무한히 오갈 수 있다.
        self._last_jump = None

    def connect_progress(self, waited: float) -> None:
        self.win.statusBar().showMessage(
            tr("카메라 정리 중... {s:.0f}초 (정리되면 자동으로 연결합니다)").format(s=waited),
            1000)
        self.win.lights["camera"].set("busy", tr("정리 중"))

    def log_progress(self, msg: str, view: str) -> None:
        """Progress that overwrites its own last line instead of stacking.

        A 1.3 GB upload prints a bar every second; appended, that buries every
        other message in the tab and makes the log useless exactly while a long
        job is running. Replacing the previous progress line keeps one live line
        and leaves the surrounding log readable.

        Deliberately not written to the log file -- the file is what gets read
        after a crash, and hundreds of superseded percentages help nobody there.
        """
        target = self.win._view(view)
        if self.win._progress_line.get(view):
            cursor = target.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.insertText(msg)
        else:
            target.appendPlainText(msg)
            self.win._progress_line[view] = True

    def bump(self, key: str, n: int = 1) -> None:
        """카운터 하나를 이번 task 와 누적 양쪽에 올린다.

        두 dict 를 따로 건드리면 반드시 한쪽만 올리는 자리가 생긴다 -- 판정
        뒤집기처럼 -1 도 있는 경로가 섞여 있어서 더 그렇다.
        """
        self.win.session.counters[key] += n
        self.win.session.cumulative[key] += n

    # ------------------------------------------------------------ 디스크
    def refresh_disk(self) -> None:
        """저장 경로의 남은 용량을 상태바에 적는다.

        Statistics 의 '디스크' 상자에서 옮겨 왔다 (2026-09-06 사용자 요청) --
        화면을 옮겨야 보이는 값이라 정작 수집 중에는 아무도 안 봤고, 다 찬
        것을 아는 시점이 저장에 실패한 뒤였다. 상태바는 늘 떠 있다.

        문턱은 이 데이터셋의 실제 크기에서 나왔다: 에피소드 하나가 20Hz
        20초 × 256² RGB 두 대 ≈ 60MB 라, 10GB 면 한 세션(150여 개)이 안
        들어가고 20GB 면 아슬아슬하다.
        """
        try:
            usage = shutil.disk_usage(
                self.win.root_edit.text().strip() or str(Path.home()))
        except OSError:
            self.win.sb_disk.setText(tr("여유 -"))
            self.win.sb_disk.setStyleSheet("color:#888;")
            return
        free_gb = usage.free / 1e9
        self.win.sb_disk.setText(tr("여유 {f:.0f} GB / {t:.0f} GB").format(
            f=free_gb, t=usage.total / 1e9))
        color = ("#e74c3c" if free_gb < DISK_CRITICAL_GB
                 else "#f39c12" if free_gb < DISK_WARN_GB else "#888")
        self.win.sb_disk.setStyleSheet(
            f"color:{color};" + (" font-weight:bold;" if free_gb < DISK_CRITICAL_GB else ""))

    # -------------------------------------------------------- 수집 이력
    def record_session(self) -> None:
        """방금 끝난 연결 세션을 이력 파일에 한 줄로 남긴다.

        Disconnect 마다 부른다. 아무것도 저장하지 않은 세션은 남기지 않는다
        -- 연결만 했다 끊은 것은 수집이 아니고, 그런 줄이 섞이면 세션 수로
        나누는 값이 전부 흐려진다.

        시간은 ``counters["t0"]`` 에서 온다. 그 값은 on_connected 에서
        0 으로 돌아가므로 정확히 이번 연결 구간이고, 그 구간이 곧 조작자가
        자리에 앉아 있던 시간이다 (리셋·재배치 포함).
        """
        st = self.win.session.counters
        if not st["saved"] or self.win.session.history_written:
            return
        self.win.session.history_written = True
        rec = SessionRecord(
            run=self.win.run_id,
            started=self.win.session.started_iso or now_iso(),
            ended=now_iso(),
            collector=self.win.collector_edit.text().strip(),
            # 이 세션이 **실제로 쓴** 폴더. Dataset 화면의 폴더 선택
            # (dataset_root) 은 구경하러 다른 데를 볼 수 있어서, 그것으로
            # 이름을 붙이면 엉뚱한 데이터셋의 순위에 줄이 들어간다.
            dataset=self._session_dataset(),
            # 아직 worker 가 살아 있는 동안 불린다 (on_worker_finished 의
            # 맨 앞) -- 그래야 이 세션이 무슨 scene 이었는지 알 수 있다.
            scene=str(self.win.scene_ops.session_scene_id() or ""),
            station=getattr(self.win, "station_name", ""),
            saved=int(st["saved"]), success=int(st["success"]),
            failed=int(st["failed"]), discarded=int(st["discarded"]),
            frames=int(st["frames"]),
            seconds=max(0.0, time.monotonic() - st["t0"]),
        )
        try:
            append_session(rec)
        except OSError as e:
            # 이력을 못 남기는 것이 수집을 막을 이유는 못 된다 -- 로그로만.
            self.win.log(f"[이력] 저장 실패: {e}")
            return
        self.win.log(tr("[이력] {n}개 / {m:.0f}분 (분당 {r:.2f}) — {p}").format(
            n=rec.saved, m=rec.seconds / 60, r=rec.per_minute, p=history_path()))
        self.refresh_history()

    def _session_dataset(self) -> str:
        """순위표와 이력이 말하는 "이 데이터셋"의 이름 = 수집 저장 경로의
        폴더명. 기록 중이면 그 파일이 놓인 폴더가 더 정확하다 -- 세션 도중에
        저장 경로 칸을 건드려도 이미 쓰고 있는 파일은 안 옮겨간다."""
        path = self.win.session.active_file_path
        if path is not None:
            return Path(path).parent.name
        return Path(self.win.root_edit.text().strip() or ".").name

    def refresh_history(self) -> None:
        """이력 파일을 읽어 수집자 순위표를 채운다.

        순위표는 **지금 데이터셋**만 센다. task 가 다르면 에피소드 하나에
        드는 시간이 다르므로, 다른 데이터셋의 속도와 한 줄에 세우면 비교가
        아니라 착시가 된다 (collection_history.leaderboard 참고).
        """
        if not hasattr(self.win, "board_tree"):
            return
        rows = load_sessions()
        dataset = self._session_dataset()
        self.win.board_tree.clear()
        me = self.win.collector_edit.text().strip()
        board = leaderboard(rows, dataset=dataset)
        for rank, pace in enumerate(board, start=1):
            item = QTreeWidgetItem([
                f"{rank}. {pace.collector}",
                f"{pace.per_minute:.2f}",
                str(pace.saved),
                f"{100 * pace.success_rate:.0f}%",
                f"{pace.seconds / 3600:.1f}h",
            ])
            item.setToolTip(0, tr("세션 {s}개 · 마지막 {l}").format(
                s=pace.sessions, l=pace.last or "-"))
            # 내 줄만 굵게. 순위표를 보는 이유가 "나는 어디쯤인가"라서,
            # 줄이 늘어나면 그 답을 찾는 것부터 일이 된다.
            if me and pace.collector == me:
                _bold_row(item)
            self.win.board_tree.addTopLevelItem(item)
        if board:
            self.win.board_hint.setText(tr("{d} · 수집자 {n}명 · 세션 {s}개").format(
                d=dataset or "-", n=len(board), s=sum(p.sessions for p in board)))
        else:
            # 비어 있을 때 "아직 없습니다" 만 말하면 막다른 길이다. 실제로는
            # **경로를 한 칸 위로 두었을 때** 가장 자주 빈다 -- 큐레이션 하려고
            # 데이터 경로를 libero_datasets(부모)로 바꾸면 순위표가 통째로
            # 사라진다 (2026-09-12 조작자: "갑자기 수집 속도 순위가 비었네요?
            # 원래 좀 차있었는데"). 이력이 있는 폴더 이름을 대면 곧바로 답이
            # 된다. 순위표는 데이터셋을 섞지 않는다는 규칙은 그대로다.
            others = sorted({r.dataset for r in rows if r.dataset and r.dataset != dataset})
            self.win.board_hint.setText(
                tr("{d} 의 이력이 없습니다 — 이력이 있는 폴더: {o} "
                   "(데이터 경로를 그 폴더로 두면 보입니다)").format(
                       d=dataset or "-", o=", ".join(others))
                if others else
                tr("{d} 의 이력이 아직 없습니다 — 세션을 끝내면(Disconnect) 한 줄이 쌓입니다.")
                .format(d=dataset or "-"))


    def on_summary(self, summary) -> None:
        # 해제는 여기서 하지 않는다 -- 정상 종료에만 오는 신호다. 실제 해제는
        # 모든 종료 경로에서 오는 finished(collection.on_worker_finished)가 맡는다.
        self.win.log(f"[세션 요약] {summary}")

    # ------------------------------------------------------------- 재분석
    def mark_stats_stale(self) -> None:
        """데이터가 바뀌었다고 표시하고, 곧 다시 분석하도록 예약한다.

        저장·삭제·재판정·트림·폴더 변경이 전부 여기로 온다. 예약이 한 번에
        몰리지 않게 단발 타이머를 다시 걸어 뭉갠다 -- 실패만 골라 20개를
        지우면 신호가 20번 오는데, 그때마다 전체 스캔을 돌 이유가 없다.
        """
        self.win.session.stats_stale = True
        timer = getattr(self.win, "analysis_timer", None)
        if timer is not None:
            timer.start(ANALYSIS_DEBOUNCE_MS)

    def auto_refresh_analysis(self) -> None:
        """조작자가 [다시 분석] 을 누르지 않아도 최신값이 보이게 한다
        (2026-09-06 사용자 요청).

        두 가지를 지킨다:

        * **세션 중에는 안 돈다.** 스캔은 파일을 읽기로 여는데 기록 중인
          파일은 saver 가 쥐고 있어서, 열리지 않으면 ``scan_dataset`` 이
          그 파일을 조용히 건너뛴다 -- 지금 찍고 있는 것만 빠진 통계가
          가장 나쁘다. 세션이 끝나면 그 자리에서 다시 돈다.
        * **보고 있지 않으면 안 돈다.** 결과가 보이는 화면은 Statistics 와
          Dataset(Analysis 탭) 뿐이다. Configure 에서 도는 스캔은 아무도
          못 보면서 디스크만 훑는다.
        """
        if self.win.worker is not None:
            return
        if self.win._activity not in ("stats", "dataset"):
            return
        try:
            self.refresh_analysis()
        except Exception as e:  # noqa: BLE001
            # **활동 전환을 끌고 죽지 않는다.** 자동 분석은 편의 기능이라
            # 실패해도 상관없지만(2026-09-07 사용자), 예외가 여기서 새면
            # Dataset 활동에 들어가는 것 자체가 막힌다 -- 필드가 적은 파일에서
            # 실제로 그랬다 (KeyError: 'frames'). [다시 분석] 은 그대로 두어,
            # 사람이 눌렀을 때는 이유가 그대로 보이게 한다.
            self.win.log(f"[통계] 자동 분석을 건너뜁니다: "
                         f"{type(e).__name__}: {e}")

    def refresh_analysis(self, force: bool = False) -> None:
        """Rescans every .hdf5's actions. Only a few KB per episode, so this is
        rebuilt from disk rather than cached -- a cache would go stale the
        moment a session records another take.

        ``force`` 는 조작자가 직접 [다시 분석] 을 누른 경우다. 그 밖에는
        마지막 스캔 뒤에 데이터가 바뀌었을 때만 돈다 -- 화면을 오갈 때마다
        같은 스캔을 되풀이하면 트리를 고를 때마다 몇백 ms 씩 멎는다.
        """
        if not force and self.win.session.stats and not self.win.session.stats_stale:
            return
        # Dataset 페이지의 폰더 선택을 따른다 (수집 경로 하드코딩 제거) --
        # scene 파일도 함께 스캔한다.
        root = self.win.dataset_ops.dataset_root()
        files = hdf5_files(root) + [str(p) for p in iter_scene_files(root)]
        if not files:
            self.win.session.stats_stale = False
            # 요약 텍스트 줄은 없앴다 (2026-09-12) -- 남은 한 줄에 적는다.
            self.win.stats_hint.setText(
                tr("{r} 에 *_demo.hdf5 / scene_*.hdf5 가 없습니다.").format(r=root))
            return
        t0 = time.monotonic()
        # UI 스레드에서 돈다 -- 파일 27개 1,987 에피소드에 1.6초다. 스레드로
        # 옮기는 대신 그동안 그렇다고 말한다 (분석은 사람이 기다려도 되는
        # 일이고, 결과를 곧바로 쓰는 곳이 많아 비동기로 만들면 "아직 없음"
        # 갈래가 여기저기 생긴다).
        set_busy(self.win, tr("{n}개 파일을 분석하는 중").format(n=len(files)))
        try:
            self.win.session.stats = scan_dataset(files)
        finally:
            set_busy(self.win)
        self.win.session.stats_stale = False
        self.win._summary = summarize(self.win.session.stats)
        dt = time.monotonic() - t0
        s = self.win._summary
        # 데이터셋 전체의 규모와 판정은 **로그로만** 남긴다. 화면 맨 위에
        # 텍스트로 붙여 두었더니 후보 목록의 칸들과 같은 숫자를 두 번 말하는
        # 셈이었다 (조작자, 2026-09-12). 지금 목록 기준 개수는 목록 아래
        # 한 줄이 말한다.
        self.win.log(f"[분석] {len(files)}개 파일 / {s['n']}개 에피소드 ({dt:.2f}s) — {s['verdict']}")

        means = [e.mean_da for e in self.win.session.stats]
        self.win.da_hist.set_values(means, [(s["p50"], tr("중앙값")), (s["p99"], "p99")])

        lens = [e.seconds for e in self.win.session.stats]
        self.win.len_min_spin.blockSignals(True)
        self.win.len_max_spin.blockSignals(True)
        self.win.len_min_spin.setRange(0, int(max(lens) * 10) + 5)
        self.win.len_max_spin.setRange(0, int(max(lens) * 10) + 5)
        self.win.len_min_spin.setValue(0)
        self.win.len_max_spin.setValue(int(max(lens) * 10) + 5)
        self.win.len_min_spin.blockSignals(False)
        self.win.len_max_spin.blockSignals(False)
        self.refresh_rank_list()

    def filtered_stats(self) -> list:
        lo = self.win.len_min_spin.value() / 10.0
        hi = self.win.len_max_spin.value() / 10.0
        if lo > hi:
            lo, hi = hi, lo
        self.win.len_label.setText(f"{lo:.1f}~{hi:.1f}s")
        # 범위는 **왼쪽 패널이 정본이다** (조작자, 2026-09-11): Scene 콤보가
        # 파일을, Instruction 목록이 지시문을 고른다. 여기에 별도의 그룹
        # 콤보를 두면 같은 축이 두 군데가 된다 -- 큐레이션은 오직
        # (씬 → 지시문) 안에서만 한다.
        path = self.win.dataset_ops.selected_file()
        out = [e for e in self.win.session.stats if lo <= e.seconds <= hi]
        out = [e for e in out if path is None or e.path == str(path)]
        # EpisodeStat 에는 instruction_id 가 없고 e.task 가 지시문 문장이다.
        # 지금 걸러진 에피소드(_gallery_shown)의 문장 집합으로 판정한다 --
        # Gallery 가 이미 instruction_id 로 걸러 놓은 것과 같은 집합이다.
        lst = getattr(self.win, "instruction_list", None)
        it = lst.currentItem() if lst is not None else None
        iid = it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None
        if iid is not None:
            shown = getattr(self.win, "_gallery_shown", []) or []
            if shown:
                sentences = {e["instruction"] for e in shown}
                out = [e for e in out if e.task in sentences]
        return out

    def refresh_rank_list(self) -> None:
        if not self.win.session.stats:
            return
        # 정렬 기준은 콤보가 정한다. 기본은 **에피소드 순** -- 왼쪽 목록·중간
        # 격자와 같은 순서라야 세 화면을 오갈 때 헷갈리지 않는다. 기준을 바꾼
        # 것이 눈에 띄는 것도 이점이다 (예전 기본값은 '급함' 이라, 처음부터
        # 다른 순서인 줄 모르고 볼 수 있었다).
        #
        # 그래도 정렬 자체는 남긴다: 상황에 따라 이상치를 찾는 수단이다
        # (조작자, 2026-09-12) -- "늘어짐" 으로 녹화를 늦게 끝낸 것을,
        # "짧음" 으로 2~3틱짜리를 위로 끌어올린다.
        key = (self.win.rank_combo.currentData()
               if hasattr(self.win, "rank_combo") else None)
        score = {
            "fast": lambda e: -e.task_dev,
            "slow": lambda e: e.task_dev,
            "still": lambda e: -e.still_frac,
            "short": lambda e: e.n_frames,
            "long": lambda e: -e.n_frames,
        }.get(key) or (lambda e: _episode_sort_key(e.demo))
        rows = sorted(self.filtered_stats(), key=score)[:60]
        self.win.rank_tree.clear()
        for e in rows:
            # scene 은 **짧은 이름**으로 적는다: "scene_022" 가 아니라 "S022"
            # (조작자, 2026-09-12). 파일에 적힌 scene_id 가 원래 그 형식이고
            # (metadata/scene_id = "S022"), 목록의 첫 칸은 150px 뿐이라 긴
            # 이름은 정작 구분해야 할 episode 번호를 밀어낸다. scene_id 가
            # 없는 legacy 파일만 파일명으로 돌아간다.
            where = e.scene or Path(e.path).stem[:22]
            item = QTreeWidgetItem([
                f"{where} · {e.demo}",
                f"{e.task_dev:+.4f}", f"{100 * e.still_frac:.0f}%",
                f"{e.seconds:.1f}s"])
            item.setData(0, Qt.ItemDataRole.UserRole, (e.path, e.demo))
            # 밴드 밖은 차이 칸만 물들인다 -- 행 전체를 칠하면 실패(빨강)와
            # 겹쳐서 둘 다 안 읽힌다.
            if e.task_dev > TASK_DEV_LIMIT:
                item.setForeground(1, Qt.GlobalColor.red)
            elif e.task_dev < -TASK_DEV_LIMIT:
                item.setForeground(1, Qt.GlobalColor.blue)
            if e.success is False:
                item.setForeground(0, Qt.GlobalColor.red)
            # 튄 것은 **줄 전체**에 옅은 바탕을 깔아 목록에서 바로 보이게
            # 한다 (조작자, 2026-09-12: "늘어짐 하나가 있다는데 어디있는건지
            # 모르겠어"). 글자색이 아니라 바탕이라, 실패(빨간 글자)와 겹쳐도
            # 둘 다 읽힌다.
            if e.flagged:
                tint = QColor("#c0392b" if e.task_dev > 0 else "#2e6fb7")
                tint.setAlpha(46)
                for c in range(4):
                    item.setBackground(c, QBrush(tint))
            self.win.rank_tree.addTopLevelItem(item)
        # **무엇의 후보인지**는 상자 제목이 말한다. 회색 설명 줄에 섞어
        # 두었더니 정작 필요한 한 마디(판정선)가 묻혔다 (조작자, 2026-09-12).
        eps = self.filtered_stats()
        box = getattr(self.win, "filt_box", None)
        if box is not None:
            more = tr(" · 위 {m}개만").format(m=len(rows)) if len(rows) < len(eps) else ""
            box.setTitle(tr("큐레이션 후보 — {s}{m}").format(
                s=self.scope_label(), m=more))
        flagged = [e for e in eps if e.flagged]
        band = tr("±{d} 밖 = 급함(빨강 바탕) / 늘어짐(파랑 바탕)").format(d=TASK_DEV_LIMIT)
        self.win.stats_hint.setText(
            tr("{b} — 지금 목록에 {k}개").format(b=band, k=len(flagged))
            if flagged else tr("{b} — 지금 목록에는 없음").format(b=band))
        self.refresh_dim_dist()

    def scope_label(self) -> str:
        """지금 목록이 **무엇의 목록인지** 한 줄로.

        왼쪽 패널이 고른 것을 되읽지 않고 **걸러진 결과에서 뽑는다** -- 화면
        두 곳이 각자 "지금 범위"를 계산하면 언젠가 갈라진다. 지시문이 하나로
        좁혀졌으면 그 문장을, 아직 씬만 골랐으면 문장 수를 말한다.
        """
        eps = self.filtered_stats()
        if not eps:
            return tr("빈 목록")
        scenes = {e.scene for e in eps if e.scene}
        tasks = {e.task for e in eps}
        where = next(iter(scenes)) if len(scenes) == 1 else (
            tr("씬 {n}개").format(n=len(scenes)) if scenes else tr("전체"))
        if len(tasks) == 1:
            sent = next(iter(tasks))
            what = sent if len(sent) <= 34 else sent[:33] + "…"
        else:
            what = tr("지시문 {n}개").format(n=len(tasks))
        return f"{where} · {what} · {len(eps)}개"

    def refresh_dim_dist(self) -> None:
        """차원별 σ(Δa) 분포 -- 모수는 **지금 걸러진 목록**이다.

        전체 데이터셋으로 재면 작업이 다른 에피소드의 퍼짐이 섞인다. 큐레이션은
        (씬 → 지시문) 안에서만 하므로 (조작자, 2026-09-11), 비교도 그 안에서
        해야 "이 차원이 이 작업에서 유난히 흔들린다"가 된다. 그래서 목록이
        다시 그려질 때마다 함께 다시 그린다.
        """
        box = getattr(self.win, "dim_box", None)
        if box is not None:
            box.setTitle(tr("차원별 σ(Δa) 분포 — {s}").format(s=self.scope_label()))
        eps = [e for e in self.filtered_stats() if e.per_dim_sigma is not None]
        if not eps:
            self.win.dim_bars.set_rows([])
            return
        sig = np.stack([e.per_dim_sigma for e in eps])
        self.win.dim_bars.set_rows(
            [(f"joint{i + 1}", sig[:, i]) for i in range(min(7, sig.shape[1]))])

    def show_analysis_for(self, path: str, demo: str) -> None:
        """Dataset 트리와 순위표가 공유하는 곡선 표시 경로."""
        if not path or not demo:
            return
        try:
            series = load_series(path, demo)
        except Exception as e:  # noqa: BLE001
            self.win.log(f"[분석] 시계열 로드 실패: {type(e).__name__}: {e}")
            return
        for plot, dims in self.win.series_plots.values():
            plot.set_data(series, dims)
            plot.set_cursor(None)
        stat = next((e for e in self.win.session.stats if e.key == (path, demo)), None)
        if stat is not None:
            # 고른 에피소드의 숫자는 후보 목록의 칸이 이미 같은 것을
            # 보여준다 -- 여기서 다시 문장으로 쓰지 않는다 (2026-09-12).
            self.win.da_hist.set_values(
                [e.mean_da for e in self.win.session.stats],
                [(self.win._summary["p50"], tr("중앙값")), (stat.mean_da, tr("이 에피소드"))])

    def on_select_flagged(self) -> None:
        """지금 순위표에 보이는 것 중 **밴드 밖**을 전부 고른다.

        전역 선택이 아니라 **보이는 것 안에서**다. 예전에 "튀는 것만 선택" 이
        전 파일을 훑어 고르던 때가 있었는데(2026-09-11 에 제거), 화면에 없는
        것까지 고르는 셈이라 무엇이 골라졌는지 알 수 없었다. 이제는 왼쪽
        패널(씬 → 지시문)이 좁혀 놓은 범위 안에서만 고른다 -- 같은
        (scene·문장) 안에서라야 편차가
        비교 가능하다는 것이 애초에 그룹을 나눈 이유다.

        고르기만 하고 지우지 않는다. 지우는 문은 여전히 하나다 -- 여기서
        고른 뒤 "🗑 Mark for delete" 로 표시하고 왼쪽에서 실행한다.
        """
        tree = self.win.rank_tree
        tree.clearSelection()
        picked = []
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            key = it.data(0, Qt.ItemDataRole.UserRole)
            st = next((e for e in self.win.session.stats if e.key == key), None)
            if st is not None and st.flagged:
                it.setSelected(True)
                picked.append(it)
        # 고르기만 하면 60줄 목록 어딘가에 선택이 있을 뿐이라 여전히 못 찾는다
        # (조작자, 2026-09-12). 첫 번째 것으로 **스크롤해서 세운다** -- 현재
        # 항목이 되면 곡선이 그려지고 Trim 탭도 그 에피소드를 문다.
        if picked:
            tree.setCurrentItem(picked[0])
            for it in picked:
                it.setSelected(True)
            tree.scrollToItem(picked[0])
            self._last_jump = None
            self.win.stats_hint.setText(
                tr("밴드(±{d}) 밖 {n}개를 골랐습니다 — [Trim 에서 재생] 으로 보고, "
                   "버릴 것만 🗑 Mark for delete 로 표시한 뒤 왼쪽에서 지웁니다.").format(
                       d=TASK_DEV_LIMIT, n=len(picked)))
            return
        # 지금 목록에 없으면 **어디에 있는지 찾아 데려간다.** 예전에는 "이
        # 목록에는 없습니다" 로 끝냈는데, 요약은 "늘어짐 1개" 라고 말하고
        # 있어서 조작자가 그 하나를 찾을 길이 없었다 (2026-09-12).
        rest = sorted((e for e in self.win.session.stats if e.flagged),
                      key=lambda e: (e.path, _episode_sort_key(e.demo)))
        if not rest:
            self.win.stats_hint.setText(tr("데이터셋 어디에도 밴드 밖이 없습니다."))
            return
        tgt = rest[0]
        where = tr("{s} · {t}").format(s=tgt.scene or Path(tgt.path).stem,
                                       t=tgt.task[:34])
        if self._last_jump == (tgt.path, tgt.demo):
            self._last_jump = None
            self.win.stats_hint.setText(
                tr("{w} 에 있는데 지금 목록에 안 나옵니다 — 길이(초) 슬라이더를 "
                   "넓혀 보세요.").format(w=where))
            return
        self._last_jump = (tgt.path, tgt.demo)
        if self.win.gallery_ops.go_to_episode(tgt.path, tgt.demo):
            self.win.stats_hint.setText(
                tr("지금 목록에는 없어 {w} 로 옮겼습니다 (밴드 밖 {n}개 중 첫 번째).")
                .format(w=where, n=len(rest)))
        else:
            self.win.stats_hint.setText(
                tr("지금 목록에는 없습니다 — {w} 에 있습니다 (데이터 경로가 다르면 "
                   "그 폴더로 옮기세요).").format(w=where))

    def on_rank_delete(self) -> None:
        """순위표 선택을 삭제 목록에 **표시**한다 (실행이 아니다).

        격자·트리·순위표 어디서든 표시는 자유롭고, 실제 삭제는 왼쪽 패널의
        "Delete marked" 버튼 하나뿐이다 -- 그 곳의 확인창이 배치 전체를
        보여주는 유일한 검토 순간이다.
        """
        picks = [i.data(0, Qt.ItemDataRole.UserRole) for i in self.win.rank_tree.selectedItems()]
        if not picks:
            QMessageBox.information(self.win, tr("선택 필요"),
                                    tr("삭제 목록에 넣을 에피소드를 선택하세요 (Ctrl/Shift로 여러 개)."))
            return
        for path, demo in picks:
            self.win.basket.add((path, demo))
        self.win.dataset_ops.refresh_basket_ui()
        self.win.gallery_grid.refresh_marks()

    def on_metric_help(self) -> None:
        """Shows docs/curation-metrics.md rather than a copy of it.

        The thresholds in that file are the ones episode_stats.py actually
        uses; a second prose copy inside the GUI would be the version that
        goes stale first, and the operator would have no way to tell which of
        the two was lying.
        """
        doc = Path(__file__).resolve().parents[4] / "docs" / "curation-metrics.md"
        try:
            body = doc.read_text(encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self.win, tr("지표 설명"),
                                tr("{p} 를 읽을 수 없습니다: {e}").format(p=doc, e=e))
            return
        dlg = QDialog(self.win)
        dlg.setWindowTitle(tr("지표 정의 — curation-metrics.md"))
        dlg.resize(900, 680)
        lay = QVBoxLayout(dlg)
        view = QTextBrowser()
        view.setMarkdown(body)
        view.setOpenExternalLinks(True)
        lay.addWidget(view)
        path_lbl = QLabel(str(doc))
        path_lbl.setStyleSheet("color:#888;")
        path_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(path_lbl)
        btn = QPushButton(tr("닫기"))
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec()
