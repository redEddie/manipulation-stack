"""Session ledger: counting, disk space, and collection history."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from PyQt6.QtWidgets import QTreeWidgetItem

from mstack.data.collection_history import (
    SessionRecord,
    append_session,
    history_path,
    leaderboard,
    load_sessions,
    now_iso,
)
from mstack.gui.i18n import tr

#: 저장 경로 여유가 이보다 적으면 상태바가 주황(주의) / 빨강(위험)이 된다.
#: 에피소드 하나가 60MB 안팎이라 10GB 면 한 세션이 안 들어간다.
DISK_WARN_GB = 50.0
DISK_CRITICAL_GB = 10.0


def _bold_row(item) -> None:
    """트리 한 줄 전체를 굵게. 항목은 위젯이 아니라 칸마다 글꼴을 들고 있어
    (fonts.set_bold 처럼) 위젯 하나를 고칠 수가 없다."""
    font = item.font(0)
    font.setBold(True)
    for c in range(item.columnCount()):
        item.setFont(c, font)


class HistoryOps:
    """한 수집 세션에 대한 장부 -- 세고(bump), 보여 주고(디스크), 남긴다(이력).

    셋이 한 클래스인 이유는 같은 세션 하나를 말하기 때문이다: bump 가 센 수가
    record_session 이 쓰는 줄이 되고, 그 줄들이 refresh_history 의 수집자
    순위표가 된다. 저장 경로 여유(refresh_disk)도 "이 세션이 더 들어가는가"
    라는 같은 질문이다.

    분석(StatsOps)과는 다른 일이다 -- 저쪽은 **이미 찍힌 것**의 통계이고
    이쪽은 **찍는 동안**의 장부다 (kimi 구조 감사 2026-09-12 에 갈랐다).
    """

    def __init__(self, win) -> None:
        self.win = win

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
