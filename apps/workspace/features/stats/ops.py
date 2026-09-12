"""Analysis and curation-candidate operations for the workspace window."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
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
from mstack.gui.i18n import tr
from mstack.scene.scene_format import iter_scene_files

#: 데이터가 바뀐 뒤 자동 재분석까지 기다리는 시간. 삭제 한 번이 신호 여러
#: 개를 내므로 그것들을 한 번으로 뭉치는 것이 목적이다.
ANALYSIS_DEBOUNCE_MS = 2000


def _episode_sort_key(demo: str):
    """에피소드 번호 순 정렬 키 ("episode_012" -> 12).

    숫자를 못 뽾으면 숫자 있는 것들 뒤에 문자열 순으로 둔다."""
    m = re.search(r"(\d+)", demo)
    if m:
        return (0, int(m.group(1)), "")
    return (1, 0, demo)


class StatsOps:
    """Dataset analysis: rescan, curation candidates, and per-episode curves."""

    def __init__(self, win) -> None:
        self.win = win
        #: [튀는 것만 선택] 이 방금 데려간 곳. 같은 곳으로 두 번 데려가지
        #: 않기 위한 빗장이다 -- 옮긴 목록에도 그것이 안 보이면(길이 필터에
        #: 가렸거나) 다시 옮기려 들어 무한히 오갈 수 있다.
        self._last_jump = None

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
        # 지시문은 **정본 통로**에서 읽고(gallery_ops.selected_instruction_id),
        # id -> 문장 변환만 지금 화면의 에피소드로 한다. 전에는 걸러진 목록의
        # 문장 집합으로 역추론해서, 갤러리 필터가 바뀌면 여기가 조용히
        # 갈라질 자리였다 (kimi 구조 감사, 2026-09-12).
        iid = self.win.gallery_ops.selected_instruction_id()
        if iid is not None:
            sentences = {e["instruction"]
                         for e in self.win.gallery.episodes
                         if e.get("instruction_id") == iid}
            if sentences:
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
        # 한 번만 계산해 돌린다 -- 예전에는 이 갱신 한 번에 filtered_stats()
        # 가 네 번 돌았다 (목록·힌트·scope_label·refresh_dim_dist).
        eps = self.filtered_stats()
        rows = sorted(eps, key=score)[:60]
        # 다시 그리는 동안 신호를 막는다. 안 그러면 clear() 가 "선택 없음" 을,
        # 복원이 "선택 바뀜" 을 쏘아 on_rank_selected 가 헛돈다.
        self.win.rank_tree.blockSignals(True)
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
        # 순위표 선택은 **공유 선택의 보기**다 -- 목록을 다시 그려도 창이 들고
        # 있는 선택을 그대로 비춘다. 예전에는 clear() 가 선택을 지워 버려서,
        # 선택을 바꿀 때마다 순위표만 텅 비었다.
        want = set(self.win.gallery_ops.selected_keys())
        first = None
        for i in range(self.win.rank_tree.topLevelItemCount()):
            it = self.win.rank_tree.topLevelItem(i)
            if it.data(0, Qt.ItemDataRole.UserRole) in want:
                it.setSelected(True)
                first = first or it
        if first is not None:
            self.win.rank_tree.setCurrentItem(first)
            self.win.rank_tree.scrollToItem(first)
        self.win.rank_tree.blockSignals(False)
        # **무엇의 후보인지**는 상자 제목이 말한다. 회색 설명 줄에 섞어
        # 두었더니 정작 필요한 한 마디(판정선)가 묻혔다 (조작자, 2026-09-12).
        box = getattr(self.win, "filt_box", None)
        if box is not None:
            more = tr(" · 위 {m}개만").format(m=len(rows)) if len(rows) < len(eps) else ""
            box.setTitle(tr("큐레이션 후보 — {s}{m}").format(
                s=self.scope_label(eps), m=more))
        flagged = [e for e in eps if e.flagged]
        band = tr("±{d} 밖 = 급함(빨강 바탕) / 늘어짐(파랑 바탕)").format(d=TASK_DEV_LIMIT)
        self.win.stats_hint.setText(
            tr("{b} — 지금 목록에 {k}개").format(b=band, k=len(flagged))
            if flagged else tr("{b} — 지금 목록에는 없음").format(b=band))
        self.refresh_dim_dist(eps)
    def scope_label(self, eps=None) -> str:
        """지금 목록이 **무엇의 목록인지** 한 줄로.

        왼쪽 패널이 고른 것을 되읽지 않고 **걸러진 결과에서 뽑는다** -- 화면
        두 곳이 각자 "지금 범위"를 계산하면 언젠가 갈라진다. 지시문이 하나로
        좁혀졌으면 그 문장을, 아직 씬만 골랐으면 문장 수를 말한다.
        """
        eps = self.filtered_stats() if eps is None else eps
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
    def refresh_dim_dist(self, eps=None) -> None:
        """차원별 σ(Δa) 분포 -- 모수는 **지금 걸러진 목록**이다.

        전체 데이터셋으로 재면 작업이 다른 에피소드의 퍼짐이 섞인다. 큐레이션은
        (씬 → 지시문) 안에서만 하므로 (조작자, 2026-09-11), 비교도 그 안에서
        해야 "이 차원이 이 작업에서 유난히 흔들린다"가 된다. 그래서 목록이
        다시 그려질 때마다 함께 다시 그린다.
        """
        eps = self.filtered_stats() if eps is None else eps
        box = getattr(self.win, "dim_box", None)
        if box is not None:
            box.setTitle(tr("차원별 σ(Δa) 분포 — {s}").format(s=self.scope_label(eps)))
        eps = [e for e in eps if e.per_dim_sigma is not None]
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
        picked = []
        for i in range(tree.topLevelItemCount()):
            it = tree.topLevelItem(i)
            key = it.data(0, Qt.ItemDataRole.UserRole)
            st = next((e for e in self.win.session.stats if e.key == key), None)
            if st is not None and st.flagged:
                picked.append(it)
        # **공유 선택에 넣는다.** 순위표에만 칠하면 격자·목록·우측 카드는
        # 딴 것을 가리킨 채라, 고른 것을 그 자리에서 지우거나 판정할 수
        # 없었다. 넣고 나면 refresh_rank_list 가 순위표에도 그대로 비춘다.
        if picked:
            by_name = {e["name"]: e for e in self.win.gallery.episodes}
            eps = [by_name[d] for _p, d in
                   (it.data(0, Qt.ItemDataRole.UserRole) for it in picked)
                   if d in by_name]
            if eps:
                self.win.gallery_ops.set_selection(eps, source="rank")
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
    # on_rank_delete 는 없앴다 (2026-09-12). 순위표 선택이 공유 선택이 된
    # 뒤로는 왼쪽 패널의 [🗑 Mark for delete] 와 **같은 것에 같은 일**을 해서,
    # 같은 문이 둘이 되었다. 삭제로 가는 문은 하나다 (curation_basket 참고).

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
