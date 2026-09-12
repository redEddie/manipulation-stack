"""긴 작업 하나를 **한 곳에서** 말한다 -- 무엇이 도는가, 얼마나 남았는가.

재압축·LeRobot 변환·업로드·프록시 굽기는 몇 분에서 몇십 분이 걸리고, 그동안
데이터셋 파일을 쥐고 있다. 그 사실을 화면 세 곳이 알아야 한다:

1. **상태바** -- 지금 무엇이 도는지와 남은 시간 (안 그러면 조작자는 멈춘
   건지 도는 건지 알 수 없다).
2. **잠금** -- 그 파일을 건드리는 버튼(판정·삭제·트림 확정·프록시)은 도는
   동안 눌리면 안 된다. 잠그지 않으면 h5py 가 반쯤 쓴 파일을 열게 된다.
3. **닫기 확인** -- 창을 닫으면 그 작업이 죽는다. 15분짜리 재압축이 조용히
   사라지는 일을 막는다.

셋이 같은 답을 보게 하려고 여기 하나로 모은다 (조작자 요청, 2026-09-13:
"기능을 그동안 잠그거나, 프로그램 닫는 것도 다시 묻고, 남은 시간을
알려주는 GUI"). 예전에는 "지금 바쁜가" 를 DatasetOps.busy_reason 이 따로
알고 있었고 파이프라인은 세지도 않았다.

**추정은 자식이 말해 준 만큼만 한다.** 진행률을 못 읽으면 경과 시간만 적는다
-- 거짓 ETA 는 없느니만 못하다.
"""

from __future__ import annotations

import time

from PyQt6.QtCore import QProcess

from mstack.gui.i18n import tr

#: 잠글 버튼들. 전부 **파일을 실제로 바꾸거나 통째로 읽는** 것이다.
#: 표시(Mark)는 되돌릴 수 있고 파일을 안 건드리므로 잠그지 않는다.
LOCKED_BUTTONS = (
    "verdict_ok_btn",      # 판정 -- hdf5 attrs 를 쓴다
    "verdict_fail_btn",
    "basket_exec_btn",     # 삭제 실행
    "trim_apply_btn",      # 끝 다듬기 확정
    "proxy_btn",           # 프록시 굽기 (같은 hdf5 를 통째로 읽는다)
)


def _running(proc) -> bool:
    return proc is not None and proc.state() != QProcess.ProcessState.NotRunning


def running_job(win) -> str:
    """지금 도는 긴 작업의 이름. 한가하면 빈 문자열.

    DatasetOps.busy_reason 이던 것을 옮겨 넓혔다 -- 예전에는 파이프라인과
    프록시 굽기를 못 봐서, 전체 처리가 도는 중에도 삭제 버튼이 살아 있었다.
    """
    procs = win.procs
    if procs.job_name:
        return procs.job_name
    for proc, label in ((procs.repack_process, tr("재압축")),
                        (procs.convert_process, tr("LeRobot 변환")),
                        (procs.upload_process, tr("HDF5 업로드")),
                        (getattr(procs, "pipeline_proc", None), tr("전체 처리"))):
        if _running(proc):
            return label
    return ""


def start_job(win, name: str) -> None:
    """긴 작업이 시작됐다고 알린다 (이름은 화면에 그대로 나간다)."""
    win.procs.job_name = name
    win.procs.job_t0 = time.monotonic()
    win.procs.job_frac = None
    refresh(win)


def job_progress(win, frac: "float | None") -> None:
    """자식이 말해 준 진행률(0~1). None 이면 모르는 것으로 둔다."""
    if frac is None or not win.procs.job_name:
        return
    win.procs.job_frac = frac
    refresh(win)


def end_job(win) -> None:
    win.procs.job_name = ""
    win.procs.job_frac = None
    refresh(win)


def _fmt_min(seconds: float) -> str:
    if seconds < 90:
        return tr("{s:.0f}초").format(s=max(0.0, seconds))
    return tr("{m:.0f}분").format(m=seconds / 60.0)


def job_status_text(win) -> str:
    """상태바 한 줄. 한가하면 빈 문자열."""
    name = running_job(win)
    if not name:
        return ""
    t0 = win.procs.job_t0
    elapsed = time.monotonic() - t0 if t0 else 0.0
    frac = win.procs.job_frac
    # 5% 아래에서는 추정하지 않는다 -- 시작 직후의 비율은 남은 시간을
    # 몇 시간으로도 몇 초로도 만든다.
    if frac is not None and frac >= 0.05:
        left = elapsed * (1.0 - frac) / frac
        return tr("⏳ {n} {p:.0f}% · 남은 약 {r}").format(
            n=name, p=100 * frac, r=_fmt_min(left))
    return tr("⏳ {n} {e}째").format(n=name, e=_fmt_min(elapsed))


def refresh(win) -> None:
    """상태바와 잠금을 지금 사실에 맞춘다. 타이머와 작업 시작·끝이 부른다."""
    lab = getattr(win, "sb_job", None)
    if lab is not None:
        text = job_status_text(win)
        lab.setText(text)
        lab.setVisible(bool(text))
    refresh_locks(win)


def refresh_locks(win) -> None:
    """긴 작업이 도는 동안 데이터셋을 바꾸는 버튼을 잠근다.

    풀 때는 **직접 켜지 않고 주인에게 다시 계산하게 한다** -- 그 버튼들은
    각자의 조건(자를 것이 있나, 장바구니가 비었나)으로 켜지고 꺼지므로,
    여기서 True 를 박으면 그 조건을 덮어쓴다.
    """
    why = running_job(win)
    if why:
        for name in LOCKED_BUTTONS:
            b = getattr(win, name, None)
            if b is not None:
                b.setEnabled(False)
        win.procs.job_locked = True
        return
    if getattr(win.procs, "job_locked", False):
        win.procs.job_locked = False
        for name in LOCKED_BUTTONS:
            b = getattr(win, name, None)
            if b is not None:
                b.setEnabled(True)
        # 자기 조건을 아는 쪽이 다시 정한다.
        try:
            win.trim_ops.trim_update()
            win.delete_ops.refresh_basket_ui()
        except Exception:  # noqa: BLE001 -- 창이 아직 다 안 지어졌을 수 있다
            pass


def close_blockers(win) -> list:
    """창을 닫으면 죽는 것들. 비어 있으면 그냥 닫아도 된다.

    수집 세션을 함께 세는 이유: 닫기가 세션을 끝내는 것은 맞지만, 그것이
    **의도였는지**를 묻는 값은 긴 작업과 같다 (마지막 에피소드를 저장하기
    전일 수 있다).
    """
    out = []
    job = running_job(win)
    if job:
        out.append(job)
    worker = getattr(win, "worker", None)
    if worker is not None and worker.isRunning():
        out.append(tr("수집 세션"))
    return out
