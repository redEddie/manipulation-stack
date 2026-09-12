"""긴 작업 하나를 세 곳이 같은 답으로 본다 (2026-09-13 조작자 요청).

재압축·변환·업로드·프록시 굽기는 몇 분~몇십 분이고 그동안 데이터셋 파일을
쥔다. 그 사실을 (1) 상태바가 말하고, (2) 파일을 바꾸는 버튼이 잠기고,
(3) 창을 닫을 때 한 번 묻는다. 셋이 `shared/jobs.py` 하나를 본다.

로봇도 카메라도 필요 없다 (offscreen). 창 대신 스텁에 붙여 검사한다 --
검사 대상이 순수 함수와 위젯 토글이라 진짜 창이 필요 없다.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import helpers  # noqa: E402
helpers.isolate_state()

from PyQt6.QtCore import QProcess  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

app = QApplication.instance() or QApplication([])

from apps.workspace.models import ProcessRegistry  # noqa: E402
from apps.workspace.shared import jobs  # noqa: E402
from mstack.gui.text_utils import parse_progress_fraction  # noqa: E402

# ------------------------------------------------- 1. 진행률 읽기 (순수)
# 자식마다 찍는 꼴이 다르다. 못 읽으면 None 이어야 한다 -- 그때는 화면이
# 경과 시간만 말하고, **거짓 ETA 를 만들지 않는다.**
cases = {
    "  45%|####      | 3/7 [00:12<00:14]": 0.45,
    "  진행   12.3%  (120/480 MB)  a.hdf5": 0.123,
    "[3/27] 업로드 중: scene_003.hdf5 (612 MB)": 3 / 27,
    "쓰기 완료 (3.2s) -- 내용 검증 중...": None,
    "": None,
}
for line, want in cases.items():
    got = parse_progress_fraction(line)
    if want is None:
        assert got is None, (line, got)
    else:
        assert got is not None and abs(got - want) < 1e-6, (line, got, want)
print("1. 진행률 읽기 OK (%, i/N, 못 읽으면 None)")


class _NoOp:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _FakeProc:
    def __init__(self, running: bool) -> None:
        self._r = running

    def state(self):
        return (QProcess.ProcessState.Running if self._r
                else QProcess.ProcessState.NotRunning)


class _Win:
    """jobs 가 창에서 읽는 것만 흉내 낸다."""

    def __init__(self) -> None:
        self.procs = ProcessRegistry()
        self.worker = None
        self.sb_job = QPushButton("")          # setText/setVisible 만 쓴다
        self.trim_ops = _NoOp()
        self.delete_ops = _NoOp()
        for name in jobs.LOCKED_BUTTONS:
            setattr(self, name, QPushButton(name))


# --------------------------------------------- 2. 무엇이 도는가 (한 통로)
win = _Win()
assert jobs.running_job(win) == "", jobs.running_job(win)
win.procs.repack_process = _FakeProc(True)
assert jobs.running_job(win) == "재압축", jobs.running_job(win)
win.procs.repack_process = _FakeProc(False)
assert jobs.running_job(win) == ""
# 파이프라인(전체 처리)도 센다 -- 예전 busy_reason 은 이것을 못 봤다.
win.procs.pipeline_proc = _FakeProc(True)
assert jobs.running_job(win) == "전체 처리"
win.procs.pipeline_proc = None
print("2. running_job OK (재압축·전체 처리·한가함)")

# ------------------------------------------------------- 3. 잠금과 풀기
jobs.start_job(win, "재압축 3개")
assert jobs.running_job(win) == "재압축 3개"
for name in jobs.LOCKED_BUTTONS:
    assert not getattr(win, name).isEnabled(), name
assert win.procs.job_locked is True
jobs.end_job(win)
for name in jobs.LOCKED_BUTTONS:
    assert getattr(win, name).isEnabled(), name
assert win.procs.job_locked is False
print("3. 잠금/풀기 OK (5개 버튼)")

# --------------------------------------------------- 4. 남은 시간 문구
jobs.start_job(win, "재압축")
win.procs.job_t0 = time.monotonic() - 60.0          # 1분 경과
# 90초 아래는 초로 적는다 -- "1분째" 는 45초에도 75초에도 같은 글자가 되어
# 도는지 멈췄는지 알 수 없다.
assert "60초째" in jobs.job_status_text(win), jobs.job_status_text(win)
win.procs.job_t0 = time.monotonic() - 120.0         # 2분 경과
assert "2분째" in jobs.job_status_text(win), jobs.job_status_text(win)
# 진행률을 모르면 경과만 -- 5% 아래도 추정하지 않는다 (초반 비율은 못 믿는다).
jobs.job_progress(win, 0.02)
assert "남은" not in jobs.job_status_text(win), jobs.job_status_text(win)
jobs.job_progress(win, 0.5)                          # 절반 -> 남은 약 2분
text = jobs.job_status_text(win)
assert "50%" in text and "남은 약 2분" in text, text
jobs.end_job(win)
assert jobs.job_status_text(win) == ""
assert not win.sb_job.isVisible()
print("4. 남은 시간 문구 OK (경과만 / 절반이면 남은 약 2분)")


# ------------------------------------------------- 5. 닫을 때 묻는 이유
class _Worker:
    def isRunning(self):
        return True


win2 = _Win()
assert jobs.close_blockers(win2) == []
jobs.start_job(win2, "LeRobot 변환")
win2.worker = _Worker()
assert jobs.close_blockers(win2) == ["LeRobot 변환", "수집 세션"], jobs.close_blockers(win2)
print("5. 닫기 확인의 이유 OK")

# --------------------------------------- 6. "지금 읽는 중" 표시 (set_busy)
# 켜는 것보다 **끄는 것**이 중요하다 -- 안 끄면 화면이 영원히 "읽는 중" 이라고
# 거짓말한다. 이 표시는 큐레이션 로딩이 길어서 넣은 것이다 (2026-09-12).
from PyQt6.QtWidgets import QLabel  # noqa: E402

from apps.workspace.shared.busy import set_busy  # noqa: E402

win3 = _Win()
win3.hud_busy = QLabel("")
win3.hud_busy.setVisible(False)
set_busy(win3, "27개 파일을 분석하는 중")
assert win3.hud_busy.isVisible() and "27개" in win3.hud_busy.text(), win3.hud_busy.text()
set_busy(win3)
assert not win3.hud_busy.isVisible() and win3.hud_busy.text() == "", win3.hud_busy.text()
# 라벨이 아직 없는 창(짓는 도중)에서도 죽지 않는다
set_busy(_Win(), "아무거나")
print("6. set_busy OK (켜기·끄기·라벨 없는 창)")

print("test_jobs OK")
