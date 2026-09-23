"""제어 주기를 설정으로 뺀 것에 대한 인수 시험 (issue #1).

핵심 계약은 하나다: **저장했다고 해서 도는 세션의 주기가 바뀌지 않는다.**

그래서 값이 모듈 상수가 아니라 ``WorkerConfig`` 필드여야 한다. 모듈 수준 상수는
import 시점에 굳어서, 설정을 고쳐도 이미 뜬 워커에는 닿지 않고 -- 더 나쁘게는
같은 프로세스에서 새로 만든 워커에도 안 닿는다. 세션이 자기 값을 들고 다녀야
"바꾼 값은 다시 시작 후 적용됩니다" 가 거짓말이 아니게 된다.

여기서 확인하는 것:

1. 옛 모듈 상수가 **없다** (있으면 누군가 다시 상수로 읽고 있다는 뜻)
2. 기본값이 상수 시절과 **똑같다** -- 이 변경은 동작을 바꾸면 안 된다
3. 파생값(tick 당 이동)이 주기를 바꿔도 **속도를 보존한다**
4. 잘못된 설정값이 로봇을 못 띄우게 하지 않는다 (경고 후 기본값)
5. ``teleop_substeps`` 는 정수가 아니면 거부된다
6. 워커 하나의 값을 바꿔도 다른 워커는 안 바뀐다 (= 전역 상태가 아니다)
"""
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
sys.argv = ["t"]

from mstack.config.station import (  # noqa: E402
    ControlSpec,
    StationConfig,
    _control_from,
    load_station,
)
from mstack.collect import worker as W  # noqa: E402
from mstack.collect.worker import WorkerConfig  # noqa: E402


def cfg(**kw):
    return WorkerConfig(task_name="t", language_instruction="l",
                        data_root=tempfile.gettempdir(), **kw)


# ---------------------------------------------- 1. 옛 상수가 남아 있지 않다
GONE = ["RAMP_HZ", "RAMP_PERIOD_S", "TELEOP_SUBSTEPS", "APPROACH_SPEED",
        "RAMP_STEP", "APPROACH_DONE_RAD", "HOME_SPEED", "HOME_TICK_DQ"]
still = [n for n in GONE if hasattr(W, n)]
assert not still, (
    f"모듈 상수가 남아 있다: {still} -- import 시점에 굳으므로 설정이 닿지 않는다")
print("1. 옛 모듈 상수 제거 확인 OK")

# ---------------------------------------------- 2. 기본값이 상수 시절과 같다
#
# **비교 대상은 ControlSpec 의 dataclass 기본값이다, 지금 station 파일이
# 아니다.** 이 시험이 지키려는 것은 "상수를 설정으로 옮기면서 값이 바뀌지
# 않았다" 이고, station 파일은 그 뒤로 조작자가 의도적으로 바꾸는 자리다
# (2026-09-23: 기록 120 Hz, substeps 1). 로드된 값을 여기에 박아 두면
# 설정을 바꿀 때마다 이 시험이 "회귀" 라고 거짓말을 한다.
d = ControlSpec()
assert d.ramp_hz == 100.0, d.ramp_hz
assert d.teleop_substeps == 5, d.teleop_substeps
assert d.approach_speed == 2.0, d.approach_speed
assert d.approach_done_rad == 0.10, d.approach_done_rad
assert d.home_speed == 1.2, d.home_speed
assert abs(d.ramp_period_s - 0.01) < 1e-12, d.ramp_period_s
assert abs(d.ramp_step - 0.02) < 1e-12, d.ramp_step
print("2. ControlSpec 기본값이 상수 시절과 동일 OK")

# 이 리그가 실제로 쓰는 값은 station 파일에서 오고, 파생값은 **그 값과**
# 맞아야 한다 -- 특정 숫자가 아니라 관계를 본다.
c = cfg()
assert abs(c.ramp_period_s - 1.0 / c.ramp_hz) < 1e-12, c.ramp_period_s
assert abs(c.ramp_step - c.approach_speed / c.ramp_hz) < 1e-12, c.ramp_step
assert abs(c.home_tick_dq - c.home_speed / c.ramp_hz) < 1e-12, c.home_tick_dq
assert c.command_hz == c.fps * c.teleop_substeps, (
    c.command_hz, c.fps, c.teleop_substeps)
print(f"2b. 이 리그 설정과 파생값이 일관 OK "
      f"(기록 {c.fps} Hz x {c.teleop_substeps} = 명령 {c.command_hz:.0f} Hz)")

# ---------------------------------------------- 3. 주기를 바꿔도 속도가 보존된다
fast = cfg(ramp_hz=200.0)
assert abs(fast.ramp_step * fast.ramp_hz - c.ramp_step * c.ramp_hz) < 1e-12
assert abs(fast.home_tick_dq * fast.ramp_hz - c.home_tick_dq * c.ramp_hz) < 1e-12
# tick 당 이동은 절반이어야 한다 (주기가 두 배니까)
assert abs(fast.ramp_step - c.ramp_step / 2) < 1e-12, fast.ramp_step
print(f"3. ramp_hz 100->200 에서 tick 당 이동 {c.ramp_step}->{fast.ramp_step} "
      f"(속도 {c.approach_speed} rad/s 보존) OK")

# 120 Hz 명령 (20x6, 30x4) -- 다음 데이터세트의 목표값이 성립하는지
for fps, subs in ((20, 6), (30, 4)):
    k = cfg(fps=fps, teleop_substeps=subs)
    assert k.command_hz == 120, (fps, subs, k.command_hz)
print("   20x6 = 30x4 = 120 Hz 둘 다 정수배 OK")

# ---------------------------------------------- 4. 나쁜 설정이 로봇을 막지 않는다
base = ControlSpec()
assert _control_from({"ramp_hz": 5000}, base).ramp_hz == base.ramp_hz
assert _control_from({"ramp_hz": 0}, base).ramp_hz == base.ramp_hz
assert _control_from({"home_speed": "fast"}, base).home_speed == base.home_speed
assert _control_from({"approach_speed": None}, base).approach_speed == base.approach_speed
assert _control_from({}, base) == base
assert _control_from("쓰레기", base) == base
print("4. 범위 밖/형식 오류는 경고 후 기본값 OK (읽기가 실패하지 않는다)")

# ---------------------------------------------- 5. teleop_substeps 는 정수
for bad in (5.5, 0, -1, "여섯", None):
    got = _control_from({"teleop_substeps": bad}, base).teleop_substeps
    assert got == base.teleop_substeps, (bad, got)
assert _control_from({"teleop_substeps": 6}, base).teleop_substeps == 6
assert _control_from({"teleop_substeps": 6.0}, base).teleop_substeps == 6  # 6.0 == 6
print("5. teleop_substeps 정수 강제 OK (5.5/0/-1 거부, 6 수용)")

# ---------------------------------------------- 6. 전역 상태가 아니다
a, b = cfg(), cfg(ramp_hz=250.0)
assert a.ramp_hz == 100.0 and b.ramp_hz == 250.0
assert cfg().ramp_hz == 100.0, "다른 워커를 만들었더니 값이 옮았다 -- 전역 상태다"
print("6. 워커마다 독립된 값 OK (도는 세션이 안 바뀐다)")

# ---------------------------------------------- 7. 스테이션 yaml 이 실제로 읽힌다
st = load_station(reload=True)
assert isinstance(st.control, ControlSpec)
assert st.control.teleop_substeps >= 1
assert StationConfig().control == ControlSpec(), "기본 스테이션의 control 이 기본값이 아니다"
print(f"7. 스테이션 '{st.name}' 의 control 블록 읽기 OK "
      f"(명령 {st.fps * st.control.teleop_substeps} Hz)")

print("\n제어 주기 설정 인수 통과")
