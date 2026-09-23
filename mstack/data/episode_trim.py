"""Trimming frames off the end of a recorded episode.

Why this exists
---------------
Saving is a key press, and the operator's hand is still on the leader when it
happens. Measured over 254 real episodes, the per-frame joint speed of the last
20 frames -- normalised by each episode's own median so tasks of different
reach are comparable -- looks like this:

    frame from end  -20  -18  -15  -12  -10   -8   -6   -4   -2   -1
    median x median 0.30 0.50 0.93 1.58 1.73 1.78 1.56 1.24 0.99 0.85
    p90             2.09 2.37 3.01 4.01 4.16 4.06 3.60 3.06 2.67 2.40

The take is already settling at -20 (0.30x), then speed *rises* to a bump
peaking around -10 to -7 before the recording stops. That bump is the reach for
the key, not the task, and it is the last thing a policy sees -- so it is
learned as "how to finish".

How far it is safe to cut
-------------------------
The real end of the task is the gripper release. Across the same episodes the
last gripper change sits at:

    p5 = 19,  p50 = 27,  p95 = 39,  max = 61  frames from the end

So a trim below ~19 frames leaves every recorded release intact, and anything
larger starts eating the release itself in the earliest 5%. `plan_trim` returns
that distance per episode so the caller can refuse the ones that would cut it,
rather than trusting a single global number.

The comparison is on indices, and that detail is the whole guard: the last
frame kept after cutting n is index `T-n-1`, and the release must land at or
before it. Comparing frame *counts* instead lets through exactly the n that
removes the release frame and nothing else -- the one value the guard exists
to catch. GRIPPER_MARGIN then keeps a couple of frames past it, so the take
still ends with "released, and holding still" rather than on the release tick.

Editing HDF5 in place
---------------------
Every dataset here is fixed-size (`maxshape == shape`), so `resize()` is not
available: each one is read, deleted and rewritten with the same dtype, chunks
and compression. That leaves the freed bytes as file slack -- HDF5 does not
return them to the filesystem -- which is what the existing repack step is for.
`trimmed` is written into the group's attrs so a second pass can tell an
already-trimmed take from a naturally short one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import h5py
import numpy as np

from mstack.data.frame_table import CONTROL_AXIS, axis_of, has_axis_attr

ARM_DIMS = 7
# 그리퍼 명령은 0/1 이산값이라, 이보다 큰 변화는 열림/닫힘 한 번을 뜻한다.
GRIPPER_STEP = 0.05

# ---- 한계는 **초**다, 프레임이 아니다 -------------------------------------
# 값 자체는 안 바뀌었다 -- 전부 20 Hz 기준 옛 프레임 수를 초로 옮긴 것이다
# (2 프레임 = 0.10초, 20 프레임 = 1.0초, 15 = 0.75초, 40 = 2.0초). 바뀐 것은
# **기준 주파수에 매이지 않는다**는 점이다. control 축을 120 Hz 로 올리면
# "20 프레임" 은 1.0초가 아니라 0.167초가 되어, 같은 상수가 조용히 6배
# 빡빡해진다. 카메라 축(30 fps)에서 세면 또 다른 값이 된다.
#
# 시각을 모르는 옛 파일(knu-1.0.x, timing 없음)에서는 프레임으로 되돌린다 --
# 그 파일들은 전부 20 Hz 로 찍혔으므로 아래 _FALLBACK_HZ 로 환산한다.
_FALLBACK_HZ = 20.0
#: 놓은 뒤 남겨야 할 최소 시간. 놓는 순간만 겨우 남기면 정책이 보는 "다 놓고
#: 끝난 상태"가 한 틱뿐이라, 마지막 동작이 사실상 학습되지 않는다.
GRIPPER_MARGIN_S = 0.10
#: 자르고도 남겨야 할 최소 길이. LeRobot 변환과 Δa 통계 모두 너무 짧으면 못 쓴다.
MIN_SECONDS = 1.0
#: 자동 제안이 한 번에 자를 수 있는 최대 시간.
SUGGEST_MAX_S = 0.75
#: 속도 곡선을 보여 주는 꼬리 구간.
TAIL_WINDOW_S = 2.0

# 옛 이름 -- 프레임 수로 말하던 시절의 값. 참조하는 코드가 남아 있을 수 있어
# 남기되, 판정에는 쓰지 않는다.
GRIPPER_MARGIN = int(round(GRIPPER_MARGIN_S * _FALLBACK_HZ))
MIN_FRAMES = int(round(MIN_SECONDS * _FALLBACK_HZ))


@dataclass
class TrimPlan:
    """자르기 한 번의 계획. **판정은 시각으로, 표시는 프레임으로.**

    ``n_trim`` 은 ``axis`` 축에서 세는 프레임 수다 -- 화면이 그 축을 보고
    있으므로 사람이 고르는 단위가 그것이다. 하지만 실제로 자르는 기준은
    ``t_cut`` 하나이고, 다른 축은 전부 그 시각으로 잘린다. 축마다 주파수가
    다르므로 "몇 프레임" 은 축을 떠나면 뜻이 없다.
    """

    demo: str
    n_frames: int              # axis 축의 전체 프레임 수
    n_trim: int                # axis 축에서 자를 프레임 수
    release_idx: int | None    # 마지막 그리퍼 변화가 *확립되는* control 행
    already: str | None        # 이미 자른 이력 (attrs["trimmed"])
    scene: bool = False        # scene-v1 에피소드 (표시용; 처리 경로는 동일)
    axis: str | None = None    # n_trim 을 세는 축. None 이면 단일축(knu-1.3.0)
    t_cut: float | None = None      # 남는 마지막 표본의 시각 (초)
    t_end: float | None = None      # 자르기 전 마지막 표본의 시각 (초)
    t_release: float | None = None  # 놓기가 확립된 시각 (초)
    t_start: float | None = None    # 첫 표본의 시각 (초)

    @property
    def result_frames(self) -> int:
        return self.n_frames - self.n_trim

    @property
    def timed(self) -> bool:
        """시각을 아는 파일인가. 아니면 프레임 수로 판정한다."""
        return self.t_cut is not None and np.isfinite(self.t_cut)

    @property
    def kept_seconds(self) -> float | None:
        if not self.timed or self.t_start is None:
            return None
        return float(self.t_cut - self.t_start)

    @property
    def cut_seconds(self) -> float | None:
        """잘려나가는 시간."""
        if not self.timed or self.t_end is None:
            return None
        return float(self.t_end - self.t_cut)

    @property
    def gripper_tail(self) -> int | None:
        """놓는 지점이 끝에서 몇 프레임째인가 -- **``axis`` 축의 프레임으로**.

        ``release_idx`` 는 ``actions`` 에서 찾으므로 control 축의 행 번호다.
        세는 축이 카메라면 그 번호를 그대로 빼면 안 된다: 실측
        (scene_024/episode_000) 에서 사진 211장 − control 754행 = **−543**
        이 화면에 찍혔다. 시각으로 옮긴 다음 그 축에서 센다.

        시각을 모르는 옛 파일은 축이 하나뿐이라 예전 셈이 그대로 맞다.
        """
        if self.release_idx is None:
            return None
        if not self.timed or self.t_release is None or self.t_end is None:
            return self.n_frames - self.release_idx
        span = self._axis_span
        if not span:
            return None
        return int(round((self.t_end - self.t_release) / span))

    @property
    def _axis_span(self) -> float | None:
        """``axis`` 축의 표본 간격 (초)."""
        if (self.t_end is None or self.t_start is None or self.n_frames < 2
                or self.t_end <= self.t_start):
            return None
        return (self.t_end - self.t_start) / (self.n_frames - 1)

    @property
    def max_safe_seconds(self) -> float | None:
        """놓기와 그 뒤 여유를 남기면서 잘라낼 수 있는 최대 시간."""
        if self.release_idx is None or not self.timed:
            return None
        if self.t_release is None or self.t_end is None:
            return None
        return max(0.0, float(self.t_end - self.t_release - GRIPPER_MARGIN_S))

    @property
    def max_safe_trim(self) -> int | None:
        """같은 것을 ``axis`` 축 프레임 수로. 시각을 모르면 옛 인덱스 셈."""
        if self.release_idx is None:
            return None
        secs = self.max_safe_seconds
        if secs is None:
            return max(0, self.n_frames - 1 - self.release_idx - GRIPPER_MARGIN)
        span = self._axis_span
        if not span:
            return None
        return int(max(0.0, secs) // span)

    @property
    def cuts_gripper(self) -> bool:
        """남는 끝이 '놓기 + 여유' 뒤까지 가는가.

        시각을 알면 시각으로 본다: 자르는 선이 놓기보다 GRIPPER_MARGIN_S 이상
        뒤에 있어야 한다. 이래야 축을 바꿔도 뜻이 같다 -- 프레임으로 따지면
        카메라 축(30 fps)과 control 축(20/120 Hz)에서 서로 다른 판정이 나온다.

        시각을 모르는 옛 파일은 인덱스로 따진다. 프레임 *수*와 비교하면 정확히
        하나가 어긋나서, 놓는 프레임만 딱 잘려나가는 값이 통과한다
        (실측: 120프레임/놓기 90번 에피소드에서 n=30).
        """
        if self.release_idx is None:
            return False
        if self.timed and self.t_release is not None:
            return self.t_cut < self.t_release + GRIPPER_MARGIN_S
        return self.result_frames - 1 < self.release_idx + GRIPPER_MARGIN

    @property
    def too_short(self) -> bool:
        kept = self.kept_seconds
        if kept is None:
            return self.result_frames < MIN_FRAMES
        return kept < MIN_SECONDS

    @property
    def blocked(self) -> str | None:
        # scene 에피소드도 자른다 (2026-08-14 결정: 실패/튀는 궤적 삭제와 함께
        # HDF5 큐레이션 편집 허용). 자르기는 프레임축 데이터셋만 줄이고
        # uid·번호·instruction 은 손대지 않으므로 slot 계보에 영향이 없다.
        if self.n_trim <= 0:
            return "자를 프레임 수가 0입니다"
        if self.too_short:
            kept = self.kept_seconds
            if kept is None:
                return f"남는 프레임이 {self.result_frames}개뿐입니다 (최소 {MIN_FRAMES})"
            return (f"남는 길이가 {kept:.2f}초뿐입니다 (최소 {MIN_SECONDS:g}초)")
        if self.cuts_gripper:
            if self.timed and self.t_release is not None:
                return (f"물체를 놓는 순간이 {self.t_release:.2f}초인데 "
                        f"{self.t_cut:.2f}초에서 자릅니다 — 놓은 뒤 "
                        f"{GRIPPER_MARGIN_S:g}초는 남겨야 합니다")
            return (f"물체를 놓는 프레임이 끝에서 {self.gripper_tail}번째입니다 "
                    f"— 놓은 뒤 여유를 남겨야 합니다")
        return None


def _release_idx(actions: np.ndarray) -> int | None:
    """Frame at which the last gripper change has taken effect.

    `diff[i] != 0` means the value differs between frame i and i+1, so the new
    command is first *observed* at i+1 -- that later frame is the one that has
    to survive, not i.
    """
    if actions.shape[1] <= ARM_DIMS:
        return None
    changes = np.where(np.abs(np.diff(actions[:, ARM_DIMS])) > GRIPPER_STEP)[0]
    return int(changes[-1]) + 1 if len(changes) else None


def _axis_of_name(grp, name: str) -> str | None:
    """이 데이터셋이 어느 시간축에 놓여 있는가. 모르면 None.

    세 가지로 정해진다. ``attrs["axis"]`` 가 정본이지만, 축 자체를 적는
    ``t/<축>`` 과 그 축의 부가정보인 ``meta/<축>/*`` 는 경로가 이미 축을
    말하고 있어서 태그가 붙어 있지 않다 (실측: 31개 중 12개).
    """
    ds = grp[name]
    if has_axis_attr(ds):
        return axis_of(ds)
    parts = name.split("/")
    if len(parts) == 2 and parts[0] == "t":
        return parts[1]
    if len(parts) == 3 and parts[0] == "meta":
        return parts[1]
    return None


def _keep_at(grp, t_cut: float) -> dict:
    """자르는 선 ``t_cut`` 에서 축마다 몇 행을 남기는가.

    ``side="right"`` 는 "t_cut 이하인 표본의 개수" 를 준다. 선 위에 정확히
    놓인 표본은 **남긴다** -- 그 행이 쓰는 마지막 관측이다.
    """
    keep = {a: int(np.searchsorted(grp["t"][a][:], t_cut, side="right"))
            for a in grp["t"]}
    empty = sorted(a for a, k in keep.items() if k < 1)
    if empty:
        raise ValueError(
            f"{', '.join(empty)} 축에 남는 표본이 없다 (자르는 선 {t_cut:.3f}s). "
            "잘못 자른 파일을 만드느니 멈춘다.")
    return keep


def _keep_by_axis(grp, keep_ctrl: int) -> "tuple[dict[str, int], float]":
    """축마다 몇 행을 남길지. 기준은 **행 수가 아니라 시각**이다.

    control 축에서 남는 마지막 행의 시각이 자르는 선이고, 다른 축은 그
    시각 **이하**인 표본만 남는다. 프레임 수로 자르면 안 되는 이유는
    주파수가 축마다 다르기 때문이다 -- control 5행(0.25초)은 30 fps
    카메라로 7~8장이고 100 Hz 명령으로 25행이다.

    ``side="right"`` 는 "t_cut 이하인 표본의 개수" 를 준다. 자르는 선 위에
    정확히 놓인 표본은 **남긴다** -- 그 행이 쓰는 마지막 관측이다.
    """
    t_ctrl = grp["t"][CONTROL_AXIS][:]
    t_cut = float(t_ctrl[keep_ctrl - 1])
    keep: dict[str, int] = {}
    for axis in grp["t"]:
        keep[axis] = int(np.searchsorted(grp["t"][axis][:], t_cut, side="right"))
    if keep.get(CONTROL_AXIS) != keep_ctrl:
        raise ValueError(
            f"control 축이 {keep[CONTROL_AXIS]} 행으로 잘리는데 {keep_ctrl} 행을 "
            "기대했다 -- t/control 이 단조증가가 아니다.")
    empty = sorted(a for a, k in keep.items() if k < 1)
    if empty:
        raise ValueError(
            f"{', '.join(empty)} 축에 남는 표본이 없다 (자르는 선 {t_cut:.3f}s). "
            "잘못 자른 파일을 만드느니 멈춘다.")
    return keep, t_cut


def _episode_group(f: h5py.File, demo: str):
    """legacy 는 data/demo_N, scene(scene-v1)은 루트 episode_NNN.
    에피소드 안쪽 페이로드는 동일하다. (그룹, scene 여부) 를 돌려준다."""
    if demo in f:
        return f[demo], True
    return f["data"][demo], False


def _clock(grp, axis: str | None) -> "np.ndarray | None":
    """``axis`` 축의 시각 배열. 모르면 None.

    knu-2.0.0 은 ``t/<축>`` 에 있고, 1.3.0 은 축이 하나뿐이라 ``timing/frame``
    이 그 역할을 한다 (절대 시각이라 첫 값 기준으로 옮긴다). 그보다 옛
    파일에는 아무것도 없어서 None 이고, 그때는 프레임 수로 판정한다.
    """
    t_grp = grp.get("t")
    if t_grp is not None and hasattr(t_grp, "keys"):
        name = axis or CONTROL_AXIS
        if name in t_grp:
            return t_grp[name][:]
        return None
    timing = grp.get("timing")
    if timing is not None and "frame" in timing:
        t = timing["frame"][:]
        return t - t[0]
    return None


def _plan_one(grp, demo: str, n_trim: int, axis: str | None,
              scene: bool) -> TrimPlan:
    a = grp["actions"][:]
    rel = _release_idx(a)
    t_ctrl = _clock(grp, CONTROL_AXIS)
    t_axis = _clock(grp, axis) if axis else t_ctrl
    n_axis = len(t_axis) if t_axis is not None else int(a.shape[0])
    keep = n_axis - int(n_trim)

    t_cut = t_end = t_start = t_release = None
    if t_axis is not None and len(t_axis):
        t_start, t_end = float(t_axis[0]), float(t_axis[-1])
        if 1 <= keep <= len(t_axis):
            t_cut = float(t_axis[keep - 1])
    if rel is not None and t_ctrl is not None and rel < len(t_ctrl):
        t_release = float(t_ctrl[rel])
    return TrimPlan(
        demo=demo, n_frames=n_axis, n_trim=int(n_trim), release_idx=rel,
        already=grp.attrs.get("trimmed"), scene=scene, axis=axis,
        t_cut=t_cut, t_end=t_end, t_start=t_start, t_release=t_release)


def plan_trim(path: str, demos: list[str], n_trim: int, *,
              axis: str | None = None) -> list[TrimPlan]:
    """Checks, without writing anything, what trimming `n_trim` would do.

    ``axis`` 는 ``n_trim`` 을 세는 축이다. 화면이 카메라를 보고 있으면
    ``"agent"`` 를 주면 되고, 그러면 자르는 선은 그 카메라의 프레임 시각이
    된다. None 이면 control 축 (옛 동작).
    """
    out: list[TrimPlan] = []
    with h5py.File(path, "r") as f:
        for demo in demos:
            grp, scene = _episode_group(f, demo)
            out.append(_plan_one(grp, demo, n_trim, axis, scene))
    return out


def _ctrl_hz(grp) -> float:
    """control 축의 실제 주파수. 모르면 옛 기본값(20 Hz)."""
    hz = grp.attrs.get("control_hz")
    if hz:
        return float(hz)
    t = _clock(grp, CONTROL_AXIS)
    if t is not None and len(t) > 1 and np.isfinite(t[-1]) and t[-1] > t[0]:
        return float((len(t) - 1) / (t[-1] - t[0]))
    return _FALLBACK_HZ


def tail_speed(path: str, demo: str, k: int | None = None) -> np.ndarray:
    """꼬리 구간의 프레임당 팔 속도 (에피소드 자기 중앙값 단위).

    조작자가 자를 지점을 고르는 곡선이다. 구간 길이는 **초**로 정한다 --
    control 축이 20 Hz 일 때의 40 프레임과 120 Hz 일 때의 40 프레임은 각각
    2.0초와 0.33초라, 프레임으로 고정하면 창이 조용히 6배 좁아진다.
    """
    with h5py.File(path, "r") as f:
        grp = _episode_group(f, demo)[0]
        a = grp["actions"][:]
        if k is None:
            k = max(2, int(round(TAIL_WINDOW_S * _ctrl_hz(grp))))
    v = np.abs(np.diff(a[:, :ARM_DIMS], axis=0)).max(axis=1)
    med = float(np.median(v)) or 1e-9
    return v[-k:] / med


def suggest_trim(path: str, demo: str, max_trim: int | None = None, *,
                 axis: str | None = None) -> int:
    """자동 제안: 이미 잦아든 자리에서 끝나도록 자를 프레임 수.

    끝에서부터 걸어 올라오며 속도가 에피소드 자기 중앙값보다 높은 동안만
    센다 -- 마무리 튐만 자르고 그 앞은 건드리지 않는다. 조용히 끝난 테이크면
    0 이고, 그것이 편집이 필요 없는 테이크의 답이다.

    계산은 control 축에서 하고, 돌려주는 값은 ``axis`` 축의 프레임 수다 --
    화면이 그 축으로 세고 있기 때문이다. 상한도 초로 정한다.
    """
    with h5py.File(path, "r") as f:
        grp = _episode_group(f, demo)[0]
        hz = _ctrl_hz(grp)
        t_ctrl = _clock(grp, CONTROL_AXIS)
        t_axis = _clock(grp, axis) if axis else t_ctrl
    cap = int(round(SUGGEST_MAX_S * hz)) if max_trim is None else int(max_trim)
    cap = max(0, cap)
    v = tail_speed(path, demo, k=cap + 1)
    n = 0
    for x in reversed(v):
        if x <= 1.0 or n >= cap:
            break
        n += 1
    if n == 0 or axis is None or t_ctrl is None or t_axis is None:
        return n
    # control 축의 n 프레임을 시각으로 옮기고, 그 시각을 axis 축 프레임 수로.
    keep_ctrl = len(t_ctrl) - n
    if keep_ctrl < 1:
        return 0
    t_cut = float(t_ctrl[keep_ctrl - 1])
    return int(len(t_axis) - np.searchsorted(t_axis, t_cut, side="right"))


def trim_tail(path: str, demo: str, n_trim: int, *,
              axis: str | None = None) -> int:
    """Drops the last `n_trim` frames of one episode. Returns the new length.

    ``n_trim`` 은 ``axis`` 축에서 센다 (기본값은 control). 어느 축으로 세든
    실제로 자르는 기준은 그 축에서 남는 마지막 표본의 **시각** 하나이고,
    나머지 축은 전부 그 시각으로 잘린다.

    Raises ValueError if the plan is blocked -- the caller is expected to have
    shown `TrimPlan.blocked` to the operator first, so reaching here with a bad
    plan is a bug, not a user mistake.
    """
    plan = plan_trim(path, [demo], n_trim, axis=axis)[0]
    if plan.blocked:
        raise ValueError(plan.blocked)
    keep = plan.result_frames
    with h5py.File(path, "a") as f:
        grp, _scene = _episode_group(f, demo)   # 양포맷 동일 처리
        names: list[str] = []
        grp.visititems(
            lambda name, obj: names.append(name)
            if isinstance(obj, h5py.Dataset) else None)

        # 어느 데이터셋을 **몇 행까지** 남기는가. **쓰기 전에 전부 정한다** --
        # 중간에 멈추면 일부만 잘린 파일이 남고, 그건 되돌릴 수 없다.
        per_axis = any(has_axis_attr(grp[n]) for n in names)
        targets: dict[str, int] = {}
        if per_axis:
            # knu-2.0.0: 축마다 주파수가 달라 "뒤에서 n 프레임" 을 그대로
            # 적용할 수 없다. control 축에서 남는 마지막 행의 시각을 자르는
            # 선으로 삼고, 각 축은 그 시각 이하만 남긴다.
            if "t" not in grp or CONTROL_AXIS not in grp["t"]:
                # 축 태그는 붙어 있는데 시간축이 없다. 그러면 자를 선을 정할
                # 방법이 없고, 행 수로 자르면 카메라가 control 과 다른 만큼
                # 잘린다 -- 잘못 자른 파일을 만드느니 멈춘다.
                raise ValueError(
                    "axis 태그는 있는데 t/control 이 없다 -- 시각으로 자를 "
                    "기준이 없다. 축마다 주파수가 달라 행 수로는 자를 수 없다.")
            if axis is None:
                # control 축에서 세는 경우. 단조성까지 함께 확인한다.
                keep_axis, t_cut = _keep_by_axis(grp, keep)
            else:
                if plan.t_cut is None:
                    raise ValueError(
                        f"{axis} 축에서 자를 선을 정하지 못했다 "
                        f"(남길 프레임 {keep}).")
                t_cut = float(plan.t_cut)
                keep_axis = _keep_at(grp, t_cut)
            unknown = []
            for name in names:
                ax = _axis_of_name(grp, name)
                if ax is None:
                    unknown.append(name)
                elif ax in keep_axis:
                    targets[name] = keep_axis[ax]
                else:
                    unknown.append(f"{name} (axis={ax!r} 에 t/{ax} 가 없다)")
            if unknown:
                # 프레임축인데 축을 모르는 것이 있으면 그것만 안 잘려 남는다 --
                # 길이가 어긋난 파일은 조용히 잘못 읽힌다. 멈추는 편이 낫다.
                raise ValueError(
                    "축을 알 수 없는 데이터셋이 있다: " + ", ".join(unknown) +
                    ". attrs['axis'] 를 붙이거나 t/<축> · meta/<축>/ 아래로 "
                    "옮겨야 한다.")
        else:
            # knu-1.3.0 이하: 축이 하나뿐이라 길이가 곧 판별이다.
            keep_axis, t_cut = {CONTROL_AXIS: keep}, None
            targets = {n: keep for n in names if grp[n].shape[0] == plan.n_frames}

        for name, n_keep in targets.items():
            ds = grp[name]
            if ds.shape[0] == n_keep:
                continue            # 이 축은 자를 것이 없다
            data = ds[:n_keep]
            spec = {"dtype": ds.dtype, "chunks": ds.chunks,
                    "compression": ds.compression,
                    "compression_opts": ds.compression_opts}
            if ds.chunks is not None:
                # 청크의 첫 축이 남길 길이보다 크면 생성이 실패한다.
                spec["chunks"] = (min(ds.chunks[0], n_keep), *ds.chunks[1:])
            # **attrs 를 그대로 옮긴다.** 지우고 다시 만들면 attrs 도 같이
            # 사라지는데, knu-2.0.0 에서는 그중 하나가 axis 다 -- 태그를 잃은
            # 파일은 다음 트림에서 "축을 알 수 없다"로 막히고, frame_table 도
            # 그 계열을 control 축으로 잘못 읽는다.
            attrs = dict(ds.attrs)
            del grp[name]
            new = grp.create_dataset(name, data=data, **spec)
            for k, v in attrs.items():
                new.attrs[k] = v
        # **num_samples 는 언제나 control 행 수다.** 세는 축이 카메라일 수
        # 있으므로 keep 을 그대로 쓰면 안 된다 -- 화면은 agent 프레임으로
        # 세지만 이 attr 은 에피소드 길이로 읽히는 값이다.
        keep = int(keep_axis.get(CONTROL_AXIS, keep))
        grp.attrs["num_samples"] = keep
        prev = grp.attrs.get("trimmed", "")
        stamp = f"{time.strftime('%Y-%m-%d %H:%M')} -{n_trim}f"
        if t_cut is not None:
            cut = ", ".join(f"{a} {grp['t'][a].shape[0]}" for a in sorted(keep_axis))
            stamp += f" @{t_cut:.3f}s ({cut})"
        grp.attrs["trimmed"] = f"{prev}; {stamp}" if prev else stamp
        if _scene:
            # 트림은 uid·개수를 안 바꿔 어떤 개수 검사에도 안 걸린다 --
            # 편집 마커가 없으면 resume 이 트림 전 버전을 Hub 에 남긴다.
            from mstack.data.edit_marker import mark_scene_edited
            mark_scene_edited(f["metadata"])
    return keep
