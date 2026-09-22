"""One reader for both episode layouts: the frame table every consumer wants.

``knu-1.3.0`` stores one row per frame -- every series has the same length ``T``
because the 20 Hz loop sampled all of them together and wrote whatever was most
recent. ``knu-2.0.0`` stores each source at its own rate on its own time axis
(``t/agent``, ``t/wrist``, ``t/control``), because a 30 fps camera sampled at
20 Hz throws away a third of its frames and a 120 Hz leader throws away five in
six.

Consumers do not want to know which layout they got. ``frame_table()`` gives
both of them as a table, so doctor / viewers / converters / stats keep reading
rows -- and the resampling rule they were unknowingly relying on becomes one
documented function instead of something baked into the recording loop.

Why the control axis anchors the rows
-------------------------------------
Behaviour cloning learns ``observation -> action``, so the image must be
*older* than the action it is paired with. If the image were newer it would
already show the consequence of that action and the model could read the action
off the result. Measured on the 2,434-episode dataset: ``timing/action`` is
later than ``timing/<cam>_host`` in **33,226 of 33,226 rows**, i.e. the current
recording is causal by construction.

Anchoring the rows on the control axis keeps that property -- the action lands
exactly on a row and the image is the most recent one at or before it.
Anchoring on a camera axis inverts it for about half the rows, so this module
refuses to hand back actions in that mode rather than quietly producing a
leaking table.

``align``: what "most recent" means
-----------------------------------
Two clocks describe the same frame (see ``mstack/comm/camera_node.py``):

* ``t_device`` -- librealsense's own timestamp, domain ``global_time``, i.e.
  the camera clock mapped onto the host clock. Closest to when the photons
  were collected.
* ``t_host`` -- ``time.time()`` when the node received the frame.

They differ by a fixed per-model pipeline delay: measured 15.03 ms (D455 agent,
sd 0.20) and 8.60 ms (D405 wrist, sd 0.07). The spread is a tenth of a
millisecond, so this is a constant, not jitter.

* ``align="arrival"`` (default) selects by ``t_host`` -- **the rule deployment
  actually uses.** ``fr3_policy_client`` asks each camera for whatever is
  latest in its buffer, which is arrival order, so a table built this way puts
  the same per-camera skew into training that inference will see.
* ``align="capture"`` selects by ``t_device``, removing the 6.43 ms difference
  between the two cameras. Physically truer, and what a different rig with
  different pipeline delays would want.

Both are kept because "reproduce this rig" and "adapt to another rig" are both
requirements; the raw clocks are stored so the choice stays reversible.

``image_age`` is always measured against ``t_device`` regardless of ``align``:
it answers "how old is what this image shows", which does not depend on how the
frame was selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

#: 시간축 이름 -> 그 축에 실리는 1.3.0 의 timing 접두사.
#: 1.3.0 은 축이 없지만 `timing/<접두사>_host` 로 같은 정보를 갖고 있다.
CAMERA_AXES = {"agent": "agentview", "wrist": "eye_in_hand"}
CONTROL_AXIS = "control"

ALIGN_ARRIVAL = "arrival"
ALIGN_CAPTURE = "capture"
ALIGNS = (ALIGN_ARRIVAL, ALIGN_CAPTURE)


@dataclass
class FrameTable:
    """One row per sample, whatever the file's layout was."""

    #: (T,) 행 시각, 에피소드 시작 기준 초.
    t: np.ndarray
    #: obs 데이터셋 이름 -> (T, ...). 축이 달라도 전부 T 행이다.
    obs: dict
    #: (T, ...) 또는 None (anchor 가 control 이 아닐 때).
    actions: Optional[np.ndarray] = None
    #: 축 이름 -> (T,) 행 시각과 그 이미지가 실제로 찍힌 시각의 차 (초).
    #: 항상 t_device 기준이다 -- align 과 무관하게 "내용이 얼마나 오래됐나" 다.
    image_age: dict = field(default_factory=dict)
    #: 축 이름 -> 원본 표본 수. 1.3.0 에서는 카메라가 **만든** 프레임 수의
    #: 추정치(frame_no 범위)라, 저장된 행 수와 비교하면 얼마나 버렸는지 보인다.
    n_source: dict = field(default_factory=dict)
    rate: Optional[float] = None
    version: str = ""
    align: str = ALIGN_ARRIVAL

    def __len__(self) -> int:
        return int(self.t.shape[0])


def _episode_version(ep: Any) -> str:
    """이 에피소드가 어느 레이아웃인가. 판정은 ``t/`` 그룹의 유무로 한다 --
    attr 이 없거나 틀린 파일이 있어도 **구조가 진실**이기 때문이다.

    버전 문자열은 보고용이다. scene 파일은 그것을 에피소드가 아니라
    ``metadata`` 그룹에 적으므로 거기까지 올라가서 찾는다.
    """
    v = str(ep.attrs.get("dataset_version", "") or "")
    if not v:
        try:
            meta = ep.file.get("metadata")
            if meta is not None:
                v = str(meta.attrs.get("dataset_version", "") or "")
        except Exception:  # noqa: BLE001 - 보고용 값이라 못 읽어도 진행한다
            pass
    if _is_v2(ep):
        return v or "knu-2.0.0"
    return v or "knu-1.x"


def _is_v2(ep: Any) -> bool:
    return "t" in ep and hasattr(ep["t"], "keys")


def _collect_obs(grp: Any) -> dict:
    """``obs`` 아래 1단계 데이터셋들. 하위 그룹은 내려가지 않는다 --
    지금 스키마에 obs 하위 그룹이 없고, 생기면 여기서 명시적으로 다뤄야 한다."""
    out = {}
    obs = grp.get("obs")
    if obs is None:
        return out
    for k in obs.keys():
        d = obs[k]
        if hasattr(d, "shape"):
            out[k] = d
    return out


# ------------------------------------------------------------------ knu-1.3.0


def _table_v1(ep: Any, rate: Optional[float], align: str) -> FrameTable:
    """1.3.0 은 이미 표다. 재구성할 것이 없고, 시각만 채워 준다."""
    obs = {k: d[:] for k, d in _collect_obs(ep).items()}
    actions = ep["actions"][:] if "actions" in ep else None
    n_rows = len(next(iter(obs.values()))) if obs else (len(actions) if actions is not None else 0)

    timing = ep.get("timing")
    if timing is not None and "frame" in timing:
        t_abs = timing["frame"][:]
        t = t_abs - t_abs[0]
    else:
        # 옛 파일 (knu-1.0.x): 프레임 시각이 없다. 기록 주파수를 모르므로
        # 균등 간격을 **지어내지 않고** 인덱스를 초로 쓰지도 않는다 -- rate 를
        # 준 경우에만 그 주기로 채운다.
        if rate is None:
            raise ValueError(
                "이 에피소드에는 timing/frame 이 없어 행 시각을 모른다 "
                "(knu-1.3.0 이전). rate 를 넘기면 그 주기로 채운다.")
        t = np.arange(n_rows, dtype=np.float64) / float(rate)
        t_abs = None

    measured = None
    if n_rows > 1 and t[-1] > t[0]:
        measured = (n_rows - 1) / (t[-1] - t[0])
    if rate is not None and measured is not None and abs(measured - rate) > 0.5:
        raise ValueError(
            f"이 파일은 약 {measured:.1f} Hz 로 기록됐는데 rate={rate} 를 요청했다. "
            "1.3.0 에는 다시 뽑을 원본이 없어 재표본할 수 없다 -- 없는 것을 "
            "만들어 주느니 거부한다.")

    image_age, n_source = {}, {}
    if timing is not None and t_abs is not None:
        for axis, pre in CAMERA_AXES.items():
            dev = timing.get(f"{pre}_device")
            host = timing.get(f"{pre}_host")
            ref = dev if dev is not None else host
            if ref is not None:
                image_age[axis] = t_abs - ref[:]
            fn = timing.get(f"{pre}_frame_no")
            if fn is not None and len(fn) > 1:
                # 카메라가 만든 프레임 수(추정) -- 저장된 행 수와의 차이가
                # 폐기량이다. 1.3.0 에서 이 값을 볼 수 있는 유일한 경로다.
                n_source[axis] = int(fn[-1] - fn[0]) + 1
    n_source[CONTROL_AXIS] = n_rows

    return FrameTable(t=t, obs=obs, actions=actions, image_age=image_age,
                      n_source=n_source, rate=measured if rate is None else rate,
                      version=_episode_version(ep), align=align)


# ------------------------------------------------------------------ knu-2.0.0


def _axis_of(ds: Any, default: str = CONTROL_AXIS) -> str:
    return str(ds.attrs.get("axis", default) or default)


def _select_clock(ep: Any, axis: str, align: str) -> np.ndarray:
    """그 축에서 짝짓기에 쓸 시각. arrival 이면 도착, capture 면 장치 시각."""
    t_axis = ep["t"][axis][:]
    if align == ALIGN_CAPTURE or axis == CONTROL_AXIS:
        return t_axis                      # t/ 는 장치 시각이다 (설계 §1.3)
    host = ep.get(f"meta/{axis}/host")
    if host is None:
        # 도착 시각을 안 남긴 파일 -- 장치 시각으로 대신하되 조용히 하지 않는다.
        raise ValueError(
            f"align='arrival' 인데 meta/{axis}/host 가 없다. "
            "align='capture' 로 읽거나, 그 파일을 만든 수집기를 확인해라.")
    return host[:]


def _table_v2(ep: Any, rate: Optional[float], anchor: str, align: str) -> FrameTable:
    t_ctrl = ep["t"][CONTROL_AXIS][:]
    n_ctrl = len(t_ctrl)

    if anchor != CONTROL_AXIS:
        # 카메라 기준: 행이 이미지 시각 위에 놓이고 액션은 ZOH 가 된다 ->
        # 절반의 행에서 액션이 이미지보다 **먼저**가 되어 누출이다 (모듈 문서).
        rows = np.arange(len(ep["t"][anchor]))
        t_rows = ep["t"][anchor][:]
        keep_actions = False
    else:
        step = 1
        if rate is not None:
            ctrl_hz = ep.attrs.get("control_hz")
            if ctrl_hz is None and n_ctrl > 1:
                ctrl_hz = (n_ctrl - 1) / (t_ctrl[-1] - t_ctrl[0])
            ratio = float(ctrl_hz) / float(rate)
            step = int(round(ratio))
            if step < 1 or abs(ratio - step) > 1e-6:
                raise ValueError(
                    f"rate={rate} 가 명령 주기 {float(ctrl_hz):.3f} Hz 를 정수로 "
                    "나누지 못한다. 정수배가 아니면 위상이 미끄러져 '몇 번째 "
                    "tick 이 이 행인가' 가 행마다 달라진다 -- 기록기의 정수배 "
                    "규칙과 같은 이유로 거부한다.")
        rows = np.arange(0, n_ctrl, step)
        t_rows = t_ctrl[rows]
        keep_actions = True

    obs, image_age = {}, {}
    n_source = {a: len(ep["t"][a]) for a in ep["t"].keys()}

    for name, ds in _collect_obs(ep).items():
        axis = _axis_of(ds)
        if axis == CONTROL_AXIS and anchor == CONTROL_AXIS:
            obs[name] = ds[:][rows]
            continue
        if axis == anchor:
            obs[name] = ds[:][rows]
            continue
        clock = _select_clock(ep, axis, align)
        # 최신·인과: 행 시각 **이하** 에서 가장 최근. 보간하지 않는다.
        idx = np.searchsorted(clock, t_rows, side="right") - 1
        if (idx < 0).any():
            # 첫 행보다 앞선 표본이 하나도 없는 축이 있다. 그 행들은 버린다 --
            # 없는 관측을 0 이나 첫 프레임으로 채우면 조용히 거짓이 된다.
            first = int(np.argmax(idx >= 0))
            rows, t_rows, idx = rows[first:], t_rows[first:], idx[first:]
            obs = {k: v[first:] for k, v in obs.items()}
        obs[name] = ds[:][idx]
        if axis in CAMERA_AXES:
            image_age[axis] = t_rows - ep["t"][axis][:][idx]

    actions = None
    if keep_actions and "actions" in ep:
        actions = ep["actions"][:][rows]

    ft = FrameTable(t=t_rows, obs=obs, actions=actions, image_age=image_age,
                    n_source=n_source, rate=rate, version=_episode_version(ep),
                    align=align)
    _check_causal(ft, anchor)
    return ft


# ------------------------------------------------------------------ 공통


def _check_causal(ft: FrameTable, anchor: str) -> None:
    """이미지가 액션보다 **먼저**인지. 실측으로 33,226/33,226 성립하는 성질이라,
    깨지면 데이터가 아니라 이 모듈의 버그다."""
    if ft.actions is None or anchor != CONTROL_AXIS:
        return
    for axis, age in ft.image_age.items():
        if len(age) and float(np.min(age)) < 0:
            bad = int(np.argmin(age))
            raise AssertionError(
                f"{axis} 이미지가 행 {bad} 에서 액션보다 나중이다 "
                f"({float(age[bad])*1000:.2f} ms) -- 그 이미지에는 행동의 "
                "결과가 찍혀 있어 모델이 결과에서 행동을 역산할 수 있다.")


def frame_table(ep: Any, rate: Optional[float] = None, *,
                anchor: str = CONTROL_AXIS,
                align: str = ALIGN_ARRIVAL) -> FrameTable:
    """에피소드를 한 행 = 한 표본의 표로 돌려준다.

    Args:
        ep: 열린 HDF5 에피소드 그룹.
        rate: 행 주파수(Hz). ``None`` 이면 기록된 그대로. 2.0.0 에서는 명령
            주기의 약수여야 하고, 1.3.0 에서는 그 파일의 주파수와 같아야 한다.
        anchor: 행을 어느 축 위에 놓을지. 기본 ``"control"`` -- 유일하게
            액션을 함께 돌려주는 모드다 (모듈 문서의 인과성 참고).
        align: ``"arrival"`` (기본, 배포와 같은 규칙) 또는 ``"capture"``.

    Raises:
        ValueError: 정수배가 아닌 rate, 1.3.0 의 재표본 요청, 모르는 인자.
        AssertionError: 인과성이 깨진 표가 만들어졌을 때 (= 이 모듈의 버그).
    """
    if align not in ALIGNS:
        raise ValueError(f"align 은 {ALIGNS} 중 하나여야 한다: {align!r}")
    if anchor != CONTROL_AXIS and anchor not in CAMERA_AXES:
        raise ValueError(
            f"anchor 는 'control' 또는 {tuple(CAMERA_AXES)} 여야 한다: {anchor!r}")
    if _is_v2(ep):
        return _table_v2(ep, rate, anchor, align)
    if anchor != CONTROL_AXIS:
        raise ValueError(
            "knu-1.x 에는 축이 하나뿐이라 anchor 를 고를 수 없다 "
            "(모든 계열이 이미 같은 행에 있다).")
    return _table_v1(ep, rate, align)
