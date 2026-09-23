"""knu-1.3.0 에피소드를 같은 내용의 knu-2.0.0 으로 옮긴다 (테스트 공용).

**왜 이것이 먼저 필요한가.** 지금 2.0.0 파일이 하나도 없다. 그래서 소비자를
``frame_table`` 로 바꿔도 스위트는 맞든 틀리든 통과한다 -- 추상화만 늘고 검증은
0 이 된다. 이 헬퍼가 있어야 "두 레이아웃에서 같은 결과가 나오는가" 를 단언할 수
있고, 그 단언이 C5 의 유일한 안전장치다.

**아무것도 지어내지 않는다.** 실제 에피소드가 고른 프레임과 그 실제 시각만
옮긴다. 한때 버려진 프레임을 "고른 것보다 33.3 ms 앞" 에 합성해 넣으려 했는데,
123개 중 67개가 앞 행의 시각보다 앞에 놓였다 -- ``timing/frame`` 은 카메라를 읽은
*뒤* 찍히므로 그 사이(중앙 1.16 ms)에 도착한 프레임이 실재하고, 어디에 두어야
하는지는 파일만으로 알 수 없다. 그래서 솎아내기 동작은 합성 격자로 따로 본다
(``test_frame_table.py``).

그 결과 이 헬퍼가 만드는 2.0.0 은 **축마다 길이가 같다**. 레이아웃(축·`t/`·
`meta/`·`axis` attr)은 진짜 2.0.0 이지만 솎아낼 것이 없다는 뜻이고, 그래서
"같은 내용이면 같은 표가 나온다" 를 정확히 물을 수 있다.
"""
from __future__ import annotations

import h5py
import numpy as np

#: 1.3.0 의 timing 접두사 -> 2.0.0 의 시간축 이름.
AXIS_OF_CAM = {"agentview": "agent", "eye_in_hand": "wrist"}
#: 그 카메라의 이미지 데이터셋 이름.
IMAGE_OF_CAM = {"agentview": "agentview_rgb", "eye_in_hand": "eye_in_hand_rgb"}


def write_v2(ep1: h5py.Group, path: str, *, group: str = "e",
             control_hz: float = 20.0) -> str:
    """``ep1`` (knu-1.3.0 에피소드) 를 ``path`` 에 2.0.0 으로 쓴다.

    Args:
        ep1: 열린 1.3.0 에피소드 그룹. ``timing/frame`` 과 카메라별
            ``timing/<cam>_device`` / ``_host`` 가 있어야 한다.
        path: 만들 파일 경로.
        group: 에피소드 그룹 이름.
        control_hz: ``control_hz`` attr 로 찍을 값.

    Returns:
        ``path`` 그대로.
    """
    tim = ep1["timing"]
    t_row = tim["frame"][:]
    t0 = float(t_row[0])

    with h5py.File(path, "w") as g:
        e = g.create_group(group)
        e.attrs["dataset_version"] = "knu-2.0.0"
        e.attrs["t0_wall"] = t0
        e.attrs["control_hz"] = float(control_hz)
        e.attrs["num_samples"] = int(len(t_row))
        # 1.3.0 의 에피소드 attrs 를 그대로 옮긴다 -- 소비자가 읽는 것들
        # (instruction, success, collector, scene_id ...) 이 여기 있다.
        for k, v in ep1.attrs.items():
            if k not in e.attrs:
                e.attrs[k] = v

        tg = e.create_group("t")
        mg = e.create_group("meta")
        tg.create_dataset("control", data=t_row - t0)
        for cam, axis in AXIS_OF_CAM.items():
            dev = tim.get(f"{cam}_device")
            host = tim.get(f"{cam}_host")
            if dev is None or host is None:
                continue
            # t/ 는 **장치 시각** 이다 (DESIGN_knu_2_0_0 §1.3) -- 도착 시각을
            # 쓰면 기종마다 다른 고정 전송 지연이 축에 섞여 들어간다.
            tg.create_dataset(axis, data=dev[:] - t0)
            mg.create_dataset(f"{axis}/host", data=host[:] - t0)
            for extra in ("frame_no", "node_seq"):
                src = tim.get(f"{cam}_{extra}")
                if src is not None:
                    mg.create_dataset(f"{axis}/{extra}", data=src[:])
        rs = tim.get("robot_state")
        if rs is not None:
            mg.create_dataset("control/robot_state", data=rs[:] - t0)

        # obs: 카메라는 자기 축, 나머지는 control 축
        img_axis = {IMAGE_OF_CAM[c]: a for c, a in AXIS_OF_CAM.items()
                    if a in tg}
        o = e.create_group("obs")
        for name in ep1["obs"].keys():
            src = ep1[f"obs/{name}"]
            d = o.create_dataset(name, data=src[:])
            d.attrs["axis"] = img_axis.get(name, "control")
        for name in ("actions", "actions_ee", "rewards", "dones"):
            if name in ep1:
                d = e.create_dataset(name, data=ep1[name][:])
                d.attrs["axis"] = "control"
    return path


def axes_of(path: str, group: str = "e") -> dict:
    """만든 파일의 축별 길이. 시험이 "정말 축이 나뉘었나" 를 확인할 때 쓴다."""
    with h5py.File(path, "r") as g:
        return {k: int(len(g[f"{group}/t/{k}"])) for k in g[f"{group}/t"].keys()}


def assert_same_table(a, b, *, what: str = "") -> None:
    """두 FrameTable 이 같은 내용인지. 다르면 **어디가** 다른지 말한다."""
    assert len(a) == len(b), f"{what}: 행 수가 다르다 {len(a)} vs {len(b)}"
    assert set(a.obs) == set(b.obs), (
        f"{what}: obs 키가 다르다 {sorted(set(a.obs) ^ set(b.obs))}")
    for k in a.obs:
        assert np.array_equal(a.obs[k], b.obs[k]), f"{what}: obs/{k} 가 다르다"
    if a.actions is None or b.actions is None:
        assert a.actions is b.actions, f"{what}: actions 유무가 다르다"
    else:
        assert np.array_equal(a.actions, b.actions), f"{what}: actions 가 다르다"
