"""트림이 프레임 축을 **길이로 추정하지 않는다** (C5a).

옛 코드:

    if ds.shape[0] != plan.n_frames:
        continue        # 프레임축이 아닌 데이터셋은 건드리지 않는다

``plan.n_frames`` 는 ``len(actions)`` 다. knu-1.3.0 은 모든 계열이 같은 길이라
이 추정이 맞았다. knu-2.0.0 은 축마다 길이가 다르므로 **카메라가 이 조건에
걸려 조용히 건너뛰어진다** -- 액션만 잘리고 이미지는 안 잘린 에피소드가 되고,
에러도 경고도 없다.

제대로 자르는 방법은 시각 구간이고, 그것은 ``test_trim_time.py`` 가 본다.
여기 남은 것은 **자를 선을 정할 수 없는 경우**다: 축 태그는 붙어 있는데
``t/control`` 이 없는 파일. 그러면 행 수로 자르는 수밖에 없는데, 120 Hz 의
27 tick 은 0.225초이고 30 fps 카메라로는 7장이라 27 을 그대로 적용하면
카메라가 0.9초를 더 잃는다. 잘못 자른 파일을 만드느니 멈추는 쪽이 낫다.

여기서 확인하는 것:
1. 1.3.0 은 예전과 **똑같이** 잘린다 (모든 계열이 함께)
2. 축 태그가 있는데 시간축이 없으면 거부하고, **파일이 하나도 안 바뀐다**
   -- 반쯤 자르다 멈추면 되돌릴 수 없다
"""
import os
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
sys.argv = ["t"]

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from mstack.data.episode_trim import trim_tail  # noqa: E402
from mstack.data.frame_table import axis_of, has_axis_attr  # noqa: E402

TMP = tempfile.mkdtemp(prefix="trimaxis-")
N = 40


def make(path, *, with_axis: bool):
    """demo_0 하나. 그리퍼를 5 에서 잡고 20 에서 놓는다 -- 릴리스 가드가
    막지 않도록 끝에서 충분히 떨어뜨린다."""
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["next_demo_idx"] = 1
        g = d.create_group("demo_0")
        g.attrs["num_samples"] = N
        a = np.zeros((N, 8), "f4")
        a[:, :7] = np.linspace(0, 1, N)[:, None]
        a[5:20, -1] = 1.0
        ds = g.create_dataset("actions", data=a)
        o = g.create_group("obs")
        img = o.create_dataset("agentview_rgb", data=np.zeros((N, 4, 4, 3), "u1"))
        jnt = o.create_dataset("joint_states", data=np.zeros((N, 7), "f4"))
        if with_axis:
            ds.attrs["axis"] = "control"
            img.attrs["axis"] = "agent"
            jnt.attrs["axis"] = "control"
    return path


def shapes(path):
    with h5py.File(path, "r") as f:
        g = f["data/demo_0"]
        return (g["actions"].shape[0], g["obs/agentview_rgb"].shape[0],
                g["obs/joint_states"].shape[0], int(g.attrs["num_samples"]))


def main() -> None:
    # ------------------------------------------------ 1. 1.3.0 은 그대로
    p = make(os.path.join(TMP, "v1.h5"), with_axis=False)
    keep = trim_tail(p, "demo_0", 5)
    got = shapes(p)
    assert keep == 35 and got == (35, 35, 35, 35), (keep, got)
    print(f"1. knu-1.3.0: 5프레임 트림 -> 모든 계열이 {keep} 로 함께 잘린다 OK")

    # ------------------------------------------------ 2. 시간축 없는 축 태그는 거부
    p2 = make(os.path.join(TMP, "v2.h5"), with_axis=True)
    before = shapes(p2)
    try:
        trim_tail(p2, "demo_0", 5)
    except ValueError as e:
        msg = str(e)
        assert "t/control" in msg, f"이유를 안 밝힌다: {msg}"
    else:
        raise AssertionError(
            "축 태그만 있고 t/control 이 없는데 잘랐다 -- 자를 선을 정할 "
            "방법이 없으므로 행 수로 잘렸다는 뜻이고, 그러면 카메라가 "
            "control 과 다른 만큼 잘린다")
    after = shapes(p2)
    assert after == before == (N, N, N, N), (
        f"거부했는데 파일이 바뀌었다 {before} -> {after} -- 반쯤 자르다 "
        "멈추면 되돌릴 수 없다")
    print(f"2. axis 태그 + t/ 없음: 거부하고 파일은 {after[0]}프레임 그대로 OK")

    # ------------------------------------------------ 3. axis_of 자체
    with h5py.File(p2, "r") as f:
        g = f["data/demo_0"]
        assert axis_of(g["obs/agentview_rgb"]) == "agent"
        assert axis_of(g["actions"]) == "control"
        assert has_axis_attr(g["actions"])
    with h5py.File(p, "r") as f:
        g = f["data/demo_0"]
        # 1.3.0 은 attr 이 없고, 축이 하나뿐이라 기본값이 답이다
        assert not has_axis_attr(g["actions"])
        assert axis_of(g["obs/agentview_rgb"]) == "control"
    print("3. axis_of 는 명시된 축을 읽고, 없으면 기본값을 준다 OK")

    print("\n트림 축 판별 인수 통과")


if __name__ == "__main__":
    main()
