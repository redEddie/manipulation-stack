"""LeRobot 변환기가 두 레이아웃에서 **같은 프레임**을 만드는가.

변환기는 여섯 계열을 같은 ``t`` 로 동시에 인덱싱하는, 저장소에서 가장
"한 행 = 한 프레임" 모양인 코드다:

    n = actions.shape[0]
    for t in range(n):
        frame["observation.state"] = concat([arr[t] for arr in state_arrays])
        frame["observation.images.agent"] = agent_rgb[t]
        ...

knu-2.0.0 은 계열마다 길이가 다르므로 이 루프가 성립하지 않는다. 그 조인을
``frame_table`` 이 맡고, 여기서는 **그 결과가 옛 것과 같은지**를 본다.

LeRobot 전체를 돌리면 AV1 인코딩에 몇 분이 걸린다. 그런데 비교하고 싶은 것은
인코딩이 아니라 ``ds.add_frame(frame)`` 에 들어가는 그 dict 다 -- 그래서 같은
조립을 여기서 재현해 비교한다. 인코더는 우리 코드가 아니다.

``rate`` 를 넘기는 것이 변환기에서는 옳다. 변환기의 계약이 "이 주파수로
내보낸다" 이므로, 20 Hz 로 기록한 파일에서 30 Hz 를 요청하면 만들어낼 원본이
없다 -- 조용히 20 Hz 짜리를 30 Hz 라고 찍어 내보내는 대신 거부한다.
(``episode_stats`` 나 ``replay_episode`` 는 반대다. 그쪽은 기록된 그대로를
원하므로 ``rate`` 를 주지 않는다.)
"""
import os
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = ["t"]

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from mstack.data.dataset_schema import (  # noqa: E402
    OBS_AGENTVIEW_RGB,
    OBS_COMMANDED_GRIPPER_STATES,
    OBS_COMMANDED_JOINT_STATES,
    OBS_EYE_IN_HAND_RGB,
    OBS_GRIPPER_STATES,
    OBS_JOINT_STATES,
)
from mstack.data.frame_table import frame_table  # noqa: E402
from v2_fixture import write_v2  # noqa: E402

TMP = tempfile.mkdtemp(prefix="convframes-")
DATASET = Path.home() / "libero_datasets" / "fr3-tabletop" / "scene_023.hdf5"

STATE_PARTS = [OBS_JOINT_STATES, OBS_GRIPPER_STATES]
CMD_PARTS = [OBS_COMMANDED_JOINT_STATES, OBS_COMMANDED_GRIPPER_STATES]


def build_frames(grp, fps=20.0):
    """변환기의 프레임 조립을 그대로. 이미지 크롭/리사이즈는 뺀다 --
    그건 이 변경과 무관하고, 원본 프레임이 같으면 결과도 같다."""
    keys = list(STATE_PARTS) + list(CMD_PARTS) + [OBS_AGENTVIEW_RGB,
                                                  OBS_EYE_IN_HAND_RGB]
    ft = frame_table(grp, rate=fps, keys=keys)
    out = []
    for t in range(len(ft)):
        out.append({
            "action": ft.actions[t].astype("float32"),
            "observation.state": np.concatenate(
                [ft.obs[p][t] for p in STATE_PARTS]).astype("float32"),
            "observation.commanded_state": np.concatenate(
                [ft.obs[p][t] for p in CMD_PARTS]).astype("float32"),
            "observation.images.agent": ft.obs[OBS_AGENTVIEW_RGB][t],
            "observation.images.wrist": ft.obs[OBS_EYE_IN_HAND_RGB][t],
        })
    return out, ft


def main() -> None:
    if not DATASET.exists():
        print(f"건너뜀 -- {DATASET} 없음 (이 시험은 실제 에피소드가 필요하다)")
        return

    with h5py.File(DATASET, "r") as f:
        for name in ("episode_000", "episode_050"):
            if name not in f:
                continue
            ep = f[name]
            a, ft1 = build_frames(ep)

            p = os.path.join(TMP, f"{name}.h5")
            # **2.0.0 쪽은 앞 N 행만 만든다.** 주장은 "두 레이아웃이 같은
            # 프레임을 낸다" 이고 그것은 겹치는 구간에서 확인하면 된다 --
            # 1.3.0 쪽(a)은 전부 만들어 두고 앞 N 개와 견준다. 전부 변환하면
            # 에피소드당 이미지 228 MB 를 gzip 으로 다시 쓰게 된다.
            N = 24
            write_v2(ep, p, n=N)
            with h5py.File(p, "r") as g:
                b, ft2 = build_frames(g["e"])

            n = ep["actions"].shape[0]
            assert len(a) == n and len(b) == N, (len(a), len(b), n, N)
            for i, (x, y) in enumerate(zip(a[:N], b)):
                for k in x:
                    assert np.array_equal(x[k], y[k]), (
                        f"{name} 프레임 {i} 의 {k!r} 가 두 레이아웃에서 다르다 "
                        "-- 변환하면 Hub 데이터셋이 조용히 달라진다")

            # 두 레이아웃이 **똑같이 틀린** 경우를 잡으려면 원본과도 비교해야 한다
            assert np.array_equal(
                np.stack([r["observation.images.agent"] for r in a]),
                ep[f"obs/{OBS_AGENTVIEW_RGB}"][:]), f"{name}: agent 가 원본과 다르다"
            assert np.array_equal(
                np.stack([r["observation.images.wrist"] for r in a]),
                ep[f"obs/{OBS_EYE_IN_HAND_RGB}"][:]), f"{name}: wrist 가 원본과 다르다"
            assert np.array_equal(
                np.stack([r["action"] for r in a]),
                ep["actions"][:]), f"{name}: action 이 원본과 다르다"
            print(f"1. {name}: {n}프레임 x 5키가 두 레이아웃 + 직접 읽기와 일치 OK")

        # ---- fps 불일치는 거부한다
        ep = f["episode_000"]
        try:
            build_frames(ep, fps=30.0)
        except ValueError as e:
            assert "20" in str(e) and "30" in str(e), str(e)
        else:
            raise AssertionError(
                "20 Hz 로 기록한 파일에서 30 Hz 를 만들어냈다 -- 없는 원본을 "
                "지어냈거나, 20 Hz 짜리를 30 Hz 라고 찍어 내보낸 것이다")
        print("2. 20 Hz 파일 + --fps 30 은 거부한다 OK")

        # ---- actions_ee 가 표에 실린다 (루트 계열)
        ft = frame_table(ep, rate=20.0, keys=[OBS_JOINT_STATES])
        assert set(ft.extra) >= {"rewards", "dones"}, set(ft.extra)
        for k, v in ft.extra.items():
            assert len(v) == len(ft), f"extra/{k} 가 {len(v)}행, 표는 {len(ft)}행"
        print(f"3. 루트 계열이 같은 행으로 실린다 OK ({sorted(ft.extra)})")

    print("\n변환기 프레임 등가 통과")


if __name__ == "__main__":
    main()
