"""C5 소비자 치환의 본체 -- 세 소비자가 두 레이아웃에서 같은 결과를 내는지.

frame_table 로 바꾼 소비자 세 곳(load_series / export_episode /
load_trajectory)을 실제 에피소드(knu-1.3.0)와 v2_fixture.write_v2 로 만든
같은 내용의 knu-2.0.0 파일에 각각 돌려, 결과를 np.array_equal 로 단언한다.
지금 2.0.0 파일이 하나도 없으므로 "스위트가 통과한다" 는 치환의 증거가 될 수
없다 -- 이 단언이 "두 레이아웃에서 같은 결과가 나오는가" 의 유일한 안전장치다.

더해, 1.3.0 에서의 결과를 예전 직접 읽기(h5py 원시 슬라이스)와도 맞춘다.
write_v2 가 원본을 읽어 2.0.0 을 만들므로, frame_table 의 1.3.0 경로가
데이터를 망가뜨리면 2.0.0 쪽도 똑같이 망가져 "두 레이아웃이 서로 같다" 는
단언만으로는 못 잡는 구멍이 생기기 때문이다.

데이터셋이 없는 환경에서는 세 단언을 모두 건 너뛰되, 조용히 통과시키지 않고
이유를 print 한다 (test_frame_table.py 와 같은 계약).
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

import scripts.analyze.replay_episode as replay_mod  # noqa: E402
import scripts.analyze.visualize_libero_demo as viz_mod  # noqa: E402
from mstack.data.dataset_schema import (  # noqa: E402
    OBS_AGENTVIEW_RGB,
    OBS_COMMANDED_GRIPPER_STATES,
    OBS_COMMANDED_JOINT_STATES,
    OBS_EYE_IN_HAND_RGB,
    OBS_GRIPPER_STATES,
    OBS_JOINT_STATES,
)
from mstack.data.episode_stats import load_series  # noqa: E402
from mstack.data.frame_table import frame_table  # noqa: E402
from v2_fixture import write_v2  # noqa: E402

TMP = tempfile.mkdtemp(prefix="c5-consumers-")
DATASET = Path.home() / "libero_datasets" / "fr3-tabletop" / "scene_023.hdf5"
EP = "episode_000"

SERIES_KEYS = ("state", "commanded", "action")


class _FakeWriter:
    """VideoWriter 를 흉내 내 export_episode 가 밀어 넣는 프레임을 가로챈다.

    mp4 인코딩을 거치면 손실 압축 때문에 원본의 작은 차이가 지워질 수 있으므로,
    코덱 입력 프레임을 그대로 잡아 두 레이아웃을 정확히 비교한다.
    """
    instances = []

    def __init__(self, *args, **kwargs):
        self.frames = []
        _FakeWriter.instances.append(self)

    def write(self, frame):
        self.frames.append((frame.shape, frame.tobytes()))

    def release(self):
        pass


def _export_frames(grp, out_path):
    """export_episode 를 돌리고 VideoWriter 에 들어간 프레임 목록을 돌려준다."""
    real_cv2 = viz_mod.cv2
    _FakeWriter.instances = []

    class _FakeCV2:
        VideoWriter = _FakeWriter
        VideoWriter_fourcc = staticmethod(real_cv2.VideoWriter_fourcc)
        cvtColor = staticmethod(real_cv2.cvtColor)
        COLOR_RGB2BGR = real_cv2.COLOR_RGB2BGR

    viz_mod.cv2 = _FakeCV2
    try:
        viz_mod.export_episode(grp, out_path, fps=20)
    finally:
        viz_mod.cv2 = real_cv2
    assert _FakeWriter.instances, "VideoWriter 가 만들어지지 않았다"
    return _FakeWriter.instances[0].frames


def _assert_same(a, b, what):
    assert np.array_equal(a, b), (
        f"{what}: 1.3.0 과 2.0.0 의 결과가 다르다 -- 소비자를 frame_table 로 "
        "바꾸면 데이터가 조용히 달라진다")
    assert a.dtype == b.dtype, (what, a.dtype, b.dtype)


def main() -> None:
    if not DATASET.exists():
        print(f"건너뜀 -- {DATASET} 없음 (load_series / export_episode / "
              "load_trajectory 단언 전부)")
        return

    p2 = os.path.join(TMP, "v2.h5")

    with h5py.File(DATASET, "r") as f:
        ep1 = f[EP]
        write_v2(ep1, p2, group=EP)

        # ---- 1. load_series: state/commanded/action/n 이 두 레이아웃에서 같다
        s1 = load_series(str(DATASET), EP)
        s2 = load_series(p2, EP)
        for k in SERIES_KEYS:
            assert s1[k] is not None and s2[k] is not None, k
            _assert_same(s1[k], s2[k], f"load_series[{k!r}]")
        assert s1["n"] == s2["n"] == len(s1["action"]), (
            s1["n"], s2["n"], len(s1["action"]))
        # 1.3.0 에서의 결과가 예전 직접 읽기와 같다 (치환이 결과를 바꾸지 않음)
        raw = ep1["obs"]
        want_state = np.concatenate(
            [raw[OBS_JOINT_STATES][:], raw[OBS_GRIPPER_STATES][:]], axis=1)
        want_cmd = np.concatenate(
            [raw[OBS_COMMANDED_JOINT_STATES][:],
             raw[OBS_COMMANDED_GRIPPER_STATES][:]], axis=1)
        _assert_same(s1["state"], want_state, "load_series[state] vs 직접 읽기")
        _assert_same(s1["commanded"], want_cmd, "load_series[commanded] vs 직접 읽기")
        _assert_same(s1["action"], ep1["actions"][:], "load_series[action] vs 직접 읽기")
        print(f"1. load_series: state/commanded/action/n 이 두 레이아웃에서 "
              f"같고 1.3.0 은 직접 읽기와 같다 ({s1['n']}행) OK")

        # ---- 2. export_episode: VideoWriter 로 들어가는 프레임이 같다
        frames1 = _export_frames(ep1, Path(TMP) / "v1.mp4")
        with h5py.File(p2, "r") as g:
            frames2 = _export_frames(g[EP], Path(TMP) / "v2.mp4")
        assert frames1 and len(frames1) == len(frames2), (
            len(frames1), len(frames2))
        for i, (a, b) in enumerate(zip(frames1, frames2)):
            assert a == b, (
                f"export_episode: 프레임 {i} 이 두 레이아웃에서 다르다 "
                f"(shape {a[0]} vs {b[0]}, 바이트 동일 {a[1] == b[1]})")
        # 1.3.0 의 프레임이 예전 직접 읽기로 붙인 것과 같다
        agent = ep1["obs"][OBS_AGENTVIEW_RGB]
        wrist = ep1["obs"][OBS_EYE_IN_HAND_RGB]
        cv2 = viz_mod.cv2
        for i in range(0, len(frames1), 7):   # 매 프레임이 아니라 7번째마다
            want = cv2.cvtColor(
                np.concatenate([agent[i], wrist[i]], axis=1),
                cv2.COLOR_RGB2BGR)
            got = frames1[i]
            assert got[0] == want.shape and got[1] == want.tobytes(), (
                f"export_episode: 프레임 {i} 이 1.3.0 직접 읽기와 다르다")
        print(f"2. export_episode: {len(frames1)}프레임 전부 두 레이아웃에서 "
              "같고 1.3.0 은 직접 읽기와 같다 OK")

        # ---- 3. load_trajectory: 로봇에 보내는 q/grip 이 두 레이아웃에서 같다
        t1 = replay_mod.load_trajectory(Path(DATASET), EP)
        t2 = replay_mod.load_trajectory(Path(p2), EP)
        _assert_same(t1["q"], t2["q"], "load_trajectory['q']")
        _assert_same(t1["grip"], t2["grip"], "load_trajectory['grip']")
        assert t1["source"] == t2["source"], (t1["source"], t2["source"])
        assert t1["instruction"] == t2["instruction"]
        assert t1["uid"] == t2["uid"]
        # 1.3.0 의 q/grip 이 예전 직접 읽기와 같다 (원본 코드도 asarray 로
        # float64 로 바꿔 돌려줬으므로 dtype 비교는 값이 아니라 float64 로)
        assert t1["q"].dtype == t1["grip"].dtype == np.float64
        assert np.array_equal(t1["q"], ep1["obs"][OBS_COMMANDED_JOINT_STATES][:]), \
            "load_trajectory[q] 가 1.3.0 직접 읽기와 다르다"
        assert np.array_equal(t1["grip"],
                              ep1["obs"][OBS_COMMANDED_GRIPPER_STATES][:, 0]), \
            "load_trajectory[grip] 이 1.3.0 직접 읽기와 다르다"
        assert t1["source"] == "commanded_joint_states", t1["source"]
        print(f"3. load_trajectory: q/grip ({t1['source']}, {len(t1['q'])}틱) 이 "
              "두 레이아웃에서 같고 1.3.0 은 직접 읽기와 같다 OK")

    _no_timing_and_other_rates()


    print("\nc5_consumers 통과")



def _no_timing_and_other_rates() -> None:
    """rate 를 하드코딩하면 안 되는 이유 두 가지를 못박는다.

    (a) ``timing/frame`` 이 없는 옛 파일이 **실재한다** -- 지금 데이터셋의 앞
        scene 들이 전부 그렇다. 여기서 막히면 옛 에피소드를 읽는 모든 소비자가
        죽는다. 행은 진짜이므로 돌려주고, 행 시각만 NaN 으로 둔다.
    (b) 20 Hz 를 못박으면 **30 Hz 로 기록한 파일이 거부된다.** 30 Hz 는 다음
        데이터셋의 목표 주파수다.
    """
    import tempfile as _tf
    tmp = _tf.mkdtemp(prefix="c5rate-")

    def make(path, hz, timing, n=30):
        with h5py.File(path, "w") as f:
            g = f.create_group("data").create_group("demo_0")
            g.attrs["num_samples"] = n
            g.create_dataset("actions", data=np.zeros((n, 8), "f4"))
            o = g.create_group("obs")
            o.create_dataset(OBS_JOINT_STATES, data=np.zeros((n, 7), "f4"))
            o.create_dataset(OBS_GRIPPER_STATES, data=np.zeros((n, 1), "f4"))
            if timing:
                t = g.create_group("timing")
                t.create_dataset("frame", data=1.7e9 + np.arange(n) / hz)
        return path

    for hz, timing, label in ((20, True, "20 Hz"), (30, True, "30 Hz"),
                              (20, False, "timing 없는 옛 파일")):
        p = os.path.join(tmp, f"r{hz}{timing}.h5")
        make(p, hz, timing)
        s = load_series(p, "demo_0")
        assert s["n"] == 30, (label, s["n"])
        with h5py.File(p, "r") as f:
            ft = frame_table(f["data/demo_0"])
        if timing:
            assert ft.rate is not None and abs(ft.rate - hz) < 0.5, (label, ft.rate)
            assert np.isfinite(ft.t).all(), label
        else:
            # 균등 간격을 **지어내지 않는다** -- NaN 은 쓰는 순간 드러나지만
            # 지어낸 격자는 조용히 퍼진다.
            assert ft.rate is None, label
            assert not np.isfinite(ft.t).any(), label
    print("4. rate 를 안 박는다: 20/30 Hz 와 timing 없는 옛 파일 모두 읽힌다 OK")


if __name__ == "__main__":
    main()
