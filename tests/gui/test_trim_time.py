"""knu-2.0.0 에피소드를 **시각 구간**으로 자르는가 (C5a 재설계).

축마다 주파수가 다르므로 "뒤에서 n 프레임" 을 모든 축에 똑같이 적용할 수
없다. 실측 (scene_024/episode_000):

    control   99행  50.00 ms
    agent    150장  33.37 ms
    wrist    149장  33.34 ms
    command  499행  10.01 ms

control 5행은 0.25초이고, 그 구간에 카메라는 7~8장, 명령은 25행 들어 있다.
그래서 자르는 선은 **control 에서 남는 마지막 행의 시각**이고, 다른 축은 그
시각 이하만 남는다.

여기서 확인하는 것:

1. control 은 정확히 요청한 만큼 줄어든다
2. 다른 축은 **시각으로** 줄어든다 (프레임 수 비례가 아니라)
3. 남은 모든 축의 마지막 표본이 자르는 선 이하다 (인과성)
4. ``axis`` 태그가 살아남는다 -- 지우고 다시 만드는 경로라 잃기 쉽다
5. 잘린 파일을 frame_table 이 그대로 읽는다
6. 1.3.0 파일은 예전 경로 그대로 (행 수로 자른다)
"""
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from mstack.data.episode_trim import plan_trim, trim_tail  # noqa: E402
from mstack.data.frame_table import frame_table, has_axis_attr  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="trimtime_"))

# 실측한 scene_024/episode_000 의 배치 그대로. 실기 파일은 600 MB 라 스위트에서
# 복사하지 않고, 축 주파수와 태그 구조만 같게 만든다 (이미지는 1x1 로 줄인다).
HZ = {"control": 20.0, "agent": 30.0, "wrist": 30.0, "command": 100.0}
CTRL_OBS = {"ee_pos": 3, "ee_ori": 3, "ee_states": 6, "ee_wrench": 6,
            "gripper_states": 1, "joint_states": 7, "joint_torques": 7,
            "commanded_joint_states": 7, "commanded_gripper_states": 1}


def build(path: Path, n_ctrl: int) -> None:
    dur = (n_ctrl - 1) / HZ["control"]
    t = {a: np.arange(0.0, dur + 1e-12, 1.0 / hz) for a, hz in HZ.items()}
    t["control"] = np.arange(n_ctrl) / HZ["control"]
    # 카메라는 자유 실행이라 control 보다 조금 먼저 시작한다 (실측 -0.041s).
    t["agent"] = t["agent"] - 0.041
    t["wrist"] = t["wrist"] - 0.021

    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs["dataset_version"] = "knu-2.0.0"
        e = f.create_group("episode_000")
        e.attrs["num_samples"] = n_ctrl
        e.attrs["control_hz"] = HZ["control"]

        tg = e.create_group("t")
        for a, v in t.items():
            d = tg.create_dataset(a, data=v)
            if a == "control":
                d.attrs["axis"] = "control"       # 실기 파일과 같게

        def tag(ds, axis):
            ds.attrs["axis"] = axis
            return ds

        n = {a: len(v) for a, v in t.items()}
        # 그리퍼는 초반에 한 번만 바뀐다 -- 꼬리를 잘라도 놓기가 안 잘린다
        act = np.zeros((n_ctrl, 8), dtype=np.float32)
        act[:, :7] = np.linspace(0, 1, n_ctrl)[:, None]
        act[n_ctrl // 3:, 7] = 1.0
        tag(e.create_dataset("actions", data=act), "control")
        tag(e.create_dataset("rewards", data=np.zeros(n_ctrl, np.float32)), "control")
        tag(e.create_dataset("dones", data=np.zeros(n_ctrl, np.uint8)), "control")

        obs = e.create_group("obs")
        for name, dim in CTRL_OBS.items():
            tag(obs.create_dataset(name, data=np.zeros((n_ctrl, dim), np.float32)),
                "control")
        for name, axis in (("agentview_rgb", "agent"), ("eye_in_hand_rgb", "wrist")):
            tag(obs.create_dataset(name, shape=(n[axis], 1, 1, 3), dtype=np.uint8,
                                   chunks=(1, 1, 1, 3)), axis)

        cmd = e.create_group("command")
        tag(cmd.create_dataset("joint_positions",
                               data=np.zeros((n["command"], 7), np.float32)), "command")
        tag(cmd.create_dataset("gripper",
                               data=np.zeros((n["command"], 1), np.float32)), "command")

        mg = e.create_group("meta")
        for axis in ("control", "agent", "wrist"):
            g_ = mg.create_group(axis)
            for key in (("action", "robot_state") if axis == "control"
                        else ("host", "frame_no", "node_seq")):
                g_.create_dataset(key, data=np.arange(n[axis], dtype=np.float64))


# ---------------------------------------------- 합성 2.0.0 파일
src = TMP / "scene_900.hdf5"
build(src, n_ctrl=99)

with h5py.File(src, "r") as f:
    g = f["episode_000"]
    before = {a: g["t"][a].shape[0] for a in g["t"]}
    t_before = {a: g["t"][a][:] for a in g["t"]}
    tagged_before = {n for n in ("actions", "obs/agentview_rgb",
                                 "obs/eye_in_hand_rgb", "command/joint_positions")
                     if has_axis_attr(g[n])}
print("자르기 전:", before)
assert tagged_before, "축 태그가 처음부터 없으면 이 테스트는 의미가 없다"

N_TRIM = 5
keep_ctrl = before["control"] - N_TRIM
t_cut = float(t_before["control"][keep_ctrl - 1])

plan = plan_trim(str(src), ["episode_000"], N_TRIM)[0]
assert plan.blocked is None, plan.blocked
new_n = trim_tail(str(src), "episode_000", N_TRIM)

with h5py.File(src, "r") as f:
    g = f["episode_000"]
    after = {a: g["t"][a].shape[0] for a in g["t"]}
    t_after = {a: g["t"][a][:] for a in g["t"]}

    # ---------------------------------------------- 1. control
    assert new_n == keep_ctrl == after["control"], (new_n, keep_ctrl, after)
    assert int(g.attrs["num_samples"]) == keep_ctrl, g.attrs["num_samples"]
    print(f"1. control {before['control']} -> {after['control']} "
          f"(-{N_TRIM}행, 자르는 선 {t_cut:.3f}s) OK")

    # ---------------------------------------------- 2. 시각으로 줄었다
    for axis in ("agent", "wrist", "command"):
        want = int(np.searchsorted(t_before[axis], t_cut, side="right"))
        assert after[axis] == want, (axis, after[axis], want)
        # 프레임 수 비례로 잘랐다면 이 값이 나온다 -- 그것과 달라야 의미가 있다
        naive = before[axis] - N_TRIM
        assert after[axis] != naive or axis == "control", (
            f"{axis}: 시각 기준({want})과 행 기준({naive})이 같아 구분이 안 된다")
    print("2. 다른 축은 시각으로 줄었다: " + ", ".join(
        f"{a} {before[a]}->{after[a]} (행 기준이면 {before[a] - N_TRIM})"
        for a in ("agent", "wrist", "command")) + " OK")

    # ---------------------------------------------- 3. 인과성
    for axis, t in t_after.items():
        assert t[-1] <= t_cut + 1e-9, (axis, t[-1], t_cut)
    print(f"3. 모든 축의 마지막 표본이 {t_cut:.3f}s 이하 OK")

    # ---------------------------------------------- 4. 축 태그 생존
    lost = [n for n in tagged_before if not has_axis_attr(g[n])]
    assert not lost, f"지우고 다시 만들며 axis 태그를 잃었다: {lost}"
    print(f"4. axis 태그 {len(tagged_before)}개 모두 살아남았다 OK")

    # ---------------------------------------------- 5. 다시 읽힌다
    ft = frame_table(g)
    assert len(ft) == keep_ctrl, (len(ft), keep_ctrl)
    for axis, age in ft.image_age.items():
        assert (age >= 0).all(), f"{axis}: 이미지가 행보다 나중이다 (누출)"
    print(f"5. frame_table 이 {len(ft)}행으로 읽는다, image_age 음수 0 OK")

    assert "@" in g.attrs["trimmed"], g.attrs["trimmed"]
    print(f"   이력: {g.attrs['trimmed']}")

# ---------------------------------------------- 6. 1.3.0 은 예전 경로
legacy = TMP / "scene_901.hdf5"
build(legacy, n_ctrl=60)
# 축 태그와 t/ 를 걷어내 1.3.0 모양으로 되돌린다
with h5py.File(legacy, "a") as f:
    g = f["episode_000"]
    n = g["actions"].shape[0]
    for name in ("obs/agentview_rgb", "obs/eye_in_hand_rgb"):
        del g[name]
    del g["t"], g["meta"], g["command"]
    f["metadata"].attrs["dataset_version"] = "knu-1.2.2"
    names = []
    g.visititems(lambda k, o: names.append(k) if isinstance(o, h5py.Dataset) else None)
    for k in names:
        g[k].attrs.pop("axis", None)

n_legacy = trim_tail(str(legacy), "episode_000", 5)
with h5py.File(legacy, "r") as f:
    assert n_legacy == n - 5 == f["episode_000"]["actions"].shape[0], n_legacy
print(f"6. 1.3.0 파일은 예전 경로 그대로: {n} -> {n_legacy} OK")

shutil.rmtree(TMP, ignore_errors=True)
print("\n시각 기준 트림 인수 통과")
