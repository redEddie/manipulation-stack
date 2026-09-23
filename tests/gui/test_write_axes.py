"""기록기가 축을 나눠 쓰고, 그것을 frame_table 이 읽는가 (knu-2.0.0).

C6 의 나머지 절반이다. 앞 절반(`test_capture_all`)은 카메라 프레임을 하나도
안 버리고 모으는 것이었고, 여기서는 **그것을 파일에 축으로 쓰는 것**이다.

지켜야 하는 것:

1. **카메라가 control 격자에 눌리지 않는다.** 30 fps 로 온 30장이 20 Hz 20행에
   맞춰 20장으로 줄면 C6 이 한 일이 없다.
2. **축이 명시된다** (`attrs["axis"]`). 길이로 추정하면 축이 갈린 순간부터
   틀리고, 그 추정이 트림에서 실제로 카메라를 놓쳤다.
3. **왕복.** 쓴 것을 `frame_table` 이 읽어 옛 표와 같은 규칙으로 짝지어야
   한다 -- 쓰는 쪽과 읽는 쪽이 갈리면 둘 다 맞아 보이면서 데이터가 틀어진다.
4. **목표 주기와 달성 주기가 나란히 남는다** (C2). 계열마다 시각이 있어 주기를
   못 맞춰도 데이터는 안 상하지만, 남기지 않으면 "그때 몇 Hz 였지" 를 못 푼다.
5. **노출이 프레임마다 남는다** (C7). 되돌릴 수 없는 손실이고 비용이 0 이다 --
   손목 카메라의 실측 노출 33 ms 가 프레임 주기와 거의 같아, 기록이 없으면
   그 뭉갬을 사후에 정량화할 수도 t_device 의 기준점을 보정할 수도 없다.
6. **capture 가 없으면 옛 구조 그대로.** 모든 테스트와 연습 모드가 그 경로다.

축 시각에 **장치 시각**을 쓰는 것이 여기서 드러난다: 같은 파일을
``align="arrival"`` 과 ``align="capture"`` 로 읽으면 **다른 프레임**이 골라진다.
같은 것이 골라지면 두 모드를 둘 이유가 없다.
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

from mstack.data.dataset_schema import (  # noqa: E402
    TIMING_ACTION,
    TIMING_FRAME,
    DatasetSchemaConfig,
)
from mstack.data.frame_table import (  # noqa: E402
    axis_of,
    frame_table,
    has_axis_attr,
    n_frames,
    stream,
)
from mstack.data.libero_format import (  # noqa: E402
    LiberoEpisodeBuffer,
    write_episode_payload,
)

TMP = tempfile.mkdtemp(prefix="writeaxes-")
T0 = 1.7e9
N_CTRL, HZ_CTRL, HZ_CAM = 20, 20.0, 30.0
SUBSTEPS = 6          # 20 x 6 = 120 Hz 명령
N_CAM = int(N_CTRL / HZ_CTRL * HZ_CAM)
#: 카메라 도착이 장치 시각보다 이만큼 늦다 (실측 D455 15.03 ms / D405 8.60 ms).
PIPE_MS = 0.015


def build(with_capture: bool, roles=("agent", "wrist")):
    sc = DatasetSchemaConfig()
    buf = LiberoEpisodeBuffer(sc)
    if with_capture:
        # 무장한 역할은 20 Hz 폴링본을 **버퍼에 쌓지 않는다** -- 쌓으면 같은
        # 에피소드를 두 번 들고 있게 된다 (capture 1.11 GB + 폴링 0.74 GB,
        # 뒤엣것은 아무도 안 읽는다).
        buf.capture_roles = set(roles)
    for i in range(N_CTRL):
        t = T0 + i / HZ_CTRL
        buf.add_frame(
            agentview_rgb=np.zeros((8, 8, 3), "u1"),
            eye_in_hand_rgb=np.zeros((8, 8, 3), "u1"),
            joint_positions=np.full(7, i, dtype=float), gripper_position=0.0,
            ee_pos_quat=np.array([0, 0, 0, 0, 0, 0, 1.0]), gripper_closed=False,
            joint_velocities=np.zeros(7), timestamp=t,
            commanded_joint_positions=np.full(7, i, dtype=float),
            commanded_gripper=0.0,
            timing={TIMING_FRAME: t, TIMING_ACTION: t + 0.001})
    buf.control_hz = HZ_CTRL
    # 명령은 기록보다 SUBSTEPS 배 자주 나간다 -- 그 전부가 명령 축에 남아야 한다
    for i in range(N_CTRL * SUBSTEPS):
        buf.add_command(T0 + i / (HZ_CTRL * SUBSTEPS),
                        np.full(7, i, dtype=np.float32), 0.0)
    if with_capture:
        # 값에 프레임 번호를 넣어 둔다 -- 어느 것이 골라졌는지 값만 보면 안다.
        for axis in roles:
            buf.set_capture(axis, [
                (T0 + j / HZ_CAM + PIPE_MS, np.full((8, 8, 3), j, dtype="u1"),
                 {"t_device": T0 + j / HZ_CAM, "frame_no": 100 + j, "seq": j,
                  "t_domain": "global_time", "exposure": 8000 + j})
                for j in range(N_CAM)])
    p = os.path.join(TMP, f"cap{int(with_capture)}.h5")
    with h5py.File(p, "w") as f:
        write_episode_payload(f.create_group("e"), buf, sc, success=True)
    return p


def main() -> None:
    # ============================================ 1. 카메라가 안 눌린다
    p = build(with_capture=True)
    with h5py.File(p, "r") as f:
        e = f["e"]
        axes = {k: len(e[f"t/{k}"]) for k in e["t"]}
        assert axes == {"control": N_CTRL, "command": N_CTRL * SUBSTEPS,
                        "agent": N_CAM, "wrist": N_CAM}, axes
        assert e["obs/agentview_rgb"].shape[0] == N_CAM, (
            f"카메라가 {e['obs/agentview_rgb'].shape[0]}장으로 눌렸다 -- "
            f"{N_CAM}장이 와야 C6 이 한 일이 있다")
        assert e["actions"].shape[0] == N_CTRL
        print(f"1. 축이 갈렸다 {axes} -- 이미지 {N_CAM}장, 명령 "
              f"{N_CTRL * SUBSTEPS}행, control {N_CTRL}행 OK")

        # ============================================ 2. 축이 명시된다
        assert axis_of(e["obs/agentview_rgb"]) == "agent"
        assert axis_of(e["obs/eye_in_hand_rgb"]) == "wrist"
        for name in ("actions", "rewards", "dones"):
            assert has_axis_attr(e[name]), f"{name} 에 축이 안 붙었다"
            assert axis_of(e[name]) == "control", name
        for name in e["obs"]:
            assert has_axis_attr(e[f"obs/{name}"]), f"obs/{name} 에 축이 안 붙었다"
        assert n_frames(e) == N_CTRL, n_frames(e)
        print("2. 모든 데이터셋에 axis 가 붙었다 (control/agent/wrist) OK")

        # ============================================ 3. meta 와 t0
        assert "t0_wall" in e.attrs
        for axis in ("agent", "wrist"):
            g = e[f"meta/{axis}"]
            assert set(g.keys()) >= {"host", "frame_no", "node_seq"}, set(g.keys())
            assert g.attrs.get("clock") == "global_time"
            assert len(g["host"]) == N_CAM
        assert "action" in e["meta/control"], sorted(e["meta/control"].keys())
        assert "timing" not in e, "timing/ 이 남아 있다 -- 같은 값이 두 곳에 있다"
        print("3. meta/<축>/{host,frame_no,node_seq} + t0_wall, timing/ 없음 OK")

        # ============================================ 4. 왕복
        for align, clock in (("arrival", "meta/agent/host"), ("capture", "t/agent")):
            ft = frame_table(e, align=align)
            got = ft.obs["agentview_rgb"][:, 0, 0, 0]
            idx = np.searchsorted(e[clock][:], e["t/control"][:], side="right") - 1
            want = idx[idx >= 0]          # 채울 수 없는 앞 행은 버려진다
            assert np.array_equal(got, want), (align, got[:6], want[:6])
            age = ft.image_age["agent"]
            assert (age >= 0).all(), f"{align}: 이미지가 행보다 나중이다"
            print(f"4. 왕복 {align:8s}: {len(ft)}행, 고른 인덱스 {got[:6].tolist()}, "
                  f"이미지 나이 중앙 {np.median(age) * 1000:.1f} ms, 음수 0 OK")

        # align 이 실제로 갈려야 장치 시각을 축으로 둔 의미가 있다
        a = frame_table(e, align="arrival").obs["agentview_rgb"][:, 0, 0, 0]
        c = frame_table(e, align="capture").obs["agentview_rgb"][:, 0, 0, 0]
        assert not (len(a) == len(c) and np.array_equal(a, c)), (
            "arrival 과 capture 가 같은 프레임을 골랐다 -- 축에 장치 시각을 "
            "두고 도착 시각을 meta 에 남긴 이유가 사라진다")
        print("5. align 이 다른 프레임을 고른다 (장치 시각 축 + 도착 시각 meta) OK")

        ds, t = stream(e, "agentview_rgb")
        assert ds.shape[0] == N_CAM and len(t) == N_CAM
        print(f"6. stream() 이 카메라 축 그대로 {N_CAM}장을 준다 OK")

        # ---- 명령 축: 기록보다 SUBSTEPS 배 많다
        n_cmd = N_CTRL * SUBSTEPS
        assert len(e["t/command"]) == n_cmd, len(e["t/command"])
        assert e["command/joint_positions"].shape[0] == n_cmd
        assert axis_of(e["command/joint_positions"]) == "command"
        hz = float(e.attrs["command_hz_actual"])
        assert abs(hz - HZ_CTRL * SUBSTEPS) < 1.0, hz
        # control 축의 actions 는 **안 바뀐다** -- 옛 소비자가 보던 의미 그대로
        assert e["actions"].shape[0] == N_CTRL
        ds_c, t_c = stream(e, "command/joint_positions")
        assert ds_c.shape[0] == n_cmd and len(t_c) == n_cmd
        print(f"6b. 명령 축 {n_cmd}행 ({hz:.0f} Hz) vs control {N_CTRL}행 "
              f"({N_CTRL / n_cmd:.0%} 만 기록됐었다) OK")

        # ---- 목표 주기와 달성 주기 (C2)
        assert abs(e.attrs["control_hz"] - HZ_CTRL) < 1e-9, e.attrs["control_hz"]
        got_hz = float(e.attrs["control_hz_actual"])
        assert abs(got_hz - HZ_CTRL) < 0.5, got_hz
        print(f"7. 목표 {e.attrs['control_hz']:.1f} Hz / 달성 {got_hz:.2f} Hz 가 "
              "나란히 기록된다 OK (갈라지면 목표를 못 맞춘 것)")

        # ---- 노출이 실린다 (C7)
        assert "exposure" in e["meta/agent"], sorted(e["meta/agent"].keys())
        assert len(e["meta/agent/exposure"]) == N_CAM
        print(f"8. meta/<축>/exposure 가 {N_CAM}장 전부에 기록된다 OK")

    # ============================================ 7. capture 없으면 옛 구조
    p2 = build(with_capture=False)
    with h5py.File(p2, "r") as f:
        e = f["e"]
        assert "t" not in e, "capture 가 없는데 축을 만들었다"
        assert "command" not in e, "옛 경로인데 명령 축이 생겼다"
        assert "timing" in e, "옛 경로인데 timing/ 이 없다"
        assert e["obs/agentview_rgb"].shape[0] == N_CTRL
        assert not has_axis_attr(e["actions"])
        ft = frame_table(e)
        assert len(ft) == N_CTRL and ft.version == "knu-1.x", (len(ft), ft.version)
    print("9. capture 가 없으면 옛 한 행 = 한 프레임 그대로 OK")

    # ============================================ 10. 폴링본을 안 쌓는다
    sc = DatasetSchemaConfig()
    probe = LiberoEpisodeBuffer(sc)
    probe.capture_roles = {"agent", "wrist"}
    probe.add_frame(
        agentview_rgb=np.zeros((8, 8, 3), "u1"),
        eye_in_hand_rgb=np.zeros((8, 8, 3), "u1"),
        joint_positions=np.zeros(7), gripper_position=0.0,
        ee_pos_quat=np.array([0, 0, 0, 0, 0, 0, 1.0]), gripper_closed=False,
        joint_velocities=np.zeros(7), timestamp=T0)
    assert probe.agentview_rgb == [] and probe.eye_in_hand_rgb == [], (
        "capture 가 무장됐는데 폴링본을 쌓았다 -- 같은 에피소드를 두 번 "
        "들고 있게 되고, _process_image 의 copy 가 50 ms 예산 안에서 돈다")
    print("10. 무장된 역할은 20 Hz 폴링본을 안 쌓는다 OK")

    # ============================================ 11. 한쪽만 실패해도 안 잃는다
    p3 = build(with_capture=True, roles=("agent",))
    with h5py.File(p3, "r") as f:
        e = f["e"]
        a, w = e["obs/agentview_rgb"], e["obs/eye_in_hand_rgb"]
        assert a.shape[0] == N_CAM and axis_of(a) == "agent"
        assert w.shape[0] == N_CTRL, (
            f"wrist 가 {w.shape[0]}장 -- capture 에 실패한 카메라의 이미지가 "
            "통째로 사라졌다")
        assert axis_of(w) == "control", axis_of(w)
        ft = frame_table(e)
        assert all(len(v) == len(ft) for v in ft.obs.values())
    print(f"11. 한쪽만 capture 실패: agent {N_CAM}장(agent축) + "
          f"wrist {N_CTRL}장(control축), 잃은 것 없음 OK")

    print("\n축별 기록 인수 통과")


if __name__ == "__main__":
    main()
