"""frame_table 인수 시험 -- 두 레이아웃을 한 표로 읽는다.

이 모듈이 지켜야 하는 것은 셋이다:

1. **충실도.** knu-1.3.0 파일은 이미 표이므로 바이트 그대로 나와야 한다. 그리고
   같은 내용을 2.0.0 로 담으면 그 표를 **정확히** 되살려야 한다. 되살리지 못하면
   소비자 40곳을 이 함수로 바꾸는 순간 데이터가 조용히 달라진다.
2. **인과성.** 이미지가 액션보다 먼저여야 한다. 실측으로 33,226/33,226 성립하는
   성질이라, 깨진 표가 나오면 그것은 데이터가 아니라 이 모듈의 버그다.
3. **거부.** 할 수 없는 일은 조용히 근사하지 말고 멈춘다 -- 정수배가 아닌 rate,
   1.3.0 의 재표본, 카메라 기준 + 액션 요청.

시험 A 는 **아무것도 지어내지 않는다**: 실제 에피소드가 고른 프레임과 그 실제
시각만으로 2.0.0 을 만들고, 원래 표가 되돌아오는지 본다. 버려진 프레임을 추측해
끼워 넣으려 했더니 그 추측이 틀렸다 -- ``timing/frame`` 은 카메라를 읽은 *뒤*
찍히므로 그 사이(중앙 1.16 ms)에 도착한 프레임이 있고, 어디에 두어야 하는지
파일만으로는 알 수 없다. 그래서 솎아내기 동작은 시험 B 의 명확한 합성 격자로
따로 확인한다.
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

from mstack.data.frame_table import frame_table  # noqa: E402

TMP = tempfile.mkdtemp(prefix="frametable-")
DATASET = Path.home() / "libero_datasets" / "fr3-tabletop" / "scene_023.hdf5"


def _v2_from_v1(ep1, path):
    """실제 1.3.0 에피소드를 같은 내용의 2.0.0 로 옮긴다 (지어내는 값 없음)."""
    tim = ep1["timing"]
    t_row = tim["frame"][:]
    t0 = t_row[0]
    with h5py.File(path, "w") as g:
        e = g.create_group("e")
        e.attrs["dataset_version"] = "knu-2.0.0"
        e.attrs["t0_wall"] = t0
        e.attrs["control_hz"] = 20.0
        tg, mg = e.create_group("t"), e.create_group("meta")
        tg.create_dataset("control", data=t_row - t0)
        for axis, pre in (("agent", "agentview"), ("wrist", "eye_in_hand")):
            tg.create_dataset(axis, data=tim[f"{pre}_device"][:] - t0)
            mg.create_dataset(f"{axis}/host", data=tim[f"{pre}_host"][:] - t0)
        o = e.create_group("obs")
        for k, axis in (("agentview_rgb", "agent"), ("eye_in_hand_rgb", "wrist"),
                        ("joint_states", "control"), ("gripper_states", "control")):
            d = o.create_dataset(k, data=ep1[f"obs/{k}"][:])
            d.attrs["axis"] = axis
        d = e.create_dataset("actions", data=ep1["actions"][:])
        d.attrs["axis"] = "control"


def _synthetic(path, *, n_ctrl=20, ctrl_hz=20.0, cam_hz=30.0, offset=0.015):
    """깨끗한 합성 격자. 카메라 프레임마다 그 인덱스를 값으로 넣어 두므로,
    어느 프레임을 골랐는지 값만 보면 알 수 있다."""
    t_ctrl = np.arange(n_ctrl) / ctrl_hz
    n_cam = int(t_ctrl[-1] * cam_hz) + 1
    # 카메라 프레임을 제어 tick 과 겹치지 않게 반 칸 앞에 둔다 -> 어느 것이
    # 골라져야 하는지가 한 가지로 정해진다.
    t_cam = np.arange(n_cam) / cam_hz - 0.5 / cam_hz
    with h5py.File(path, "w") as g:
        e = g.create_group("e")
        e.attrs["dataset_version"] = "knu-2.0.0"
        e.attrs["control_hz"] = ctrl_hz
        tg, mg = e.create_group("t"), e.create_group("meta")
        tg.create_dataset("control", data=t_ctrl)
        tg.create_dataset("agent", data=t_cam)
        mg.create_dataset("agent/host", data=t_cam + offset)
        o = e.create_group("obs")
        d = o.create_dataset("agentview_rgb",
                             data=np.arange(n_cam, dtype=np.int32).reshape(-1, 1))
        d.attrs["axis"] = "agent"
        d = o.create_dataset("joint_states",
                             data=np.arange(n_ctrl, dtype=np.int32).reshape(-1, 1))
        d.attrs["axis"] = "control"
        d = e.create_dataset("actions",
                             data=np.arange(n_ctrl, dtype=np.int32).reshape(-1, 1))
        d.attrs["axis"] = "control"
    return t_ctrl, t_cam


def main() -> None:
    # ============================================ 1. 1.3.0 은 바이트 그대로
    if not DATASET.exists():
        print(f"1-4. 건너뜀 -- {DATASET} 없음")
    else:
        with h5py.File(DATASET, "r") as f:
            ep1 = f["episode_000"]
            ft = frame_table(ep1)
            assert np.array_equal(ft.obs["agentview_rgb"], ep1["obs/agentview_rgb"][:])
            assert np.array_equal(ft.actions, ep1["actions"][:])
            assert all(len(v) == len(ft) for v in ft.obs.values())
            assert abs(ft.rate - 20.0) < 0.5, ft.rate
            print(f"1. knu-1.3.0 은 바이트 그대로 ({len(ft)}행, {ft.rate:.1f} Hz) OK")

            # 이미지 나이가 음수가 아니다 = 인과적이다
            for axis, age in ft.image_age.items():
                assert age.min() >= 0, (axis, age.min())
            ages = {a: float(np.median(v)) * 1000 for a, v in ft.image_age.items()}
            print(f"2. 이미지 나이(장치 시각 기준, 중앙 ms) {ages}, 음수 0개 OK")

            # 버린 양이 보인다 -- 1.3.0 에서 이것을 볼 수 있는 유일한 경로
            kept, made = len(ft), ft.n_source["agent"]
            assert made > kept, (made, kept)
            print(f"3. 폐기량이 드러난다: 카메라 {made}장 중 {kept}장 저장 "
                  f"({1 - kept / made:.1%} 폐기) OK")

            # ---- 시험 A: 같은 내용의 2.0.0 이 그 표를 정확히 되살린다
            p = os.path.join(TMP, "v2.h5")
            for name in ("episode_000", "episode_037", "episode_099"):
                if name not in f:
                    continue
                _v2_from_v1(f[name], p)
                for align in ("arrival", "capture"):
                    with h5py.File(p, "r") as g:
                        ft2 = frame_table(g["e"], align=align)
                    for k in ("agentview_rgb", "eye_in_hand_rgb"):
                        assert np.array_equal(ft2.obs[k], f[name][f"obs/{k}"][:]), (
                            f"{name}/{align}: {k} 가 1.3.0 과 다르다 -- 소비자를 "
                            "이 함수로 바꾸면 데이터가 조용히 달라진다")
                    assert np.array_equal(ft2.actions, f[name]["actions"][:])
            print("4. 같은 내용의 knu-2.0.0 이 1.3.0 표를 정확히 되살린다 "
                  "(3 에피소드 x arrival/capture) OK")

    # ============================================ 5. 솎아내기 (합성 격자)
    p = os.path.join(TMP, "syn.h5")
    t_ctrl, t_cam = _synthetic(p, n_ctrl=20, ctrl_hz=20.0, cam_hz=30.0)
    with h5py.File(p, "r") as g:
        ft = frame_table(g["e"], align="capture")
    got = ft.obs["agentview_rgb"].ravel()
    want = np.searchsorted(t_cam, t_ctrl, side="right") - 1
    assert np.array_equal(got, want), (got, want)
    assert np.array_equal(ft.actions.ravel(), np.arange(20))
    print(f"5. ZOH: 행마다 '그 시각 이하의 가장 최근' 프레임을 고른다 OK "
          f"(고른 인덱스 {got[:6].tolist()}...)")

    # 30 fps 원본에서 20 Hz 행이면 프레임의 1/3 이 안 쓰인다
    unused = set(range(len(t_cam))) - set(got.tolist())
    assert unused, "합성이 잘못됐다 -- 버려지는 프레임이 없다"
    print(f"   카메라 {len(t_cam)}장 중 {len(set(got.tolist()))}장 사용, "
          f"{len(unused)}장 미사용 OK")

    # ============================================ 6. align 이 실제로 갈린다
    with h5py.File(p, "r") as g:
        a = frame_table(g["e"], align="arrival").obs["agentview_rgb"].ravel()
        c = frame_table(g["e"], align="capture").obs["agentview_rgb"].ravel()
    assert not np.array_equal(a, c), (
        "arrival 과 capture 가 같은 프레임을 골랐다 -- 고정 지연이 짝짓기에 "
        "영향을 안 준다면 두 모드를 둘 이유가 없다")
    print(f"6. align 이 짝짓기를 바꾼다: arrival {a[:6].tolist()} vs "
          f"capture {c[:6].tolist()} OK")

    # ============================================ 7. 데시메이트는 정수배만
    with h5py.File(p, "r") as g:
        ft10 = frame_table(g["e"], rate=10.0, align="capture")
        assert len(ft10) == 10, len(ft10)
        assert np.array_equal(ft10.actions.ravel(), np.arange(0, 20, 2))
        for bad in (7.0, 12.0, 20.0001):
            try:
                frame_table(g["e"], rate=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"rate={bad} 가 통과했다 -- 정수배가 아니다")
    print("7. rate=10 은 2칸씩 솎고, 7/12/20.0001 은 거부 OK")

    # ============================================ 8. 거부해야 할 것들
    with h5py.File(p, "r") as g:
        e = g["e"]
        # 카메라 기준 + 액션 = 누출
        ft_cam = frame_table(e, anchor="agent", align="capture")
        assert ft_cam.actions is None, (
            "카메라 기준인데 액션을 돌려줬다 -- 절반의 행에서 액션이 이미지보다 "
            "먼저가 되어 모델이 결과에서 행동을 역산할 수 있다")
        # 첫 카메라 프레임은 첫 제어 tick 보다 앞이라(-16.7 ms) 그 행을 채울
        # 제어 표본이 없다. **없는 관측을 지어내는 대신 그 행을 버린다** --
        # 0 이나 첫 표본으로 채우면 조용히 거짓이 된다.
        n_fillable = int((t_cam >= t_ctrl[0]).sum())
        assert len(ft_cam) == n_fillable, (len(ft_cam), n_fillable)
        assert len(ft_cam) < len(t_cam), "채울 수 없는 행이 그대로 남았다"
        assert all(len(v) == len(ft_cam) for v in ft_cam.obs.values())
        for kw in ({"align": "nearest"}, {"anchor": "몸통"}):
            try:
                frame_table(e, **kw)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{kw} 가 통과했다")
    print("8. 카메라 기준은 액션을 안 준다 (누출 방지), 모르는 인자는 거부 OK")

    # ============================================ 9. host 없는 파일 + arrival
    p2 = os.path.join(TMP, "nohost.h5")
    _synthetic(p2)
    with h5py.File(p2, "a") as g:
        del g["e/meta/agent/host"]
    with h5py.File(p2, "r") as g:
        frame_table(g["e"], align="capture")          # capture 는 된다
        try:
            frame_table(g["e"], align="arrival")
        except ValueError as e:
            assert "host" in str(e)
        else:
            raise AssertionError("도착 시각이 없는데 arrival 이 통과했다")
    print("9. 도착 시각이 없으면 arrival 을 거부하고 이유를 말한다 OK")

    print("\nframe_table 인수 통과")


if __name__ == "__main__":
    main()
