"""Gripper stroke and parts in scene metadata (2026-09-29).

The normalised gripper column is ``1 - width / gripper_max_width``. That
stroke is measured by homing the hand at every recording Connect (UMI fingers:
74.55 mm, not the 80 mm the station table assumed), and the visible parts
(mount / finger / pad colours) are recorded next to it.

1. station: gripper_parts parse, unknown parts are dropped, save round-trips
2. new file: measured stroke, source and parts are written and read back
3. resume, no episodes: the composer's station fallback is replaced
4. resume, with episodes: same hand passes; other stroke / parts refused
5. gripper_synth takes the stroke instead of assuming 80 mm
"""
import json
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from mstack.config import station as st  # noqa: E402
from mstack.data.dataset_schema import (  # noqa: E402
    META_GRIPPER_MAX_WIDTH, META_GRIPPER_MAX_WIDTH_SOURCE, META_GRIPPER_PARTS,
    SCHEMA_VERSION)
from mstack.data.gripper_synth import synth_gripper_states  # noqa: E402
from mstack.scene.props import active_prop_ids  # noqa: E402
from mstack.scene.scene_format import (  # noqa: E402
    SceneMetadata, SceneWriter, read_scene_metadata, scene_filename)

OBJ = ["OBJ-CUP-WHT-02"]
LAYOUT = {"grid": [3, 3], "placements": {"OBJ-CUP-WHT-02": {"zone": [0, 0]}}}
UMI_PARTS = {"mount": "white", "finger": "black", "pad": "black"}

# ---- 1. station ------------------------------------------------------------
umi = st.load_station("umi-finger").robot
assert umi.gripper == "franka_hand_umi", umi.gripper
assert umi.gripper_parts_dict == UMI_PARTS, umi.gripper_parts_dict
assert st.load_station("knu-eng7").robot.gripper_parts_dict == {}
assert "franka_hand_umi_yellow" not in st.GRIPPERS, "색이 키에 남아 있다"
assert st._gripper_parts_from({"gripper_parts": {"Mount": "White", "strap": "red"}},
                              ()) == (("mount", "white"),), "모르는 부품은 빠진다"
# 마법사가 저장해도 그리퍼가 franka_hand 로 되돌아가지 않는다
tmpd = Path(tempfile.mkdtemp(prefix="gripper_station_"))
_orig_path = st.station_path
st.station_path = lambda name: tmpd / f"{name}.yaml"
try:
    import dataclasses

    cfg = dataclasses.replace(st.load_station("umi-finger"), name="gripper-save-test")
    st.save_station(cfg)
    text = (tmpd / f"{cfg.name}.yaml").read_text(encoding="utf-8")
    assert "gripper: franka_hand_umi" in text, text
    assert "finger: black" in text and "mount: white" in text, text
finally:
    st.station_path = _orig_path
print("1. station: gripper_parts 파싱·모르는 부품 제외·저장 왕복 OK")


def _md(sid, **kw):
    return SceneMetadata(scene_id=sid, objects=OBJ, layout=LAYOUT,
                         dataset_version=SCHEMA_VERSION,
                         payload_mass=0.85, payload_com=[0.0, 0.0, 0.03],
                         reset_pose="libero", reset_qpos=[0.0] * 7,
                         provenance_source="live", **kw)


MEASURED = {"name": "franka_hand_umi", "max_width": 0.07455,
            "parts": UMI_PARTS, "source": "measured"}

with tempfile.TemporaryDirectory() as d:
    root = Path(d)

    # ---- 2. new file with the measured hand --------------------------------
    sid = "SAAAAAAA1"
    SceneWriter(root, metadata=_md(
        sid, gripper="franka_hand_umi", gripper_max_width=0.07455,
        gripper_max_width_source="measured", gripper_parts=UMI_PARTS),
        known_prop_ids=active_prop_ids()).close()
    with h5py.File(root / scene_filename(sid), "r") as f:
        a = dict(f["metadata"].attrs)
    assert abs(float(a[META_GRIPPER_MAX_WIDTH]) - 0.07455) < 1e-9
    assert a[META_GRIPPER_MAX_WIDTH_SOURCE] == "measured"
    assert json.loads(a[META_GRIPPER_PARTS]) == UMI_PARTS
    back = read_scene_metadata(root / scene_filename(sid))
    assert back.gripper_parts == UMI_PARTS and back.gripper_max_width_source == "measured"
    print("2. 새 파일: 실측 최대 벌림·출처·부품 기록/읽기 OK")

    # ---- 3. composer file (station fallback), resumed with no episodes -----
    sid = "SAAAAAAA2"
    SceneWriter(root, metadata=_md(sid, gripper="franka_hand_umi",
                                   gripper_max_width=0.08,
                                   gripper_max_width_source="station"),
                known_prop_ids=active_prop_ids()).close()
    w = SceneWriter(root, scene_id=sid, resume=True,
                    session_version=SCHEMA_VERSION, session_gripper=MEASURED)
    w.close()
    back = read_scene_metadata(root / scene_filename(sid))
    assert abs(back.gripper_max_width - 0.07455) < 1e-9, back.gripper_max_width
    assert back.gripper_max_width_source == "measured"
    assert back.gripper_parts == UMI_PARTS
    print("3. 빈 scene 이어찍기: 설정값(80 mm)을 실측값(74.55 mm)으로 바꾼다 OK")

    # ---- 4. episodes already recorded ---------------------------------------
    with h5py.File(root / scene_filename(sid), "a") as f:
        f.create_group("episode_000")          # _resume_gripper 는 이름만 본다
    same = dict(MEASURED, max_width=0.0747)    # 다시 homing 한 같은 손
    SceneWriter(root, scene_id=sid, resume=True, session_version=SCHEMA_VERSION,
                session_gripper=same).close()
    assert abs(read_scene_metadata(root / scene_filename(sid)).gripper_max_width
               - 0.07455) < 1e-9, "에피소드가 있는 파일의 값을 바꿨다"
    for other, why in ((dict(MEASURED, max_width=0.08), "최대 벌림"),
                       (dict(MEASURED, parts={**UMI_PARTS, "finger": "yellow"}), "부품"),
                       (dict(MEASURED, name="franka_hand", parts={}), "그리퍼")):
        try:
            SceneWriter(root, scene_id=sid, resume=True,
                        session_version=SCHEMA_VERSION, session_gripper=other).close()
        except ValueError as e:
            assert why in str(e), (why, str(e))
        else:
            raise AssertionError(f"{why} 가 다른데 이어찍기를 받았다")
    print("4. 에피소드 있는 scene: 같은 손은 통과, 벌림·부품·그리퍼가 다르면 거부 OK")

# ---- 5. gripper_synth uses the given stroke ---------------------------------
# closes at frame 10 and settles 90% closed; the ramp moves in mm/s, so the
# same ramp is a different fraction of a 74.55 mm stroke than of an 80 mm one
acts = np.zeros((80, 8)); acts[10:, -1] = 1.0
meas = np.zeros(80); meas[60:] = 0.9
out80 = synth_gripper_states(acts, meas)
out74 = synth_gripper_states(acts, meas, max_width_mm=74.55)
assert out80[0] == 0.0 and out74[0] == 0.0          # 열림은 어느 폭에서도 0
assert not np.allclose(out80, out74), "폭 인자가 무시된다"
print("5. gripper_synth 가 폭을 인자로 받는다 OK")

print("\n그리퍼 메타데이터 인수 통과")
