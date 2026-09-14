"""파일이 **무엇이 만들었는지**를 말한다 (knu-1.2.2, 2026-09-13 조작자 요청).

물리 셋업(station·payload·reset)은 적어 왔는데 그것을 돌린 소프트웨어는
어디에도 없었다. 정책이 이상하게 움직일 때 첫 질문이 "이 데이터는 어느
컨트롤러, 어느 수집기 코드에서 나왔나" 인데 답이 사람 기억뿐이었다.

여기서 지키는 것:

1. 읽은 값만 적는다 -- 못 읽은 항목은 **attrs 에 아예 없다** (knu-1.1.0 이
   모든 프레임 0 인 힘 필드를 적었다가 아무도 그 값을 믿을 수 없게 된 것이
   이 규칙의 출처다).
2. 판번호를 모르면 그 버전 도장을 **찍지 않는다** (파일이 갖지 않은 것을
   가졌다고 주장하지 않는다).
3. `provenance_source` 가 live / backfilled 를 구분한다.
"""
import os
import sys
import tempfile
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import helpers  # noqa: E402
helpers.isolate_state()

from mstack.data.dataset_schema import (  # noqa: E402
    META_COLLECTOR_COMMIT,
    META_FR3_SYSTEM_VERSION,
    META_PROVENANCE_SOURCE,
    META_PYLIBFRANKA_VERSION,
    SCHEMA_FIELDS,
    SCHEMA_VERSION,
)
from mstack.data.provenance import collector_commit  # noqa: E402
from mstack.scene.props import active_prop_ids  # noqa: E402
from mstack.scene.scene_format import (  # noqa: E402
    SceneMetadata,
    SceneWriter,
    read_scene_metadata,
)

OBJ = ["OBJ-CUP-WHT-02"]
LAYOUT = {"grid": [3, 3], "placements": {OBJ[0]: {"zone": [0, 0]}}}

# ---- 1. 기본 기록 버전이 판번호를 요구한다 ----
assert SCHEMA_VERSION == "knu-1.2.2", SCHEMA_VERSION
need = SCHEMA_FIELDS[SCHEMA_VERSION]["metadata_attrs"]
assert META_COLLECTOR_COMMIT in need and META_PROVENANCE_SOURCE in need, need
# 로봇이 답해야만 아는 것들은 **요구하지 않는다** -- 시뮬레이터나 로봇이 꺼진
# 세션에도 파일은 찍혀야 한다.
assert META_PYLIBFRANKA_VERSION not in need and META_FR3_SYSTEM_VERSION not in need
# 세 갈래 모두 한 칸씩 올라간 판이 있다 (지금 데이터셋에 셋이 다 살아 있다).
for v in ("knu-1.0.1", "knu-1.1.2", "knu-1.2.2"):
    assert META_COLLECTOR_COMMIT in SCHEMA_FIELDS[v]["metadata_attrs"], v
print("1. knu-1.2.2 가 요구하는 것 / 요구하지 않는 것 OK")

# ---- 2. git 커밋을 읽는다 ----
sha = collector_commit()
assert sha and len(sha.split("-")[0]) == 12, sha
assert collector_commit(repo_root="/definitely/not/a/repo") == "", "없는 저장소는 빈 문자열"
print(f"2. 수집기 커밋 읽기 OK ({sha})")

with tempfile.TemporaryDirectory() as d:
    root = Path(d)

    # ---- 3. 아는 것만 적힌다 ----
    md = SceneMetadata(
        scene_id="S000", objects=OBJ, layout=LAYOUT,
        dataset_version=SCHEMA_VERSION,
        payload_mass=0.85, payload_com=[0.0, 0.0, 0.03],
        reset_pose="libero", reset_qpos=[0.0] * 7,
        collector_commit=sha, pylibfranka_version="0.21.2",
        fr3_system_version="5.10.0", provenance_source="live")
    SceneWriter(root, metadata=md, known_prop_ids=active_prop_ids()).close()
    with h5py.File(root / "scene_000.hdf5", "r") as f:
        attrs = dict(f["metadata"].attrs)
    assert attrs[META_COLLECTOR_COMMIT] == sha
    assert attrs[META_PYLIBFRANKA_VERSION] == "0.21.2"
    assert attrs[META_PROVENANCE_SOURCE] == "live"
    assert "fr3_system_build" not in attrs, "안 준 값이 적혔다"
    assert attrs["dataset_version"] == SCHEMA_VERSION, attrs["dataset_version"]
    back = read_scene_metadata(root / "scene_000.hdf5")
    assert back.collector_commit == sha and back.provenance_source == "live"
    assert back.fr3_system_build is None
    print("3. 아는 것만 적고, 읽으면 그대로 돌아온다 OK")

    # ---- 4. 판번호를 모르면 그 도장을 안 찍는다 ----
    md2 = SceneMetadata(
        scene_id="S001", objects=OBJ, layout=LAYOUT,
        dataset_version=SCHEMA_VERSION,
        payload_mass=0.85, payload_com=[0.0, 0.0, 0.03],
        reset_pose="libero", reset_qpos=[0.0] * 7)      # provenance 없음
    w = SceneWriter(root, metadata=md2, known_prop_ids=active_prop_ids())
    w.close()
    with h5py.File(root / "scene_001.hdf5", "r") as f:
        v = str(f["metadata"].attrs["dataset_version"])
        assert META_COLLECTOR_COMMIT not in f["metadata"].attrs
    assert v == "knu-1.2.1", v      # 한 칸 내려서 찍힌다
    assert "knu-1.2.2" in (w.version_note or ""), w.version_note
    print(f"4. 판번호 없으면 {v} 로 내려 찍고 이유를 남긴다 OK")

print("test_provenance OK")
