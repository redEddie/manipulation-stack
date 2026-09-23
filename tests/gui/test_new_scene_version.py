"""구성기로 만든 빈 scene 이 **처음부터 최신 버전**으로 찍히는가.

2026-09-23 S024: [✚ 새 Scene 만들기] 로 만든 빈 scene 이 ``knu-1.1.1`` 로
찍혔다. 버튼이 로봇에 붙기 전에 파일을 만드는데, 스키마가 metadata 로
요구하는 세 값(부하 모델·리셋 자세·판번호 출처)은 그 시점에 아무도 채우지
않았기 때문이다. ``stampable_version`` 이 "못 채우는 요구가 있으면 내린다"
규칙대로 2.0.0 -> 1.1.1 까지 내려갔다.

1.x 시절엔 무해했다 -- Connect 가 ``_resume_version`` 으로 도장을 올려
주었다. 2.0.0 부터는 MAJOR 가 달라 이어찍기 자체가 거부되므로, 그 파일은
**영원히 녹화할 수 없는 빈 scene** 이 된다.

여기서 확인하는 것:

1. 세 값을 채우면 새 scene 이 곧바로 최신 버전으로 찍힌다
2. 그 파일에 최신 버전 세션으로 이어찍을 수 있다 (MAJOR 가 같으므로)
3. 부하 모델을 모르면 도장이 MAJOR 를 넘어 내려간다 -- 그래서 구성기는
   ``metadata_pending`` 으로 그 판단을 Connect 로 미룬다
4. 리셋 자세와 판번호는 **로봇 없이** 채워진다 (station 설정과 git)
5. 노드 없이 만든 빈 scene 도 요청한 버전으로 찍히고, Connect 가 부하
   모델을 채운다 -- #47 방어는 없어진 것이 아니라 그 자리로 옮겨갔다
6. Connect 도 모르면 **녹화를 시작하지 않는다** (빈 scene 일 때만)
"""
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from mstack.collect.session_meta import (  # noqa: E402
    apply_to_metadata,
    offline_provenance,
    reset_pose_from_station,
)
from mstack.data.dataset_schema import SCHEMA_VERSION, parse_schema_version  # noqa: E402
from mstack.scene.scene_format import (  # noqa: E402
    SceneMetadata,
    SceneWriter,
    stampable_version,
)

TMP = Path(tempfile.mkdtemp(prefix="newscene_"))


def _md(sid="S024"):
    return SceneMetadata(
        scene_id=sid, objects=["OBJ-CUP-BLU-02"],
        layout={"grid": [3, 3], "placements": {"OBJ-CUP-BLU-02": {"zone": [1, 1]}}},
        station="knu-eng7", dataset_version=SCHEMA_VERSION)


# ---------------------------------------------- 4. 로봇 없이 아는 것
reset = reset_pose_from_station()
prov = offline_provenance()
assert reset and reset.get("name") and reset.get("qpos"), (
    f"리셋 자세는 station 설정에서 와야 한다 (로봇 불필요): {reset!r}")
assert prov.get("collector_commit"), (
    f"수집기 커밋은 git 에서 와야 한다 (로봇 불필요): {prov!r}")
print(f"4. 로봇 없이: 리셋={reset['name']} · 커밋={prov['collector_commit'][:8]} OK")

# ---------------------------------------------- 1. 세 값이 있으면 최신으로
payload = {"mass": 0.85, "com": [0.0, 0.0, 0.05]}
md = _md()
apply_to_metadata(md, payload, reset, prov)
assert md.provenance_source == "live", md.provenance_source

root = TMP / "ds"
w = SceneWriter(root, metadata=md, session_version=SCHEMA_VERSION)
assert w.metadata.dataset_version == SCHEMA_VERSION, (
    f"새 scene 이 {w.metadata.dataset_version} 로 찍혔다 (요청 {SCHEMA_VERSION}) -- "
    f"{w.version_note}")
assert not w.version_note, f"내려 찍을 이유가 없는데 사유가 남았다: {w.version_note}"
w.close()
print(f"1. 새 scene 이 곧바로 {SCHEMA_VERSION} 로 찍힌다 OK")

# ---------------------------------------------- 2. 그 파일에 이어찍을 수 있다
w2 = SceneWriter(root, scene_id="S024", resume=True,
                 session_version=SCHEMA_VERSION,
                 session_payload=payload, session_reset=reset,
                 session_provenance=prov)
assert w2.metadata.dataset_version == SCHEMA_VERSION
w2.close()
print("2. 같은 버전 세션으로 이어찍기 OK (MAJOR 가 같다)")

# ---------------------------------------------- 3. 부하를 모르면 MAJOR 를 넘어 내려간다
down = stampable_version(SCHEMA_VERSION, False, bool(reset), bool(prov))
want_major = parse_schema_version(SCHEMA_VERSION)[0]
assert down != SCHEMA_VERSION, "부하 없이도 최신이 찍힌다면 #47 방어가 없어진 것이다"
assert parse_schema_version(down)[0] != want_major, (
    f"{down} 이 같은 MAJOR 라면 이 테스트의 전제가 틀렸다")
print(f"3. 부하 모델이 없으면 {SCHEMA_VERSION} -> {down} (MAJOR 가 넘어간다)")

# 그 파일이 실제로 못 쓰는 파일이 되는지까지 확인한다 (원래 증상)
md3 = _md("S025")
apply_to_metadata(md3, None, reset, prov)
w3 = SceneWriter(root, metadata=md3, session_version=SCHEMA_VERSION)
assert w3.metadata.dataset_version == down, w3.metadata.dataset_version
w3.close()
try:
    SceneWriter(root, scene_id="S025", resume=True,
                session_version=SCHEMA_VERSION,
                session_payload=payload, session_reset=reset,
                session_provenance=prov)
    raise AssertionError("MAJOR 가 다른데 이어찍기가 열렸다 -- 가드가 사라졌다")
except ValueError as e:
    assert "MAJOR" in str(e), e
print("   내려 찍힌 파일은 이어찍기가 거부된다 -- 이것이 S024 의 증상이다 OK")

# ---------------------------------------------- 5. pending: 노드 없이 만들어도 최신
md5 = _md("S026")
apply_to_metadata(md5, None, reset, prov)          # 부하 모델 없음 (노드 없음)
w5 = SceneWriter(root, metadata=md5, session_version=SCHEMA_VERSION,
                 metadata_pending=True)
assert w5.metadata.dataset_version == SCHEMA_VERSION, (
    f"노드 없이 만든 빈 scene 이 {w5.metadata.dataset_version} 로 찍혔다")
assert "부하 모델" in w5.version_note, w5.version_note
w5.close()
print(f"5. 노드 없이 만들어도 {SCHEMA_VERSION} 로 찍힌다 (부하는 Connect 로 미룸) OK")

# Connect 가 채운다
w6 = SceneWriter(root, scene_id="S026", resume=True,
                 session_version=SCHEMA_VERSION,
                 session_payload=payload, session_reset=reset,
                 session_provenance=prov)
assert w6.metadata.dataset_version == SCHEMA_VERSION
w6.close()
import h5py  # noqa: E402

with h5py.File(root / "scene_026.hdf5", "r") as f:
    got = float(f["metadata"].attrs["payload_mass"])
assert abs(got - payload["mass"]) < 1e-9, got
print(f"   Connect 가 부하 모델({got} kg)을 채웠다 OK")

# ---------------------------------------------- 6. 끝까지 모르면 거절한다
md7 = _md("S027")
apply_to_metadata(md7, None, reset, prov)
SceneWriter(root, metadata=md7, session_version=SCHEMA_VERSION,
            metadata_pending=True).close()
try:
    SceneWriter(root, scene_id="S027", resume=True,
                session_version=SCHEMA_VERSION,
                session_payload=None, session_reset=reset,
                session_provenance=prov)
    raise AssertionError(
        "부하 모델을 끝까지 모르는데 녹화가 열렸다 -- #47 방어가 사라졌다")
except ValueError as e:
    assert "payload_mass" in str(e), e
print("6. Connect 도 모르면 녹화를 시작하지 않는다 OK (#47 방어가 여기로 옮겨왔다)")

print("\n새 scene 버전 도장 인수 통과")
