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
# 요구하는 것은 **하나**다: 이 값들이 어디서 왔는지.
assert META_PROVENANCE_SOURCE in need, need
# 나머지는 요구하지 않는다. 로봇 판번호는 팔이 꺼진 세션에서 못 읽고,
# 커밋은 **옛 파일에서 복구할 수 없다** -- 요구하면 이미 찍힌 파일이 영원히
# 이 버전에 못 올라간다 (닥터가 채워 올릴 길이 막힌다).
for optional in (META_COLLECTOR_COMMIT, META_PYLIBFRANKA_VERSION,
                 META_FR3_SYSTEM_VERSION):
    assert optional not in need, optional
# 세 갈래 모두 한 칸씩 올라간 판이 있다 (지금 데이터셋에 셋이 다 살아 있다).
for v in ("knu-1.0.1", "knu-1.1.2", "knu-1.2.2"):
    assert META_PROVENANCE_SOURCE in SCHEMA_FIELDS[v]["metadata_attrs"], v
print("1. knu-1.2.2 가 요구하는 것 하나 / 요구하지 않는 것 셋 OK")

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

    # ---- 5. 닥터가 옛 파일의 칸을 채워 도장을 올린다 ----
    # 스크립트를 따로 두지 않는다 (조작자, 2026-09-13): 닥터가 이미 "같은
    # 데이터셋의 다른 scene 에 적힌 값으로 채우고 올리는" 기계를 갖고 있다.
    from mstack.scene.schema_doctor import (  # noqa: E402
        diagnose,
        fill_and_raise,
        known_versions,
    )

    old = SceneMetadata(
        scene_id="S002", objects=OBJ, layout=LAYOUT,
        dataset_version="knu-1.2.1",
        payload_mass=0.85, payload_com=[0.0, 0.0, 0.03],
        reset_pose="libero", reset_qpos=[0.0] * 7)
    SceneWriter(root, metadata=old, known_prop_ids=active_prop_ids()).close()
    path = root / "scene_002.hdf5"
    assert diagnose(path).stamped == "knu-1.2.1"

    # 출처는 **같은 데이터셋의 다른 scene** 이다 (S000 이 live 로 갖고 있다).
    known = known_versions(root)
    assert known == {"pylibfranka_version": "0.21.2",
                     "fr3_system_version": "5.10.0"}, known
    got = fill_and_raise(path, versions=known, source="다른 scene")
    assert got == "knu-1.2.2", got
    with h5py.File(path, "r") as f:
        a = dict(f["metadata"].attrs)
    assert a[META_PYLIBFRANKA_VERSION] == "0.21.2"
    assert str(a[META_PROVENANCE_SOURCE]).startswith("backfilled"), a[META_PROVENANCE_SOURCE]
    assert "다른 scene" in str(a[META_PROVENANCE_SOURCE])
    # 커밋은 **안 채운다** -- 다른 scene 의 커밋은 이 파일의 커밋이 아니다.
    assert META_COLLECTOR_COMMIT not in a, "남의 커밋을 이 파일에 적었다"
    print(f"5. 닥터가 채워 올린다 knu-1.2.1 → {got} (backfilled 로 표시) OK")

    # ---- 6. live 는 덮지 않는다 ----
    before = read_scene_metadata(root / "scene_000.hdf5").provenance_source
    fill_and_raise(root / "scene_000.hdf5", versions={"fr3_system_version": "9.9.9"})
    after = read_scene_metadata(root / "scene_000.hdf5")
    assert after.provenance_source == before == "live", (before, after.provenance_source)
    assert after.fr3_system_version == "5.10.0", after.fr3_system_version
    print("6. 수집 시점에 적힌 값(live)은 나중 추정으로 안 덮인다 OK")

    # ---- 7. 고른 것만 올라간다 (선택이 정본) ----
    # 시스템이 미리 골라 주지 않는다 (조작자, 2026-09-14): 무엇을 건드릴지는
    # 선택이 말하고, 시스템은 "같이 올라갈 수 있는가" 만 본다.
    class _Diag:
        def __init__(self, sid, stamped, satisfied, error="", can_restamp=False):
            self.scene_id, self.stamped, self.satisfied = sid, stamped, satisfied
            self.error, self.can_restamp = error, can_restamp

    import types  # noqa: E402

    from apps.workspace.features.doctor.ops import DoctorOps  # noqa: E402

    ops = DoctorOps.__new__(DoctorOps)          # 위젯 없이 로직만 쓴다
    ops.win = types.SimpleNamespace()
    ops._root = lambda: root
    ops._path = lambda sid: root / f"scene_{int(sid[1:]):03d}.hdf5"
    ops._fill_sources = lambda: (None, None, False, known_versions(root))
    ops._versions_from = "테스트"

    # S002 는 방금 1.2.2 로 올라갔으니 더 갈 곳이 없고, 새로 만든 S003 은 간다.
    md3 = SceneMetadata(
        scene_id="S003", objects=OBJ, layout=LAYOUT,
        dataset_version="knu-1.2.1",
        payload_mass=0.85, payload_com=[0.0, 0.0, 0.03],
        reset_pose="libero", reset_qpos=[0.0] * 7)
    SceneWriter(root, metadata=md3, known_prop_ids=active_prop_ids()).close()

    up, skip = ops.upgrade_targets([
        _Diag("S003", "knu-1.2.1", "knu-1.2.1"),
        _Diag("S002", "knu-1.2.2", "knu-1.2.2"),
        _Diag("S009", "?", "?", error="없는 파일"),
    ])
    assert [x[0] for x in up] == ["S003"], up
    assert up[0][2] == "knu-1.2.2", up
    reasons = dict(skip)
    assert "S002" in reasons and "S009" in reasons, reasons
    assert "읽지" in reasons["S009"], reasons["S009"]
    print("7. 고른 것마다 따로 계산하고, 못 가는 것은 이유와 함께 건너뛴다 OK")

    # ---- 8. 판번호는 **수집 세션 없이도** 읽힌다 ----
    # 처음에는 "다른 scene 에 적힌 값" 만 출처로 삼아서, 한 번 수집해야만 옛
    # 파일을 올릴 수 있었다 (조작자 지적, 2026-09-14). 부하 모델 경로를 그대로
    # 따라한 탓이다 -- 판번호는 리그에 물어보면 그 자리에서 나온다.
    from mstack.data.provenance import _desk_version, robot_versions  # noqa: E402

    # Desk 응답은 **JSON 문자열 하나**다 (줄바꿈이 진짜 개행이 아니라 \n 두 글자).
    import mstack.data.provenance as prov_mod  # noqa: E402

    class _Fake:
        def __init__(self, body): self._b = body.encode()
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    real = prov_mod.urllib.request.urlopen
    prov_mod.urllib.request.urlopen = lambda *a, **k: _Fake(
        '"5.10.0\\nec764230\\n340b9610\\n"')
    try:
        got = _desk_version("10.0.0.9", 1.0)
    finally:
        prov_mod.urllib.request.urlopen = real
    assert got == {"fr3_system_version": "5.10.0",
                   "fr3_system_build": "ec764230 340b9610"}, got
    # 로봇이 없으면 그 항목만 빠지고 나머지는 읽는다
    assert _desk_version("", 1.0) == {}
    # 스위치가 켜져 있으면 **아무것도 묻지 않는다** -- 인수 스위트가 실험실
    # 로봇의 전원 상태에 따라 달라지면 안 된다.
    assert os.environ.get(prov_mod.NO_RIG_QUERY_ENV)
    assert robot_versions(timeout=0.2) == {}, "스위치가 켜졌는데 리그에 물었다"
    print("8. 판번호를 리그에서 직접 읽는다 (JSON 문자열 파싱 포함) OK")

    # ---- 9. 잠긴 파일 에러를 사람 말로 ----
    # HDF5 는 쓰기로 열 때 배타 잠금을 요구해서, 그 파일을 **읽고 있는**
    # 프로세스 하나만 있어도 errno 11 로 튕긴다. 원문만 보면 디스크가 고장
    # 난 것처럼 읽힌다 (2026-09-14 조작자 질문).
    locked = OSError(
        "[Errno 11] Unable to synchronously open file (unable to lock file, "
        "errno = 11, error message = 'Resource temporarily unavailable')")
    why = DoctorOps._why_failed(locked)
    assert "다른 프로그램이 열고" in why, why
    assert "errno" not in why, why
    other = DoctorOps._why_failed(ValueError("무언가 다른 것"))
    assert other == "ValueError: 무언가 다른 것", other
    print("9. 잠긴 파일 에러를 사람 말로 OK")

print("test_provenance OK")
