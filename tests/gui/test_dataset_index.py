"""dataset_index — scene HDF5 에서 파생하는 3종 TSV 색인 검증.

- 임시 디렉터리에 작은 scene HDF5 2개를 만들고 scenes/cells/episodes.tsv 의
  헤더와 행 수·집계 값을 본다.
- attr 이 빠진 에피소드(옛 스키마 시늄)가 있어도 죽지 않고 ``-`` 로 남는다.
- 깨진 scene 파일 하나가 나머지 색인을 망치지 않는다.
- 진입점 스크립트(``--out``)가 모듈과 같은 TSV 를 만든다.

Qt 없이 돈다 — 로봇/카메라 불필요.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

WT = str(Path(__file__).resolve().parents[2])   # 리포 루트
sys.path.insert(0, WT)

from mstack.scene.dataset_index import (  # noqa: E402
    CELLS_HEADER,
    EPISODES_HEADER,
    SCENES_HEADER,
    build_dataset_index,
)
from mstack.scene.scene_format import (  # noqa: E402
    SceneMetadata,
    SceneWriter,
    scene_filename,
)

TMP = Path(tempfile.mkdtemp(prefix="dsindex_"))
SID_A, SID_B, SID_C = "SAAAAAAA1", "SBBBBBBB1", "SCCCCCCC1"   # 불투명 ID (고정)


def _cleanup():
    shutil.rmtree(TMP, ignore_errors=True)


def _make_scene(root: Path, scene_id: str, objects: list,
                placements: dict, slots: list) -> None:
    """slots: [(instruction_id, instruction, [success, ...])] -- success 목록만큼
    에피소드를 저장한다."""
    md = SceneMetadata(scene_id=scene_id, objects=objects,
                       layout={"grid": [3, 3], "placements": placements})
    w = SceneWriter(root, metadata=md, collector="test")
    r = np.random.default_rng(0)
    for iid, text, successes in slots:
        for ok in successes:
            w.start_episode()
            for _ in range(3):
                w.add_frame(
                    agentview_rgb=r.integers(0, 255, (8, 8, 3), dtype=np.uint8),
                    eye_in_hand_rgb=r.integers(0, 255, (8, 8, 3), dtype=np.uint8),
                    joint_positions=r.standard_normal(7).astype(np.float32),
                    gripper_position=0.5, ee_pos_quat=np.zeros(7),
                    gripper_closed=False,
                    commanded_joint_positions=r.standard_normal(7).astype(
                        np.float32),
                    commanded_gripper=0.0)
            w.save_buffer(w.detach_buffer(), instruction=text,
                          instruction_id=iid, success=ok, collector="test")
    w.close()


ds = TMP / "fr3-tabletop"
ds.mkdir()
_make_scene(
    ds, SID_A,
    ["OBJ-CUP-BLU-01", "OBJ-BOWLL-YEL-01"],
    {"OBJ-CUP-BLU-01": {"zone": [0, 0]}, "OBJ-BOWLL-YEL-01": {"zone": [0, 1]}},
    [("I000", "pick up the blue cup and place it inside the large yellow bowl",
      [True, False]),
     ("I001", "drag the blue cup next to the large yellow bowl", [True])])
_make_scene(
    ds, SID_B, ["OBJ-DRAWER-01"],
    {"OBJ-DRAWER-01": {"zone": [0, 0]}},
    [("I002", "open the top drawer", [True, True])])

# 옛 스키마 시늄: SID_A 의 두 번째 에피소드에서 attr 을 뗀다.
with h5py.File(ds / scene_filename(SID_A), "a") as f:
    g = f["episode_001"]
    for attr in ("success", "collector", "timestamp"):
        del g.attrs[attr]

out = TMP / "out"
result = build_dataset_index(ds, out_dir=out)
assert not result.errors, result.errors

# ---- 1. 파일 3종 + 헤더 ----
for kind, header in (("scenes", SCENES_HEADER), ("cells", CELLS_HEADER),
                     ("episodes", EPISODES_HEADER)):
    p = out / f"{kind}.tsv"
    assert p.is_file(), p
    first = p.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert first == list(header), (kind, first)
print("1 통과: 3종 TSV 생성 + 헤더 정확")

# ---- 2. scenes.tsv: scene 2행, ordinal 은 만든 순서 1부터, objects 쉼표 연결 ----
# scene ID 가 불투명해지면서 "몇 번째"는 ID 에서 읽히지 않는다 -- 그 자리를
# ordinal 열이 맡고, scene_id 열에는 불투명 ID 가 그대로 든다.
scene_lines = (out / "scenes.tsv").read_text(encoding="utf-8").splitlines()
assert len(scene_lines) == 3, scene_lines          # 헤더 + scene 2
row = scene_lines[1].split("\t")
assert row[0] == "1" and row[1] == SID_A, row      # ordinal = 만든 순서
assert row[3] == "3" and row[4].isdigit(), row     # n_episodes, bytes
assert row[6] == "OBJ-CUP-BLU-01,OBJ-BOWLL-YEL-01", row
row2 = scene_lines[2].split("\t")
assert row2[0] == "2" and row2[1] == SID_B, row2
print("2 통과: scenes.tsv 2행 + ordinal(만든 순서)/n_episodes/bytes/objects (쉼표 연결)")

# ---- 3. cells.tsv: 칸 집계 + 스킬/역할 해석 ----
# 이 데이터셋엔 instructions.json 이 없다 — kind 는 전부 기본값 task 다.
cell_lines = (out / "cells.tsv").read_text(encoding="utf-8").splitlines()
assert len(cell_lines) == 4, cell_lines            # 헤더 + 칸 3
cells = {l.split("\t")[1]: l.split("\t") for l in cell_lines[1:]}
row = cells["I000"]
assert row[0] == SID_A, row
assert row[2] == "task", row                       # 계획 없음 -> 기본값 task
assert row[3] == "pick-inside", row                # skill_of 정본 분류
assert row[4] == "OBJ-CUP-BLU-01", row             # 조작 물체 oid
assert row[5] == "OBJ-BOWLL-YEL-01", row           # 목적지 oid
assert row[6] == "2" and row[7] == "1", row        # success 뺀 에피소드는 n_ok 에 안 센다
row = cells["I001"]
assert row[2] == "task", row
assert row[3] == "drag-next_to" and row[4] == "OBJ-CUP-BLU-01", row
assert row[5] == "OBJ-BOWLL-YEL-01" and row[7] == "1", row
row = cells["I002"]
assert row[2] == "task", row
assert row[3] == "drawer-open" and row[4] == "OBJ-DRAWER-01", row
assert row[5] == "-", row                          # 목적지 없는 지시문은 -
assert row[6] == "2" and row[7] == "2", row
print("3 통과: cells.tsv 칸 단위 집계 + kind(계획 없음=task) + 스킬/물체/목적지")

# ---- 4. episodes.tsv: attr 빠진 에피소드는 - 로 남고 죽지 않는다 ----
ep_lines = (out / "episodes.tsv").read_text(encoding="utf-8").splitlines()
assert len(ep_lines) == 6, ep_lines                # 헤더 + 에피소드 5
ep1 = ep_lines[2].split("\t")                      # episode_001 (attr 뺀 것)
assert ep1[4] == "3", ep1                          # n_frames = actions.shape[0]
assert ep1[3] == "task", ep1                       # 계획 없음 -> 기본값 task
assert ep1[5] == "-" and ep1[6] == "-" and ep1[7] == "-", ep1
ep0 = ep_lines[1].split("\t")
assert ep0[5] == "1" and ep0[6] == "test", ep0     # 정상 에피소드는 그대로
print("4 통과: attr 이 빠진 에피소드는 - 로 남고 나머지는 정상")

# ---- 5. 진입점 스크립트: --out 이 같아야 한다 ----
out_cli = TMP / "out_cli"
r = subprocess.run(
    [sys.executable, str(Path(WT) / "scripts" / "analyze" /
                         "build_dataset_index.py"), str(ds), "--out", str(out_cli)],
    capture_output=True, text=True)
assert r.returncode == 0, r.stderr
for kind in ("scenes", "cells", "episodes"):
    assert (out_cli / f"{kind}.tsv").is_file(), kind
    assert (out_cli / f"{kind}.tsv").read_text(encoding="utf-8") == \
           (out / f"{kind}.tsv").read_text(encoding="utf-8"), kind
print("5 통과: 스크립트 --out 출력이 모듈과 동일")

# ---- 6. 깨진 파일 하나가 나머지를 망치지 않는다 ----
(ds / "scene_002.hdf5").write_bytes(b"not an hdf5")
result2 = build_dataset_index(ds)
assert len(result2.errors) == 1 and result2.errors[0][0] == "scene_002.hdf5", \
    result2.errors
assert len(result2.scenes) == 2 and len(result2.episodes) == 5, \
    (len(result2.scenes), len(result2.episodes))
print("6 통과: 깨진 scene 파일 건어너뛰기 + errors 기록")

# ---- 7. 계획의 reset 슬롯이 kind 에 반영된다 ----
# kind 는 HDF5 에서 오지 않는다 — 에피소드 attrs 에 kind 필드가 없고 앞으로도
# 없을 수 있다. 값은 계획(instructions.json)의 슬롯에서만 온다는 것을 본다.
ds2 = TMP / "fr3-reset"
ds2.mkdir()
_make_scene(
    ds2, SID_C,
    ["OBJ-CUP-BLU-01", "OBJ-BOWLL-YEL-01"],
    {"OBJ-CUP-BLU-01": {"zone": [0, 0]}, "OBJ-BOWLL-YEL-01": {"zone": [0, 1]}},
    [("I000", "pick up the blue cup and place it inside the large yellow bowl",
      [True]),
     ("I001", "reset", [True])])
(ds2 / "instructions.json").write_text(json.dumps({
    "plan_version": 1,
    "scenes": [{
        "scene_id": SID_C,
        "slots": [
            {"instruction_id": "I000",
             "instruction": "pick up the blue cup and place it inside the "
                            "large yellow bowl",
             "target": 1},
            {"instruction_id": "I001", "instruction": "reset",
             "target": 1, "kind": "reset"},
        ],
    }],
}, ensure_ascii=False) + "\n", encoding="utf-8")
out2 = TMP / "out_reset"
result3 = build_dataset_index(ds2, out_dir=out2)
assert not result3.errors, result3.errors
cells2 = {l.split("\t")[1]: l.split("\t") for l in
          (out2 / "cells.tsv").read_text(encoding="utf-8").splitlines()[1:]}
assert cells2["I000"][2] == "task", cells2["I000"]
assert cells2["I001"][2] == "reset", cells2["I001"]
eps2 = (out2 / "episodes.tsv").read_text(encoding="utf-8").splitlines()[1:]
kinds2 = sorted(l.split("\t")[3] for l in eps2)
assert kinds2 == ["reset", "task"], kinds2
print("7 통과: 계획의 reset 슬롯이 kind=reset 으로 색인에 반영")

# ---- 8. 깨진 계획 파일이 색인을 죽이지 않고 kind 는 전부 task ----
(ds2 / "instructions.json").write_text("{ broken json", encoding="utf-8")
result4 = build_dataset_index(ds2)
assert not result4.errors, result4.errors          # scene 파일 자체는 정상
kinds4 = {c["kind"] for c in result4.cells} | {e["kind"] for e in result4.episodes}
assert kinds4 == {"task"}, kinds4
print("8 통과: 깨진 계획은 kind 전부 task 로 폰백 + 색인 완료")

print("\ndataset_index 검증 통과")
_cleanup()
import os  # noqa: E402

# os._exit 는 버퍼를 비우지 않는다 -- 먼저 비운다 (run_all.sh 는 종료 코드만 본다).
sys.stdout.flush()
os._exit(0)
