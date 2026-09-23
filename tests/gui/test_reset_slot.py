"""reset 을 **번호가 아니라 역할로** 표시하고, 같은 에피소드로 찍는가.

"각 scene 의 0번 지시문을 reset 으로 예약" 은 두 가지 이유로 못 한다:

* 지금 25개 scene 전부가 ``I000`` 을 실제 작업에 쓰고 있다 (실측:
  instructions.json 의 I000 슬롯 25개). 예약하면 같은 ID 가 만든 시점에
  따라 다른 뜻이 되고, 그것이 2,434 에피소드에 걸친 조용한 충돌이 된다.
* 불투명 scene/instruction ID 로 옮겨가면 "0번" 이라는 **위치 자체가**
  없어진다.

그래서 슬롯에 ``kind`` 필드를 둔다. 여기서 확인하는 것:

1. 필드가 없으면 ``task`` -- 이미 있는 계획 파일이 그대로 읽힌다
2. ``kind="reset"`` 이 읽히고, 모르는 값은 거부된다
3. reset 문장은 **통일 문법 검사를 받지 않는다** -- "reset" 은 지칭할
   물체도 관계도 없어서, 검사에 걸면 매번 경고가 뜬다
4. 색인이 칸·에피소드마다 kind 를 싣는다 (train/eval 에서 걸러내는 값)
5. 수집기가 reset 을 **리더 대신 홈 궤적이 모는 같은 기록 루프**로 찍는다
"""
import json
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

import numpy as np  # noqa: E402

from mstack.scene.collection_plan import (  # noqa: E402
    KIND_RESET,
    KIND_TASK,
    KINDS,
    PlanSlot,
    load_plan,
)

TMP = Path(tempfile.mkdtemp(prefix="resetslot_"))
TASK = "pick up the blue cup and place it inside the large blue bowl"


def write_plan(slots) -> Path:
    p = TMP / "instructions.json"
    p.write_text(json.dumps(
        {"plan_version": 1, "scenes": [{"scene_id": "S000", "slots": slots}]}))
    return p


# ---------------------------------------------- 1. 필드가 없으면 task
plan = load_plan(write_plan([
    {"instruction_id": "I000", "instruction": TASK, "target": 10}]))
slot = plan.slots_for("S000")[0]
assert slot.kind == KIND_TASK and not slot.is_reset, slot
assert PlanSlot("I000", TASK, 1).kind == KIND_TASK
print("1. kind 를 안 적은 슬롯은 task -- 기존 계획 파일이 그대로 읽힌다 OK")

# ---------------------------------------------- 2. reset 이 읽히고 오타는 막힌다
plan = load_plan(write_plan([
    {"instruction_id": "I000", "instruction": TASK, "target": 10},
    {"instruction_id": "I001", "instruction": "reset", "target": 2,
     "kind": "reset"}]))
kinds = {s.instruction_id: s.kind for s in plan.slots_for("S000")}
assert kinds == {"I000": KIND_TASK, "I001": KIND_RESET}, kinds
assert plan.slots_for("S000")[1].is_reset
try:
    load_plan(write_plan([
        {"instruction_id": "I000", "instruction": TASK, "target": 1,
         "kind": "bogus"}]))
    raise AssertionError("모르는 kind 가 통과했다 -- 오타가 조용히 task 가 된다")
except ValueError as e:
    assert "kind" in str(e), e
print(f"2. kind={KINDS} 만 받는다 (오타는 거부) OK")

# **번호는 예약되지 않는다**: reset 이 I001 이어도 되고 I000 이 task 여도 된다
assert kinds["I000"] == KIND_TASK, "I000 이 reset 으로 예약돼 버렸다"
print("   I000 은 여전히 task -- 번호가 아니라 필드가 역할을 정한다 OK")

# ---------------------------------------------- 3. reset 은 문법 검사를 안 받는다
plan = load_plan(write_plan([
    {"instruction_id": "I000", "instruction": TASK, "target": 10},
    {"instruction_id": "I001", "instruction": "reset", "target": 2,
     "kind": "reset"}]))
assert not plan.warnings, (
    f"reset 문장이 문법 경고를 냈다: {plan.warnings} -- 지칭할 물체도 관계도 "
    "없는 문장이라 검사 대상이 아니다")
# 같은 문장을 task 로 두면 경고가 난다 -- 검사 자체는 살아 있다는 뜻
noisy = load_plan(write_plan([
    {"instruction_id": "I000", "instruction": "reset", "target": 1}]))
assert noisy.warnings, "task 슬롯의 비문법 문장이 경고를 안 낸다 -- 검사가 죽었다"
print("3. reset 은 문법 검사 제외, task 는 그대로 검사 OK")

# ---------------------------------------------- 4. 색인이 kind 를 싣는다
from mstack.scene.dataset_index import (  # noqa: E402
    CELLS_HEADER,
    EPISODES_HEADER,
    _kind_of,
    _plan_kind_map,
)

assert "kind" in CELLS_HEADER and "kind" in EPISODES_HEADER, (
    CELLS_HEADER, EPISODES_HEADER)
write_plan([
    {"instruction_id": "I000", "instruction": TASK, "target": 10},
    {"instruction_id": "I001", "instruction": "reset", "target": 2,
     "kind": "reset"}])
kmap = _plan_kind_map(TMP)
assert _kind_of(kmap, "S000", "I001") == KIND_RESET, kmap
assert _kind_of(kmap, "S000", "I000") == KIND_TASK, kmap
# 계획을 못 읽어도 색인은 만들어져야 한다 (kind 는 부가정보다)
assert _kind_of(_plan_kind_map(TMP / "nope"), "S000", "I001") == KIND_TASK
print("4. 색인이 칸·에피소드마다 kind 를 싣고, 계획이 없으면 task 로 떨어진다 OK")

# ---------------------------------------------- 5. 수집기는 같은 루프로 찍는다
from mstack.collect.worker import CollectionWorker  # noqa: E402

src = (Path(WT) / "mstack/collect/worker.py").read_text()
assert "def _record_episode(self, home_traj" in src, (
    "_record_episode 가 홈 궤적을 안 받는다 -- reset 을 따로 만든 루프로 "
    "찍으면 주기·시간축·저장 경로가 갈라져 색인과 트림이 특별 취급해야 한다")
i = src.index("def _record_episode(")
body = src[i:i + 9000]
assert "if home_traj is not None:" in body, "액션 출처가 갈리지 않는다"
assert "drop_guard.update" in body, "리더 경로의 낙하 감시가 사라졌다"
# 낙하 감시는 리더 경로에만 걸려야 한다 -- 홈 궤적은 리더가 몰지 않는다
guard_at = body.index("drop_guard.update")
assert body.index("if home_traj is not None:") < guard_at, (
    "홈 궤적에도 리더 낙하 감시가 걸린다 -- 리더가 팔을 몰고 있지 않으므로 "
    "그 판정의 전제가 성립하지 않는다")
for name in ("cmd_record_reset", "_record_reset", "_action_from_q"):
    assert hasattr(CollectionWorker, name), name
# 예약은 한 번만 소모된다
assert "self._reset_armed = False" in src, "예약이 소모되지 않는다 (매번 찍힌다)"
print("5. reset 은 액션 출처만 바꾼 같은 기록 루프로 찍힌다 (예약 1회 소모) OK")

print("\nreset 역할 표시 인수 통과")
