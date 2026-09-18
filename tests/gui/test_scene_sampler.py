"""후보 생성기: 크기를 먼저 뽑고 그 예산 안에서 채운다 (2026-09-18).

  1. 물체 수가 object_count 범위 전체에 퍼진다 -- 예전에는 종류·색을 먼저 뽑아
     그 결과가 크기였고, 상한 5에서 5물체가 60%였으며 3종(최소 6개)은
     구조적으로 만들어질 수 없었다
  2. 짝 종류를 3종까지 고르고, 서랍·트레이 같은 단일 종류로 예산을 채워
     2-2-1-1 같은 조합이 나온다
  3. 내놓는 후보는 언제나 규칙을 만족한다 (pair_if_present 포함)
  4. 같은 seed 는 같은 후보 -- scene provenance

로봇도 화면도 필요 없다.
"""
import random
import sys
from collections import Counter
from pathlib import Path

WT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, WT)

from mstack.scene.instruction_grammar import PICKABLE_CATS  # noqa: E402
from mstack.scene.props import props_by_id  # noqa: E402
from mstack.scene.sampler import MAX_OBJECTS, MIN_OBJECTS, generate_candidate  # noqa: E402
from mstack.scene.scene_rules import check  # noqa: E402

props = props_by_id()
N = 200
mds = [generate_candidate(props, random.Random(0)) for _ in range(1)]
rng = random.Random(7)
mds = [generate_candidate(props, rng) for _ in range(N)]

sizes = Counter(len(m.objects) for m in mds)
cats = Counter(len({props[o].category for o in m.objects}) for m in mds)

# ---- 1. 크기가 범위 전체에 퍼진다
assert set(sizes) == set(range(MIN_OBJECTS, MAX_OBJECTS + 1)), sizes
assert min(sizes.values()) >= N / (MAX_OBJECTS - MIN_OBJECTS + 1) / 3, sizes
print(f"1. 물체 수가 {MIN_OBJECTS}~{MAX_OBJECTS} 전체에 퍼진다 {dict(sorted(sizes.items()))} OK")

# ---- 2. 종류 수: 3종 이상이 나온다 (상한이 6 이상일 때 4종도)
assert cats.get(3, 0) > 0, cats
if MAX_OBJECTS >= 6:
    assert cats.get(4, 0) > 0, cats
print(f"2. 종류 수 {dict(sorted(cats.items()))} OK")

# ---- 3. 규칙을 언제나 만족한다
for m in mds:
    v = check(m, props)
    assert not v, (m.objects, v)
    by_cat = Counter(props[o].category for o in m.objects)
    assert any(c in PICKABLE_CATS for c in by_cat), by_cat
    assert MIN_OBJECTS <= len(m.objects) <= MAX_OBJECTS
    # 격자 칸은 겹치지 않는다
    zones = [tuple(s["zone"]) for s in m.layout["placements"].values()]
    assert len(set(zones)) == len(zones) == len(m.objects)
print("3. 모든 후보가 규칙을 만족하고 칸이 겹치지 않는다 OK")

# ---- 4. 같은 seed = 같은 후보
a = [generate_candidate(props, random.Random(11)).objects for _ in range(3)]
b = [generate_candidate(props, random.Random(11)).objects for _ in range(3)]
assert a == b, (a, b)
print("4. 같은 seed 는 같은 후보 OK")

print("test_scene_sampler 통과")
