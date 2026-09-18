"""scene 후보 생성 -- 인벤토리 제약 안의 무작위 조합과 배치.

두 진입점이 있고, 워크플로 두 가지에 각각 대응한다:

- :func:`generate_candidate`  물체 조합까지 새로 뽑는다 (전체 추천).
- :func:`place_objects`       물체 집합은 주어지고 배치만 뽑는다
  (사람이 소품을 고른 뒤 배치만 추천받는 워크플로).

배치는 :mod:`mstack.scene.placement_solver` (CP-SAT)가 만든다 -- 규칙을
제약으로 걸고 실행 가능한 배치를 받으므로, 무작위로 뽑아 놓고 버리는
재시도가 없다. 커버리지 목적함수는 다음 단계에서 솔버 쪽에 들어간다
(recommender-v3-plan.md D2).
"""

from __future__ import annotations

import random

from mstack.scene.placement_solver import enumerate_placements, solve_placement
from mstack.scene.scene_format import SceneMetadata
from mstack.scene.scene_rules import (
    check,
    object_count_range,
    violations_by_section,
)
from mstack.scene.signature import GRID

#: compose 규칙만 먼저 보기 위한 더미 배치용 칸 순서 (배치는 솔버가 정한다).
_CELLS = [(r, c) for r in range(GRID[0]) for c in range(GRID[1])]

#: 씬의 물체 개수 범위. 정본은 scene_rules.yaml 의 object_count 규칙이고,
#: count 커버리지 축의 서포트와 같은 값을 본다. 규칙 yaml 을 고치면
#: 프로세스 재시작이 필요하다 (_default_rules 캐시와 같은 정책).
MIN_OBJECTS, MAX_OBJECTS = object_count_range()

__all__ = ["MIN_OBJECTS", "MAX_OBJECTS", "generate_candidate",
           "place_objects", "all_placements"]

_DESCRIPTION = "(추천안 -- 채택 시 배치 의도를 적어주세요)"


def _md(objects: list, zones: dict, scene_id: str) -> SceneMetadata:
    """{물체: (행,열)} 배치를 SceneMetadata 로. 물체 순서는 objects 를 따른다."""
    return SceneMetadata(
        scene_id=scene_id,
        objects=list(objects),
        layout={"grid": list(GRID),
                "placements": {o: {"zone": list(zones[o])} for o in objects}},
        description=_DESCRIPTION,
    )


def generate_candidate(props: dict, rng: random.Random,
                       scene_id: str = "S999",
                       max_attempts: int = 200) -> SceneMetadata:
    """인벤토리 제약 안의 무작위 scene: 등장 category 는 색 다른 2개 이상
    (pair_if_present), pickable 최소 한 종류, 물체 수는 object_count 규칙
    범위에서 균등하게 뽑은 뒤 그 예산 안에서 채운다, 존 비충돌.
    configs/scenes/scene_rules.yaml 규칙을 만족하지 않으면 재시도한다.

    2026-08-27 일반화: category 목록을 하드코딩(cup/small_bowl/large_bowl/
    drawer)하지 않고 인벤토리에서 파생한다 -- 활성 색이 2개 이상인
    category 는 "짝" 후보(등장 시 2~3색), 색이 하나뿐인 category(drawer/
    tray)는 확률적 단일 추가. pickable 판정은 문법(PICKABLE_CATS)이 정본.
    새 소품은 props.yaml + NOUN_MAP 등록만으로 추천 대상이 된다."""
    from mstack.scene.instruction_grammar import PICKABLE_CATS
    from mstack.scene.scene_rules import stack_pair_categories

    stack_cats = stack_pair_categories()
    active = [p for p in props.values() if not p.retired]
    # pair_if_present 규칙(2026-08-24) 아래에서는 물체 단위 무작위 뽑기가
    # 거의 다 기각된다 -- category 단위로 "짝"을 뽑는다.
    by_cat: dict = {}
    for p_ in active:
        by_cat.setdefault(p_.category, {}).setdefault(p_.color, []).append(p_)
    paired_cats = sorted(c for c in by_cat if len(by_cat[c]) >= 2)
    single_cats = sorted(c for c in by_cat if len(by_cat[c]) == 1)
    if not any(c in PICKABLE_CATS for c in paired_cats):
        raise ValueError("인벤토리에 2색 이상인 pickable category 가 없다")
    for _ in range(max_attempts):
        # 크기를 **먼저** 뽑는다 (2026-09-18). 예전에는 종류를 1~2개 뽑고 색을
        # 2~3개씩 붙인 결과가 크기가 됐는데, 그러면 상한이 5일 때 3종(최소 6개)이
        # 구조적으로 불가능했고 크기 분포도 5에 몰렸다 (후보 200개 중 121개).
        # 크기를 균등하게 뽑고 그 예산 안에서 채우면 count 커버리지 축이 원하는
        # 분포와 생성 분포가 같아지고, 2-2-1-1 (짝 2종 + 서랍·트레이) 같은
        # 조합도 나온다.
        target = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
        max_pairs = min(len(paired_cats), target // 2)
        if max_pairs < 1:
            continue
        cats = rng.sample(paired_cats, rng.randint(1, max_pairs))
        if not any(c in PICKABLE_CATS for c in cats):
            continue
        # 짝 종류는 **색 2개**로 시작한다 -- pair_if_present 의 최소값이고,
        # 나머지 예산은 아래에서 무엇에 쓸지 무작위로 정한다.
        picked = []
        used_colors: dict = {}
        for c in cats:
            colors = list(by_cat[c])
            rng.shuffle(colors)
            used_colors[c] = colors[:2]
            picked += [rng.choice(by_cat[c][col]) for col in colors[:2]]
        # 남은 예산 채우기: 짝 종류에 색 하나 더 / 단일 종류(서랍·트레이·
        # 커트러리) / 동일 외형 쌍(포갤 수 있는 그릇, 씬당 1쌍).
        singles_left = list(single_cats)
        rng.shuffle(singles_left)
        twin_used = False
        while len(picked) < target:
            fillers = []
            for c in cats:
                rest = [col for col in by_cat[c] if col not in used_colors[c]]
                if rest:
                    fillers.append(("color", c, rest))
            if singles_left:
                fillers.append(("single", singles_left[0], None))
            if not twin_used:
                twins = [(p_, q) for p_ in picked if p_.category in stack_cats
                         for q in by_cat[p_.category][p_.color] if q.id != p_.id]
                if twins:
                    fillers.append(("twin", None, twins))
            if not fillers:
                break
            kind, cat, extra = rng.choice(fillers)
            if kind == "color":
                col = rng.choice(extra)
                used_colors[cat].append(col)
                picked.append(rng.choice(by_cat[cat][col]))
            elif kind == "single":
                singles_left.pop(0)
                picked.append(rng.choice(by_cat[cat][next(iter(by_cat[cat]))]))
            else:
                twin_used = True
                picked.append(rng.choice(extra)[1])
        if not MIN_OBJECTS <= len(picked) <= MAX_OBJECTS:
            continue
        ids = [p.id for p in picked]
        # 물체 구성 규칙은 배치와 무관하므로 솔버를 부르기 전에 먼저 거른다.
        probe = _md(ids, dict(zip(ids, _CELLS)), scene_id)
        if violations_by_section(probe, props)["compose"]:
            continue
        zones = solve_placement(ids, props, seed=rng.randrange(2 ** 31))
        if zones is None:
            continue                       # 규칙상 놓을 자리가 없는 조합
        md = _md(ids, zones, scene_id)
        # 최종 게이트: 추천기는 규칙을 어긴 scene 을 절대 내보내지 않는다.
        # (솔버와 check() 의 동등성은 selftest 가 전수로 검사하지만, 실행
        #  시점에도 한 번 더 확인하는 값이 파싱 한 번보다 크다.)
        if not check(md, props):
            return md
    raise ValueError(
        f"규칙을 만족하는 후보를 {max_attempts}회 시도 중 생성하지 못함"
    )


def place_objects(objects: list, props: dict, rng: random.Random,
                  scene_id: str = "S999") -> SceneMetadata:
    """물체 집합이 정해졌을 때 규칙을 만족하는 배치 하나.

    **배치 규칙만** 본다. 물체 구성 자체의 위반(compose: 예를 들어 컵이 한
    개뿐)은 어떤 배치로도 고칠 수 없으므로 여기서 후보를 버리는 근거가 되지
    않는다 -- 사람이 고른 조합을 추천기가 조용히 거부하면 "왜 아무것도 안
    나오는지" 알 수 없게 된다. 그런 위반은
    :func:`mstack.scene.scene_rules.violations_by_section` 으로 호출자가
    읽어 사용자에게 보여준다.
    """
    if not objects:
        raise ValueError("배치할 물체가 없다")
    zones = solve_placement(objects, props, seed=rng.randrange(2 ** 31))
    if zones is None:
        raise ValueError(
            "배치 규칙을 만족하는 배치가 없다 -- 이 물체 구성은 격자에 "
            "들어갈 수 없다 (예: 키 큰 소품이 여러 열을 비운다)")
    return _md(objects, zones, scene_id)


def all_placements(objects: list, props: dict,
                   scene_id: str = "S999") -> list:
    """실행 가능한 배치 **전부** [SceneMetadata, ...].

    무작위 표본이 아니라 가능한 배치 전체다 -- 배치만 추천하는 워크플로는
    후보를 뽑을 필요 없이 전수에서 고른다.
    """
    if not objects:
        raise ValueError("배치할 물체가 없다")
    return [_md(objects, z, scene_id)
            for z in enumerate_placements(objects, props)]
