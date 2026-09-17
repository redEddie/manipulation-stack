"""소품 위치 편향 검정 — 물체 종류가 격자 칸을 예측하는가 (논문용).

audit_scene_diversity.py 의 position 균등성은 **종류를 무시한** 칸 사용
빈도라, "컵은 늘 왼쪽 열" 같은 종류–위치 결합을 못 잡는다. 여기서는 그
결합을 상호정보량으로 재고 순열 검정으로 우연과 구별한다.

    U = H(Z) / log|Z|                       칸 사용 균등성 (1 = 9칸 고르게)
    I(K;Z) = H(K) + H(Z) - H(K,Z)           종류 K 와 칸 Z 의 결합 (0 = 독립)
    p = (#{I_null >= I_obs} + 1) / (N + 1)  칸 라벨을 섞었을 때 이만큼 나올 확률

역할 세 가지를 따로 잰다:
    all  scene 에 놓인 모든 물체
    obj  지시문이 조작하는 물체 (pick/drag/tidy 의 대상, drawer 동사는 서랍)
    tgt  지시문의 목적지 물체

**순열은 물리 배치 (scene, 물체) 단위로 섞는다.** 한 scene 에서 같은 지시문을
10번 수집해도 배치는 한 번 정해진 것이다. 에피소드를 하나씩 섞으면(단순
순열) 10개가 독립 관측인 척 되어 무작위 기준값이 비현실적으로 낮아지고,
결합이 없어도 p < 0.001 이 나온다 (가짜 반복). 두 결과를 나란히 찍어
그 차이 자체를 보여준다. 에피소드 가중치는 정책이 학습에서 보는 분포라
유지한다.

사용:
    python scripts/analyze/audit_position_bias.py --root ~/libero_datasets/fr3-tabletop
    python scripts/analyze/audit_position_bias.py --selftest
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mstack.scene.instruction_grammar import (  # noqa: E402
    _DRAG_RE,
    _PICK_RE,
    _TIDY_RE,
    _zone,
    resolve_reference,
)
from mstack.scene.props import props_by_id  # noqa: E402
from mstack.scene.scene_format import (  # noqa: E402
    iter_scene_files,
    list_scene_episodes,
    read_scene_metadata,
)
from mstack.scene.signature import GRID  # noqa: E402

ROLES = ("all", "obj", "tgt")


def instruction_roles(sentence: str, md, props: dict) -> tuple:
    """지시문 -> (조작 물체 oid, 목적지 oid). 못 정하면 그 자리가 None.

    정규식 그룹에는 관사가 빠져 있다 ("small blue bowl") -- resolve_reference 는
    "the ..." 로 시작하는 지칭 구를 받으므로 붙여서 넘긴다.
    """
    s = sentence.strip().strip('"')
    if s in ("open the top drawer", "close the top drawer"):
        d = [o for o in md.objects if o in props and props[o].category == "drawer"]
        return (d[0] if d else None), None
    for rx in (_PICK_RE, _DRAG_RE, _TIDY_RE):
        m = rx.match(s)
        if m is None:
            continue
        qual = m.groupdict().get("qual")
        obj = "the " + m.group("obj") + (" " + qual if qual else "")
        return (resolve_reference(obj, md, props),
                resolve_reference("the " + m.group("tgt"), md, props))
    return None, None


def collect(root: Path, props: dict) -> tuple:
    """역할별 {(scene, oid): [종류, 칸, 에피소드 수]} 와 해석 실패 수."""
    units = {r: {} for r in ROLES}
    unresolved = Counter()
    for p in iter_scene_files(root):
        md = read_scene_metadata(p)
        for oid in md.objects:
            z = _zone(md, oid)
            if oid in props and z is not None:
                units["all"][(md.scene_id, oid)] = [props[oid].category, tuple(z), 1]
        for e in list_scene_episodes(p):
            text = str(e.get("instruction", ""))
            o, t = instruction_roles(text, md, props)
            wants_tgt = "drawer" not in text or "into the" in text
            for role, oid in (("obj", o), ("tgt", t)):
                if role == "tgt" and not wants_tgt:
                    continue                  # 서랍 열기/닫기엔 목적지가 없다
                z = _zone(md, oid) if oid else None
                if z is None:
                    unresolved[role] += 1
                    continue
                u = units[role].setdefault(
                    (md.scene_id, oid), [props[oid].category, tuple(z), 0])
                u[2] += 1
    return units, unresolved


def _entropy(c: Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values() if v) if n else 0.0


def mutual_info(cats: list, cells: list, weights: list) -> float:
    """가중 I(K;Z) [nats]."""
    joint, kc, zc = Counter(), Counter(), Counter()
    for k, z, w in zip(cats, cells, weights):
        joint[(k, z)] += w
        kc[k] += w
        zc[z] += w
    return _entropy(kc) + _entropy(zc) - _entropy(joint)


def uniformity(cells: list, weights: list, n_cells: int) -> float:
    c = Counter()
    for z, w in zip(cells, weights):
        c[z] += w
    return _entropy(c) / math.log(n_cells)


def _perm(cats, cells, weights, iters, rng) -> tuple:
    obs = mutual_info(cats, cells, weights)
    sh = list(cells)
    ge = 0
    total = 0.0
    for _ in range(iters):
        rng.shuffle(sh)
        v = mutual_info(cats, sh, weights)
        total += v
        ge += v >= obs - 1e-12
    return obs, total / iters, (ge + 1) / (iters + 1)


def test_role(unit_list: list, iters: int, seed: int) -> dict:
    """unit_list = [[종류, 칸, 에피소드 수], ...] 한 역할분."""
    cats = [u[0] for u in unit_list]
    cells = [u[1] for u in unit_list]
    w = [u[2] for u in unit_list]
    n_cells = GRID[0] * GRID[1]
    # 단순 순열: 에피소드를 하나하나 독립 관측으로 펼친다 (가짜 반복)
    flat_k = [k for k, n in zip(cats, w) for _ in range(n)]
    flat_z = [z for z, n in zip(cells, w) for _ in range(n)]
    naive = _perm(flat_k, flat_z, [1] * len(flat_k), iters, random.Random(seed))
    # 군집 순열: 물리 배치 단위로 섞고 에피소드 가중치는 유지
    cluster = _perm(cats, cells, w, iters, random.Random(seed))
    return {"units": len(unit_list), "episodes": sum(w),
            "U": uniformity(cells, w, n_cells), "I": cluster[0],
            "naive_null": naive[1], "naive_p": naive[2],
            "cluster_null": cluster[1], "cluster_p": cluster[2],
            "per_cat": {k: (sum(1 for c in cats if c == k),
                            uniformity([z for c, z in zip(cats, cells) if c == k],
                                       [1] * cats.count(k), n_cells))
                        for k in sorted(set(cats))}}


def report(units: dict, unresolved: Counter, iters: int, seed: int) -> dict:
    out = {}
    print(f"순열 {iters}회 · seed {seed} · 해석 못 한 에피소드 {dict(unresolved)}\n")
    print("역할  배치  에피소드   U      I     | 단순 순열 null   p      "
          "| 군집 순열 null   p")
    for role in ROLES:
        r = test_role(list(units[role].values()), iters, seed)
        out[role] = r
        print(f"{role:4s} {r['units']:5d} {r['episodes']:8d}  {r['U']:.3f} "
              f"{r['I']:.3f} |      {r['naive_null']:.3f}  {r['naive_p']:.4f} "
              f"|      {r['cluster_null']:.3f}  {r['cluster_p']:.4f}")
    print("\n종류별 칸 균등성 (물리 배치 단위, 조작 물체)")
    for k, (n, u) in out["obj"]["per_cat"].items():
        print(f"  {k:11s} n={n:3d}  U={u:.2f}")
    print("\n판정은 군집 순열 p 로 한다. all 역할은 에피소드 가중치가 1 이라 두 p 가 같다.")
    return out


def _selftest() -> None:
    cells = [(r, c) for r in range(GRID[0]) for c in range(GRID[1])]
    rng = random.Random(3)
    cats = ["cup", "bowl", "tray"]

    # 1. 종류가 칸을 결정하면(컵=0행, 그릇=1행, 트레이=2행) 군집 p 가 작다
    dep = [[k, (i, rng.randrange(3)), 5] for _ in range(20)
           for i, k in enumerate(cats)]
    r = test_role(dep, 500, 0)
    assert r["cluster_p"] < 0.01, r

    # 2. 독립 배치에 에피소드를 20번씩 복제하면 단순 순열은 거짓 양성,
    #    군집 순열은 아니다 -- 이 스크립트가 존재하는 이유
    ind = [[rng.choice(cats), rng.choice(cells), 20] for _ in range(24)]
    r = test_role(ind, 500, 0)
    assert r["naive_p"] < 0.01, r
    assert r["cluster_p"] > 0.05, r

    # 3. 지시문 해석: 관사 없는 정규식 그룹도 물체로 이어진다
    from mstack.scene.scene_format import SceneMetadata
    props = props_by_id()
    md = SceneMetadata(
        scene_id="S000",
        objects=["OBJ-BOWLS-BLU-01", "OBJ-BOWLL-WHT-01", "OBJ-DRAWER-01"],
        layout={"grid": [3, 3], "placements": {
            "OBJ-BOWLS-BLU-01": {"zone": [0, 1]},
            "OBJ-BOWLL-WHT-01": {"zone": [1, 0]},
            "OBJ-DRAWER-01": {"zone": [0, 0]}}})
    o, t = instruction_roles(
        "pick up the small blue bowl and place it inside the large white bowl",
        md, props)
    assert (o, t) == ("OBJ-BOWLS-BLU-01", "OBJ-BOWLL-WHT-01"), (o, t)
    assert instruction_roles("open the top drawer", md, props) == ("OBJ-DRAWER-01", None)
    print("position bias selftest 통과")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path.home() / "libero_datasets")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    props = props_by_id()
    units, unresolved = collect(args.root, props)
    if not units["all"]:
        print(f"{args.root} 에 scene 파일이 없다")
        return
    report(units, unresolved, args.iters, args.seed)


if __name__ == "__main__":
    main()
