"""Prop position bias test -- does an object's category predict its grid cell?
(analysis for the paper)

The position uniformity in audit_scene_diversity.py counts cell usage while
ignoring category, so it cannot see a category-position coupling such as
"cups always sit in the left column". This script measures that coupling with
mutual information and separates it from chance with a permutation test.

    U = H(Z) / log|Z|                       cell-usage uniformity (1 = all 9 cells evenly)
    I(K;Z) = H(K) + H(Z) - H(K,Z)           coupling of category K and cell Z (0 = independent)
    p = (#{I_null >= I_obs} + 1) / (N + 1)  chance of an I this large once cell labels are shuffled

Three roles are measured separately:
    all  every object placed in a scene
    obj  the object an instruction manipulates (pick/drag/tidy; the drawer for drawer verbs)
    tgt  the destination object of an instruction

**Permutations shuffle physical placements, (scene, object), not episodes.**
Collecting the same instruction ten times in one scene still decides the
placement once. Shuffling episodes one by one (naive permutation) treats those
ten as independent observations, drives the null unrealistically low, and gives
p < 0.001 with no coupling at all (pseudo-replication). Both results are printed
side by side so the gap itself is visible. Episode weights are kept because they
are the distribution a policy sees in training.

Usage:
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
    """Instruction -> (manipulated oid, destination oid); None where unresolved.

    The regex groups carry no article ("small blue bowl"), while
    resolve_reference expects a phrase starting with "the", so it is prepended.
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
    """Per role {(scene, oid): [category, cell, episode count]}, plus unresolved counts."""
    units = {r: {} for r in ROLES}
    unresolved = Counter()
    for p in iter_scene_files(root):
        try:
            md = read_scene_metadata(p)
            episodes = list_scene_episodes(p)
        except OSError as e:          # locked by a live collection -- skip, never wait
            print(f"[경고] {p.name} 읽기 실패 ({type(e).__name__}) -- 제외")
            continue
        for oid in md.objects:
            z = _zone(md, oid)
            if oid in props and z is not None:
                units["all"][(md.scene_id, oid)] = [props[oid].category, tuple(z), 1]
        for e in episodes:
            text = str(e.get("instruction", ""))
            o, t = instruction_roles(text, md, props)
            wants_tgt = "drawer" not in text or "into the" in text
            for role, oid in (("obj", o), ("tgt", t)):
                if role == "tgt" and not wants_tgt:
                    continue                  # drawer open/close has no destination
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
    """Weighted I(K;Z) in nats."""
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
    """unit_list = [[category, cell, episode count], ...] for one role."""
    cats = [u[0] for u in unit_list]
    cells = [u[1] for u in unit_list]
    w = [u[2] for u in unit_list]
    n_cells = GRID[0] * GRID[1]
    # naive: expand episodes into independent observations (pseudo-replication)
    flat_k = [k for k, n in zip(cats, w) for _ in range(n)]
    flat_z = [z for z, n in zip(cells, w) for _ in range(n)]
    naive = _perm(flat_k, flat_z, [1] * len(flat_k), iters, random.Random(seed))
    # cluster: shuffle per physical placement, keep episode weights
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

    # 1. when category fixes the row (cup=0, bowl=1, tray=2) the cluster p is small
    dep = [[k, (i, rng.randrange(3)), 5] for _ in range(20)
           for i, k in enumerate(cats)]
    r = test_role(dep, 500, 0)
    assert r["cluster_p"] < 0.01, r

    # 2. independent placements replicated 20 episodes each: the naive test
    #    reports a false positive, the cluster test does not -- the reason
    #    this script exists
    ind = [[rng.choice(cats), rng.choice(cells), 20] for _ in range(24)]
    r = test_role(ind, 500, 0)
    assert r["naive_p"] < 0.01, r
    assert r["cluster_p"] > 0.05, r

    # 3. instruction parsing: article-less regex groups still resolve
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
    # colorless references (tray, cutlery) resolve too -- otherwise every tidy
    # and "... the wooden tray" episode silently drops out of the counts
    md2 = SceneMetadata(
        scene_id="S001", objects=["OBJ-CUTLERY-SET-01", "OBJ-TRAY-01"],
        layout={"grid": [3, 3], "placements": {
            "OBJ-CUTLERY-SET-01": {"zone": [2, 0]},
            "OBJ-TRAY-01": {"zone": [0, 2]}}})
    assert instruction_roles("tidy the cutlery into the wooden tray", md2, props) \
        == ("OBJ-CUTLERY-SET-01", "OBJ-TRAY-01")
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
