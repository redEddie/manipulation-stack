"""Scene generator comparison -- naive random vs. the recommender (analysis for the paper).

Both generators build the same number of scenes from the same active prop
inventory, and each set is measured with the same yardsticks:

    U_position       cell-usage uniformity over all placed objects
    I(K;Z), p        category-position coupling and its permutation p
                     (one unit per placement -- see audit_position_bias.py)
    U_category, U_color, U_count
                     normalized entropy over the full support, so an unused
                     category/color/count counts as a zero bin

Generators:
    random       object count uniform in [MIN_OBJECTS, MAX_OBJECTS], props drawn
                 uniformly without replacement, cells drawn uniformly without
                 replacement. No scene rules.
    recommender  scenes added one at a time, each from recommend_detailed(k=1)
                 given the scenes picked so far -- the way the GUI is used.

Each generator is repeated over --reps seeds; the spread across seeds is what
separates a real difference from luck. The collected dataset is printed as a
reference row when --root has scene files.

Usage:
    python scripts/analyze/compare_scene_generators.py --root ~/libero_datasets/fr3-tabletop
    python scripts/analyze/compare_scene_generators.py --reps 100 --workers 4 --csv out.csv
    python scripts/analyze/compare_scene_generators.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_position_bias import _perm, uniformity  # noqa: E402
from mstack.scene.props import props_by_id  # noqa: E402
from mstack.scene.scene_diversity import (  # noqa: E402
    MAX_OBJECTS,
    MIN_OBJECTS,
    axis_coverage,
    axis_support,
    coverage_uniformity,
    recommend_detailed,
    signature,
)
from mstack.scene.scene_format import (  # noqa: E402
    SceneMetadata,
    iter_scene_files,
    read_scene_metadata,
)
from mstack.scene.signature import GRID  # noqa: E402

GENERATORS = ("random", "recommender")
METRICS = ("U_position", "I", "p", "U_category", "U_color", "U_count")


def _cells() -> list:
    return [(r, c) for r in range(GRID[0]) for c in range(GRID[1])]


def generate_random(n_scenes: int, props: dict, seed: int) -> list:
    rng = random.Random(seed)
    active = sorted(p.id for p in props.values() if not p.retired)
    out = []
    for i in range(n_scenes):
        n = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
        objs = rng.sample(active, n)
        zones = rng.sample(_cells(), n)
        out.append(SceneMetadata(
            scene_id=f"S{i:03d}", objects=objs,
            layout={"grid": list(GRID), "placements": {
                o: {"zone": list(z)} for o, z in zip(objs, zones)}}))
    return out


def generate_recommender(n_scenes: int, props: dict, seed: int) -> list:
    out: list = []
    for i in range(n_scenes):
        picked = recommend_detailed(out, props, k=1, seed=seed * 1000 + i,
                                    scene_id=f"S{i:03d}")
        if not picked:
            break
        out.append(picked[0]["md"])
    return out


def measure(mds: list, props: dict, iters: int, seed: int) -> dict:
    sigs = [signature(md, props) for md in mds]
    cats = [k for s in sigs for k, _ in s.placements]
    cells = [z for s in sigs for _, z in s.placements]
    ones = [1] * len(cells)
    obs, _null, p = _perm(cats, cells, ones, iters, random.Random(seed))
    uni = coverage_uniformity(axis_coverage(sigs), axis_support(props))
    return {"U_position": uniformity(cells, ones, GRID[0] * GRID[1]),
            "I": obs, "p": p, "U_category": uni["category"],
            "U_color": uni["color"], "U_count": uni["count"]}


def _run_one(job: tuple) -> dict:
    gen, seed, n_scenes, iters = job
    props = props_by_id()
    fn = generate_random if gen == "random" else generate_recommender
    mds = fn(n_scenes, props, seed)
    return {"generator": gen, "seed": seed, "scenes": len(mds),
            **measure(mds, props, iters, seed)}


def _summary(rows: list) -> None:
    print(f"{'':12s} " + "  ".join(f"{m:>18s}" for m in METRICS))
    for gen in GENERATORS:
        rs = [r for r in rows if r["generator"] == gen]
        if not rs:
            continue
        cols = []
        for m in METRICS:
            v = sorted(r[m] for r in rs)
            if m == "p":
                share = sum(x < 0.05 for x in v) / len(v)
                cols.append(f"med {statistics.median(v):.2f} <.05:{share:4.0%}")
            else:
                sd = statistics.stdev(v) if len(v) > 1 else 0.0
                cols.append(f"{statistics.mean(v):.3f} ± {sd:.3f}")
        print(f"{gen:12s} " + "  ".join(f"{c:>18s}" for c in cols))


def _selftest() -> None:
    props = props_by_id()
    a = generate_random(6, props, 1)
    assert [m.objects for m in a] == [m.objects for m in generate_random(6, props, 1)]
    for md in a:
        assert MIN_OBJECTS <= len(md.objects) <= MAX_OBJECTS
        zs = [tuple(s["zone"]) for s in md.layout["placements"].values()]
        assert len(set(zs)) == len(zs)
    rows = [_run_one(("random", 1, 6, 200)), _run_one(("recommender", 1, 3, 200))]
    for r in rows:
        for m in METRICS:
            assert 0.0 <= r[m] <= 1.0 or m == "I", r
    assert rows[1]["scenes"] == 3, rows[1]
    print("compare_scene_generators selftest 통과")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path.home() / "libero_datasets")
    ap.add_argument("--scenes", type=int, default=None,
                    help="scenes per set (default: as many as --root has, else 29)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    props = props_by_id()
    real = []
    for p in iter_scene_files(args.root):
        try:
            real.append(read_scene_metadata(p))
        except OSError as e:          # locked by a live collection -- skip
            print(f"[경고] {p.name} 읽기 실패 ({type(e).__name__}) -- 제외")
    n_scenes = args.scenes or len(real) or 29
    print(f"scene {n_scenes}개 × {args.reps}회 · 순열 {args.iters}회\n")

    jobs = [(g, s, n_scenes, args.iters) for g in GENERATORS
            for s in range(args.reps)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_run_one, jobs))
    _summary(rows)
    if real:
        r = measure(real, props, args.iters, 0)
        print(f"{'collected':12s} " + "  ".join(
            f"{('p ' + format(r[m], '.2f')) if m == 'p' else format(r[m], '.3f'):>18s}"
            for m in METRICS))
    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["generator", "seed", "scenes", *METRICS])
            w.writeheader()
            w.writerows(rows)
        print(f"\n회차별 값: {args.csv}")


if __name__ == "__main__":
    main()
