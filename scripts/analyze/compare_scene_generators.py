"""Scene generator comparison -- random vs. recommender, with and without scene rules
(analysis for the paper).

Every condition builds sets of the same number of scenes from the same active
prop inventory, and every set is measured with the same yardsticks. A condition
is a generator crossed with a rule switch:

    random       object count k ~ U{MIN_OBJECTS..MAX_OBJECTS}, k props drawn
                 uniformly without replacement, k cells drawn uniformly without
                 replacement.
    random_cat   (sensitivity baseline) same, but each object first draws a
                 category uniformly, then a prop within it -- the inventory is
                 uneven (6 cups, 1 drawer), so plain `random` is category-uneven
                 by construction.
    recommender  scenes added one at a time with recommend_detailed(k=1) given
                 the scenes picked so far, which is how the GUI is used.
    recommender_shuffle
                 (diagnostic, not in the default set) the recommender with the
                 feasible placements shuffled before refine_placement scores
                 them. Coverage gain is category-agnostic, so permutations of
                 the same objects over the same cells tie exactly and the first
                 one in solver enumeration order wins; shuffling breaks those
                 ties at random. If the recommender's category-cell coupling
                 disappears here, tie-breaking is its cause.

    rules=on     random: rejection sampling until scene_rules.check() passes.
                 recommender: the shipped code path.
    rules=off    random: no check. recommender: the same code path with every
                 compose/placement rule removed except object_count (which
                 defines scene size, so both arms keep the same count support).

Measures per set (see docs in the paper report for formulas):

    U_*            Pielou evenness H/log|S| over the full support S
                   (unused bins count), plug-in and Miller-Madow corrected
    E_simpson_*    (1/sum p^2)/|S|, a second evenness form
    U_catcolor     evenness over (category, color) pairs -- NOT an axis the
                   recommender optimizes
    I, dI, p_perm  category-cell mutual information, shuffle-corrected
                   excess I - E[I_null], and permutation p (placement units)
    p_chi2, V      Pearson chi-square p and Cramer's V (cross-check only;
                   expected counts are small)
    *_nodrawer     the coupling measures without the one placement-ruled
                   category (drawer: ban_zones + tall-column rules)
    violate        share of scenes that break the full rule set

Usage:
    python scripts/analyze/compare_scene_generators.py --root ~/libero_datasets/fr3-tabletop
    python scripts/analyze/compare_scene_generators.py --reps 100 --workers 8 --out DIR
    python scripts/analyze/compare_scene_generators.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_position_bias import mutual_info  # noqa: E402
import mstack.scene.placement_solver as _solver  # noqa: E402
import mstack.scene.scene_rules as _rules  # noqa: E402
import mstack.scene.selector as _selector  # noqa: E402
from mstack.scene.props import props_by_id  # noqa: E402
from mstack.scene.scene_diversity import (  # noqa: E402
    MAX_OBJECTS,
    MIN_OBJECTS,
    axis_coverage,
    axis_support,
    recommend_detailed,
    signature,
)
from mstack.scene.scene_format import (  # noqa: E402
    SceneMetadata,
    iter_scene_files,
    read_scene_metadata,
)
from mstack.scene.signature import GRID  # noqa: E402

GENERATORS = ("random", "random_cat", "recommender")
DIAGNOSTIC_GENERATORS = ("recommender_shuffle",)
_ENUMERATE = _selector.enumerate_placements
AXES = ("position", "category", "color", "count")
PLACEMENT_RULED = frozenset({"drawer"})

_FULL_RULES = _rules._default_rules()
_NO_RULES = {"version": _FULL_RULES.get("version", 1),
             "compose": [e for e in _FULL_RULES["compose"]
                         if e.get("rule") == "object_count"],
             "placement": []}


def set_rules(on: bool) -> None:
    """Swap the rule set every consumer reads (checker, solver, sampler).

    Both modules look the loader up by global name at call time, so rebinding
    the module attribute is enough; the solver imported the name itself.
    """
    data = _FULL_RULES if on else _NO_RULES
    _rules._default_rules = lambda: data
    _solver._default_rules = lambda: data


def set_tie_shuffle(seed: "int | None") -> None:
    """Shuffle the selector's feasible placements (seeded), or restore."""
    if seed is None:
        _selector.enumerate_placements = _ENUMERATE
        return
    rng = random.Random(seed)

    def shuffled(*a, **kw):
        out = _ENUMERATE(*a, **kw)
        rng.shuffle(out)
        return out
    _selector.enumerate_placements = shuffled


# ------------------------------------------------------------------ generators
def _cells() -> list:
    return [(r, c) for r in range(GRID[0]) for c in range(GRID[1])]


def _md(i: int, objs: list, zones: list) -> SceneMetadata:
    return SceneMetadata(
        scene_id=f"S{i:03d}", objects=list(objs),
        layout={"grid": list(GRID), "placements": {
            o: {"zone": list(z)} for o, z in zip(objs, zones)}})


def _draw_uniform(rng, active: list, by_cat: dict) -> list:
    return rng.sample(active, rng.randint(MIN_OBJECTS, MAX_OBJECTS))


def _draw_category_first(rng, active: list, by_cat: dict) -> list:
    k = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
    pools = {c: list(v) for c, v in by_cat.items()}
    out = []
    while len(out) < k:
        c = rng.choice(sorted(c for c, v in pools.items() if v))
        out.append(pools[c].pop(rng.randrange(len(pools[c]))))
    return out


def generate_random(n_scenes: int, props: dict, seed: int, rules_on: bool,
                    category_first: bool = False, max_tries: int = 100_000) -> tuple:
    """-> (scenes, draws). draws/n_scenes is the inverse acceptance rate."""
    rng = random.Random(seed)
    active = sorted(p.id for p in props.values() if not p.retired)
    by_cat: dict = {}
    for oid in active:
        by_cat.setdefault(props[oid].category, []).append(oid)
    draw = _draw_category_first if category_first else _draw_uniform
    out, draws = [], 0
    while len(out) < n_scenes:
        for _ in range(max_tries):
            draws += 1
            objs = draw(rng, active, by_cat)
            md = _md(len(out), objs, rng.sample(_cells(), len(objs)))
            if not rules_on or not _rules.check(md, props):
                out.append(md)
                break
        else:
            raise RuntimeError("rejection sampling did not converge")
    return out, draws


def generate_recommender(n_scenes: int, props: dict, seed: int) -> list:
    out: list = []
    for i in range(n_scenes):
        picked = recommend_detailed(out, props, k=1, seed=seed * 1000 + i,
                                    scene_id=f"S{i:03d}")
        if not picked:
            break
        out.append(picked[0]["md"])
    return out


# --------------------------------------------------------------------- measures
def _entropy(counts: list) -> float:
    n = sum(counts)
    return -sum(v / n * math.log(v / n) for v in counts if v) if n else 0.0


def evenness(counter: Counter, support: set) -> dict:
    """Pielou J (plug-in and Miller-Madow) and Simpson evenness over support."""
    counts = [counter.get(s, 0) for s in support]
    n = sum(counts)
    size = len(support)
    if size <= 1 or n == 0:
        return {"U": 1.0 if size <= 1 else 0.0, "U_mm": 0.0, "E_simpson": 0.0}
    h = _entropy(counts)
    used = sum(1 for v in counts if v)
    h_mm = h + (used - 1) / (2 * n)
    simpson = sum((v / n) ** 2 for v in counts)
    return {"U": h / math.log(size),
            "U_mm": min(1.0, h_mm / math.log(size)),
            "E_simpson": (1.0 / simpson) / size}


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes 6.2)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, df: int) -> float:
    return _gammaincc(df / 2.0, x / 2.0)


def chi_square(cats: list, cells: list) -> dict:
    joint = Counter(zip(cats, cells))
    kc, zc = Counter(cats), Counter(cells)
    n = len(cats)
    x2 = sum((joint.get((k, z), 0) - kc[k] * zc[z] / n) ** 2 / (kc[k] * zc[z] / n)
             for k in kc for z in zc)
    r, c = len(kc), len(zc)
    df = (r - 1) * (c - 1)
    v = math.sqrt(x2 / (n * (min(r, c) - 1))) if min(r, c) > 1 else 0.0
    return {"chi2": x2, "df": df, "p_chi2": chi2_sf(x2, df) if df else 1.0, "V": v}


def coupling(cats: list, cells: list, iters: int, seed: int) -> dict:
    ones = [1] * len(cats)
    obs = mutual_info(cats, cells, ones)
    rng = random.Random(seed)
    sh = list(cells)
    nulls = []
    for _ in range(iters):
        rng.shuffle(sh)
        nulls.append(mutual_info(cats, sh, ones))
    ge = sum(v >= obs - 1e-12 for v in nulls)
    null_mean = sum(nulls) / iters
    return {"I": obs, "I_null": null_mean, "dI": obs - null_mean,
            "p_perm": (ge + 1) / (iters + 1), **chi_square(cats, cells)}


def measure(mds: list, props: dict, iters: int, seed: int) -> dict:
    sigs = [signature(md, props) for md in mds]
    hist = axis_coverage(sigs)
    support = axis_support(props)
    row = {"scenes": len(mds),
           "placements": sum(len(s.placements) for s in sigs),
           "objects_per_scene": statistics.mean(len(md.objects) for md in mds)}
    for ax in AXES:
        for k, v in evenness(hist[ax], support[ax]).items():
            row[f"{k}_{ax}"] = v
    cc_support = {(p.category, p.color) for p in props.values() if not p.retired}
    cc = Counter((t[0], t[1]) for s in sigs for t in s.triples)
    row["U_catcolor"] = evenness(cc, cc_support)["U"]
    cats = [k for s in sigs for k, _ in s.placements]
    cells = [z for s in sigs for _, z in s.placements]
    row.update(coupling(cats, cells, iters, seed))
    keep = [i for i, k in enumerate(cats) if k not in PLACEMENT_RULED]
    nd = coupling([cats[i] for i in keep], [cells[i] for i in keep], iters, seed)
    row.update({f"{k}_nodrawer": nd[k] for k in ("I", "dI", "p_perm", "V")})
    viol = [_violations(md, props) for md in mds]
    row["violate"] = sum(bool(v) for v in viol) / len(mds)
    for name in sorted({n for v in viol for n in v}):
        row[f"violate_{name}"] = sum(name in v for v in viol) / len(mds)
    return row


def _violations(md, props) -> set:
    return {v.split(":", 1)[0] for v in _rules.check(md, props, _FULL_RULES)}


# ------------------------------------------------------------------------- jobs
def run_job(job: tuple) -> dict:
    gen, rules_on, seed, n_scenes, iters = job
    props = props_by_id()
    set_rules(rules_on)
    set_tie_shuffle(seed + 7919 if gen == "recommender_shuffle" else None)
    try:
        draws = None
        if gen.startswith("recommender"):
            mds = generate_recommender(n_scenes, props, seed)
        else:
            mds, draws = generate_random(n_scenes, props, seed, rules_on,
                                         category_first=(gen == "random_cat"))
        row = measure(mds, props, iters, seed)
    finally:
        set_rules(True)
        set_tie_shuffle(None)
    row.update({"generator": gen, "rules": "on" if rules_on else "off",
                "seed": seed, "n": n_scenes,
                "acceptance": (n_scenes / draws) if draws else None})
    row["_placements"] = [(k, z) for md in mds
                          for k, z in signature(md, props).placements]
    return row


# ------------------------------------------------------------------ statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def ks_uniform(ps: list) -> tuple:
    """One-sample KS against U(0,1): (D, asymptotic p)."""
    xs = sorted(ps)
    n = len(xs)
    d = max(max((i + 1) / n - x, x - i / n) for i, x in enumerate(xs))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
                for j in range(1, 101))
    return d, max(0.0, min(1.0, p))


def mann_whitney(a: list, b: list) -> dict:
    """Two-sided Mann-Whitney U (normal approx., tie-corrected) + Cliff's delta."""
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pooled)
    ties = 0.0
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[t] = r
        m = j - i + 1
        ties += m ** 3 - m
        i = j + 1
    n1, n2 = len(a), len(b)
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    z = (u1 - n1 * n2 / 2) / sd if sd else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"U": u1, "z": z, "p": p, "cliffs_delta": 2 * u1 / (n1 * n2) - 1}


def bootstrap_diff(a: list, b: list, iters: int = 5000, seed: int = 0) -> tuple:
    rng = random.Random(seed)
    diffs = sorted(statistics.mean(rng.choices(a, k=len(a)))
                   - statistics.mean(rng.choices(b, k=len(b)))
                   for _ in range(iters))
    return (diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1])


def _pct(v: list, q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def describe(values: list) -> dict:
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p2.5": _pct(values, 0.025), "median": statistics.median(values),
            "p97.5": _pct(values, 0.975)}


def pooled_coupling(placements: list, iters: int, seed: int) -> dict:
    """All sets of one condition pooled -- very high power for systematic coupling."""
    cats = [k for k, _ in placements]
    cells = [z for _, z in placements]
    res = coupling(cats, cells, iters, seed)
    n = len(cats)
    kc, zc = Counter(cats), Counter(cells)
    joint = Counter(placements)
    resid = []
    for k in kc:
        for z in zc:
            e = kc[k] * zc[z] / n
            resid.append((round((joint.get((k, z), 0) - e) / math.sqrt(e), 2),
                          k, z, joint.get((k, z), 0), round(e, 1)))
    resid.sort(key=lambda t: -abs(t[0]))
    per_cat = {k: {f"{z[0]},{z[1]}": joint.get((k, z), 0) for z in sorted(zc)}
               for k in sorted(kc)}
    return {**res, "n": n, "top_residuals": resid[:8], "per_category_cells": per_cat}


METRIC_KEYS = ("objects_per_scene", "U_position", "U_mm_position", "E_simpson_position",
               "U_category", "U_mm_category", "E_simpson_category",
               "U_color", "U_mm_color", "E_simpson_color",
               "U_count", "U_mm_count", "E_simpson_count", "U_catcolor",
               "I", "I_null", "dI", "V", "dI_nodrawer", "V_nodrawer", "violate")
P_KEYS = ("p_perm", "p_chi2", "p_perm_nodrawer")


def summarize(rows: list, iters: int) -> dict:
    out: dict = {"conditions": {}, "comparisons": {}}
    conds = sorted({(r["n"], r["generator"], r["rules"]) for r in rows})
    for cond in conds:
        rs = [r for r in rows if (r["n"], r["generator"], r["rules"]) == cond]
        entry = {"reps": len(rs),
                 "metrics": {k: describe([r[k] for r in rs]) for k in METRIC_KEYS}}
        for k in P_KEYS:
            ps = [r[k] for r in rs]
            hits = sum(p < 0.05 for p in ps)
            d, ksp = ks_uniform(ps)
            entry[k] = {"median": statistics.median(ps), "share_lt_05": hits / len(ps),
                        "wilson95": wilson(hits, len(ps)), "ks_D": d, "ks_p": ksp}
        acc = [r["acceptance"] for r in rs if r["acceptance"] is not None]
        if acc:
            entry["acceptance"] = describe(acc)
        viol_keys = sorted({k for r in rs for k in r if k.startswith("violate_")})
        entry["violate_by_rule"] = {k[8:]: statistics.mean(r.get(k, 0.0) for r in rs)
                                    for k in viol_keys}
        pooled = [pl for r in rs for pl in r["_placements"]]
        entry["pooled"] = pooled_coupling(pooled, min(iters, 1000), 0)
        out["conditions"]["|".join(map(str, cond))] = entry
    pairs = []
    for n in sorted({c[0] for c in conds}):
        for rules in ("on", "off"):
            pairs.append(((n, "recommender", rules), (n, "random", rules)))
            pairs.append(((n, "recommender", rules), (n, "random_cat", rules)))
        for rules in ("on", "off"):
            pairs.append(((n, "recommender_shuffle", rules), (n, "recommender", rules)))
            pairs.append(((n, "recommender_shuffle", rules), (n, "random", rules)))
        for gen in GENERATORS + DIAGNOSTIC_GENERATORS:
            pairs.append(((n, gen, "on"), (n, gen, "off")))
    for a, b in pairs:
        ra = [r for r in rows if (r["n"], r["generator"], r["rules"]) == a]
        rb = [r for r in rows if (r["n"], r["generator"], r["rules"]) == b]
        if not ra or not rb:
            continue
        comp = {}
        for k in METRIC_KEYS:
            va, vb = [r[k] for r in ra], [r[k] for r in rb]
            comp[k] = {"diff": statistics.mean(va) - statistics.mean(vb),
                       "boot95": bootstrap_diff(va, vb), **mann_whitney(va, vb)}
        out["comparisons"][f"{'|'.join(map(str, a))} vs {'|'.join(map(str, b))}"] = comp
    return out


def print_summary(summary: dict) -> None:
    cols = ("objects_per_scene", "U_position", "U_category", "U_color", "U_count",
            "U_catcolor", "dI", "V", "violate")
    print(f"{'condition':26s} " + " ".join(f"{c[:11]:>13s}" for c in cols)
          + f" {'p_perm med/<.05':>17s}")
    for name, e in summary["conditions"].items():
        m = e["metrics"]
        cells = " ".join(f"{m[c]['mean']:6.3f}±{m[c]['sd']:.3f}" for c in cols)
        pp = e["p_perm"]
        print(f"{name:26s} {cells} {pp['median']:8.2f}/{pp['share_lt_05']:4.0%}")


# ----------------------------------------------------------------------- selftest
def _selftest() -> None:
    # statistics against textbook values
    assert abs(chi2_sf(3.841459, 1) - 0.05) < 1e-4
    assert abs(chi2_sf(18.307038, 10) - 0.05) < 1e-4
    mw = mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
    assert mw["cliffs_delta"] == -1.0 and mw["p"] < 0.02, mw
    d, p = ks_uniform([(i + 0.5) / 200 for i in range(200)])
    assert d < 0.01 and p > 0.99, (d, p)
    assert evenness(Counter({1: 5, 2: 5}), {1, 2})["U"] == 1.0
    assert abs(evenness(Counter({1: 10}), {1, 2, 3})["U"]) < 1e-12

    props = props_by_id()
    a, _ = generate_random(8, props, 1, rules_on=False)
    b, _ = generate_random(8, props, 1, rules_on=False)
    assert [m.objects for m in a] == [m.objects for m in b]
    on, draws = generate_random(8, props, 2, rules_on=True)
    assert all(not _rules.check(m, props) for m in on) and draws >= 8
    # the switch reaches the solver: without rules a drawer may sit in a banned zone
    set_rules(False)
    try:
        assert _rules._default_rules() is _NO_RULES
        assert _solver._default_rules() is _NO_RULES
    finally:
        set_rules(True)
    assert _rules._default_rules() is _FULL_RULES
    row = run_job(("recommender", True, 0, 3, 100))
    assert row["scenes"] == 3 and row["violate"] == 0.0, row
    row = run_job(("random", False, 0, 10, 100))
    assert 0 <= row["U_position"] <= 1 and 0 <= row["p_perm"] <= 1
    row = run_job(("recommender_shuffle", True, 0, 3, 100))
    assert row["scenes"] == 3 and row["violate"] == 0.0, row
    assert _selector.enumerate_placements is _ENUMERATE
    print("compare_scene_generators selftest 통과")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path.home() / "libero_datasets")
    ap.add_argument("--scenes", type=int, nargs="*", default=None,
                    help="scene counts per set (default: as many as --root has, else 29)")
    ap.add_argument("--generators", nargs="*", default=list(GENERATORS),
                    choices=GENERATORS + DIAGNOSTIC_GENERATORS)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None,
                    help="directory for rows.csv, summary.json, collected.json")
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
    ns = args.scenes or [len(real) or 29]
    jobs = [(g, on, s, n, args.iters) for n in ns for g in args.generators
            for on in (True, False) for s in range(args.reps)]
    print(f"scene {ns} × {args.reps}회 × {len(args.generators)}생성기 × 규칙 on/off "
          f"· 순열 {args.iters}회 · 작업 {len(jobs)}개\n")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_job, jobs))
    summary = summarize(rows, args.iters)
    print_summary(summary)
    collected = measure(real, props, args.iters, 0) if real else None
    if collected:
        print(f"\ncollected ({len(real)} scenes): " + json.dumps(
            {k: round(v, 3) for k, v in collected.items()
             if isinstance(v, float)}, ensure_ascii=False))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r if not k.startswith("_")})
        with (args.out / "rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: r.get(k) for k in keys} for r in rows)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1, default=str))
        if collected:
            (args.out / "collected.json").write_text(
                json.dumps(collected, ensure_ascii=False, indent=1))
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
