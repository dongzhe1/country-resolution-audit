#!/usr/bin/env python3
"""What changes when the same measure is read at two resolutions."""

from __future__ import annotations

import csv
import random
import statistics
import sys
from pathlib import Path

from facts import emit

def _find_reference():
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference()

CUTS = (0.10, 0.20, 0.25, 0.33)
DRAWS = 2000


def midrank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def gdp_coverage(res: Path, states) -> float:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sample import universe


    unresolved = universe(res)
    for r in csv.DictReader(open(REF / "gtap_regions_mepc82.csv")):
        if r.get("iso3") and r.get("resolution") == "individual":
            unresolved.discard(r["iso3"])
    gdp = {}
    for r in csv.DictReader(open(res / "country_indicators.csv")):
        try:
            gdp[r["country"]] = float(r["gdp_usd"])
        except (ValueError, KeyError):
            continue
    seen = {s["iso"] for s in states}
    num = den = 0.0
    for iso in unresolved:
        g = gdp.get(iso, 0.0)
        if g <= 0:
            continue
        den += g
        if iso in seen:
            num += g
    return num / den if den else float("nan")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    res = Path(sys.argv[1]).expanduser().resolve()

    ind = {r["country"]: r for r in csv.DictReader(open(res / "country_indicators.csv"))}
    states = []
    for r in csv.DictReader(open(res / "aggregate_rows_reconstructed.csv")):
        c = ind.get(r["iso3"])
        if c is None:
            continue
        try:
            gdp, co2 = float(c["gdp_usd"]), float(r["co2_t"])
        except (ValueError, KeyError):
            continue
        if gdp <= 0 or co2 <= 0:
            continue
        states.append({"iso": r["iso3"], "name": c.get("name", r["iso3"]),
                       "co2": co2, "gdp": gdp, "own": co2 / (gdp / 1e6),
                       "adm": [x.strip() for x in r["admissible_rows"].split(";")]})
    n = len(states)
    print(f"unresolved states with exposure and output: {n}")
    print(f"admissible to more than one row: "
          f"{sum(1 for s in states if len(s['adm']) > 1)}\n")

    def group_values(assign):
        tot = {}
        for s in states:
            t = tot.setdefault(assign[s["iso"]], [0.0, 0.0])
            t[0] += s["co2"]; t[1] += s["gdp"]
        return {row: v[0] / (v[1] / 1e6) for row, v in tot.items() if v[1] > 0}

    def compare(assign):
        gv = group_values(assign)
        own = [s["own"] for s in states]
        grp = [gv[assign[s["iso"]]] for s in states]
        ro, rg = midrank([-v for v in own]), midrank([-v for v in grp])
        out = {"shift_median": statistics.median(abs(a - b) for a, b in zip(ro, rg)),
               "shift_max": max(abs(a - b) for a, b in zip(ro, rg))}
        for c in CUTS:
            k = max(1, round(c * n))
            thr_o = sorted(own, reverse=True)[k - 1]
            thr_g = sorted(grp, reverse=True)[k - 1]
            so = {s["iso"] for s, v in zip(states, own) if v >= thr_o}
            sg = {s["iso"] for s, v in zip(states, grp) if v >= thr_g}
            out[c] = (len(so & sg), len(so), len(sg), len(so - sg))
        return out

    ref = {s["iso"]: s["adm"][0] for s in states}
    r0 = compare(ref)
    print("REFERENCE MAPPING: one measure, two resolutions")
    print(f"  rank movement: median {r0['shift_median']:.0f} places of {n}, "
          f"max {r0['shift_max']:.0f}")
    print(f"  {'cut':>6} {'k':>4} {'country':>9} {'group':>7} {'both':>6} {'missed':>7}")
    for c in CUTS:
        both, so, sg, miss = r0[c]
        print(f"  {c:>6.0%} {max(1, round(c*n)):>4} {so:>9} {sg:>7} {both:>6} {miss:>7}")

    k = max(1, round(0.25 * n))
    gv = group_values(ref)
    thr_o = sorted((s["own"] for s in states), reverse=True)[k - 1]
    thr_g = sorted((gv[ref[s["iso"]]] for s in states), reverse=True)[k - 1]
    top_o = [s for s in states if s["own"] >= thr_o]
    grp_sel = {s["iso"] for s in states if gv[ref[s["iso"]]] >= thr_g}
    missed = [s for s in top_o if s["iso"] not in grp_sel]
    print(f"\nin the country-level top quarter, missed by the group reading "
          f"({len(missed)} of {len(top_o)}):")
    for s in sorted(missed, key=lambda x: -x["own"])[:8]:
        print(f"  {s['iso']}  {s['name'][:24]:<24} own {s['own']:7.1f}  "
              f"row {gv[ref[s['iso']]]:7.1f}  {ref[s['iso']]}")

    rng = random.Random(0)
    draws = [compare({s["iso"]: (s["adm"][0] if len(s["adm"]) == 1
                                 else rng.choice(s["adm"])) for s in states})
             for _ in range(DRAWS)]

    def band(key, idx=None):
        v = sorted((d[key] if idx is None else d[key][idx]) for d in draws)
        return v[int(.025 * len(v))], statistics.median(v), v[int(.975 * len(v))]

    print(f"\nACROSS {DRAWS:,} SAMPLED MAPPINGS  (2.5% / median / 97.5%)")
    for label, args in (("median rank movement", ("shift_median", None)),
                        ("missed at the quartile", (0.25, 3)),
                        ("selected by both", (0.25, 0))):
        lo, mid, hi = band(*args)
        print(f"  {label:<26} {lo:.0f} / {mid:.0f} / {hi:.0f}")

    both, so, sg, miss = r0[0.25]


    cov = gdp_coverage(res, states)
    print(f"\nthe row denominators cover {cov:.1%} of the output of the states "
          f"the assessment leaves unresolved")

    emit(res, "consequence", {
        "gdp_coverage_pct": round(100 * cov, 1),
        "states": n, "shift_median": round(r0["shift_median"]),
        "shift_max": round(r0["shift_max"]), "quartile_k": k,
        "country_picks": so, "group_picks": sg, "both_picks": both, "missed": miss,


        "decile_k": max(1, round(0.10 * n)),
        "decile_group_picks": r0[0.10][2],


        "fifth_k": max(1, round(0.20 * n)),
        "fifth_group_picks": r0[0.20][2],
        "fifth_both": r0[0.20][0],
        "missed_lo": round(band(0.25, 3)[0]), "missed_hi": round(band(0.25, 3)[2]),
        "shift_median_lo": round(band("shift_median", None)[0]),
        "shift_median_hi": round(band("shift_median", None)[2]),
    })

    out = res / "decision_consequence.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "name", "row", "own_t_per_musd", "row_t_per_musd",
                    "in_country_top_quarter", "in_group_top_quarter"])
        for s in sorted(states, key=lambda x: -x["own"]):
            w.writerow([s["iso"], s["name"], ref[s["iso"]], f"{s['own']:.3f}",
                        f"{gv[ref[s['iso']]]:.3f}", s["own"] >= thr_o,
                        s["iso"] in grp_sel])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
