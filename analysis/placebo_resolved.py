#!/usr/bin/env python3
"""Does the assessment's OWN output vary within the groups it publishes?

    python placebo_resolved.py <results_dir>
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

from aggregate_rows import MIN_MEMBERS, admissible
from facts import emit


def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference()
SCENARIO = "21"
MIN_FOR_ROW = MIN_MEMBERS


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "t", "yes")


def load_states():
    """iso3 -> the attributes admissible() needs, for M49 states only."""
    reg = {r["iso3"]: r for r in csv.DictReader(open(REF / "country_regions.csv"))}
    out = {}
    for r in csv.DictReader(open(REF / "country_status.csv")):
        iso = r["iso3"]
        if not iso or iso not in reg:
            continue
        out[iso] = {
            "region": reg[iso]["region"],
            "subregion": reg[iso]["subregion"],
            "intermediate_region": reg[iso]["intermediate_region"],
            "sids": truthy(r["sids"]),
            "ldc": truthy(r["ldc"]),
            "landlocked": truthy(r["landlocked"]),
        }
    return out


def spread_pp(values):
    """Ninetieth minus tenth percentile, linearly interpolated, in points."""
    v = sorted(values)

    def q(p):
        i = p * (len(v) - 1)
        lo = int(i)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (i - lo) * (v[hi] - v[lo])

    return q(0.90) - q(0.10)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    res = Path(sys.argv[1]).expanduser().resolve()

    iso_of = {r["gtap_region"]: r["iso3"]
              for r in csv.DictReader(open(REF / "gtap_regions_mepc82.csv"))
              if r["resolution"] == "individual" and r["iso3"]}
    effects = {}
    for r in csv.DictReader(open(REF / "gtap_gdp_impact.csv")):
        if r["scenario"] != SCENARIO or r["resolution"] != "individual":
            continue
        iso = iso_of.get(r["gtap_region"])
        if iso and r["gdp_2050"]:
            effects[iso] = float(r["gdp_2050"])

    states = load_states()
    missing = sorted(set(effects) - set(states))
    print(f"individually resolved economies with a {SCENARIO} GDP effect: "
          f"{len(effects)}")
    if missing:
        print(f"  no M49 attributes, dropped: {', '.join(missing)}")

    by = {}
    for iso, eff in effects.items():
        if iso not in states:
            continue
        by.setdefault(admissible(states[iso])[0], []).append((iso, eff))

    print(f"\nWITHIN-GROUP SPREAD OF THE ASSESSMENT'S OWN 2050 GDP EFFECT")
    print(f"  scenario {SCENARIO}; groups of {MIN_FOR_ROW}+ resolved economies")
    print(f"  {'group':<52} {'n':>3} {'median':>8} {'p90-p10':>9} "
          f"{'min':>7} {'max':>7}")
    spreads, rows = [], []
    for row, members in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(members) < MIN_FOR_ROW:
            continue
        vals = [e for _, e in members]
        sp = spread_pp(vals)
        spreads.append(sp)
        worst = min(members, key=lambda m: m[1])
        mildest = max(members, key=lambda m: m[1])
        rows.append((row, len(members), statistics.median(vals), sp,
                     worst, mildest))
        print(f"  {row:<52} {len(members):>3} {statistics.median(vals):>8.3f} "
              f"{sp:>9.3f} {worst[1]:>7.2f} {mildest[1]:>7.2f}")

    if not spreads:
        sys.exit("no group reached the minimum size")

    med = statistics.median(spreads)
    print(f"\n  median group spans {med:.2f} percentage points between its "
          f"tenth and ninetieth percentile member")
    print(f"  widest {max(spreads):.2f} pp, narrowest {min(spreads):.2f} pp")

    widest = max(rows, key=lambda r: r[3])
    print(f"\n  widest group is {widest[0]!r}: "
          f"{widest[4][0]} at {widest[4][1]:+.2f}% and "
          f"{widest[5][0]} at {widest[5][1]:+.2f}%")
    signs = sum(1 for r in rows if r[4][1] * r[5][1] < 0)
    if signs:
        print(f"  {signs} of {len(rows)} groups contain members whose "
              f"published effects differ in SIGN")

    between = spread_pp([r[2] for r in rows])
    RESIDUAL = "Rest of the world"
    sub = [r for r in rows if r[0] != RESIDUAL]
    if sub:
        w2 = statistics.median([r[3] for r in sub])
        b2 = spread_pp([r[2] for r in sub])
        print(f"\n  excluding the residual group ({RESIDUAL}):")
        print(f"    within {w2:.2f} pp   between {b2:.2f} pp   ratio {w2/b2:.2f}"
              f"   ({len(sub)} groups)")

    print(f"\n  spread ACROSS group medians:  {between:.2f} pp")
    print(f"  spread WITHIN the median group: {med:.2f} pp  "
          f"({med / between:.2f}x the between-group spread)")

    print(f"\n  {len(rows)} groups, {sum(r[1] for r in rows)} resolved "
          f"economies placed")

    with open(res / "placebo_resolved.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n", "median", "spread_pp",
                    "worst_iso", "worst_pct", "mildest_iso", "mildest_pct"])
        for row, n, med_v, sp, worst, mildest in rows:
            w.writerow([row, n, f"{med_v:.4f}", f"{sp:.4f}",
                        worst[0], f"{worst[1]:.4f}", mildest[0], f"{mildest[1]:.4f}"])
    print(f"  -> placebo_resolved.csv ({len(rows)} rows)")

    emit(res, "placebo_resolved", {
        "scenario": int(SCENARIO),
        "n_economies": sum(r[1] for r in rows),
        "n_groups": len(rows),
        "spread_median_pp": f"{med:.2f}",
        "spread_max_pp": round(max(spreads), 2),
        "spread_min_pp": round(min(spreads), 2),
        "widest_group": widest[0],
        "widest_worst_iso": widest[4][0],
        "widest_worst_pct": round(widest[4][1], 2),
        "widest_mildest_iso": widest[5][0],
        "widest_mildest_pct": round(widest[5][1], 2),
        "spread_between_pp": round(between, 2),
        "within_over_between": round(med / between, 2),
        "within_over_between_no_residual": round(w2 / b2, 2) if sub else float("nan"),
        "groups_sign_split": signs,
    })


if __name__ == "__main__":
    main()
