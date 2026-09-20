#!/usr/bin/env python3
"""What country resolution costs a decision about who receives support."""

from __future__ import annotations

import csv
import itertools
import statistics
import sys
from pathlib import Path

from aggregate_rows import MIN_MEMBERS, admissible, gtap_units
from facts import emit

def _find_reference():
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference()
SCHEMES = ("All economies", "Developing economies, LDCs and SIDS", "SIDS and LDCs")
DISBURSEMENT_SCENARIOS = ("26", "31", "32", "46")
YEAR = "gdp_2050"


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def spread_pp(values):
    v = sorted(values)
    n = len(v)

    def q(p):
        i = p * (n - 1)
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)
    return q(0.9) - q(0.1)


def load_states():
    geo = {r["iso3"]: r for r in csv.DictReader(open(REF / "country_regions.csv"))}
    out = {}
    for r in csv.DictReader(open(REF / "country_status.csv")):
        g = geo.get(r["iso3"])
        if g:
            out[r["iso3"]] = {
                "sids": truthy(r.get("sids")), "ldc": truthy(r.get("ldc")),
                "landlocked": truthy(r.get("landlocked")),
                "region": g["region"], "subregion": g["subregion"],
                "intermediate_region": g["intermediate_region"]}
    return out


def load_weights(results: Path, path: Path | None):
    if path is not None:
        w = {}
        for r in csv.DictReader(open(path)):
            iso = r.get("iso3") or r.get("country") or r.get("Region")
            val = r.get("gdp") or r.get("value") or r.get("2050")
            try:
                if iso and val:
                    w[iso] = float(val)
            except ValueError:
                continue
        return w, path.name
    w = {}
    for r in csv.DictReader(open(results / "country_indicators.csv")):
        try:
            g = float(r["gdp_usd"])
        except (ValueError, KeyError):
            continue
        if g > 0:
            w[r["country"]] = g
    return w, "World Bank GDP"


def mappings(isos, states):
    units = gtap_units()
    by_unit = {}
    for i in isos:
        by_unit.setdefault(units.get(i, i), []).append(i)
    fixed, free = {}, []
    for members in by_unit.values():
        adm = admissible(states[members[0]])
        if len(adm) == 1:
            for i in members:
                fixed[i] = adm[0]
        else:
            free.append((members, adm))
    for combo in itertools.product(*[adm for _, adm in free]):
        out = dict(fixed)
        for (members, _), pick in zip(free, combo):
            for i in members:
                out[i] = pick
        yield out


def row_figures(assign, values, weight):
    by = {}
    for iso, row in assign.items():
        if iso in values:
            by.setdefault(row, []).append(iso)
    by = {r: m for r, m in by.items() if len(m) >= MIN_MEMBERS}
    fig = {}
    for row, members in by.items():
        tot = sum(weight(i) for i in members)
        fig[row] = (sum(values[i] * weight(i) for i in members) / tot
                    if tot else statistics.fmean(values[i] for i in members))
    return fig, by


def assess(assign, values, weight):
    fig, by = row_figures(assign, values, weight)
    if len(fig) < 2:
        return None
    between = max(fig.values()) - min(fig.values())
    within = [spread_pp([values[i] for i in m]) for m in by.values()]
    grand = statistics.fmean(values[i] for m in by.values() for i in m)
    ss_tot = sum((values[i] - grand) ** 2 for m in by.values() for i in m)
    ss_between = sum(len(m) * (fig[r] - grand) ** 2 for r, m in by.items())
    return {
        "between": between, "within_median": statistics.median(within),
        "wider": sum(1 for w in within if w > between), "rows": len(fig),
        "members": sum(len(m) for m in by.values()),
        "within_share": 1 - ss_between / ss_tot if ss_tot else float("nan")}


def main():
    args = sys.argv[1:]
    wpath = None
    if "--weights" in args:
        i = args.index("--weights")
        wpath = Path(args[i + 1]).expanduser()
        del args[i:i + 2]
    if len(args) != 1:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir> [--weights FILE]")
    results = Path(args[0]).expanduser().resolve()

    reg = list(csv.DictReader(open(REF / "gtap_regions_mepc82.csv")))
    iso_of = {r["gtap_region"]: r["iso3"] for r in reg
              if r["resolution"] == "individual" and r["iso3"]}
    res_of = {r["gtap_region"]: r["resolution"] for r in reg}
    impact = list(csv.DictReader(open(REF / "gtap_gdp_impact.csv")))
    states = load_states()
    gdp, wname = load_weights(results, wpath)

    isos = sorted(i for i in iso_of.values() if i in states)
    maps = list(mappings(isos, states))
    print(f"economies with a published individual effect and M49 attributes: "
          f"{len(isos)}")
    print(f"admissible mappings onto the row names: {len(maps)} "
          f"(enumerated, not sampled)")
    print(f"weights: {wname}\n")

    weightings = (("GDP-weighted", lambda i: gdp.get(i, 0.0)),
                  ("equal-weighted", lambda i: 1.0))
    scenarios = sorted({r["scenario"] for r in impact
                        if r["disbursement"] == "none"}, key=int)

    summary = {}
    for label, weight in weightings:
        print(f"ALLOCATION INPUT -- {label}\n")
        print(f"  {'scen':>5} {'between':>8} {'within (median, over mappings)':>32} "
              f"{'wider':>8} {'share':>7}")
        btw, wth, wider, share, nrows, nmem = [], [], [], [], [], []
        for scen in scenarios:
            vals = {}
            for r in impact:
                if r["scenario"] == scen and r["disbursement"] == "none":
                    iso = iso_of.get(r["gtap_region"])
                    if iso in states and r[YEAR]:
                        vals[iso] = float(r[YEAR])
            got = [a for a in (assess(m, vals, weight) for m in maps) if a]
            if not got:
                continue
            b = [g["between"] for g in got]
            w = [g["within_median"] for g in got]
            x = [g["wider"] for g in got]
            s = [g["within_share"] for g in got]
            btw += b
            wth += w
            wider += x
            share += s
            nrows += [g["rows"] for g in got]
            nmem += [g["members"] for g in got]
            print(f"  {scen:>5} {statistics.median(b):>8.3f} "
                  f"{statistics.median(w):>19.3f} [{min(w):.3f}-{max(w):.3f}]"
                  f"{statistics.median(x):>5.0f}/{got[0]['rows']:<3}"
                  f"{statistics.median(s):>7.0%}")
        summary[label] = {
            "between": statistics.median(btw), "within": statistics.median(wth),
            "wider": statistics.median(wider), "share": statistics.median(share),
            "rows": statistics.median(nrows),
            "members": statistics.median(nmem)}
        r = summary[label]
        print(f"\n  The median row is internally {r['within']:.3f} pp wide; the "
              f"whole range between row figures is {r['between']:.3f} pp.")
        print(f"  Within-row share of the variance: {r['share']:.0%}\n")


    print("REALISED BENEFIT -- the assessment's own rows\n")
    print(f"  {'scen':>5} {'scheme':<36} {'largest beneficiary':<26} "
          f"{'groups in top 10':>16}")
    in_top, targeted, first_agg, first_names = [], [], 0, set()
    for scen in DISBURSEMENT_SCENARIOS:
        base = {r["gtap_region"]: float(r[YEAR]) for r in impact
                if r["scenario"] == scen and r["disbursement"] == "none" and r[YEAR]}
        for scheme in SCHEMES:
            ben = {r["gtap_region"]: float(r[YEAR]) - base[r["gtap_region"]]
                   for r in impact if r["scenario"] == scen
                   and r["disbursement"] == scheme and r[YEAR]
                   and r["gtap_region"] in base}
            order = sorted(ben.items(), key=lambda kv: -kv[1])
            n_agg = sum(1 for n, _ in order[:10] if res_of.get(n) == "aggregate")
            in_top.append(n_agg)
            if scheme == "SIDS and LDCs":
                targeted.append(n_agg)
            first_agg += res_of.get(order[0][0]) == "aggregate"
            first_names.add(order[0][0])
            print(f"  {scen:>5} {scheme:<36} {order[0][0][:24]:<26} {n_agg:>13}/10")
    print(f"\n  A group row is the largest single beneficiary in "
          f"{first_agg}/{len(in_top)} scenario-scheme combinations.")
    print(f"  Group rows are 12 of 111 rows but take a median of "
          f"{statistics.median(in_top):.0f} of the ten largest benefits.")


    ref = maps[0]
    vals21 = {}
    for r in impact:
        if r["scenario"] == "21" and r["disbursement"] == "none":
            iso = iso_of.get(r["gtap_region"])
            if iso in states and r[YEAR]:
                vals21[iso] = float(r[YEAR])
    fig21, by21 = row_figures(ref, vals21, lambda i: gdp.get(i, 0.0))
    name_of = {v: k for k, v in iso_of.items()}
    with open(results / "disbursement_rows.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "iso3", "economy", "gdp_effect_2050",
                    "row_figure", "members"])
        for row, ms in sorted(by21.items(), key=lambda kv: -len(kv[1])):
            for i in sorted(ms, key=lambda x: vals21[x]):
                w.writerow([row, i, name_of.get(i, i), f"{vals21[i]:.3f}",
                            f"{fig21[row]:.3f}", len(ms)])
    with open(results / "disbursement_benefit.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "scheme", "row", "resolution", "benefit_2050"])
        for scen in DISBURSEMENT_SCENARIOS:
            base = {r["gtap_region"]: float(r[YEAR]) for r in impact
                    if r["scenario"] == scen and r["disbursement"] == "none"
                    and r[YEAR]}
            for scheme in SCHEMES:
                for r in impact:
                    if (r["scenario"] == scen and r["disbursement"] == scheme
                            and r[YEAR] and r["gtap_region"] in base):
                        w.writerow([scen, scheme, r["gtap_region"],
                                    res_of.get(r["gtap_region"], ""),
                                    f"{float(r[YEAR]) - base[r['gtap_region']]:.3f}"])
    print(f"\nwrote disbursement_rows.csv, disbursement_benefit.csv")

    g, e = summary["GDP-weighted"], summary["equal-weighted"]
    emit(results, "disbursement_resolution", {
        "economies": len(isos), "mappings": len(maps),
        "scenarios": len(scenarios), "weights": wname,


        "between_pp": f"{g['between']:.3f}",
        "within_pp": f"{g['within']:.3f}",
        "rows": int(g["rows"]),
        "rows_wider": int(g["wider"]),
        "within_share_pct": round(100 * g["share"]),
        "within_share_equal_pct": round(100 * e["share"]),
        "benefit_first_is_group": first_agg,


        "benefit_first_row": (sorted(first_names)[0]
                              if len(first_names) == 1 else ""),
        "benefit_groups_in_top_ten": int(statistics.median(in_top)),
        "benefit_groups_in_top_ten_targeted": int(statistics.median(targeted)),


        "min_members": MIN_MEMBERS,
        "economies_in_rows": int(g["members"]),
        "benefit_scenarios": len(DISBURSEMENT_SCENARIOS),
        "benefit_combinations": len(in_top),
    })


if __name__ == "__main__":
    main()
