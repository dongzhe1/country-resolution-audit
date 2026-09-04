#!/usr/bin/env python3
"""Which states sit inside each of the twelve regional rows -- reconstructed.

    python aggregate_rows.py /path/to/results_dir [--order alt]
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from pathlib import Path

from facts import emit


def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


ROWS = [
    "Rest of American SIDS",
    "Rest of Asia",
    "Rest of Asian SIDS and LDCs",
    "Rest of Caribbean",
    "Rest of Europe",
    "Rest of LDC in Africa",
    "Rest of Middle East and North Africa",
    "Rest of Oceania",
    "Rest of Western Africa",
    "Rest of developing economies in Europe and Central Asia",
    "Rest of landlocked economies in Africa",
    "Rest of the world",
]

MENA_SUB = {"Northern Africa", "Western Asia"}
MIN_FOR_PERCENTILE = 5
DRAWS = 4000
MIN_MEMBERS = 3
EECA_SUB = {"Central Asia", "Eastern Europe"}


REF = _find_reference()


def gtap_units():
    """Map each state to the GTAP 11 region it enters the aggregation as.

    An aggregation can combine GTAP regions but never split one, so the
    members of a composite all land in the same row.  That is what makes
    the set of admissible mappings small enough to enumerate.
    """
    import csv as _csv
    unit = {}
    for r in _csv.DictReader(open(REF / "gtap11_composite_members.csv")):
        if r["iso3"]:
            unit[r["iso3"]] = r["gtap_code"]
    return unit


def composite_names():
    """gtap_code -> the composite's own name, for the rows that pass through."""
    import csv as _csv
    return {r["gtap_code"]: r["gtap_name"]
            for r in _csv.DictReader(open(REF / "gtap11_regions.csv"))
            if r["kind"] == "composite"}


def admissible(r) -> list[str]:
    """Every row whose NAME the state satisfies, not one chosen by priority."""
    reg, sub, inter = r["region"], r["subregion"], r["intermediate_region"]
    sids, ldc, land = r["sids"], r["ldc"], r["landlocked"]
    out = []
    if reg == "Oceania":
        out.append("Rest of Oceania")
    if inter == "Caribbean":
        out.append("Rest of Caribbean")
    if reg == "Americas" and sids:
        out.append("Rest of American SIDS")
    if reg == "Asia" and (sids or ldc):
        out.append("Rest of Asian SIDS and LDCs")
    if reg == "Africa" and ldc:
        out.append("Rest of LDC in Africa")
    if inter == "Western Africa":
        out.append("Rest of Western Africa")
    if reg == "Africa" and land:
        out.append("Rest of landlocked economies in Africa")
    if sub in MENA_SUB:
        out.append("Rest of Middle East and North Africa")
    if reg == "Asia" and sub not in MENA_SUB and sub not in EECA_SUB:
        out.append("Rest of Asia")
    if reg == "Europe" and sub not in EECA_SUB:
        out.append("Rest of Europe")
    if sub in EECA_SUB:
        out.append("Rest of developing economies in Europe and Central Asia")
    return out or ["Rest of the world"]


def dispersion(values):
    """Spread within a row: the ratio of its ninetieth to its tenth percentile."""
    v = sorted(values)
    if v[0] <= 0:
        return float("inf"), "p90/p10"

    def q(p):
        i = p * (len(v) - 1)
        lo = int(i)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (i - lo) * (v[hi] - v[lo])

    return q(0.90) / q(0.10), "p90/p10"


def load(results: Path):
    ref = _find_reference()
    geo = {r["iso3"]: r for r in csv.DictReader(open(ref / "country_regions.csv"))}
    st = {r["iso3"]: r for r in csv.DictReader(open(ref / "country_status.csv"))}
    ind = {r["country"]: r for r in csv.DictReader(open(results / "country_indicators.csv"))}
    out, nogeo = [], []
    from sample import in_sample, m49_codes
    m49 = m49_codes()
    for r in csv.DictReader(open(results / "resolution_gap.csv")):
        if r["resolved"] == "True" or not in_sample(r, m49):
            continue
        iso = r["iso3"]
        try:
            tpm = float(r["t_per_musd"])
        except (TypeError, ValueError):
            continue
        g = geo.get(iso)
        if g is None:
            nogeo.append(iso)
            continue
        u = st.get(iso, {})
        out.append({"iso": iso, "tpm": tpm, "co2": float(r["co2_t"]),
                    "sids": u.get("sids") == "TRUE", "ldc": u.get("ldc") == "TRUE",
                    "landlocked": u.get("landlocked") == "TRUE",
                    "region": g["region"], "subregion": g["subregion"],
                    "intermediate_region": g["intermediate_region"],
                    "name": ind.get(iso, {}).get("name", iso)})
    return out, nogeo


def summarise(members, label):
    print(f"\n{label}")
    print(f"  {'row':<50} {'n':>3} {'median':>8} {'spread':>8} {'kind':>8} "
          f"{'top':>5}")
    spreads = []
    for row in ROWS:
        m = [x for x in members if x["row"] == row]
        if not m:
            continue
        v = [x["tpm"] for x in m]
        sp, kind = dispersion(v)
        co2 = [x["co2"] for x in m]
        top = max(co2) / sum(co2) if sum(co2) else float("nan")
        if len(m) >= MIN_MEMBERS:
            spreads.append(sp)
        print(f"  {row:<50} {len(m):>3} {statistics.median(v):>8.3g} "
              f"{sp:>8.1f} {kind:>8} {top:>4.0%}")
    return spreads


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    results = Path(args[0]).expanduser().resolve()
    members, nogeo = load(results)
    if nogeo:
        print(f"no M49 record, dropped: {nogeo}")
    unit_of = gtap_units()
    comp_name = composite_names()
    for m in members:
        m["unit"] = unit_of.get(m["iso"], m["iso"])
    units = {}
    for m in members:
        units.setdefault(m["unit"], []).append(m)

    for uid, ms in units.items():
        if uid in comp_name and comp_name[uid] in ROWS:
            adm = [comp_name[uid]]
        elif uid in comp_name:
            sets = [set(admissible(m)) for m in ms]
            adm = [r for r in ROWS if all(r in x for x in sets)] or ["Rest of the world"]
        else:
            adm = admissible(ms[0])
        for m in ms:
            m["adm"] = adm

    print(f"unresolved sovereign states with an exposure figure: {len(members)}")
    print(f"GTAP units they belong to: {len(units)} "
          f"({sum(1 for u in units if u in comp_name)} composite blocks, "
          f"{sum(1 for u in units if u not in comp_name)} individual economies)")
    fixed = sum(1 for u, ms in units.items()
                if u in comp_name and comp_name[u] in ROWS for _ in [0])
    print(f"blocks whose row is published by GTAP, not reconstructed: {fixed}")

    amb = [u for u, ms in units.items() if len(ms[0]["adm"]) > 1]
    total = 1
    for u, ms in units.items():
        total *= len(ms[0]["adm"])
    print(f"units admissible to exactly one row: {len(units) - len(amb)}")
    print(f"units admissible to more than one:   {len(amb)}")
    print(f"admissible mappings: {total:,g}"
          + ("  -- enumerated" if total <= DRAWS else f"  -- sampling {DRAWS:,}"))

    for m in members:
        m["row"] = m["adm"][0]
    ref_spreads = summarise(members, "ONE ADMISSIBLE MAPPING (first-listed row)")

    import itertools
    free = [(u, ms) for u, ms in units.items() if len(ms[0]["adm"]) > 1]
    enumerate_all = total <= 200_000
    if enumerate_all:
        combos = itertools.product(*[ms[0]["adm"] for _, ms in free])
        label = f"ACROSS ALL {total:,} ADMISSIBLE MAPPINGS (enumerated)"
    else:
        rng = random.Random(0)
        combos = ([rng.choice(ms[0]["adm"]) for _, ms in free] for _ in range(DRAWS))
        label = f"ACROSS ADMISSIBLE MAPPINGS ({DRAWS:,} draws)"
    for uid, ms in units.items():
        if len(ms[0]["adm"]) == 1:
            for m in ms:
                m["row"] = ms[0]["adm"][0]

    med, mn, nrows = [], [], []
    for combo in combos:
        for (uid, ms), pick in zip(free, combo):
            for m in ms:
                m["row"] = pick
        by = {}
        for m in members:
            by.setdefault(m["row"], []).append(m["tpm"])
        sp = [dispersion(v)[0] for v in by.values() if len(v) >= MIN_MEMBERS]
        if sp:
            med.append(statistics.median(sp))
            mn.append(min(sp))
            nrows.append(len(sp))

    def band(v):
        q = sorted(v)
        return q[int(.025 * len(q))], statistics.median(q), q[int(.975 * len(q))]

    print(f"\n{label}")
    print(f"  {'quantity':<38} {'2.5%':>8} {'median':>8} {'97.5%':>8}")
    for name, v in (("rows with 3+ members", nrows),
                    ("median within-row spread", med),
                    ("narrowest within-row spread", mn)):
        lo, m_, hi = band(v)
        print(f"  {name:<38} {lo:>8.1f} {m_:>8.1f} {hi:>8.1f}")
    print("\n  The claim the write-up can make is the one that holds across this\n"
          "  whole set, not the one a chosen priority order happens to give.")

    n_comp = sum(1 for u in units if u in comp_name)
    n_pub = sum(1 for u in units if u in comp_name and comp_name[u] in ROWS)
    ref_by = {}
    for m in members:
        ref_by.setdefault(m["adm"][0], []).append(m)
    conc = sum(1 for v in ref_by.values()
               if len(v) >= MIN_MEMBERS
               and max(x["co2"] for x in v) / sum(x["co2"] for x in v) > 0.5)
    conc_rows = sum(1 for v in ref_by.values() if len(v) >= MIN_MEMBERS)

    emit(results, "aggregate_rows", {
        "concentrated_rows": conc,
        "spread_ref_median": round(statistics.median(ref_spreads), 1) if ref_spreads else float("nan"),
        "rows_summarised": conc_rows,
        "units": len(units),
        "unit_blocks": n_comp,
        "unit_individual": len(units) - n_comp,
        "blocks_published": n_pub,
        "states": len(members),
        "ambiguous": len(amb),
        "admissible_mappings": (f"${total/10**int(math.log10(total)):.1f}"
                                rf"\times10^{{{int(math.log10(total))}}}$"),
        "draws": len(med),
        "spread_median_lo": round(band(med)[0], 1),
        "spread_median_mid": round(band(med)[1], 1),
        "spread_median_hi": round(band(med)[2], 1),
        "spread_min_lo": round(band(mn)[0], 1),
        "spread_min_mid": round(band(mn)[1], 1),
        "rows_3plus_mid": round(band(nrows)[1]),
    })

    for m in members:
        m["row"] = m["adm"][0]
    out = results / "aggregate_rows_reconstructed.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "name", "reconstructed_row", "n_admissible",
                    "admissible_rows", "sids", "ldc", "t_per_musd", "co2_t"])
        for m in sorted(members, key=lambda x: (x["row"], -x["tpm"])):
            w.writerow([m["iso"], m["name"], m["row"], len(m["adm"]),
                        "; ".join(m["adm"]), m["sids"], m["ldc"],
                        f"{m['tpm']:.3f}", f"{m['co2']:.0f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
