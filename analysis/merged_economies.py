#!/usr/bin/env python3
"""Which economies did this study merge that its own database keeps separate?

    python merged_economies.py <results_dir>
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from facts import emit
from sample import m49_codes, territories


def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference()


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "t", "yes")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    res = Path(sys.argv[1]).expanduser().resolve()

    gtap = list(csv.DictReader(open(REF / "gtap11_regions.csv")))
    individual = {r["gtap_code"] for r in gtap if r["kind"] == "individual"}
    composite = {r["gtap_code"] for r in gtap if r["kind"] == "composite"}
    if len(gtap) != 160 or len(individual) != 141 or len(composite) != 19:
        sys.exit(f"gtap11_regions.csv has {len(individual)}+{len(composite)}; "
                 "the published Data Base has 141+19")

    kept = {r["iso3"] for r in csv.DictReader(open(REF / "gtap_regions_mepc82.csv"))
            if r["resolution"] == "individual" and r["iso3"]}
    stray = kept - individual
    if stray:
        sys.exit(f"resolved rows not in the GTAP individual set: {sorted(stray)}")

    merged = individual - kept
    print(f"GTAP 11 individual economies: {len(individual)}")
    print(f"  kept as their own row:      {len(kept)}")
    print(f"  merged into a group:        {len(merged)}")

    status = {r["iso3"]: r for r in csv.DictReader(open(REF / "country_status.csv"))}
    m49, terr = m49_codes(), territories(res)
    sovereign = {c for c in merged if c in m49 and c not in terr}
    other = sorted(merged - sovereign)

    ldc = sorted(c for c in sovereign if truthy(status[c]["ldc"]))
    sids = sorted(c for c in sovereign if truthy(status[c]["sids"]))
    land = sorted(c for c in sovereign if truthy(status[c]["landlocked"]))

    print(f"\nOf the {len(merged)} merged, {len(sovereign)} are sovereign states "
          f"on the UN M49 list.")
    print(f"  least developed:   {len(ldc):>2}   {' '.join(ldc)}")
    print(f"  small island:      {len(sids):>2}   {' '.join(sids) or '--'}")
    print(f"  landlocked:        {len(land):>2}   {' '.join(land)}")
    if other:
        print(f"  not sovereign states on that list, reported separately: "
              f"{' '.join(other)}")

    kept_sov = {c for c in kept if c in m49 and c not in terr}
    for lab, key in (("least developed", "ldc"), ("small island", "sids")):
        m = sum(1 for c in sovereign if truthy(status[c][key]))
        k = sum(1 for c in kept_sov if truthy(status[c][key]))
        tot = m + k
        if tot:
            print(f"\n  {lab}: {m} of {tot} such economies GTAP represents "
                  f"individually were merged ({100*m/tot:.0f}%)")
    allm = len(sovereign)
    alltot = allm + len(kept_sov)
    print(f"  all sovereign: {allm} of {alltot} were merged "
          f"({100*allm/alltot:.0f}%)")

    with open(res / "merged_economies.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gtap_code", "name", "sovereign_m49", "ldc", "sids",
                    "landlocked"])
        names = {r["gtap_code"]: r["gtap_name"] for r in gtap}
        for c in sorted(merged):
            s = status.get(c, {})
            w.writerow([c, names[c], c in sovereign,
                        truthy(s.get("ldc")), truthy(s.get("sids")),
                        truthy(s.get("landlocked"))])
    print(f"\nwrote {res / 'merged_economies.csv'}")

    emit(res, "merged", {
        "gtap_individual": len(individual),
        "gtap_composite": len(composite),
        "kept": len(kept),
        "merged": len(merged),
        "merged_sovereign": len(sovereign),
        "merged_ldc": len(ldc),
        "merged_sids": len(sids),
        "merged_landlocked": len(land),
        "merged_share_pct": round(100 * allm / alltot),
        "ldc_merged_share_pct": round(
            100 * len(ldc) / max(len(ldc) + sum(
                1 for c in kept_sov if truthy(status[c]["ldc"])), 1)),
    })


if __name__ == "__main__":
    main()
