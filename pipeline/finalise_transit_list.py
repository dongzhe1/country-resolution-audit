#!/usr/bin/env python3
"""Combine the automatic classification with the recorded human decisions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _find_reference():
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference() / "transit_decisions.csv"


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <work_dir>")
    work = Path(sys.argv[1]).expanduser().resolve()

    ev_path = work / "transit_evidence.csv"
    if not ev_path.exists():
        sys.exit(f"missing {ev_path} -- run transit_evidence.py first")
    ev = pd.read_csv(ev_path, dtype={"anchorage_id": str})
    if not REF.exists():
        sys.exit(f"missing {REF}")
    dec = pd.read_csv(REF)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transit_evidence import classify, MIN_VISITS

    big = ev[ev["visits"] >= MIN_VISITS].copy()
    big = big[big.get("flagged", True).astype(bool)] if "flagged" in big else big
    big["verdict"] = big.apply(classify, axis=1)

    uncertain = big[big["verdict"] == "UNCERTAIN"]
    key = dict(zip(dec["name"].str.strip().str.upper(), dec["decision"]))
    why = dict(zip(dec["name"].str.strip().str.upper(), dec["reason"]))
    missing = [n for n in uncertain["name"].fillna("").str.strip().str.upper()
               if n not in key]
    if missing:
        sys.exit("UNCERTAIN anchorages with no recorded decision:\n  "
                 + "\n  ".join(missing)
                 + f"\nAdd them to {REF} with a reason before proceeding.")

    unc_names = uncertain["name"].fillna("").str.strip().str.upper()
    uncertain = uncertain.assign(decision=[key[n] for n in unc_names])

    auto_drop = big[big["verdict"].isin(("WAIT", "THROUGH"))]
    manual_drop = uncertain[uncertain["decision"] == "drop"]
    kept = pd.concat([big[big["verdict"] == "DEST"],
                      uncertain[uncertain["decision"] == "keep"]])

    small = ev[(ev["visits"] < MIN_VISITS)]
    if "flagged" in small:
        small = small[small["flagged"].astype(bool)]

    ids = set(auto_drop["anchorage_id"].dropna().astype(str))
    ids |= set(manual_drop["anchorage_id"].dropna().astype(str))
    ids |= set(small["anchorage_id"].dropna().astype(str))

    out = work / "transit_anchorage_ids_final.txt"
    out.write_text("\n".join(sorted(ids)) + "\n")

    total = int(ev["visits"].sum())
    print(f"excluded anchorages: {len(ids):,}")
    print(f"  on evidence (WAIT/THROUGH)  {len(auto_drop):>4}  "
          f"{int(auto_drop['visits'].sum()):>9,} visits")
    print(f"  by recorded decision        {len(manual_drop):>4}  "
          f"{int(manual_drop['visits'].sum()):>9,} visits")
    print(f"  below the test threshold    {len(small):>4}  "
          f"{int(small['visits'].sum()):>9,} visits")
    dropped = (auto_drop['visits'].sum() + manual_drop['visits'].sum()
               + small['visits'].sum())
    print(f"  total                             {int(dropped):>9,} visits "
          f"({100*dropped/total:.2f}% of all)")
    print(f"\nretained despite the name flag: {len(kept):,} anchorages, "
          f"{int(kept['visits'].sum()):,} visits")

    print(f"\n--- methods table: judgement calls ---")
    for _, r in uncertain.sort_values("visits", ascending=False).iterrows():
        print(f"\n  {r['name']} ({r['country']}, {int(r['visits']):,} visits) "
              f"-> {r['decision'].upper()}")
        print(f"    {why[str(r['name']).strip().upper()]}")

    print(f"\nwrote {out}")
    print(f"\nnext: python build_voyages.py {work} --drop-anchorages {out} "
          f"--out voyages_final.csv.gz"
          f"\n      python compare_transit.py {work} voyages.csv.gz "
          f"voyages_final.csv.gz")


if __name__ == "__main__":
    main()
