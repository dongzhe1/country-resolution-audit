#!/usr/bin/env python3
"""Fetch the public inputs this analysis needs, and say where the rest comes from.

    python download_data.py [--out data]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

THETIS = "https://mrv.emsa.europa.eu/#public/emission-report"
WDI = "https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=400&date={year}"
M49 = "https://unstats.un.org/unsd/methodology/m49/overview/"

INDICATORS = {"NY.GDP.MKTP.CD": "gdp_usd", "SP.POP.TOTL": "population",
              "NY.GDP.MKTP.PP.CD": "gdp_ppp_intl"}


def fetch(url: str, dest: Path, label: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())
        print(f"  {label:<28} {dest} ({dest.stat().st_size / 1024:.0f} kB)")
        return True
    except Exception as e:
        print(f"  {label:<28} FAILED: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--year", default=2025, type=int,
                    help="WDI reference year (the write-up used 2025)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    print("World Bank indicators (public API)")
    ok = 0
    for ind, name in INDICATORS.items():
        ok += fetch(WDI.format(ind=ind, year=a.year),
                    a.out / f"wdi_{name}_{a.year}.json", name)

    print("\nNot fetched automatically:")
    print(f"  AIS port calls      pipeline/pull_port_visits.py, GFW_TOKEN required")
    print(f"                      registration: https://globalfishingwatch.org/our-apis/")
    print(f"  EU MRV returns      {THETIS}")
    print(f"                      save the annual CSVs under <out>/mrv/")
    print(f"  Vessel particulars  commercial licence; schema in the README")
    print(f"  UN M49 list         {M49}")
    print(f"                      already in reference/country_status.csv")

    print(f"\n{ok}/{len(INDICATORS)} public files fetched. To run the code without "
          f"any of this:\n  python generate_fake_data.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
