#!/usr/bin/env python3
"""Exposure year by year, and how complete the record is for each state.

    python state_temporal_coverage.py /path/to/gfw_port_visits
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from state_exposure import (ATTRIBUTION_SPLIT, estimate_co2, find_seaweb,
                            read_particulars)

GAP_DAYS = 180
MIN_VOYAGES_FOR_GAP = 4


def per_year(df: pd.DataFrame) -> pd.DataFrame:
    """Country x year attributed CO2, on the same 50/50 split as the main table."""
    df = df.copy()
    df["year"] = pd.to_datetime(df["dep_time"], errors="coerce").dt.year
    df = df[df["year"].notna()]
    df["year"] = df["year"].astype(int)

    dep = df[["dep_country", "year", "co2_t"]].rename(columns={"dep_country": "country"})
    dep["co2_t"] = dep["co2_t"] * ATTRIBUTION_SPLIT
    arr = df[["arr_country", "year", "co2_t"]].rename(columns={"arr_country": "country"})
    arr["co2_t"] = arr["co2_t"] * (1 - ATTRIBUTION_SPLIT)
    both = pd.concat([dep, arr], ignore_index=True).dropna(subset=["country"])
    out = (both.groupby(["country", "year"])["co2_t"].sum()
                .reset_index().sort_values(["country", "year"]))
    return out


def gappy_vessels(df: pd.DataFrame) -> pd.Series:
    """IMOs whose observed history contains an interior silence over GAP_DAYS."""
    d = df[["imo", "dep_time"]].copy()
    d["dep_time"] = pd.to_datetime(d["dep_time"], errors="coerce")
    d = d.dropna(subset=["dep_time"]).sort_values(["imo", "dep_time"])
    d["gap"] = d.groupby("imo")["dep_time"].diff().dt.total_seconds() / 86400.0
    n = d.groupby("imo")["dep_time"].size()
    longest = d.groupby("imo")["gap"].max()
    eligible = n[n >= MIN_VOYAGES_FOR_GAP].index
    return (longest.reindex(eligible) > GAP_DAYS).rename("gappy")


def coverage(df: pd.DataFrame, gappy: pd.Series) -> pd.DataFrame:
    d = df.copy()
    d["gappy"] = d["imo"].map(gappy).fillna(False)
    dep = d[["dep_country", "imo", "co2_t", "gappy"]].rename(columns={"dep_country": "country"})
    dep["co2_t"] = dep["co2_t"] * ATTRIBUTION_SPLIT
    arr = d[["arr_country", "imo", "co2_t", "gappy"]].rename(columns={"arr_country": "country"})
    arr["co2_t"] = arr["co2_t"] * (1 - ATTRIBUTION_SPLIT)
    both = pd.concat([dep, arr], ignore_index=True).dropna(subset=["country"])

    g = both.groupby("country")
    out = pd.DataFrame({
        "co2_t": g["co2_t"].sum(),
        "vessels": g["imo"].nunique(),
        "co2_from_gappy_t": both[both["gappy"]].groupby("country")["co2_t"].sum(),
        "gappy_vessels": both[both["gappy"]].groupby("country")["imo"].nunique(),
    }).fillna(0.0)
    out["gappy_co2_share"] = out["co2_from_gappy_t"] / out["co2_t"].replace(0, np.nan)
    out["gappy_vessel_share"] = out["gappy_vessels"] / out["vessels"].replace(0, np.nan)
    return out.sort_values("co2_t", ascending=False)


def parse_args(argv, name):
    voy, rest, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--voyages":
            voy, i = argv[i + 1], i + 2
        elif a.startswith("--voyages="):
            voy, i = a.split("=", 1)[1], i + 1
        else:
            rest.append(a); i += 1
    if len(rest) != 1 or voy is None:
        sys.exit(f"usage: {name} <gfw_port_visits_dir> --voyages NAME\n"
                 f"  --voyages is required and deliberately has no default; "
                 f"the write-up uses voyages_routed.csv.gz.\n"
                 f"  See the docstring for what the four tables are and why "
                 f"they are not interchangeable.")
    return Path(rest[0]).expanduser().resolve(), voy


def main():
    work, voy_name = parse_args(sys.argv[1:], Path(sys.argv[0]).name)

    path = work / voy_name
    if not path.exists():
        sys.exit(f"{path} does not exist")
    voy = pd.read_csv(path, dtype={"imo": str})
    print(f"voyage table: {voy_name}")
    print(f"voyages: {len(voy):,}  vessels: {voy['imo'].nunique():,}")
    sw = read_particulars(find_seaweb(work))
    df = estimate_co2(voy, sw, work, voy_name)

    yr = per_year(df)
    yr.to_csv(work / "state_exposure_by_year.csv", index=False)
    print(f"\nwrote state_exposure_by_year.csv  "
          f"({yr['country'].nunique()} states x {yr['year'].nunique()} years)")
    tot = yr.groupby("year")["co2_t"].sum() / 1e6
    print("  global attributed Mt CO2 by year:")
    for y, v in tot.items():
        print(f"    {y}  {v:8,.0f}")
    print("  the last year is partial; treat it separately, do not annualise it")

    gappy = gappy_vessels(df)
    print(f"\nvessels with >= {MIN_VOYAGES_FOR_GAP} voyages: {len(gappy):,}   "
          f"of which gappy (silent > {GAP_DAYS}d mid-history): "
          f"{int(gappy.sum()):,} ({gappy.mean():.1%})")

    cov = coverage(df, gappy)
    cov.to_csv(work / "state_coverage.csv")
    print(f"wrote state_coverage.csv  ({len(cov)} states)")
    print("\n  states whose emissions lean most on gappy vessels "
          "(exposure most likely understated):")
    top = cov[cov["co2_t"] > cov["co2_t"].quantile(0.25)]
    for c, r in top.nlargest(10, "gappy_co2_share").iterrows():
        print(f"    {c}  {r['gappy_co2_share']:.1%} of CO2 from gappy ships, "
              f"{int(r['vessels']):,} vessels")
    print(f"\n  median across states: {cov['gappy_co2_share'].median():.1%}")
    print("  Join this to resolution_gap.csv and test whether the share is "
          "higher\n  for unresolved, SIDS or LDC states. If it is, the write-up "
          "must say so.")


if __name__ == "__main__":
    main()
