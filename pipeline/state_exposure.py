#!/usr/bin/env python3
"""State-level exposure to a per-tonne-CO2 maritime measure."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SFOC_G_PER_KWH = 190.0
CO2_PER_FUEL_T = 3.114
AUX_FRACTION_AT_SEA = 0.05
MIN_LOAD, MAX_LOAD = 0.05, 0.90
DEFAULT_SERVICE_SPEED_KN = 14.0
ATTRIBUTION_SPLIT = 0.5
CARBON_PRICE_USD_PER_T = 100.0

CIA_CASE_STUDIES = ["ARG", "BLZ", "BRA", "CHL", "COK",
                    "PER", "ZAF", "TGO", "TON", "USA"]

SEAWEB_COLS = ["lrnoimo_ship_no", "gross_tonnage", "deadweight",
               "shiptype_group", "year_of_build", "flag_name",
               "total_kilowattsof_main_engines", "speedservice"]


def env_path(var: str):
    v = os.environ.get(var, "").strip()
    if not v:
        return None
    p = Path(v).expanduser().resolve()
    if not p.exists():
        sys.exit(f"{var}={v} does not exist")
    return p


def load_calibration(work: Path, sw_types: pd.Series,
                     voy_name: str, strict: bool = True) -> pd.Series:
    path = work / "type_calibration.csv"
    if not path.exists():
        print("no type_calibration.csv -- running UNCALIBRATED")
        return pd.Series(1.0, index=sw_types.index)
    cal = pd.read_csv(path)
    if "fitted_on" in cal.columns:
        on = str(cal["fitted_on"].iloc[0])
        if on != voy_name and strict:
            sys.exit(f"type_calibration.csv was fitted on {on} but this run "
                     f"uses {voy_name}.\n  Refit with "
                     f"calibrate_types.py --voyages {voy_name}, or pass the "
                     f"matching table.")
        if on != voy_name:
            print(f"  calibration fitted on {on}, applied to {voy_name} "
                  f"-- deliberate, this run compares tables")
    else:
        print("  type_calibration.csv does not say which voyage table it was "
              "fitted on -- refit to record it")
    glob = cal.loc[cal["shiptype_group"] == "__global__", "factor"]
    glob = float(glob.iloc[0]) if len(glob) else 1.0
    fmap = dict(zip(cal["shiptype_group"], cal["factor"]))
    fmap.pop("__global__", None)
    f = sw_types.map(fmap)
    n_typed = int(f.notna().sum())
    f = f.fillna(glob)
    print(f"calibration: {len(fmap)} types, {n_typed:,}/{len(f):,} voyages "
          f"({n_typed/max(len(f),1):.0%}) matched a type factor; "
          f"rest use global {glob:.2f}")
    return f


def find_seaweb(work: Path, explicit: Path | None = None) -> Path:
    sw_path = explicit or env_path("SEAWEB_PATH")
    if sw_path is None:
        for c in (work.parent / "seaweb" / "ship_info.csv",
                  work / "ship_info.csv",
                  work.parent / "ship_info.csv"):
            if c.exists():
                sw_path = c
                break
    if sw_path is None:
        sys.exit("ship_info.csv not found. Pass --seaweb /path/to/ship_info.csv\n"
                 "  (or set SEAWEB_PATH, which --seaweb overrides).")
    return sw_path


def parse_args(argv, name):
    voy, out, seaweb, rest, i = "voyages.csv.gz", "state_exposure.csv", None, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--voyages":
            voy, i = argv[i + 1], i + 2
        elif a.startswith("--voyages="):
            voy, i = a.split("=", 1)[1], i + 1
        elif a == "--out":
            out, i = argv[i + 1], i + 2
        elif a.startswith("--out="):
            out, i = a.split("=", 1)[1], i + 1
        elif a == "--seaweb":
            seaweb, i = Path(argv[i + 1]).expanduser().resolve(), i + 2
        elif a.startswith("--seaweb="):
            seaweb, i = Path(a.split("=", 1)[1]).expanduser().resolve(), i + 1
        else:
            rest.append(a); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <gfw_port_visits_dir> "
                 f"[--voyages NAME] [--out NAME] [--seaweb PATH]")
    return Path(rest[0]).expanduser().resolve(), voy, out, seaweb


def estimate_co2(voy, sw, work, voy_name):
    df = voy.merge(sw, on="imo", how="left")
    matched = df["me_kw"].notna().sum()
    print(f"matched to particulars: {matched:,}/{len(df):,} voyages "
          f"({100*matched/len(df):.1f}%)")

    svc = df["service_kn"].where(df["service_kn"].between(5, 30),
                                 DEFAULT_SERVICE_SPEED_KN)
    load = (df["implied_speed_kn"] / svc) ** 3
    load = load.clip(MIN_LOAD, MAX_LOAD)
    p_used = df["me_kw"] * load
    p_aux = df["me_kw"] * AUX_FRACTION_AT_SEA
    df["fuel_t"] = (p_used + p_aux) * df["sea_hours"] * SFOC_G_PER_KWH / 1e6
    df["cal"] = load_calibration(work, df["shiptype_group"],
                                 voy_name).to_numpy()
    df["fuel_t"] = df["fuel_t"] * df["cal"]
    df["co2_t"] = df["fuel_t"] * CO2_PER_FUEL_T
    df["cost_usd"] = df["co2_t"] * CARBON_PRICE_USD_PER_T
    df = df[df["co2_t"].notna() & (df["co2_t"] > 0)]
    print(f"voyages with CO2 estimate: {len(df):,}  "
          f"total {df['co2_t'].sum()/1e6:,.1f} Mt CO2")
    return df


def read_particulars(sw_path):
    sw = pd.read_csv(sw_path, usecols=SEAWEB_COLS,
                     dtype={"lrnoimo_ship_no": str}, low_memory=False)
    sw = sw.rename(columns={"lrnoimo_ship_no": "imo",
                            "total_kilowattsof_main_engines": "me_kw",
                            "speedservice": "service_kn"})
    for c in ("gross_tonnage", "deadweight", "me_kw", "service_kn", "year_of_build"):
        sw[c] = pd.to_numeric(sw[c], errors="coerce")
    return sw


def main():
    work, voy_name, out_name, seaweb = parse_args(sys.argv[1:], Path(sys.argv[0]).name)

    voy = pd.read_csv(work / voy_name, dtype={"imo": str})
    print(f"voyage table: {voy_name}")
    print(f"voyages: {len(voy):,}  vessels: {voy['imo'].nunique():,}")

    sw_path = find_seaweb(work, seaweb)
    print(f"vessel particulars: {sw_path}")
    sw = read_particulars(sw_path)

    df = estimate_co2(voy, sw, work, voy_name)

    dep = df[["dep_country", "co2_t", "cost_usd", "distance_nm"]].copy()
    dep.columns = ["country", "co2_t", "cost_usd", "distance_nm"]
    dep[["co2_t", "cost_usd"]] *= ATTRIBUTION_SPLIT
    dep["voyages"] = ATTRIBUTION_SPLIT

    arr = df[["arr_country", "co2_t", "cost_usd", "distance_nm"]].copy()
    arr.columns = ["country", "co2_t", "cost_usd", "distance_nm"]
    arr[["co2_t", "cost_usd"]] *= (1 - ATTRIBUTION_SPLIT)
    arr["voyages"] = 1 - ATTRIBUTION_SPLIT

    both = pd.concat([dep, arr], ignore_index=True).dropna(subset=["country"])
    agg = both.groupby("country").agg(
        co2_t=("co2_t", "sum"),
        cost_usd=("cost_usd", "sum"),
        voyages=("voyages", "sum"),
        mean_distance_nm=("distance_nm", "mean"),
    ).sort_values("co2_t", ascending=False)
    agg["co2_share_pct"] = 100 * agg["co2_t"] / agg["co2_t"].sum()
    agg["rank"] = range(1, len(agg) + 1)

    ind = work / "country_indicators.csv"
    if ind.exists():
        ci = pd.read_csv(ind)
        need = [c for c in ("country", "population", "gdp_usd") if c in ci.columns]
        if len(need) < 3:
            sys.exit(f"{ind} must have columns country, population, gdp_usd; "
                     f"found {list(ci.columns)}")
        before = agg.index.copy()
        agg = agg.join(ci[need].set_index("country"), how="left")
        assert agg.index.equals(before), "country index lost during the join"
        matched = int(agg["population"].notna().sum())
        agg["cost_per_capita_usd"] = agg["cost_usd"] / agg["population"]
        agg["cost_share_of_gdp_pct"] = 100 * agg["cost_usd"] / agg["gdp_usd"]
        print(f"normalised by population and GDP "
              f"({matched}/{len(agg)} states matched an indicator)")
    else:
        print(f"(no {ind.name}; absolute exposure only -- add "
              "country,population,gdp_usd to normalise)")

    out = work / out_name
    agg.to_csv(out)
    print(f"\nwrote {out}   ({len(agg):,} states)\n")

    print("top 20 by attributed CO2")
    print(agg.head(20)[["rank", "co2_t", "co2_share_pct", "voyages"]]
          .to_string(float_format=lambda x: f"{x:,.1f}"))

    print("\nwhere the CIA case-study countries sit")
    missing = [c for c in CIA_CASE_STUDIES if c not in agg.index]
    present = agg.loc[[c for c in CIA_CASE_STUDIES if c in agg.index]]
    if len(present):
        print(present[["rank", "co2_t", "co2_share_pct"]]
              .to_string(float_format=lambda x: f"{x:,.1f}"))
        cov = present["co2_share_pct"].sum()
        print(f"\n  case studies cover {cov:.1f}% of attributed CO2 "
              f"across {len(present)} of {len(agg)} states")
    if missing:
        print(f"  not present in data: {missing}")


if __name__ == "__main__":
    main()
