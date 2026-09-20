#!/usr/bin/env python3
"""Calibrate the bottom-up emission model per ship type against reported MRV."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_distance import prepare, FIT_YEARS, TEST_YEARS, MIN_N
from validate_mrv import (MIN_LOAD, MAX_LOAD, SFOC_G_PER_KWH, CO2_PER_FUEL_T,
                          AUX_FRACTION_AT_SEA, env_path, find, load_mrv)


def ship_years(df: pd.DataFrame, mrv: pd.DataFrame) -> pd.DataFrame:
    load = ((df["implied_speed_kn"] / df["svc"]) ** 3).clip(MIN_LOAD, MAX_LOAD)
    co2 = ((df["me_kw"] * load + df["me_kw"] * AUX_FRACTION_AT_SEA)
           * df["sea_hours"] * SFOC_G_PER_KWH / 1e6 * CO2_PER_FUEL_T)
    est = (df.assign(co2_t=co2)
             .groupby(["imo", "year"], as_index=False)
             .agg(est_co2_t=("co2_t", "sum"), grp=("shiptype_group", "first")))
    m = est.merge(mrv, on=["imo", "year"], how="inner")
    m = m[(m["reported_co2_t"] > 0) & (m["est_co2_t"] > 0)].copy()
    m["ratio"] = m["est_co2_t"] / m["reported_co2_t"]
    return m


def spread(s: pd.Series) -> float:
    return float(s.max() / s.min()) if len(s) and s.min() > 0 else np.nan


BASELINE = "voyages.csv.gz"


def parse_args(argv, name):
    voy, rest, i = "voyages_routed.csv.gz", [], 0
    allow = False
    while i < len(argv):
        if argv[i] == "--voyages":
            voy, i = argv[i + 1], i + 2
        elif argv[i].startswith("--voyages="):
            voy, i = argv[i].split("=", 1)[1], i + 1
        elif argv[i] == "--allow-baseline":
            allow, i = True, i + 1
        else:
            rest.append(argv[i]); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <work_dir> [--voyages FILE] [--allow-baseline]")
    if voy == BASELINE and not allow:
        sys.exit(f"{name}: refusing to fit on {BASELINE}.\n"
                 f"  That table is the baseline -- all anchorages, great-circle\n"
                 f"  distance -- and the write-up's emissions do not come from it.\n"
                 f"  Pass --voyages voyages_routed.csv.gz, or --allow-baseline\n"
                 f"  if you really mean to reproduce the superseded calibration.")
    return Path(rest[0]).expanduser().resolve(), voy


def main():
    work, voy_name = parse_args(sys.argv[1:], Path(sys.argv[0]).name)
    print(f"voyage table: {voy_name}")

    df = prepare(work, voy_name)
    mrv_path = env_path("MRV_DIR") or find(work, "../mrv", "mrv")
    if mrv_path is None:
        sys.exit("set MRV_DIR to the THETIS directory")
    mrv = load_mrv(mrv_path)

    fit = ship_years(df[df["year"].isin(FIT_YEARS)], mrv)
    test = ship_years(df[df["year"].isin(TEST_YEARS)], mrv)
    print(f"\nfit {FIT_YEARS}: {len(fit):,} ship-years   "
          f"test {TEST_YEARS}: {len(test):,} ship-years")

    g = fit.groupby("grp")["ratio"].agg(["size", "median"])
    usable = g[g["size"] >= MIN_N]
    global_factor = 1.0 / fit["ratio"].median()
    cal = pd.DataFrame({
        "shiptype_group": usable.index,
        "n_fit": usable["size"].to_numpy(),
        "fit_median_ratio": usable["median"].to_numpy(),
        "factor": 1.0 / usable["median"].to_numpy(),
    })
    print(f"\ntypes calibrated: {len(cal)}  (n >= {MIN_N} ship-years)")
    print(f"global fallback factor: {global_factor:.2f}  "
          f"(median ratio {fit['ratio'].median():.2f})")

    fmap = dict(zip(cal["shiptype_group"], cal["factor"]))
    for name, d in (("fit", fit), ("test", test)):
        d["corr_type"] = d["ratio"] * d["grp"].map(fmap).fillna(global_factor)
        d["corr_global"] = d["ratio"] * global_factor

    rows = []
    for name, d in (("fit", fit), ("test", test)):
        t = d.groupby("grp").agg(n=("ratio", "size"),
                                 raw=("ratio", "median"),
                                 per_type=("corr_type", "median"),
                                 global_only=("corr_global", "median"))
        t = t[t["n"] >= MIN_N]
        rows.append((name, len(t), spread(t["raw"]), spread(t["per_type"]),
                     spread(t["global_only"])))
        if name == "test":
            test_table = t

    print("\nspread across ship types (max/min of type medians)")
    print(f"  {'':<6}{'types':>7}{'raw':>9}{'per-type':>11}{'global-only':>13}")
    for name, n, raw, pt, go in rows:
        print(f"  {name:<6}{n:>7}{raw:>9.2f}{pt:>11.2f}{go:>13.2f}")

    print("\nheld-out year, worst residual types")
    print(test_table.sort_values("per_type").head(6)
          .to_string(float_format=lambda x: f"{x:,.2f}"))
    print(test_table.sort_values("per_type").tail(6)
          .to_string(float_format=lambda x: f"{x:,.2f}"))

    cal["fitted_on"] = voy_name
    cal.to_csv(work / "type_calibration.csv", index=False)
    pd.DataFrame([{"shiptype_group": "__global__", "n_fit": len(fit),
                   "fit_median_ratio": fit["ratio"].median(),
                   "factor": global_factor}]).to_csv(
        work / "type_calibration.csv", mode="a", header=False, index=False)
    print(f"\nwrote {work / 'type_calibration.csv'} "
          f"({len(cal)} types + __global__ fallback)")

    raw_t, test_pt, glob_t = rows[1][2], rows[1][3], rows[1][4]
    print("\nVERDICT")
    print(f"  held-out spread: {raw_t:.2f}x raw -> {test_pt:.2f}x per-type "
          f"-> {glob_t:.2f}x global-only")
    red = 1 - test_pt / raw_t if raw_t else float("nan")
    print(f"  held-out reduction: {red:.0%}  "
          f"(memorisation would leave the test spread near {raw_t:.2f}x)")
    if test_pt <= 1.5:
        print("  Flat out of sample. Use it, declare MRV a calibration set, and"
              "\n  cite the held-out year as the validation.")
    elif red >= 0.2:
        print("  Partial: real out-of-sample effect, not flat. Whether that is"
              "\n  enough is decided by the rank test in analyze.py, not here --"
              "\n  a bias that does not move the ranking is a limitation, not a"
              "\n  threat to a distributional claim.")
    else:
        print("  Negligible out-of-sample effect. The fit years were largely"
              "\n  memorised; do not lean on it.")
    if abs(glob_t - raw_t) < 1e-9:
        print("  A single global factor leaves the spread untouched, as it must:"
              "\n  one multiplicative constant cannot change a relative ratio.")


if __name__ == "__main__":
    main()
