#!/usr/bin/env python3
"""Can one route factor flatten the ship-type slope in the MRV comparison?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_mrv import (EEA, MIN_LOAD, MAX_LOAD, SFOC_G_PER_KWH,
                          CO2_PER_FUEL_T, AUX_FRACTION_AT_SEA,
                          DEFAULT_SERVICE_SPEED_KN, SEAWEB_COLS,
                          env_path, find, load_mrv)

K_GRID = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]
FIT_YEARS = [2022, 2023]
TEST_YEARS = [2024]
MIN_N = 20
FLAT = 1.5


def prepare(work: Path, voy_name: str = "voyages.csv.gz") -> pd.DataFrame:
    sw_path = env_path("SEAWEB_PATH") or find(
        work, "../seaweb/ship_info.csv", "ship_info.csv", "../ship_info.csv")
    if sw_path is None:
        sys.exit("set SEAWEB_PATH to ship_info.csv")

    voy = pd.read_csv(work / voy_name, dtype={"imo": str})
    voy["dep_time"] = pd.to_datetime(voy["dep_time"], errors="coerce", utc=True)
    voy["year"] = voy["dep_time"].dt.year

    sw = pd.read_csv(sw_path, usecols=SEAWEB_COLS, dtype={"lrnoimo_ship_no": str},
                     low_memory=False).rename(columns={
        "lrnoimo_ship_no": "imo",
        "total_kilowattsof_main_engines": "me_kw",
        "speedservice": "service_kn"})
    for c in ("me_kw", "service_kn", "gross_tonnage", "deadweight"):
        sw[c] = pd.to_numeric(sw[c], errors="coerce")

    df = voy.merge(sw, on="imo", how="left")
    df = df[df["me_kw"].notna() & (df["me_kw"] > 0)]
    df["svc"] = df["service_kn"].where(df["service_kn"].between(5, 30),
                                       DEFAULT_SERVICE_SPEED_KN)
    in_scope = df["dep_country"].isin(EEA) | df["arr_country"].isin(EEA)
    return df[in_scope].copy()


def ratios(df: pd.DataFrame, mrv: pd.DataFrame, k: float) -> pd.DataFrame:
    raw = (df["implied_speed_kn"] * k / df["svc"]) ** 3
    load = raw.clip(MIN_LOAD, MAX_LOAD)
    co2 = ((df["me_kw"] * load + df["me_kw"] * AUX_FRACTION_AT_SEA)
           * df["sea_hours"] * SFOC_G_PER_KWH / 1e6 * CO2_PER_FUEL_T)
    est = (df.assign(co2_t=co2, _lo=(raw < MIN_LOAD), _hi=(raw > MAX_LOAD))
             .groupby(["imo", "year"], as_index=False)
             .agg(est_co2_t=("co2_t", "sum"),
                  svc_kn=("svc", "median"),
                  clip_lo=("_lo", "mean"), clip_hi=("_hi", "mean"),
                  grp=("shiptype_group", "first")))
    m = est.merge(mrv, on=["imo", "year"], how="inner")
    m = m[(m["reported_co2_t"] > 0) & (m["est_co2_t"] > 0)].copy()
    m["ratio"] = m["est_co2_t"] / m["reported_co2_t"]
    return m


def slope(m: pd.DataFrame):
    g = m.groupby("grp").agg(n=("ratio", "size"),
                             med=("ratio", "median"),
                             svc=("svc_kn", "median"))
    g = g[g["n"] >= MIN_N]
    if len(g) < 8:
        return np.nan, np.nan, g
    return (g["med"].max() / g["med"].min(),
            g["med"].corr(g["svc"], method="spearman"), g)


def report(tag, m):
    sp, rho, _ = slope(m)
    return (f"  {tag:<10} n={len(m):>6,}  median={m['ratio'].median():5.2f}  "
            f"geo={np.exp(np.log(m['ratio']).mean()):5.2f}  "
            f"spread={sp:5.2f}x  rho(svc)={rho:+.2f}  "
            f"clip_lo={m['clip_lo'].mean():.0%}")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <work_dir>")
    work = Path(sys.argv[1]).expanduser().resolve()

    df = prepare(work)
    mrv_path = env_path("MRV_DIR") or find(work, "../mrv", "mrv")
    if mrv_path is None:
        sys.exit("set MRV_DIR to the THETIS directory")
    mrv = load_mrv(mrv_path)

    fit = df[df["year"].isin(FIT_YEARS)]
    test = df[df["year"].isin(TEST_YEARS)]
    print(f"\nfit years {FIT_YEARS}: {len(fit):,} voyages   "
          f"test years {TEST_YEARS}: {len(test):,} voyages")

    print(f"\nroute factor scan  (flat = spread <= {FLAT}x)")
    print("  k       n      median    geo   spread   rho(svc)  clip_lo")
    best_k, best_spread = None, np.inf
    for k in K_GRID:
        m = ratios(fit, mrv, k)
        sp, rho, _ = slope(m)
        star = ""
        if np.isfinite(sp) and sp < best_spread:
            best_spread, best_k, star = sp, k, "  <-"
        print(f"  {k:4.2f}  {len(m):>6,}   {m['ratio'].median():5.2f}  "
              f"{np.exp(np.log(m['ratio']).mean()):5.2f}   {sp:5.2f}x    "
              f"{rho:+.2f}     {m['clip_lo'].mean():4.0%}{star}")

    if best_k is None:
        sys.exit("\nno usable fit -- too few ship types clear MIN_N")

    print(f"\nflattest on the fit years: k = {best_k} (spread {best_spread:.2f}x)")
    print("\nheld out:")
    print(report("fit", ratios(fit, mrv, best_k)))
    print(report("test", ratios(test, mrv, best_k)))

    _, _, g = slope(ratios(test, mrv, best_k))
    print(f"\nship types on the held-out year at k={best_k}")
    print(g.sort_values("med").to_string(float_format=lambda x: f"{x:,.2f}"))

    print(f"\nVERDICT")
    sp_test, _, _ = slope(ratios(test, mrv, best_k))
    sp_base, _, _ = slope(ratios(test, mrv, 1.0))
    print(f"  slope on held-out year: {sp_base:.2f}x at k=1.0 "
          f"-> {sp_test:.2f}x at k={best_k}")
    if sp_test <= FLAT:
        print(f"  FLAT ENOUGH. One documented route factor fixes it; report k as"
              f"\n  a calibrated parameter and this held-out check as its evidence.")
    else:
        print(f"  STILL SLOPED. Distance is not the whole story -- the emission"
              f"\n  model itself has to change, or the claim has to be restricted"
              f"\n  to comparisons within a ship type.")


if __name__ == "__main__":
    main()
