#!/usr/bin/env python3
"""Validate the bottom-up voyage CO2 estimate against reported EU MRV totals."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EEA = {
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
    "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK",
    "SVN","ESP","SWE","ISL","LIE","NOR",
}

SFOC_G_PER_KWH = 190.0
CO2_PER_FUEL_T = 3.114
AUX_FRACTION_AT_SEA = 0.05
MIN_LOAD, MAX_LOAD = 0.05, 0.90
DEFAULT_SERVICE_SPEED_KN = 14.0

SEAWEB_COLS = ["lrnoimo_ship_no", "gross_tonnage", "deadweight", "shiptype_group",
               "total_kilowattsof_main_engines", "speedservice"]


def env_path(var: str) -> Path | None:
    v = os.environ.get(var, "").strip()
    if not v:
        return None
    p = Path(v).expanduser().resolve()
    if not p.exists():
        sys.exit(f"{var}={v} does not exist")
    return p


def find(work: Path, *rel) -> Path | None:
    for r in rel:
        p = (work / r).resolve()
        if p.exists():
            return p
    return None


def load_mrv(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(list(path.glob("*.csv")) + list(path.glob("*.xlsx")))
    else:
        files = [path]
    if not files:
        sys.exit(f"no CSV/XLSX under {path}")

    parts = []
    for f in files:
        try:
            d = load_mrv_file(f)
        except SystemExit as exc:
            print(f"  skipped {f.name}: {exc}")
            continue
        yrs = sorted(d["year"].dropna().unique().astype(int)) if d["year"].notna().any() else []
        print(f"  {f.name}: {len(d):,} rows, years {yrs}")
        parts.append(d)
    if not parts:
        sys.exit(f"no readable MRV file under {path}")
    all_mrv = pd.concat(parts, ignore_index=True)

    key = ["imo", "year"]
    dup = all_mrv.duplicated(key, keep=False)
    if dup.any():
        g = all_mrv[dup].groupby(key)["reported_co2_t"]
        spread = (g.max() - g.min()) / g.max().replace(0, pd.NA)
        disagree = int((spread > 0.01).sum())
        print(f"  duplicate ship-years: {int(dup.sum()):,}; "
              f"{disagree:,} disagree by >1% on reported CO2")
        if disagree:
            print("  !! duplicate exports are NOT the same data -- point "
                  "--mrv-dir at one consistent set of files")
    before = len(all_mrv)
    all_mrv = all_mrv.drop_duplicates(key, keep="first")
    print(f"MRV total: {len(all_mrv):,} ship-years "
          f"({before - len(all_mrv):,} duplicate rows dropped)")
    return all_mrv


def load_mrv_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(path, dtype=str)
    else:
        raw = pd.read_csv(path, dtype=str, sep=None, engine="python",
                          on_bad_lines="skip")
    cols = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in raw.columns}

    def pick(*needles, avoid=()):
        for orig, flat in cols.items():
            if any(n in flat for n in needles) and not any(a in flat for a in avoid):
                return orig
        return None

    c_imo = pick("imonumber", "imo")
    c_co2 = pick("totalco2emissions", "totalco2", avoid=("perdistance", "pertransport",
                                                         "onladen", "atberth"))
    c_year = pick("reportingperiod", "year")
    if not (c_imo and c_co2):
        sys.exit(f"could not find IMO / total CO2 columns in {list(raw.columns)[:15]}")
    print(f"  imo={c_imo!r}  co2={c_co2!r}  year={c_year!r}")

    out = pd.DataFrame({
        "imo": raw[c_imo].astype(str).str.extract(r"(\d{7})")[0],
        "reported_co2_t": pd.to_numeric(
            raw[c_co2].astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
            errors="coerce"),
        "year": pd.to_numeric(raw[c_year], errors="coerce") if c_year else np.nan,
    })
    return out.dropna(subset=["imo", "reported_co2_t"])


def parse_args(argv, name):
    voy, mrv_dir, seaweb, rest, i = "voyages_routed.csv.gz", None, None, [], 0
    while i < len(argv):
        if argv[i] == "--voyages":
            voy, i = argv[i + 1], i + 2
        elif argv[i].startswith("--voyages="):
            voy, i = argv[i].split("=", 1)[1], i + 1
        elif argv[i] == "--mrv-dir":
            mrv_dir, i = Path(argv[i + 1]).expanduser().resolve(), i + 2
        elif argv[i].startswith("--mrv-dir="):
            mrv_dir, i = Path(argv[i].split("=", 1)[1]).expanduser().resolve(), i + 1
        elif argv[i] == "--seaweb":
            seaweb, i = Path(argv[i + 1]).expanduser().resolve(), i + 2
        elif argv[i].startswith("--seaweb="):
            seaweb, i = Path(argv[i].split("=", 1)[1]).expanduser().resolve(), i + 1
        else:
            rest.append(argv[i]); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <gfw_port_visits_dir> [--voyages NAME] "
                 f"[--mrv-dir PATH] [--seaweb PATH]")
    return Path(rest[0]).expanduser().resolve(), voy, mrv_dir, seaweb


def main():
    work, voy_name, mrv_dir, seaweb = parse_args(sys.argv[1:], Path(sys.argv[0]).name)

    mrv_path = mrv_dir or env_path("MRV_DIR") or find(work, "../mrv", "mrv", "../../data/mrv")
    if mrv_path is None:
        sys.exit("MRV data not found. Pass --mrv-dir /path/to/mrv\n"
                 "  (or set MRV_DIR, which --mrv-dir overrides).")
    sw_path = seaweb or env_path("SEAWEB_PATH") or find(
        work, "../seaweb/ship_info.csv", "ship_info.csv", "../ship_info.csv")
    if sw_path is None:
        sys.exit("ship_info.csv not found. Pass --seaweb /path/to/ship_info.csv\n"
                 "  (or set SEAWEB_PATH, which --seaweb overrides).")
    print(f"MRV    : {mrv_path}")
    print(f"Sea-web: {sw_path}")

    print(f"voyage table: {voy_name}")
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
    svc = df["service_kn"].where(df["service_kn"].between(5, 30),
                                 DEFAULT_SERVICE_SPEED_KN)
    load = ((df["implied_speed_kn"] / svc) ** 3).clip(MIN_LOAD, MAX_LOAD)
    df["co2_t"] = ((df["me_kw"] * load + df["me_kw"] * AUX_FRACTION_AT_SEA)
                   * df["sea_hours"] * SFOC_G_PER_KWH / 1e6 * CO2_PER_FUEL_T)

    dep_eea = df["dep_country"].isin(EEA)
    arr_eea = df["arr_country"].isin(EEA)
    df["scope"] = np.where(dep_eea | arr_eea, 1.0, 0.0)
    df["scope_ets"] = np.where(dep_eea & arr_eea, 1.0,
                               np.where(dep_eea | arr_eea, 0.5, 0.0))
    df["co2_eu_t"] = df["co2_t"] * df["scope"]
    df["co2_ets_t"] = df["co2_t"] * df["scope_ets"]

    est = (df[df["co2_eu_t"] > 0]
           .groupby(["imo", "year"], as_index=False)
           .agg(est_co2_t=("co2_eu_t", "sum"),
                est_ets_scope_t=("co2_ets_t", "sum"),
                voyages=("co2_eu_t", "size"),
                one_ended=("scope_ets", lambda x: float((x == 0.5).mean())),
                svc_kn=("service_kn", "median"),
                gt=("gross_tonnage", "first"),
                dwt=("deadweight", "first"),
                grp=("shiptype_group", "first")))

    mrv = load_mrv(mrv_path)
    if mrv["year"].notna().any():
        m = est.merge(mrv, on=["imo", "year"], how="inner")
    else:
        mrv1 = mrv.groupby("imo", as_index=False)["reported_co2_t"].sum()
        m = est.groupby("imo", as_index=False).agg(
            est_co2_t=("est_co2_t", "sum"), voyages=("voyages", "sum"),
            gt=("gt", "first"), dwt=("dwt", "first"), grp=("grp", "first")
        ).merge(mrv1, on="imo", how="inner")

    m = m[(m["reported_co2_t"] > 0) & (m["est_co2_t"] > 0)]
    if m.empty:
        sys.exit("no overlap between estimated and reported CO2 -- check MRV parsing")

    m["ratio"] = m["est_co2_t"] / m["reported_co2_t"]
    m["log_ratio"] = np.log(m["ratio"])

    print(f"\nmatched ship-years: {len(m):,}   ships: {m['imo'].nunique():,}")
    q = m["ratio"].quantile([.05, .25, .5, .75, .95])
    print("\nestimate / reported")
    for k, v in q.items():
        print(f"  p{int(k*100):02d}  {v:6.2f}")
    print(f"  mean {m['ratio'].mean():6.2f}   "
          f"geometric mean {np.exp(m['log_ratio'].mean()):6.2f}")
    print(f"  total est {m['est_co2_t'].sum()/1e6:,.1f} Mt vs "
          f"reported {m['reported_co2_t'].sum()/1e6:,.1f} Mt  "
          f"({100*m['est_co2_t'].sum()/m['reported_co2_t'].sum():.0f}%)")

    print("\nby ship-type group (median ratio, n>=20)")
    g = m.groupby("grp").agg(n=("ratio", "size"),
                             median_ratio=("ratio", "median"),
                             svc_kn=("svc_kn", "median"),
                             one_ended=("one_ended", "median"))
    g = g[g["n"] >= 20].sort_values("median_ratio")
    print(g.to_string(float_format=lambda x: f"{x:,.2f}"))
    if len(g) >= 8:
        rho = g["median_ratio"].corr(g["svc_kn"], method="spearman")
        print(f"\n  Spearman(type median ratio, type service speed) = {rho:+.2f}")
        if rho <= -0.5:
            print("  strongly negative: the cubic law against a high design speed"
                  "\n  is driving the slope")
        elif rho >= 0.5:
            print("  strongly positive: unexpected -- design speed is not the"
                  "\n  mechanism it was assumed to be")
        else:
            print("  near zero: design speed does NOT order the slope, so the"
                  "\n  cubic-law explanation is refuted; look elsewhere")
        spread = g["median_ratio"].max() / g["median_ratio"].min()
        print(f"  slope spread (max/min of type medians) = {spread:.1f}x"
              f"   [flat is ~1.5x or less]")

    print("\nby size decile (DWT)")
    m["dwt_decile"] = pd.qcut(m["dwt"], 10, labels=False, duplicates="drop")
    print(m.groupby("dwt_decile").agg(
        n=("ratio", "size"), median_dwt=("dwt", "median"),
        median_ratio=("ratio", "median")
    ).to_string(float_format=lambda x: f"{x:,.2f}"))

    print("\nscope rule sensitivity (whole sample)")
    for lab, col in (("MRV, Art. 2(1): one EEA end counts in full", "est_co2_t"),
                     ("ETS rule (WRONG here): one EEA end counts 0.5",
                      "est_ets_scope_t")):
        tot = m[col].sum() / m["reported_co2_t"].sum()
        print(f"  {lab:<52} {tot:5.2f}")

    stem = Path(voy_name).name.split(".")[0]
    out = work / (f"mrv_validation_{stem}.csv" if stem != "voyages"
                  else "mrv_validation.csv")
    m.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nNOTE: at-berth emissions are in the reported figure but not in the\n"
          "voyage model, so a ratio below 1 is expected. What matters for the\n"
          "distributional claim is whether the ratio is FLAT across ship types\n"
          "and sizes -- a sloped ratio biases states by fleet composition.")


if __name__ == "__main__":
    main()
