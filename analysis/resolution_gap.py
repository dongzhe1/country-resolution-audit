#!/usr/bin/env python3
"""Which states does the IMO impact assessment not resolve, and what do they carry?"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


SIDS = {
    "ATG","BHS","BRB","BLZ","CPV","COM","COK","CUB","DMA","DOM","FJI","GRD",
    "GNB","GUY","HTI","JAM","KIR","MDV","MHL","FSM","MUS","NRU","NIU","PLW",
    "PNG","KNA","LCA","VCT","WSM","STP","SYC","SGP","SLB","SUR","TLS","TON",
    "TTO","TUV","VUT",
}
LDC = {
    "AFG","AGO","BGD","BEN","BTN","BFA","BDI","KHM","CAF","TCD","COM","COD",
    "DJI","ERI","ETH","GMB","GIN","GNB","HTI","KIR","LAO","LSO","LBR","MDG",
    "MWI","MLI","MRT","MOZ","MMR","NPL","NER","RWA","STP","SEN","SLE","SLB",
    "SOM","SSD","SDN","TLS","TGO","TUV","UGA","TZA","YEM","ZMB",
}


DEPENDENCIES = {
    "AIA","ABW","BES","BMU","VGB","CYM","CXR","CCK","COK","CUW","FLK","FRO",
    "GUF","PYF","ATF","GIB","GRL","GLP","GUM","HKG","IMN","MAC","MTQ","MYT",
    "MSR","NCL","NIU","NFK","MNP","PCN","PRI","REU","BLM","SHN","MAF","SPM",
    "SXM","SGS","TKL","TCA","VIR","WLF","ESH","JEY","GGY","ALA",
}


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <work_dir>")
    work = Path(sys.argv[1]).expanduser().resolve()

def _find_reference():
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


    ref_path = _find_reference() / "gtap_regions_mepc82.csv"
    if not ref_path.exists():
        sys.exit(f"missing {ref_path}")
    ref = pd.read_csv(ref_path)
    resolved = set(ref.loc[ref["resolution"] == "individual", "iso3"].dropna())
    n_agg = int((ref["resolution"] == "aggregate").sum())

    exp_path = work / "state_exposure.csv"
    if not exp_path.exists():
        sys.exit(f"missing {exp_path} -- run state_exposure.py first")
    e = pd.read_csv(exp_path)
    ccol = "country" if "country" in e.columns else e.columns[0]
    e = e.rename(columns={ccol: "iso3"})
    e = e[e["iso3"].notna() & (e["iso3"].astype(str).str.len() == 3)]
    e["co2_t"] = pd.to_numeric(e["co2_t"], errors="coerce")
    e = e.dropna(subset=["co2_t"]).sort_values("co2_t", ascending=False)

    total = e["co2_t"].sum()
    e["share_pct"] = 100 * e["co2_t"] / total
    e["resolved"] = e["iso3"].isin(resolved)
    e["sids"] = e["iso3"].isin(SIDS)
    e["ldc"] = e["iso3"].isin(LDC)
    e["dependency"] = e["iso3"].isin(DEPENDENCIES)

    ind_path = work / "country_indicators.csv"
    if ind_path.exists():
        ind = pd.read_csv(ind_path)
        ind["country"] = ind["country"].astype(str)


        e = e.drop(columns=[c for c in ("population", "gdp_usd", "country")
                            if c in e.columns], errors="ignore")
        e = e.merge(ind[["country", "population", "gdp_usd"]],
                    left_on="iso3", right_on="country", how="left")
        e["t_per_capita"] = e["co2_t"] / e["population"]
        e["t_per_musd"] = e["co2_t"] / (e["gdp_usd"] / 1e6)
        have = int(e["population"].notna().sum())
        print(f"indicators: {have}/{len(e)} states matched")
        miss = e[e["population"].isna()].sort_values("co2_t", ascending=False)
        if len(miss):
            print(f"  no indicator for {len(miss)} states, largest: "
                  f"{', '.join(miss['iso3'].head(8))}")
    else:
        print("no country_indicators.csv -- absolute exposure only; "
              "the normalised comparison below will be skipped")
        e["t_per_capita"] = e["t_per_musd"] = pd.NA

    unres = e[~e["resolved"]]
    print(f"GTAP regions in Add.2 annexes : {len(ref)}  "
          f"({len(resolved)} individual, {n_agg} aggregate)")
    print(f"states with observed exposure : {len(e)}")
    print(f"  individually resolved       : {int(e['resolved'].sum())}")
    print(f"  inside an aggregate         : {len(unres)}  "
          f"({100*len(unres)/len(e):.0f}% of entities)")
    dep_unres = unres[unres["dependency"]]
    print(f"    of which non-sovereign    : {len(dep_unres)}  "
          f"({', '.join(sorted(dep_unres['iso3'])[:10])}"
          f"{' ...' if len(dep_unres) > 10 else ''})")
    print(f"    sovereign, no own row     : {len(unres) - len(dep_unres)}")
    print(f"  their share of exposure     : {unres['share_pct'].sum():.1f}%")

    print("\nby status")
    for lab, mask in (("SIDS", e["sids"]), ("LDC", e["ldc"]),
                      ("neither", ~e["sids"] & ~e["ldc"])):
        d = e[mask]
        if d.empty:
            continue
        u = d[~d["resolved"]]
        print(f"  {lab:<8} {len(d):>4} states, {len(u):>4} unresolved "
              f"({100*len(u)/len(d):>3.0f}%), "
              f"{d['share_pct'].sum():>5.2f}% of exposure, "
              f"{u['share_pct'].sum():>5.2f}% unresolved")

    print(f"\nlargest states with NO individual row in the assessment")
    top = unres.head(20)
    print(f"  {'':<6}{'rank':>6}{'share %':>10}{'  status'}")
    for _, r in top.iterrows():
        st = "SIDS" if r["sids"] else ("LDC" if r["ldc"] else "")
        gr = int(e.index.get_indexer([r.name])[0]) + 1
        print(f"  {r['iso3']:<6}{gr:>6}{r['share_pct']:>10.3f}  {st}")


    if e["t_per_capita"].notna().any():
        d = e[e["t_per_capita"].notna() & ~e["dependency"]].copy()
        world_med = d["t_per_capita"].median()
        print(f"\nnormalised exposure (tonnes CO2 per capita), "
              f"{len(d)} sovereign states with indicators")
        print(f"  world median {world_med:,.2f}")
        for lab, mask in (("resolved", d["resolved"]),
                          ("unresolved", ~d["resolved"]),
                          ("SIDS", d["sids"]), ("LDC", d["ldc"])):
            x = d.loc[mask, "t_per_capita"]
            if len(x) < 3:
                continue
            print(f"  {lab:<12} n={len(x):>3}  median {x.median():>9,.2f}  "
                  f"({x.median()/world_med:>5.1f}x world)  "
                  f"p90 {x.quantile(0.9):>10,.2f}")


        g = d[d["t_per_musd"].notna()]
        gmed = g["t_per_musd"].median()
        print(f"\nnormalised exposure (tonnes CO2 per million USD of GDP), "
              f"{len(g)} states")
        print(f"  world median {gmed:,.1f}")
        for lab, mask in (("resolved", g["resolved"]),
                          ("unresolved", ~g["resolved"]),
                          ("SIDS", g["sids"]), ("LDC", g["ldc"])):
            x = g.loc[mask, "t_per_musd"]
            if len(x) < 3:
                continue
            print(f"  {lab:<12} n={len(x):>3}  median {x.median():>9,.1f}  "
                  f"({x.median()/gmed:>5.1f}x world)  "
                  f"p90 {x.quantile(0.9):>10,.1f}")

        print(f"\n  top 15 by per-capita exposure  "
              f"[*] = no individual row in the assessment")
        for _, r in d.nlargest(15, "t_per_capita").iterrows():
            st = "SIDS" if r["sids"] else ("LDC" if r["ldc"] else "")
            print(f"    {'[*]' if not r['resolved'] else '   '} {r['iso3']:<5}"
                  f"{r['t_per_capita']:>12,.1f} t/cap  {r['share_pct']:>7.3f}% "
                  f"of total  {st}")


        sids = d[d["sids"]].nlargest(20, "t_per_capita")
        if len(sids):
            print(f"\n  SIDS by per-capita exposure "
                  f"({int(d['sids'].sum())} with indicators, "
                  f"{int((d['sids'] & ~d['resolved']).sum())} unresolved)")
            print(f"    {'':<4}{'':<5}{'t/cap':>9}{'t/M$':>9}{'share %':>10}")
            for _, r in sids.iterrows():
                mark = "    " if r["resolved"] else " [*]"
                gv = (f"{r['t_per_musd']:,.1f}"
                      if pd.notna(r["t_per_musd"]) else "-")
                print(f"   {mark} {r['iso3']:<5}{r['t_per_capita']:>9,.1f}"
                      f"{gv:>9}{r['share_pct']:>10.3f}")
            res = sids[sids["resolved"]]["iso3"].tolist()
            print(f"\n    individually resolved among these: "
                  f"{', '.join(res) if res else 'none'}")

    out = work / "resolution_gap.csv"
    cols = ["iso3", "co2_t", "share_pct", "resolved", "sids", "ldc",
            "dependency", "t_per_capita", "t_per_musd"]
    e[[c for c in cols if c in e.columns]].to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nThe claim this supports is NOT that the assessment ignored States --"
          "\nit assessed 111 regions. It is that the resolution is coarsest exactly"
          "\nwhere the assessment's own finding puts the largest impacts, and that"
          "\nAIS gives an independent, physically observed measure at full state"
          "\nresolution for the states that have no row of their own.")


if __name__ == "__main__":
    main()
