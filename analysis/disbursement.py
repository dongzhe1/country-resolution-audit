#!/usr/bin/env python3
"""What the aggregation hides about who gains and who loses.

    python disbursement.py <work_dir> [--price P] [--out NAME]
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolution_gap import SIDS, LDC, DEPENDENCIES

PRICES = [60.0, 100.0, 150.0]
PASSTHROUGH = [0.5, 1.0]
SCHEMES = {
    "all": lambda d: pd.Series(True, index=d.index),
    "developing": lambda d: d["income_group"].fillna("") != "High income",
    "sids_ldc": lambda d: d["sids"] | d["ldc"],
}


def main():
    argv = sys.argv[1:]
    out_name = "disbursement.csv"
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out_name, i = argv[i + 1], i + 2
        else:
            rest.append(argv[i]); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <work_dir> [--out NAME]")
    work = Path(rest[0]).expanduser().resolve()

    gap = pd.read_csv(work / "resolution_gap.csv")
    ind = pd.read_csv(work / "country_indicators.csv")
    d = gap.merge(ind[["country", "population", "income_group"]],
                  left_on="iso3", right_on="country", how="left")
    d = d[~d["iso3"].isin(DEPENDENCIES)]
    d = d[d["population"].notna() & (d["population"] > 0)]
    d["sids"] = d["iso3"].isin(SIDS)
    d["ldc"] = d["iso3"].isin(LDC)
    d["resolved"] = d["resolved"].astype(bool)
    d["share"] = d["share_pct"] / 100.0
    print(f"{len(d)} sovereign states with exposure, population and income group")
    print(f"  individually resolved in the assessment: {int(d['resolved'].sum())}")

    total_co2 = (gap["co2_t"].sum())
    rows = []
    for price, alpha, (sname, elig) in itertools.product(
            PRICES, PASSTHROUGH, SCHEMES.items()):
        rev = total_co2 * price
        cost = d["share"] * rev * alpha
        ok = elig(d)
        pool = d.loc[ok, "population"].sum()
        disb = np.where(ok, rev * d["population"] / pool, 0.0)
        net = disb - cost
        per_cap = net / d["population"]
        for lab, mask in (("SIDS", d["sids"]), ("LDC", d["ldc"]),
                          ("all", pd.Series(True, index=d.index))):
            sub = per_cap[mask]
            if len(sub) < 3:
                continue
            grp_mean = sub.mean()
            grp_sign = np.sign(grp_mean)
            flips = int((np.sign(sub) != grp_sign).sum())
            unres = int((mask & ~d["resolved"]).sum())
            near = (np.abs(sub - grp_mean) <= np.abs(grp_mean)) if grp_mean else \
                pd.Series(False, index=sub.index)
            spread = (sub.max() - sub.min())
            rows.append(dict(price=price, passthrough=alpha, scheme=sname,
                             group=lab, n=len(sub), unresolved=unres,
                             group_mean_per_cap=grp_mean,
                             median_per_cap=sub.median(),
                             losers=int((sub < 0).sum()),
                             sign_flips=flips,
                             within_2x_of_mean=int(near.sum()),
                             spread_per_cap=spread,
                             spread_over_mean=(spread / abs(grp_mean)
                                               if grp_mean else float("nan")),
                             p10=sub.quantile(.1), p90=sub.quantile(.9)))

    r = pd.DataFrame(rows)
    r.to_csv(work / "disbursement_summary.csv", index=False)

    print(f"\nnet position per capita (USD), {len(PRICES)}x{len(PASSTHROUGH)} "
          f"price/pass-through settings")
    for sname in SCHEMES:
        print(f"\n  scheme: {sname}")
        sub = r[r["scheme"] == sname]
        for grp in ("SIDS", "LDC", "all"):
            g = sub[sub["group"] == grp]
            if g.empty:
                continue
            n = int(g["n"].iloc[0])
            print(f"    {grp:<5} n={n:>3} "
                  f"({int(g['unresolved'].iloc[0])} with no individual row)")
            print(f"          group mean {g['group_mean_per_cap'].min():>9,.0f} .. "
                  f"{g['group_mean_per_cap'].max():>9,.0f} USD/cap")
            print(f"          spread across members / |mean|: "
                  f"{g['spread_over_mean'].min():>6.1f} .. "
                  f"{g['spread_over_mean'].max():>6.1f}x")
            print(f"          members within a factor of two of the mean: "
                  f"{g['within_2x_of_mean'].min()}-{g['within_2x_of_mean'].max()}"
                  f" of {n}")
            print(f"          members with the opposite sign: "
                  f"{g['sign_flips'].min()}-{g['sign_flips'].max()}")

    price, alpha = PRICES[len(PRICES) // 2], PASSTHROUGH[-1]
    rev = total_co2 * price
    for sname, elig in SCHEMES.items():
        ok = elig(d)
        pool = d.loc[ok, "population"].sum()
        d[f"net_{sname}"] = (np.where(ok, rev * d["population"] / pool, 0.0)
                             - d["share"] * rev * alpha) / d["population"]
    keep = ["iso3", "share_pct", "resolved", "sids", "ldc", "population",
            "t_per_capita", "t_per_musd"] + [f"net_{s}" for s in SCHEMES]
    d[[c for c in keep if c in d]].to_csv(work / out_name, index=False)

    print(f"\ncentral setting: {price:.0f} USD/t, pass-through {alpha:.0%}")
    s = d[d["sids"]].sort_values("net_sids_ldc")
    print(f"\n  SIDS under the SIDS+LDC scheme, worst ten net positions "
          f"(USD per capita)")
    print(f"    {'':<4}{'':<5}{'net':>12}{'exposure t/cap':>16}  own row")
    for _, x in s.head(10).iterrows():
        print(f"   {'' if x['resolved'] else '[*]':<4} {x['iso3']:<5}"
              f"{x['net_sids_ldc']:>12,.0f}{x['t_per_capita']:>16,.1f}"
              f"  {'yes' if x['resolved'] else 'no'}")
    losers = s[s["net_sids_ldc"] < 0]
    sm = d.loc[d["sids"], "net_sids_ldc"]
    print(f"\n  {len(losers)} of {len(s)} SIDS are net losers under a scheme "
          f"that disburses to SIDS and LDCs\n"
          f"  ({int((~losers['resolved']).sum())} of them with no individual row).")
    print(f"\n  But the sharper problem is dispersion, not sign. The SIDS group "
          f"mean is\n  {sm.mean():+,.0f} USD per capita while members run from "
          f"{sm.min():+,.0f} to {sm.max():+,.0f} --\n  a spread "
          f"{(sm.max()-sm.min())/abs(sm.mean()):.0f}x the mean. "
          f"{int((np.abs(sm - sm.mean()) <= abs(sm.mean())).sum())} of {len(sm)} "
          f"members lie within a\n  factor of two of it. An aggregate row "
          f"reports that mean.")
    ldc = d.loc[d["ldc"], "net_sids_ldc"]
    print(f"\n  The LDC group behaves differently: mean {ldc.mean():+,.0f}, "
          f"{int((ldc < 0).sum())} net losers.\n"
          f"  The two groups the scheme treats alike do not behave alike, and "
          f"the one\n  that is heterogeneous is the one that is 90% aggregated.")

    print(f"\nwrote {work / out_name} and {work / 'disbursement_summary.csv'}")


if __name__ == "__main__":
    main()
