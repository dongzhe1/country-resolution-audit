#!/usr/bin/env python3
"""Is the burden structured by development and geography?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolution_gap import SIDS, LDC, DEPENDENCIES

INCOME_ORDER = ["Low income", "Lower middle income",
                "Upper middle income", "High income"]


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    cov = XtXi @ meat @ XtXi * (n / (n - k))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot else float("nan")
    return beta, np.sqrt(np.diag(cov)), r2


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <work_dir>")
    work = Path(sys.argv[1]).expanduser().resolve()

    gap = pd.read_csv(work / "resolution_gap.csv")
    exp = pd.read_csv(work / "state_exposure.csv")
    ind = pd.read_csv(work / "country_indicators.csv")

    d = (gap.merge(exp[["country", "mean_distance_nm", "voyages"]],
                   left_on="iso3", right_on="country", how="left")
            .merge(ind[["country", "population", "gdp_usd", "income_group"]],
                   on="country", how="left"))
    d = d[~d["iso3"].isin(DEPENDENCIES)]
    d["resolved"] = d["resolved"].astype(bool)
    d["sids"] = d["iso3"].isin(SIDS)
    d["ldc"] = d["iso3"].isin(LDC)
    d["gdp_pc"] = d["gdp_usd"] / d["population"]
    d = d[d["t_per_musd"].notna() & (d["t_per_musd"] > 0)
          & d["gdp_pc"].notna() & (d["gdp_pc"] > 0)
          & d["mean_distance_nm"].notna() & (d["mean_distance_nm"] > 0)]
    print(f"{len(d)} sovereign states with exposure, GDP per capita and "
          f"mean voyage distance\n")


    print("--- exposure intensity by World Bank income group ---")
    print(f"  {'':<22}{'n':>4}{'unres.':>8}{'median t/M$':>13}"
          f"{'IQR':>20}{'x high-income':>15}")
    hi = d.loc[d["income_group"] == "High income", "t_per_musd"].median()
    for g in INCOME_ORDER:
        sub = d[d["income_group"] == g]
        if sub.empty:
            continue
        q = sub["t_per_musd"].quantile([.25, .75])
        print(f"  {g:<22}{len(sub):>4}{int((~sub['resolved']).sum()):>8}"
              f"{sub['t_per_musd'].median():>13,.0f}"
              f"{f'[{q.iloc[0]:,.0f}, {q.iloc[1]:,.0f}]':>20}"
              f"{sub['t_per_musd'].median()/hi:>15,.1f}")


    y = np.log(d["co2_t"].to_numpy())
    lgdp = np.log(d["gdp_usd"].to_numpy())
    ld = np.log(d["mean_distance_nm"].to_numpy())

    print("\n--- elasticity of exposure with respect to GDP ---")
    print("  (log CO2 on log GDP; the null that matters is beta = 1, not 0)")
    for lab, X, names in (
            ("GDP only", lgdp[:, None], ["log GDP"]),
            ("GDP + remoteness", np.column_stack([lgdp, ld]),
             ["log GDP", "log mean voyage distance"])):
        b, se, r2 = ols(y, X)
        print(f"\n  {lab:<20} R2={r2:.2f}  n={len(y)}")
        for nm, bi, si in zip(names, b[1:], se[1:]):
            extra = ""
            if nm == "log GDP" and si:
                extra = f"   t vs 1: {(bi - 1) / si:+.1f}"
            print(f"      {nm:<28}{bi:+7.3f}  (se {si:.3f}){extra}")

    b, se, _ = ols(y, np.column_stack([lgdp, ld]))
    beta = b[1]
    print(f"\n  Exposure grows with GDP at an elasticity of {beta:.2f}.")
    if se[1]:
        print(f"  Distance from proportionality: {(beta - 1) / se[1]:+.1f} "
              f"standard errors.")
    print(f"  Intensity (CO2 per dollar) therefore scales as GDP^({beta - 1:+.2f}): "
          f"a state with\n  10x lower GDP carries about "
          f"{10 ** (1 - beta):.1f}x the shipping CO2 per dollar of output,\n"
          f"  holding voyage distance fixed. Association across a cross-section, "
          f"not an effect.")


    t2 = b[2] / se[2] if se[2] else float("nan")
    if abs(t2) < 2:
        verdict = ("not distinguishable from zero once GDP is controlled for;\n"
                   "  remoteness does not add to the income gradient here")
    elif b[2] > 0:
        verdict = ("exposure rises with the distance a state's trade must\n"
                   "  travel, on top of the income gradient")
    else:
        verdict = ("exposure FALLS with distance once GDP is controlled for --\n"
                   "  the opposite of the expected sign; check it before "
                   "reporting it")
    print(f"\n  Remoteness enters at {b[2]:+.2f} (se {se[2]:.2f}, t {t2:+.1f}): "
          f"{verdict}.")


    print("\n--- where the unresolved states sit on both gradients ---")
    d["heavy"] = d["t_per_musd"] > d["t_per_musd"].median()
    d["poor"] = d["gdp_pc"] < d["gdp_pc"].median()
    d["remote"] = d["mean_distance_nm"] > d["mean_distance_nm"].median()
    for lab, mask in (("above-median intensity", d["heavy"]),
                      ("below-median GDP per capita", d["poor"]),
                      ("above-median voyage distance", d["remote"]),
                      ("all three", d["heavy"] & d["poor"] & d["remote"])):
        sub = d[mask]
        base = (~d["resolved"]).mean()
        share = (~sub["resolved"]).mean() if len(sub) else float("nan")
        print(f"  {lab:<32}n={len(sub):>4}  unresolved {share:>5.0%}  "
              f"(all states {base:.0%})")

    out = work / "development_gradient.csv"
    keep = ["iso3", "income_group", "gdp_pc", "mean_distance_nm", "t_per_musd",
            "t_per_capita", "share_pct", "resolved", "sids", "ldc"]
    d[[c for c in keep if c in d]].to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nIf the income slope is strongly negative and the unresolved states"
          "\nconcentrate at the heavy end, the finding is not about shipping: it"
          "\nis that the burden of a global environmental instrument is ordered"
          "\nby development and geography, and the assessment informing it is"
          "\nblindest exactly there.")


if __name__ == "__main__":
    main()
