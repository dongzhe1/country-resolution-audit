#!/usr/bin/env python3
"""Headline analysis: is the IMO impact assessment's case-study set representative?"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CIA_CASE_STUDIES = ["ARG", "BLZ", "BRA", "CHL", "COK",
                    "PER", "ZAF", "TGO", "TON", "USA"]

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)

CO2_PER_FUEL_T = 3.114
DEFAULT_SERVICE_SPEED_KN = 14.0

GRID = {
    "split":  [0.0, 0.5, 1.0],
    "sfoc":   [175.0, 190.0, 205.0],
    "aux":    [0.03, 0.05, 0.08],
    "maxload": [0.80, 0.90],
}

SEAWEB_COLS = ["lrnoimo_ship_no", "gross_tonnage", "deadweight", "shiptype_group",
               "total_kilowattsof_main_engines", "speedservice"]


def env_path(var: str):
    v = os.environ.get(var, "").strip()
    if not v:
        return None
    p = Path(v).expanduser().resolve()
    if not p.exists():
        sys.exit(f"{var}={v} does not exist")
    return p


from state_exposure import load_calibration


def load_inputs(work: Path, voy_name: str = "voyages.csv.gz"):
    voy = pd.read_csv(work / voy_name, dtype={"imo": str})
    print(f"voyage table: {voy_name}")
    sw_path = env_path("SEAWEB_PATH")
    if sw_path is None:
        for c in (work.parent / "seaweb" / "ship_info.csv",
                  work / "ship_info.csv",
                  work.parent / "ship_info.csv"):
            if c.exists():
                sw_path = c
                break
    if sw_path is None:
        sys.exit("ship_info.csv not found. Set SEAWEB_PATH to it, e.g.\n"
             "  export SEAWEB_PATH=/path/to/ship_info.csv")
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
    df["speed_ratio_cubed"] = (df["implied_speed_kn"] / df["svc"]) ** 3
    df["cal"] = load_calibration(work, df["shiptype_group"],
                                 voy_name).to_numpy()
    return df


def exposure(df: pd.DataFrame, split, sfoc, aux, maxload,
             calibrated=True) -> pd.Series:
    load = df["speed_ratio_cubed"].clip(0.05, maxload)
    co2 = ((df["me_kw"] * load + df["me_kw"] * aux)
           * df["sea_hours"] * sfoc / 1e6 * CO2_PER_FUEL_T)
    if calibrated:
        co2 = co2 * df["cal"]
    dep = pd.DataFrame({"country": df["dep_country"], "co2": co2 * split})
    arr = pd.DataFrame({"country": df["arr_country"], "co2": co2 * (1 - split)})
    both = pd.concat([dep, arr], ignore_index=True).dropna(subset=["country"])
    return both.groupby("country")["co2"].sum().sort_values(ascending=False)


_DF = None


def _set_df(df):
    global _DF
    _DF = df


def _grid_worker(combo):
    split, sfoc, aux, maxload = combo
    return exposure(_DF, split, sfoc, aux, maxload)


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def kendall_tau_b(a: pd.Series, b: pd.Series) -> float:
    x, y = a.to_numpy(float), b.to_numpy(float)
    n = len(x)
    if n < 2:
        return float("nan")
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(n, 1)
    dx, dy = dx[iu], dy[iu]
    conc = float((dx * dy > 0).sum())
    disc = float((dx * dy < 0).sum())
    tx = float((dx == 0).sum())
    ty = float((dy == 0).sum())
    n0 = n * (n - 1) / 2
    den = np.sqrt((n0 - tx) * (n0 - ty))
    return float((conc - disc) / den) if den > 0 else float("nan")


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def permutation_coverage(shares: pd.Series, k: int, draws: int = 20000, seed=0):
    rng = np.random.default_rng(seed)
    v = shares.to_numpy()
    return np.array([rng.choice(v, size=k, replace=False).sum()
                     for _ in range(draws)])


def parse_args(argv, name):
    jobs, voy, rest, i = DEFAULT_JOBS, "voyages.csv.gz", [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--jobs":
            if i + 1 >= len(argv):
                sys.exit("--jobs needs a value")
            jobs, i = int(argv[i + 1]), i + 2
        elif a.startswith("--jobs="):
            jobs, i = int(a.split("=", 1)[1]), i + 1
        elif a == "--voyages":
            voy, i = argv[i + 1], i + 2
        elif a.startswith("--voyages="):
            voy, i = a.split("=", 1)[1], i + 1
        else:
            rest.append(a); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <work_dir> [--jobs N] [--voyages NAME]")
    return Path(rest[0]).expanduser().resolve(), max(1, jobs), voy


def main():
    work, jobs, voy_name = parse_args(sys.argv[1:], Path(sys.argv[0]).name)
    df = load_inputs(work, voy_name)
    print(f"voyages usable: {len(df):,}   vessels: {df['imo'].nunique():,}")

    combos = list(itertools.product(*GRID.values()))
    jobs = max(1, min(jobs, len(combos)))
    print(f"running {len(combos)} settings on {jobs} process(es)")
    if jobs > 1:
        with mp.Pool(jobs, initializer=_set_df, initargs=(df,)) as pool:
            results = pool.map(_grid_worker, combos)
    else:
        _set_df(df)
        results = [_grid_worker(c) for c in combos]

    rows, base = [], None
    for (split, sfoc, aux, maxload), e in zip(combos, results):
        r = pd.Series(range(1, len(e) + 1), index=e.index, name="rank")
        rows.append(r.rename(f"s{split}_f{sfoc:.0f}_a{aux}_m{maxload}"))
        if (split, sfoc, aux, maxload) == (0.5, 190.0, 0.05, 0.90):
            base = e
    ranks = pd.concat(rows, axis=1)

    distinct = ranks.T.drop_duplicates().shape[0]
    print(f"distinct rankings among {len(combos)} settings: {distinct}")
    if distinct < len(combos):
        print(f"  -> {len(combos) // max(distinct, 1)}x of the grid is redundant "
              f"for rank-based claims (SFOC is a pure scale factor); "
              f"report the effective count, not {len(combos)}")
    ranks["rank_min"] = ranks.min(axis=1)
    ranks["rank_max"] = ranks.max(axis=1)
    ranks["rank_span"] = ranks["rank_max"] - ranks["rank_min"]
    ranks.to_csv(work / "sensitivity_ranks.csv")
    if base is None:
        base = exposure(df, 0.5, 190.0, 0.05, 0.90)

    raw = exposure(df, 0.5, 190.0, 0.05, 0.90, calibrated=False)
    cal = exposure(df, 0.5, 190.0, 0.05, 0.90, calibrated=True)
    common = raw.index.intersection(cal.index)
    rr = raw[common].rank(ascending=False)
    cr = cal[common].rank(ascending=False)
    rho = spearman(rr, cr)
    tau = kendall_tau_b(rr, cr)
    print(f"\ncalibrated vs uncalibrated ranking over {len(common)} states")
    print(f"  Spearman {rho:.4f}   Kendall {tau:.4f}")
    for n in (10, 20, 30):
        if n > len(common):
            continue
        ov = len(set(raw.head(n).index) & set(cal.head(n).index))
        print(f"  top-{n:<3} overlap {ov}/{n}")
    moved = (rr - cr).abs()
    print(f"  states moving >5 ranks: {int((moved > 5).sum())}   "
          f"max move {int(moved.max())}")
    pd.DataFrame({"uncalibrated_co2_t": raw[common],
                  "calibrated_co2_t": cal[common],
                  "uncalibrated_rank": rr, "calibrated_rank": cr}).to_csv(
        work / "comparison_calibration.csv")
    if rho > 0.99 and len(set(raw.head(20).index) & set(cal.head(20).index)) >= 19:
        print("  -> the ship-type bias does NOT move the ranking. Report both,"
              "\n     lead with the calibrated one, and the 4.8x spread becomes a"
              "\n     documented limitation rather than a threat to the claim.")
    else:
        print("  -> the ship-type bias DOES move the ranking. The calibration is"
              "\n     load-bearing: every distributional result must use it, and"
              "\n     MRV must be declared a calibration set with a held-out year.")

    shares = 100 * base / base.sum()
    print(f"\nstates: {len(base):,}   settings tried: {len(combos)}")
    print(f"Gini of exposure across states: {gini(base.to_numpy()):.3f}")
    top10 = shares.head(10).sum()
    print(f"top-10 states hold {top10:.1f}% of attributed CO2")

    print("\nrank stability across all settings (span = worst minus best rank)")
    order = base.index
    for lab, sel in (("top 10", order[:10]), ("top 30", order[:30]),
                     ("top 50", order[:50]), ("all", order)):
        sp = ranks.loc[ranks.index.intersection(sel), "rank_span"]
        if not len(sp):
            continue
        print(f"  {lab:<8} n={len(sp):>3}  median span {sp.median():>4.0f}  "
              f"within 2 places {100*(sp <= 2).mean():>3.0f}%  "
              f"within 5 {100*(sp <= 5).mean():>3.0f}%")
    print("\nmost assumption-sensitive states")
    print(ranks.sort_values("rank_span", ascending=False)
          .head(10)[["rank_min", "rank_max", "rank_span"]].to_string())

    lines = []
    present = [c for c in CIA_CASE_STUDIES if c in base.index]
    missing = [c for c in CIA_CASE_STUDIES if c not in base.index]
    lines.append(f"case studies given: {CIA_CASE_STUDIES}")
    lines.append(f"present in data   : {present}")
    if missing:
        lines.append(f"absent from data  : {missing}")

    if present:
        pos = pd.DataFrame({
            "rank": [int(base.index.get_loc(c)) + 1 for c in present],
            "co2_share_pct": [shares[c] for c in present],
        }, index=present)
        pos["rank_pctile"] = 100 * pos["rank"] / len(base)
        lines.append("\n" + pos.to_string(float_format=lambda x: f"{x:,.2f}"))

        cov = pos["co2_share_pct"].sum()
        null = permutation_coverage(shares, len(present))
        p = float((null >= cov).mean())
        lines.append(f"\ncoverage by the {len(present)} case studies: {cov:.2f}% "
                     f"of attributed CO2")
        lines.append(f"random pick of {len(present)}: median {np.median(null):.2f}%, "
                     f"p95 {np.percentile(null,95):.2f}%")
        lines.append(f"one-sided p (case studies cover at least this much "
                     f"by chance): {p:.3f}")
        lines.append("\nRead this as: the case studies are informative about the "
                     "states they cover,\nbut the assessment says itself that they "
                     "'do not reflect the combined\nimpact on states'. The number "
                     "above is how much of the distribution they\nleave unaddressed.")

    txt = "\n".join(lines)
    (work / "representativeness.txt").write_text(txt + "\n")
    print("\n" + txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available -- skipping figures)")
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    r = np.arange(1, len(base) + 1)
    ax[0].plot(r, base.to_numpy() / 1e6, lw=1.2, color="#333")
    for c in present:
        i = base.index.get_loc(c)
        ax[0].scatter([i + 1], [base.iloc[i] / 1e6], s=34, zorder=3, color="#c0392b")
        ax[0].annotate(c, (i + 1, base.iloc[i] / 1e6), fontsize=8,
                       xytext=(4, 4), textcoords="offset points")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("state rank")
    ax[0].set_ylabel("attributed CO$_2$ (Mt)")
    ax[0].set_title("Exposure by state, case studies marked")

    cum = np.cumsum(np.sort(base.to_numpy())) / base.sum()
    ax[1].plot(np.linspace(0, 1, len(cum)), cum, color="#333")
    ax[1].plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999")
    ax[1].set_xlabel("cumulative share of states")
    ax[1].set_ylabel("cumulative share of CO$_2$")
    ax[1].set_title(f"Concentration (Gini {gini(base.to_numpy()):.2f})")
    fig.tight_layout()
    fig.savefig(work / "fig_exposure.png", dpi=180)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    top = ranks.head(40)
    ax2.barh(range(len(top)), top["rank_span"], color="#2c7fb8")
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top.index, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("rank span across all assumption settings")
    ax2.set_title("Rank stability, top 40 states")
    fig2.tight_layout()
    fig2.savefig(work / "fig_rank_stability.png", dpi=180)
    print(f"\nwrote fig_exposure.png, fig_rank_stability.png to {work}")


if __name__ == "__main__":
    main()
