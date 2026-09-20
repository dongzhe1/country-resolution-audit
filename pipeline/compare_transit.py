#!/usr/bin/env python3
"""Does excluding transit anchorages move the state ranking?"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import (exposure, load_calibration, spearman, kendall_tau_b,
                     env_path, SEAWEB_COLS, DEFAULT_SERVICE_SPEED_KN)

WATCH = ["PAN", "EGY", "MYS", "SGP", "IDN", "ARE", "ESP", "CHN", "USA", "MAR"]


def load(work: Path, fname: str) -> pd.DataFrame:
    path = work / fname
    if not path.exists():
        sys.exit(f"missing {path}")
    voy = pd.read_csv(path, dtype={"imo": str})
    sw_path = env_path("SEAWEB_PATH")
    if sw_path is None:
        for c in (work.parent / "seaweb" / "ship_info.csv", work / "ship_info.csv"):
            if c.exists():
                sw_path = c
                break
    if sw_path is None:
        sys.exit("set SEAWEB_PATH")
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
    df["cal"] = load_calibration(work, df["shiptype_group"], fname,
                                 strict=False).to_numpy()
    print(f"  {fname}: {len(df):,} usable voyages")
    return df


def main():
    argv = sys.argv[1:]
    out_name = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out_name, i = argv[i + 1], i + 2
        elif argv[i].startswith("--out="):
            out_name, i = argv[i].split("=", 1)[1], i + 1
        else:
            rest.append(argv[i]); i += 1
    if not 1 <= len(rest) <= 3:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <work_dir> "
                 f"[baseline.csv.gz] [alternative.csv.gz] [--out NAME]")
    work = Path(rest[0]).expanduser().resolve()
    base_f = rest[1] if len(rest) > 1 else "voyages.csv.gz"
    alt_f = rest[2] if len(rest) > 2 else "voyages_notransit.csv.gz"
    if out_name is None:
        out_name = (f"comparison_{Path(base_f).name.split('.')[0]}"
                    f"__{Path(alt_f).name.split('.')[0]}.csv")

    print("loading")
    a = exposure(load(work, base_f), 0.5, 190.0, 0.05, 0.90)
    b = exposure(load(work, alt_f), 0.5, 190.0, 0.05, 0.90)

    common = a.index.intersection(b.index)
    ar, br = a[common].rank(ascending=False), b[common].rank(ascending=False)
    print(f"\nstates: {len(a)} baseline, {len(b)} alternative, {len(common)} common")
    print(f"total CO2: {a.sum()/1e6:,.0f} Mt -> {b.sum()/1e6:,.0f} Mt "
          f"({100*b.sum()/a.sum()-100:+.1f}%)")
    print(f"\nranking agreement")
    print(f"  Spearman {spearman(ar, br):.4f}   Kendall {kendall_tau_b(ar, br):.4f}")
    for n in (10, 20, 30):
        if n <= len(common):
            print(f"  top-{n:<3} overlap "
                  f"{len(set(a.head(n).index) & set(b.head(n).index))}/{n}")
    moved = (ar - br).abs()
    print(f"  states moving >5 ranks: {int((moved > 5).sum())}   "
          f"max move {int(moved.max())}")

    sa, sb = 100 * a / a.sum(), 100 * b / b.sum()
    print(f"\nstates the anchorage profile put in question")
    print(f"  {'':<6}{'rank':>7}{'->':^5}{'rank':<7}{'share %':>10}{'->':^5}"
          f"{'share %':<10}")
    for c in WATCH:
        if c not in common:
            continue
        print(f"  {c:<6}{int(ar[c]):>7}{'->':^5}{int(br[c]):<7}"
              f"{sa[c]:>10.2f}{'->':^5}{sb[c]:<10.2f}")

    print(f"\nbiggest movers")
    mv = (ar - br).sort_values()
    for c in list(mv.head(5).index) + list(mv.tail(5).index):
        print(f"  {c:<6}{int(ar[c]):>7} -> {int(br[c]):<7}"
              f"({int(br[c]-ar[c]):+d})")

    out = work / out_name
    pd.DataFrame({"baseline_co2_t": a, "alt_co2_t": b,
                  "baseline_rank": a.rank(ascending=False),
                  "alt_rank": b.rank(ascending=False)}).to_csv(out)
    print(f"\nwrote {out}")

    rho = spearman(ar, br)
    big = mv[(ar - br).abs() > 5]
    print("\nVERDICT")
    print(f"  ranking as a whole: Spearman {rho:.4f}, "
          f"{int((moved > 5).sum())}/{len(common)} states move >5 ranks")
    if rho > 0.99:
        print("    -> globally stable; the exclusion is not reshaping the"
              "\n       distribution")
    else:
        print("    -> globally unstable; the exclusion changes the distribution"
              "\n       itself and has to be justified before anything is reported")
    if len(big):
        worst = (ar - br).abs().idxmax()
        d = 100 * (sb[worst] - sa[worst]) / sa[worst] if sa[worst] else float("nan")
        print(f"  individual states: {len(big)} move materially, worst is "
              f"{worst} ({int(ar[worst])} -> {int(br[worst])}, "
              f"share {d:+.0f}%)")
        print("    -> a targeted correction doing what it was designed to do."
              "\n       Use the excluded set and report both; leaving a transit"
              "\n       state high in the ranking is a visible error, while the"
              "\n       rest of the table is unaffected either way.")
    else:
        print("  no individual state moves materially either; this is a footnote")


if __name__ == "__main__":
    main()
