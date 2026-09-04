#!/usr/bin/env python3
"""Are canal and strait waiting anchorages being counted as port calls?

    python diagnose_anchorages.py <work_dir> [--jobs N]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)
COLS = ["end_anchorage_id", "end_anchorage_name", "end_anchorage_flag",
        "end_at_dock"]
SUSPECT = ["MYS", "PAN", "EGY", "SGP", "CHN", "USA", "ESP", "IDN", "ARE", "MAR"]


def _read(path):
    df = pd.read_csv(path, usecols=COLS, dtype=str)
    return df[df["end_anchorage_flag"].notna()]


def truthy(col: pd.Series) -> pd.Series:
    """GFW returns atDock as a JSON bool; the CSV round-trip makes it a string."""
    return col.astype(str).str.strip().str.lower().isin(("true", "1", "t", "yes"))


def parse_args(argv, name):
    jobs, rest, i = DEFAULT_JOBS, [], 0
    while i < len(argv):
        if argv[i] == "--jobs":
            jobs, i = int(argv[i + 1]), i + 2
        elif argv[i].startswith("--jobs="):
            jobs, i = int(argv[i].split("=", 1)[1]), i + 1
        else:
            rest.append(argv[i]); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <work_dir> [--jobs N]")
    return Path(rest[0]).expanduser().resolve(), max(1, jobs)


def main():
    work, jobs = parse_args(sys.argv[1:], Path(sys.argv[0]).name)
    shards = sorted(work.glob("port_visits_*.csv.gz"))
    jobs = max(1, min(jobs, len(shards)))
    print(f"reading {len(shards)} shard(s) on {jobs} process(es)")
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            frames = pool.map(_read, shards)
    else:
        frames = [_read(s) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    del frames
    df["at_dock"] = truthy(df["end_at_dock"])
    print(f"{len(df):,} port visits with a country\n")

    by_c = df.groupby("end_anchorage_flag").agg(
        visits=("at_dock", "size"), at_dock=("at_dock", "mean"))
    overall = df["at_dock"].mean()
    print(f"at-dock share overall: {overall:.1%}\n")

    top = by_c.sort_values("visits", ascending=False).head(25).copy()
    top["vs_overall"] = top["at_dock"] / overall
    print("top 25 countries by port visits")
    print(f"{'':<6}{'visits':>12}{'at_dock':>10}{'x overall':>11}")
    for k, r in top.iterrows():
        flag = "   <-- suspect" if r["at_dock"] < 0.5 * overall else ""
        print(f"{str(k):<6}{int(r['visits']):>12,}{r['at_dock']:>10.1%}"
              f"{r['vs_overall']:>11.2f}{flag}")

    print("\nbusiest anchorages in the states the ranking flags")
    for c in SUSPECT:
        d = df[df["end_anchorage_flag"] == c]
        if d.empty:
            continue
        g = (d.groupby(["end_anchorage_id", "end_anchorage_name"])
               .agg(visits=("at_dock", "size"), at_dock=("at_dock", "mean"))
               .sort_values("visits", ascending=False).head(5))
        print(f"\n  {c}  ({len(d):,} visits, {d['at_dock'].mean():.1%} at dock)")
        for (aid, nm), r in g.iterrows():
            print(f"    {str(nm)[:34]:<36}{int(r['visits']):>9,}"
                  f"{r['at_dock']:>9.1%}")

    out = work / "anchorage_profile.csv"
    by_c.sort_values("visits", ascending=False).to_csv(out)
    print(f"\nwrote {out}")
    print("\nA transit state should show a LOW at-dock share concentrated in one\n"
          "or two named anchorages. If it does, build_voyages.py must either\n"
          "require at_dock or treat a non-dock visit as a waypoint that does not\n"
          "terminate a voyage -- and the ranking has to be recomputed.")


if __name__ == "__main__":
    main()
