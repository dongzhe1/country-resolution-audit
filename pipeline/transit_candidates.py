#!/usr/bin/env python3
"""List GFW anchorages that look like transit/waiting nodes, not trade nodes.

    python transit_candidates.py <work_dir> [--jobs N]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)
COLS = ["end_anchorage_id", "end_anchorage_name", "end_anchorage_flag",
        "end_at_dock", "duration_hrs"]

TRANSIT_RE = re.compile(
    r"\b(?:ANCHORAGE|ANCH|ROADS|ROADSTEAD|OPL|LIGHTERING|"
    r"CANAL|LOCKS|WAITING|HOLDING|OFFSHORE\s+TERMINAL)\b", re.I)

NAME_EXCEPTIONS = {("USA", "ANCHORAGE")}


def _read(path):
    df = pd.read_csv(path, usecols=COLS, dtype=str, low_memory=False)
    df["duration_hrs"] = pd.to_numeric(df["duration_hrs"], errors="coerce")
    return df[df["end_anchorage_flag"].notna()]


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
    print(f"{len(df):,} visits\n")

    g = (df.groupby(["end_anchorage_id", "end_anchorage_name",
                     "end_anchorage_flag"], dropna=False)
           .agg(visits=("duration_hrs", "size"),
                median_hrs=("duration_hrs", "median"))
           .reset_index()
           .rename(columns={"end_anchorage_id": "anchorage_id",
                            "end_anchorage_name": "name",
                            "end_anchorage_flag": "country"}))

    nm = g["name"].fillna("")
    g["transit"] = nm.str.contains(TRANSIT_RE)
    exempt = [(c, str(n).strip().upper()) in NAME_EXCEPTIONS
              for c, n in zip(g["country"], nm)]
    g.loc[exempt, "transit"] = False

    total = g["visits"].sum()
    flagged = g[g["transit"]].sort_values("visits", ascending=False)
    print(f"anchorages: {len(g):,}   flagged as transit: {len(flagged):,}")
    print(f"visits at flagged anchorages: {flagged['visits'].sum():,} "
          f"({100*flagged['visits'].sum()/total:.1f}% of all visits)")
    print(f"median dwell -- flagged {flagged['median_hrs'].median():.1f} h  "
          f"vs others {g[~g['transit']]['median_hrs'].median():.1f} h")

    print(f"\ntop 30 flagged anchorages")
    print(f"  {'country':<9}{'visits':>10}{'med h':>8}  name")
    for _, r in flagged.head(30).iterrows():
        print(f"  {str(r['country']):<9}{int(r['visits']):>10,}"
              f"{r['median_hrs']:>8.1f}  {str(r['name'])[:44]}")

    print(f"\nshare of each country's visits that are flagged (top 15 affected)")
    g["flagged_visits"] = g["visits"].where(g["transit"], 0)
    per = (g.groupby("country", as_index=False)
             .agg(visits=("visits", "sum"), flagged=("flagged_visits", "sum")))
    per = per[per["visits"] >= 5000]
    per["pct"] = 100 * per["flagged"] / per["visits"]
    for _, r in per.sort_values("pct", ascending=False).head(15).iterrows():
        print(f"  {r['country']:<6}{int(r['visits']):>10,} visits  "
              f"{int(r['flagged']):>9,} flagged  {r['pct']:>5.1f}%")

    out = work / "transit_candidates.csv"
    g.sort_values("visits", ascending=False).to_csv(out, index=False)
    ids = work / "transit_anchorage_ids.txt"
    ids.write_text("\n".join(flagged["anchorage_id"].dropna().astype(str)) + "\n")
    print(f"\nwrote {out}")
    print(f"wrote {ids}  ({len(flagged):,} ids)")
    print("\nReview the flagged names before using them. Then:"
          "\n  python build_voyages.py <work_dir> --drop-anchorages "
          "<work_dir>/transit_anchorage_ids.txt"
          "\nwhich removes those visits BEFORE voyages are built, so a voyage"
          "\nspans the waypoint instead of terminating at it. Rerun the ranking"
          "\nboth ways -- if it does not move, this is a footnote, as the ship-"
          "\ntype calibration turned out to be.")


if __name__ == "__main__":
    main()
