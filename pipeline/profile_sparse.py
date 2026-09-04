#!/usr/bin/env python3
"""Characterise the vessels GFW reports almost no port visits for.

    python profile_sparse.py <work_dir>
"""

from __future__ import annotations

import collections
import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)

SPARSE_MAX_ROWS = 1
SEAWEB_COLS = ["lrnoimo_ship_no", "gross_tonnage", "deadweight", "shiptype_group",
               "year_of_build", "flag_name"]
MIN_GROUP = 30


def env_path(var: str):
    """An explicit path beats guessing: the GFW data and the reference data do not live under a common parent on this"""
    v = os.environ.get(var, "").strip()
    if not v:
        return None
    p = Path(v).expanduser().resolve()
    if not p.exists():
        sys.exit(f"{var}={v} does not exist")
    return p


def _count_shard(path):
    try:
        col = pd.read_csv(path, usecols=["imo"], dtype=str)["imo"]
    except Exception as exc:
        print(f"  !! {path.name}: {type(exc).__name__}")
        return collections.Counter()
    col = col[col.notna() & (col != "imo")]
    return collections.Counter(col.value_counts().to_dict())


def count_rows(work: Path, jobs: int) -> collections.Counter:
    shards = sorted(work.glob("port_visits_*.csv.gz"))
    jobs = max(1, min(jobs, len(shards)))
    print(f"counting {len(shards)} shard(s) on {jobs} process(es)")
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            parts = pool.map(_count_shard, shards)
    else:
        parts = [_count_shard(s) for s in shards]
    counts = collections.Counter()
    for c in parts:
        counts.update(c)
    return counts


def rate_table(df: pd.DataFrame, by: str, overall: float) -> pd.DataFrame:
    g = df.groupby(by)["sparse"].agg(["size", "sum"])
    g = g[g["size"] >= MIN_GROUP].copy()
    g["rate"] = g["sum"] / g["size"]
    g["vs_overall"] = g["rate"] / overall
    return g.sort_values("rate", ascending=False)


def parse_args(argv, name):
    """<work_dir> [--jobs N]."""
    jobs, rest, i = DEFAULT_JOBS, [], 0
    while i < len(argv):
        if argv[i] == "--jobs":
            if i + 1 >= len(argv):
                sys.exit("--jobs needs a value")
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

    counts = count_rows(work, jobs)
    ids = pd.read_csv(work / "vessel_ids.csv", dtype=str)
    ids = ids[ids["vessel_id"].notna() & (ids["vessel_id"] != "")]

    df = pd.DataFrame({"imo": ids["imo"]})
    df["visits"] = df["imo"].map(counts).fillna(0).astype(int)
    df["sparse"] = df["visits"] <= SPARSE_MAX_ROWS

    sw = pd.read_csv(sw_path, usecols=SEAWEB_COLS,
                     dtype={"lrnoimo_ship_no": str}, low_memory=False)
    df = df.merge(sw, left_on="imo", right_on="lrnoimo_ship_no", how="left")

    overall = df["sparse"].mean()
    out = [f"vessels            {len(df):,}",
           f"sparse (<= {SPARSE_MAX_ROWS} visit)  {df['sparse'].sum():,}  ({overall:.1%})",
           f"matched to Sea-web {df['shiptype_group'].notna().sum():,}",
           ""]

    df["dwt_band"] = pd.cut(df["deadweight"],
                            [0, 5e3, 2e4, 6e4, 1e5, 2e5, 1e9],
                            labels=["<5k", "5-20k", "20-60k", "60-100k",
                                    "100-200k", ">200k"])
    df["age_band"] = pd.cut(df["year_of_build"],
                            [0, 1990, 2000, 2010, 2020, 2100],
                            labels=["<1990", "90s", "00s", "10s", "20s"])

    for by, title in (("shiptype_group", "ship type"), ("dwt_band", "deadweight"),
                      ("age_band", "year built"), ("flag_name", "flag")):
        t = rate_table(df, by, overall)
        if t.empty:
            continue
        out.append(f"--- sparsity by {by} ({title}) ---")
        out.append(f"{'':<28}{'n':>8}{'sparse':>9}{'rate':>9}{'x overall':>11}")
        head = (pd.concat([t.head(10), t.tail(5)])
                if by == "flag_name" and len(t) > 15 else t)
        for k, r in head.iterrows():
            out.append(f"{str(k)[:27]:<28}{int(r['size']):>8,}"
                       f"{int(r['sum']):>9,}{r['rate']:>9.1%}{r['vs_overall']:>11.2f}")
        out.append("")

    fl = rate_table(df, "flag_name", overall)
    if len(fl) >= 10:
        top = fl.head(10)["sum"].sum()
        out.append(f"share of all sparse vessels in the 10 sparsest flags: "
                   f"{top / df['sparse'].sum():.1%}")
        out.append("")
        out.append("How to read this: if every 'x overall' falls between 0.7 "
                   "and 1.4, sparsity is")
        out.append("  spread evenly, harms no comparison between states, and "
                   "belongs in the limitations.")
        out.append("  If particular flags or regions reach 2x or more AND "
                   "account for most of the")
        out.append("  sparse vessels, that is a coverage hole: it distorts "
                   "state-level attribution")
        out.append("  and has to be fixed rather than recorded.")

    txt = "\n".join(out)
    print(txt)
    (work / "sparse_profile.txt").write_text(txt + "\n")
    print(f"\nwrote {work / 'sparse_profile.txt'}")


if __name__ == "__main__":
    main()
