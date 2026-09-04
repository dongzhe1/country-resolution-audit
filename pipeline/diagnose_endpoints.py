#!/usr/bin/env python3
"""Does a port visit start and end in the same place, and does it matter?

    python diagnose_endpoints.py <work_dir> [--jobs N]
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd


def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)
COLS = ["imo", "start", "end",
        "start_anchorage_id", "start_anchorage_flag",
        "end_anchorage_id", "end_anchorage_flag"]
REF = _find_reference()

NEGLIGIBLE = 0.001


def _read(path):
    df = pd.read_csv(path, usecols=COLS, dtype=str, low_memory=False)
    return df[df["end_anchorage_flag"].notna()
              & df["start_anchorage_flag"].notna()]


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


def resolution_status():
    """iso3 -> True if the assessment gave the state a row of its own."""
    import csv
    gtap, m49 = REF / "gtap_regions_mepc82.csv", REF / "country_status.csv"
    if not (gtap.exists() and m49.exists()):
        return {}
    resolved = {r["iso3"] for r in csv.DictReader(open(gtap))
                if r.get("iso3") and r.get("resolution") == "individual"}
    universe = {r["iso3"] for r in csv.DictReader(open(m49)) if r["iso3"]}
    return {iso: iso in resolved for iso in universe | resolved}


def record(work, visits, mismatch, crossing, to_res, to_unres):
    """Write the figures Methods quotes, so nobody retypes them from a log."""
    from facts import emit
    emit(work, "endpoints", {
        "visits": int(visits),
        "mismatch_pct": round(100 * mismatch, 2),
        "crossing_pct": round(100 * crossing, 2),
        "to_resolved": int(to_res),
        "to_unresolved": int(to_unres),
    })


def main():
    work, jobs = parse_args(sys.argv[1:], Path(sys.argv[0]).name)
    shards = sorted(work.glob("port_visits_*.csv.gz"))
    if not shards:
        sys.exit(f"no port_visits_*.csv.gz under {work}")
    jobs = max(1, min(jobs, len(shards)))
    print(f"reading {len(shards)} shard(s) on {jobs} process(es)")
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            frames = pool.map(_read, shards)
    else:
        frames = [_read(s) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    del frames

    n = len(df)
    diff_anch = df["start_anchorage_id"] != df["end_anchorage_id"]
    diff_flag = df["start_anchorage_flag"] != df["end_anchorage_flag"]
    print(f"\n{n:,} port visits with a country at both ends")
    print(f"  start and end anchorage differ: {diff_anch.sum():,} "
          f"({diff_anch.mean():.4%})")
    print(f"  start and end COUNTRY differ:   {diff_flag.sum():,} "
          f"({diff_flag.mean():.4%})   <- the ones that misattribute")

    df = df.sort_values(["imo", "start"], kind="mergesort")
    arriving = df["imo"].duplicated(keep="first")
    mis = (df["start_anchorage_flag"] != df["end_anchorage_flag"]) & arriving
    print(f"  of which terminate a voyage:    {mis.sum():,} "
          f"({mis.sum() / max(arriving.sum(), 1):.4%} of arrivals)")

    if not mis.any():
        print("\nVERDICT: no visit ends in a different country from the one it "
              "began in. The arrival side of build_voyages.py is wrong in form "
              "and right in every instance. Record this and move on.")
        record(work, n, 0.0, 0.0, 0, 0)
        return

    pairs = (df[mis].groupby(["start_anchorage_flag", "end_anchorage_flag"])
             .size().sort_values(ascending=False))
    print(f"\nWhere the two ends disagree ({len(pairs)} country pairs), "
          "top 20 by visits:")
    print("  arrived in -> charged to        visits")
    for (a, b), k in pairs.head(20).items():
        print(f"  {a:>10} -> {b:<10}  {k:>12,}")

    status = resolution_status()
    if not status:
        print("\nreference/gtap_regions_mepc82.csv not found; "
              "skipping the resolved/unresolved split")
        return

    buckets = {}
    unknown = 0
    for (a, b), k in pairs.items():
        if a not in status or b not in status:
            unknown += k
            continue
        buckets[(status[a], status[b])] = buckets.get((status[a], status[b]), 0) + k
    tot = sum(buckets.values())
    name = {True: "resolved", False: "unresolved"}
    print(f"\nAcross the comparison the write-up rests on "
          f"({tot:,} classifiable visits, {unknown:,} with a state "
          f"outside the assessment's universe):")
    crossing = 0
    for (a, b), k in sorted(buckets.items(), key=lambda kv: -kv[1]):
        mark = ""
        if a != b:
            crossing += k
            mark = "   <- crosses the boundary"
        print(f"  arrived {name[a]:>10} -> charged {name[b]:<10} "
              f"{k:>12,}{mark}")

    up = [(a, b, k) for (a, b), k in pairs.items()
          if a in status and b in status and not status[a] and status[b]]
    if up:
        print("\n  the crossing flow, unresolved -> resolved, top 10:")
        for a, b, k in sorted(up, key=lambda t: -t[2])[:10]:
            print(f"    {a:>10} -> {b:<10} {k:>12,}")

    to_resolved = sum(k for (a, b), k in pairs.items()
                      if a in status and b in status and not status[a] and status[b])
    to_unresolved = sum(k for (a, b), k in pairs.items()
                        if a in status and b in status and status[a] and not status[b])
    share_all = mis.sum() / max(arriving.sum(), 1)
    share_cross = crossing / max(arriving.sum(), 1)

    record(work, n, share_all, share_cross, to_resolved, to_unresolved)
    print(f"\n  misattributed arrivals:            {share_all:.4%}")
    print(f"  of those, crossing the boundary:   {share_cross:.4%} of arrivals")

    print()
    if share_cross < NEGLIGIBLE:
        print(f"VERDICT: below {NEGLIGIBLE:.1%} of arrivals cross the "
              "resolved/unresolved boundary. The defect is real but cannot "
              "move the published comparison. Report the rate in Methods, "
              "do not rebuild.")
    else:
        print(f"VERDICT: {share_cross:.2%} of arrivals cross the boundary, "
              f"above the {NEGLIGIBLE:.1%} threshold. Rebuild the arrival side "
              "of build_voyages.py to use the next visit's START anchorage, "
              "then re-run the pipeline. Check the direction first: if the "
              "flow is mostly unresolved -> resolved, the current numbers "
              "UNDERSTATE the write-up's finding.")


if __name__ == "__main__":
    main()
