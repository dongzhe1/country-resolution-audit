#!/usr/bin/env python3
"""Replace great-circle distance with a routed sea distance."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_voyages import haversine_nm_vec

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)
POS_COLS = ["end_anchorage_id", "lat", "lon"]
CHUNK = 2000


def _positions(path):
    df = pd.read_csv(path, usecols=POS_COLS, dtype=str, low_memory=False)
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


def _route_chunk(args):
    import searoute as sr
    out = []
    for key, alat, alon, blat, blon in args:
        try:
            r = sr.searoute((alon, alat), (blon, blat), units="naut")
            out.append((key, float(r.properties["length"]), "ok"))
        except Exception as exc:
            out.append((key, float("nan"), type(exc).__name__))
    return out


def parse_args(argv, name):
    voy, jobs, rest, i = "voyages_final.csv.gz", DEFAULT_JOBS, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--voyages":
            voy, i = argv[i + 1], i + 2
        elif a.startswith("--voyages="):
            voy, i = a.split("=", 1)[1], i + 1
        elif a == "--jobs":
            jobs, i = int(argv[i + 1]), i + 2
        elif a.startswith("--jobs="):
            jobs, i = int(a.split("=", 1)[1]), i + 1
        else:
            rest.append(a); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <work_dir> [--voyages NAME] [--jobs N]")
    return Path(rest[0]).expanduser().resolve(), voy, max(1, jobs)


def main():
    try:
        import searoute
    except ImportError:
        sys.exit("searoute is not installed.\n"
                 "  pip install searoute\n"
                 "It bundles the marine network, so nothing else is needed.")

    work, voy_name, jobs = parse_args(sys.argv[1:], Path(sys.argv[0]).name)

    shards = sorted(work.glob("port_visits_*.csv.gz"))
    read_jobs = max(1, min(jobs, len(shards)))
    print(f"anchorage positions from {len(shards)} shard(s)")
    if read_jobs > 1:
        with mp.Pool(read_jobs) as pool:
            frames = pool.map(_positions, shards)
    else:
        frames = [_positions(s) for s in shards]
    pos = (pd.concat(frames, ignore_index=True)
             .groupby("end_anchorage_id")[["lat", "lon"]].median())
    del frames
    print(f"  {len(pos):,} anchorages located")

    v = pd.read_csv(work / voy_name, usecols=["dep_port", "arr_port"], dtype=str)
    pairs = (v.dropna().drop_duplicates().reset_index(drop=True))
    print(f"{len(v):,} voyages -> {len(pairs):,} distinct anchorage pairs")

    pairs = pairs.join(pos.rename(columns={"lat": "dep_lat", "lon": "dep_lon"}),
                       on="dep_port")
    pairs = pairs.join(pos.rename(columns={"lat": "arr_lat", "lon": "arr_lon"}),
                       on="arr_port")
    have = pairs[["dep_lat", "dep_lon", "arr_lat", "arr_lon"]].notna().all(axis=1)
    print(f"  {int(have.sum()):,} pairs have both endpoints located "
          f"({int((~have).sum()):,} dropped)")
    pairs = pairs[have].reset_index(drop=True)

    pairs["gc_nm"] = haversine_nm_vec(
        pairs["dep_lat"].to_numpy(), pairs["dep_lon"].to_numpy(),
        pairs["arr_lat"].to_numpy(), pairs["arr_lon"].to_numpy())

    work_items = [(i, r.dep_lat, r.dep_lon, r.arr_lat, r.arr_lon)
                  for i, r in enumerate(pairs.itertuples())]
    chunks = [work_items[i:i + CHUNK] for i in range(0, len(work_items), CHUNK)]
    print(f"routing {len(work_items):,} pairs in {len(chunks):,} chunks "
          f"on {jobs} process(es)")

    results = {}
    status = {}
    with mp.Pool(jobs) as pool:
        for n, out in enumerate(pool.imap_unordered(_route_chunk, chunks), 1):
            for key, nm, st in out:
                results[key] = nm
                status[key] = st
            if n % max(1, len(chunks) // 20) == 0:
                print(f"  {n}/{len(chunks)} chunks "
                      f"({100*n/len(chunks):.0f}%)", flush=True)

    pairs["routed_nm"] = [results.get(i, float("nan")) for i in range(len(pairs))]
    pairs["status"] = [status.get(i, "missing") for i in range(len(pairs))]
    ok = pairs["routed_nm"].notna() & (pairs["routed_nm"] > 0)
    pairs["ratio"] = pairs["routed_nm"] / pairs["gc_nm"].where(pairs["gc_nm"] > 0)

    print(f"\nrouted {int(ok.sum()):,} of {len(pairs):,} "
          f"({100*ok.mean():.1f}%)")
    if (~ok).any():
        print("  failures by reason: "
              + ", ".join(f"{k}={v:,}" for k, v in
                          pairs.loc[~ok, "status"].value_counts().items()))
    r = pairs.loc[ok, "ratio"]
    print(f"\nrouted / great-circle")
    for q in (.05, .25, .5, .75, .95):
        print(f"  p{int(q*100):02d}  {r.quantile(q):5.2f}")
    print(f"  mean {r.mean():.2f}   weighted by nothing yet -- the voyage-level"
          f"\n  effect follows from applying this table, not from this row")
    bad = int((r < 0.98).sum())
    if bad:
        print(f"  !! {bad:,} pairs route SHORTER than the great circle; "
              f"endpoint snapping, treat as unrouted")
        pairs.loc[pairs["ratio"] < 0.98, "status"] = "shorter_than_gc"

    out = work / "route_distances.csv"
    pairs.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"\nnext: python build_voyages.py {work} --routes {out} "
          f"--drop-anchorages {work}/transit_anchorage_ids_final.txt "
          f"--out voyages_routed.csv.gz")


if __name__ == "__main__":
    main()
