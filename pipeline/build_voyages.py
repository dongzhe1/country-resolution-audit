#!/usr/bin/env python3
"""Turn GFW port-visit events into voyages."""

from __future__ import annotations

import gzip
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_SEA_HOURS = 1.0
MAX_SEA_HOURS = 24 * 120
MAX_IMPLIED_KN = 30.0
MIN_DISTANCE_NM = 1.0

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)

USE_COLS = ["imo", "start", "end", "vessel_flag",
            "start_anchorage_id", "start_anchorage_flag",
            "end_anchorage_id", "end_anchorage_flag", "lat", "lon"]


def haversine_nm(lat1, lon1, lat2, lon2):
    r = 3440.065
    p1, p2 = map(math.radians, (lat1, lat2))
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def haversine_nm_vec(lat1, lon1, lat2, lon2):
    r = 3440.065
    lat1, lon1 = np.asarray(lat1, float), np.asarray(lon1, float)
    lat2, lon2 = np.asarray(lat2, float), np.asarray(lon2, float)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _read_shard(path):
    df = pd.read_csv(path, usecols=USE_COLS, dtype=str, low_memory=False)
    df = df[df["imo"] != "imo"].copy()
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def to_utc(col):
    try:
        return pd.to_datetime(col, format="ISO8601", errors="coerce", utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(col, errors="coerce", utc=True)


def parse_args(argv, name):
    jobs, drop, out, rest, i = DEFAULT_JOBS, None, "voyages.csv.gz", [], 0
    routes = None
    while i < len(argv):
        a = argv[i]
        if a == "--jobs":
            jobs, i = int(argv[i + 1]), i + 2
        elif a.startswith("--jobs="):
            jobs, i = int(a.split("=", 1)[1]), i + 1
        elif a == "--drop-anchorages":
            drop, i = Path(argv[i + 1]).expanduser().resolve(), i + 2
        elif a.startswith("--drop-anchorages="):
            drop, i = Path(a.split("=", 1)[1]).expanduser().resolve(), i + 1
        elif a == "--routes":
            routes, i = Path(argv[i + 1]).expanduser().resolve(), i + 2
        elif a.startswith("--routes="):
            routes, i = Path(a.split("=", 1)[1]).expanduser().resolve(), i + 1
        elif a == "--out":
            out, i = argv[i + 1], i + 2
        elif a.startswith("--out="):
            out, i = a.split("=", 1)[1], i + 1
        else:
            rest.append(a); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <gfw_port_visits_dir> [--jobs N] "
                 f"[--drop-anchorages FILE] [--routes FILE] [--out NAME]")
    return Path(rest[0]).expanduser().resolve(), max(1, jobs), drop, routes, out


_DF = None


def _set_df(df):
    global _DF
    _DF = df


def _build_part(k):
    d = _DF[_DF["_part"] == k]
    if d.empty:
        return None
    d = d.sort_values(["imo", "start"], kind="mergesort")
    g = d.groupby("imo", sort=False)
    nxt = pd.DataFrame({
        "arr_time": g["start"].shift(-1),
        "arr_port": g["start_anchorage_id"].shift(-1),
        "arr_country": g["start_anchorage_flag"].shift(-1),
        "arr_lat": g["lat"].shift(-1),
        "arr_lon": g["lon"].shift(-1),
    }, index=d.index)
    v = pd.concat([
        d[["imo", "vessel_flag", "end", "end_anchorage_id",
           "end_anchorage_flag", "lat", "lon"]].rename(columns={
               "end": "dep_time", "end_anchorage_id": "dep_port",
               "end_anchorage_flag": "dep_country",
               "lat": "dep_lat", "lon": "dep_lon"}),
        nxt,
    ], axis=1).dropna(subset=["arr_time"])
    if v.empty:
        return None

    v["sea_hours"] = (v["arr_time"] - v["dep_time"]).dt.total_seconds() / 3600.0
    v["distance_nm"] = haversine_nm_vec(
        v["dep_lat"].to_numpy(), v["dep_lon"].to_numpy(),
        v["arr_lat"].to_numpy(), v["arr_lon"].to_numpy())
    v["implied_speed_kn"] = v["distance_nm"] / v["sea_hours"].where(v["sea_hours"] > 0)
    return v


def main():
    (work, jobs, drop_file, routes_file,
     out_name) = parse_args(sys.argv[1:], Path(sys.argv[0]).name)
    shards = sorted(work.glob("port_visits_*.csv.gz"))
    if not shards:
        sys.exit(f"no port_visits_*.csv.gz in {work}")

    read_jobs = max(1, min(jobs, len(shards)))
    print(f"reading {len(shards)} shard(s) on {read_jobs} process(es)")
    if read_jobs > 1:
        with mp.Pool(read_jobs) as pool:
            frames = pool.map(_read_shard, shards)
    else:
        frames = [_read_shard(s) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"  {len(df):,} port visits, {df['imo'].nunique():,} vessels")

    if drop_file is not None:
        ids = {l.strip() for l in drop_file.read_text().split() if l.strip()}
        before = len(df)
        df = df[~df["end_anchorage_id"].astype(str).isin(ids)
                & ~df["start_anchorage_id"].astype(str).isin(ids)]
        print(f"  dropped {before - len(df):,} visits at {len(ids):,} transit "
              f"anchorages ({100*(before-len(df))/max(before,1):.1f}%)")

    df["start"] = to_utc(df["start"])
    df["end"] = to_utc(df["end"])
    df = df.dropna(subset=["start", "end", "lat", "lon"]).reset_index(drop=True)

    build_jobs = max(1, min(jobs, df["imo"].nunique()))
    df["_part"] = pd.factorize(df["imo"])[0] % build_jobs
    print(f"building voyages on {build_jobs} process(es) "
          f"({df['imo'].nunique():,} vessels)")

    if build_jobs > 1:
        with mp.Pool(build_jobs, initializer=_set_df, initargs=(df,)) as pool:
            parts = pool.map(_build_part, range(build_jobs))
    else:
        _set_df(df)
        parts = [_build_part(0)]
    parts = [x for x in parts if x is not None]
    if not parts:
        sys.exit("no voyages built")
    v = pd.concat(parts, ignore_index=True)
    del parts
    _set_df(None)
    del df
    v = v.sort_values(["imo", "dep_time"], kind="mergesort").reset_index(drop=True)

    v["distance_gc_nm"] = v["distance_nm"]
    if routes_file is not None:
        rt = pd.read_csv(routes_file, dtype={"dep_port": str, "arr_port": str})
        usable = rt[(rt["status"] == "ok") & rt["routed_nm"].notna()
                    & (rt["routed_nm"] > 0)]
        v = v.merge(usable[["dep_port", "arr_port", "routed_nm"]],
                    on=["dep_port", "arr_port"], how="left")
        hit = v["routed_nm"].notna()
        v.loc[hit, "distance_nm"] = v.loc[hit, "routed_nm"]
        v["distance_source"] = np.where(hit, "routed", "great_circle")
        v = v.drop(columns=["routed_nm"])
        v["implied_speed_kn"] = (v["distance_nm"]
                                 / v["sea_hours"].where(v["sea_hours"] > 0))
        print(f"  routed distance applied to {int(hit.sum()):,} of {len(v):,} "
              f"voyages ({100*hit.mean():.1f}%); the rest keep the great circle")
        print(f"  median implied speed: "
              f"{(v['distance_gc_nm'] / v['sea_hours']).median():.1f} kn "
              f"great-circle -> {v['implied_speed_kn'].median():.1f} kn routed")
    else:
        v["distance_source"] = "great_circle"

    n0 = len(v)
    reasons = {}
    keep = pd.Series(True, index=v.index)
    for label, mask in [
        ("sea_hours_too_short", v["sea_hours"] < MIN_SEA_HOURS),
        ("sea_hours_too_long", v["sea_hours"] > MAX_SEA_HOURS),
        ("distance_too_short", v["distance_nm"] < MIN_DISTANCE_NM),
        ("implied_speed_too_high", v["implied_speed_kn"] > MAX_IMPLIED_KN),
    ]:
        m = mask.fillna(True) & keep
        reasons[label] = int(m.sum())
        keep &= ~m
    v = v[keep]

    out = work / out_name
    cols = ["imo", "vessel_flag", "dep_port", "dep_country", "dep_time",
            "arr_port", "arr_country", "arr_time", "sea_hours",
            "distance_nm", "distance_gc_nm", "distance_source",
            "implied_speed_kn"]
    v[cols].to_csv(out, index=False, compression="gzip")

    print(f"\nvoyages built: {len(v):,} (from {n0:,} candidate legs)")
    for k, n in reasons.items():
        print(f"  dropped {n:>9,}  {k}")
    print(f"\n  vessels with >=1 voyage : {v['imo'].nunique():,}")
    print(f"  distinct departure ports: {v['dep_port'].nunique():,}")
    print(f"  distinct countries      : "
          f"{pd.concat([v['dep_country'], v['arr_country']]).nunique():,}")
    print(f"  median distance (nm)    : {v['distance_nm'].median():,.0f}")
    print(f"  median implied speed    : {v['implied_speed_kn'].median():.1f} kn")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
