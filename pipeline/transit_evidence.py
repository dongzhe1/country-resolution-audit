#!/usr/bin/env python3
"""Separate waiting anchorages from anchorages where cargo is actually worked."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_voyages import haversine_nm_vec, to_utc
from transit_candidates import TRANSIT_RE, NAME_EXCEPTIONS

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)
COLS = ["imo", "start", "end", "duration_hrs", "lat", "lon",
        "end_anchorage_id", "end_anchorage_name", "end_anchorage_flag"]
NEAR_NM = 100.0
MIN_VISITS = 200


def _read(path):
    df = pd.read_csv(path, usecols=COLS, dtype=str, low_memory=False)
    for c in ("lat", "lon", "duration_hrs"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["imo"] != "imo"]


WAIT_SAME, WAIT_NEAR = 0.60, 0.50
THROUGH_DETOUR = 1.25
THROUGH_MAX_HRS = 48.0


def classify(r) -> str:
    if r["same_country"] >= WAIT_SAME and r["near"] >= WAIT_NEAR:
        return "WAIT"
    if r["detour"] > THROUGH_DETOUR:
        return "DEST"
    if (r["near"] < 0.35 and r["next_nm"] > 400
            and r["med_hrs"] <= THROUGH_MAX_HRS):
        return "THROUGH"
    return "UNCERTAIN"


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

    df["start"] = to_utc(df["start"])
    df = df.dropna(subset=["start", "lat", "lon", "end_anchorage_flag"])
    df = df.sort_values(["imo", "start"], kind="mergesort")
    print(f"{len(df):,} visits, {df['imo'].nunique():,} vessels\n")

    g = df.groupby("imo", sort=False)
    df["nxt_flag"] = g["end_anchorage_flag"].shift(-1)
    df["nxt_lat"] = g["lat"].shift(-1)
    df["nxt_lon"] = g["lon"].shift(-1)
    df["prv_lat"] = g["lat"].shift(1)
    df["prv_lon"] = g["lon"].shift(1)
    d = df.dropna(subset=["nxt_flag", "nxt_lat", "nxt_lon",
                          "prv_lat", "prv_lon"]).copy()
    lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
    nla, nlo = d["nxt_lat"].to_numpy(), d["nxt_lon"].to_numpy()
    pla, plo = d["prv_lat"].to_numpy(), d["prv_lon"].to_numpy()
    d["next_nm"] = haversine_nm_vec(lat, lon, nla, nlo)
    d["same_country"] = d["end_anchorage_flag"] == d["nxt_flag"]
    d["near"] = d["next_nm"] < NEAR_NM

    legs = haversine_nm_vec(pla, plo, lat, lon) + d["next_nm"].to_numpy()
    through = haversine_nm_vec(pla, plo, nla, nlo)
    d["detour"] = np.clip(legs / np.maximum(through, 10.0), 1.0, 50.0)

    a = (d.groupby(["end_anchorage_id", "end_anchorage_name", "end_anchorage_flag"],
                   dropna=False)
           .agg(visits=("next_nm", "size"),
                med_hrs=("duration_hrs", "median"),
                same_country=("same_country", "mean"),
                near=("near", "mean"),
                next_nm=("next_nm", "median"),
                detour=("detour", "median"))
           .reset_index()
           .rename(columns={"end_anchorage_id": "anchorage_id",
                            "end_anchorage_name": "name",
                            "end_anchorage_flag": "country"}))

    nm = a["name"].fillna("")
    a["flagged"] = nm.str.contains(TRANSIT_RE)
    exempt = [(c, str(n).strip().upper()) in NAME_EXCEPTIONS
              for c, n in zip(a["country"], nm)]
    a.loc[exempt, "flagged"] = False

    big = a[a["visits"] >= MIN_VISITS]
    ref = big[~big["flagged"]]
    print(f"reference: {len(ref):,} unflagged anchorages with >= {MIN_VISITS} visits")
    print(f"  same_country {ref['same_country'].median():.2f}   "
          f"near(<{NEAR_NM:.0f}nm) {ref['near'].median():.2f}   "
          f"next_nm {ref['next_nm'].median():,.0f}   "
          f"detour {ref['detour'].median():.2f}\n")

    fl = big[big["flagged"]].sort_values("visits", ascending=False)
    print(f"name-flagged anchorages with >= {MIN_VISITS} visits ({len(fl)})")
    print(f"  {'ctry':<5}{'visits':>9}{'med h':>8}{'same':>7}{'near':>7}"
          f"{'next nm':>9}{'detour':>8}  verdict  name")
    for _, r in fl.iterrows():
        print(f"  {str(r['country']):<5}{int(r['visits']):>9,}{r['med_hrs']:>8.1f}"
              f"{r['same_country']:>7.2f}{r['near']:>7.2f}{r['next_nm']:>9,.0f}"
              f"{r['detour']:>8.2f}  {classify(r):<7}  {str(r['name'])[:32]}")

    fl = fl.assign(verdict=fl.apply(classify, axis=1))
    total_visits = int(a["visits"].sum())

    print(f"\n{'verdict':<10}{'anchorages':>11}{'visits':>11}{'% of all':>10}")
    for v in ("WAIT", "THROUGH", "UNCERTAIN", "DEST"):
        d = fl[fl["verdict"] == v]
        print(f"{v:<10}{len(d):>11}{int(d['visits'].sum()):>11,}"
              f"{100*d['visits'].sum()/total_visits:>9.2f}%")

    auto = fl[fl["verdict"].isin(("WAIT", "THROUGH"))]
    small = a[(a["visits"] < MIN_VISITS) & a["flagged"]]
    drop_ids = set(auto["anchorage_id"].dropna().astype(str))
    drop_ids |= set(small["anchorage_id"].dropna().astype(str))
    out_ids = work / "transit_anchorage_ids_evidence.txt"
    out_ids.write_text("\n".join(sorted(drop_ids)) + "\n")

    unc = fl[fl["verdict"] == "UNCERTAIN"]
    (work / "transit_uncertain.csv").write_text(
        unc.to_csv(index=False))
    print(f"\n{len(unc)} anchorages need a human decision, "
          f"{int(unc['visits'].sum()):,} visits "
          f"({100*unc['visits'].sum()/total_visits:.2f}% of all) -- "
          f"immaterial either way,\nbut the choice belongs in the methods "
          f"section, not in a threshold:")
    for _, r in unc.sort_values("visits", ascending=False).iterrows():
        print(f"    {str(r['country']):<5}{int(r['visits']):>8,}  "
              f"detour {r['detour']:.2f}  {str(r['name'])[:38]}")

    a.sort_values("visits", ascending=False).to_csv(
        work / "transit_evidence.csv", index=False)
    print(f"\nname-only list: {int(fl['visits'].sum()):,} visits from large "
          f"anchorages\nevidence list: {int(auto['visits'].sum()):,} "
          f"(+ {len(small):,} small ones below the test threshold)")
    print(f"\nwrote {work / 'transit_evidence.csv'}")
    print(f"wrote {out_ids}  ({len(drop_ids):,} ids)")
    print("\nWAIT       the ship proceeds into the adjacent port -- drop"
          "\nTHROUGH    on the route, next call far, ship did not put in"
          "\n           anywhere near (canal, bunkering) -- drop"
          "\nDEST       large detour: the ship went out of its way and came"
          "\n           back. Cargo was worked here -- KEEP; dropping it"
          "\n           understates the state"
          "\nUNCERTAIN  detour ~1 like an ordinary port (reference median"
          "\n           1.10), but no positive evidence either way -- decide"
          "\n           by name and record the reason"
          "\n\nCheck the DEST rows by name before accepting the list. Small"
          "\nanchorages below the visit threshold stay in the drop list: too"
          "\nrare to test, too small to matter.")


if __name__ == "__main__":
    main()
