#!/usr/bin/env python3
"""Bundle every derived table the analysis needs into one small directory.

    python export_derived_tables.py <work_dir> [--out DIR] [--jobs N]
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_JOBS = int(os.environ.get("N_JOBS") or os.cpu_count() or 4)

COPY = [
    "state_exposure.csv",
    "resolution_gap.csv",
    "sensitivity_ranks.csv",
    "type_calibration.csv",
    "transit_evidence.csv",
    "comparison_calibration.csv",
    "comparison_transit.csv",
    "comparison_routing.csv",
    "anchorage_profile.csv",
    "country_indicators.csv",
    "development_gradient.csv",
    "disbursement.csv",
    "disbursement_summary.csv",
    "sparse_profile.txt",
    "representativeness.txt",
    "transit_anchorage_ids_final.txt",
]
REDUCE = ["mrv_validation.csv"]
ROUTE_RATIO_BINS = np.concatenate([np.arange(1.0, 2.0, 0.02),
                                   np.arange(2.0, 5.01, 0.1)])

SPEED_BINS = np.arange(0, 31, 0.5)
DIST_BINS = np.concatenate([np.arange(0, 1000, 25), np.arange(1000, 12001, 250)])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv, name):
    out, jobs, rest, i, no_tar = None, DEFAULT_JOBS, [], 0, False
    while i < len(argv):
        a = argv[i]
        if a == "--out":
            out, i = Path(argv[i + 1]), i + 2
        elif a.startswith("--out="):
            out, i = Path(a.split("=", 1)[1]), i + 1
        elif a == "--jobs":
            jobs, i = int(argv[i + 1]), i + 2
        elif a.startswith("--jobs="):
            jobs, i = int(a.split("=", 1)[1]), i + 1
        elif a == "--no-tar":
            no_tar, i = True, i + 1
        else:
            rest.append(a); i += 1
    if len(rest) != 1:
        sys.exit(f"usage: {name} <work_dir> [--out DIR] [--jobs N] [--no-tar]")
    work = Path(rest[0]).expanduser().resolve()
    return (work, (out or work / "derived_tables").expanduser().resolve(),
            max(1, jobs), no_tar)


def summarise_voyages(work: Path, out: Path, name: str, tag: str) -> dict:
    """Distributions and per-year counts, from the table that stays behind."""
    path = work / name
    if not path.exists():
        print(f"  !! {path.name} missing; voyage summaries skipped")
        return {}
    cols = ["imo", "dep_country", "arr_country", "dep_time",
            "sea_hours", "distance_nm", "implied_speed_kn"]
    v = pd.read_csv(path, usecols=cols, dtype=str, low_memory=False)
    for c in ("sea_hours", "distance_nm", "implied_speed_kn"):
        v[c] = pd.to_numeric(v[c], errors="coerce")
    v["year"] = pd.to_datetime(v["dep_time"], format="ISO8601",
                               errors="coerce", utc=True).dt.year
    print(f"  voyages: {len(v):,}")

    for name, col, bins in (("speed", "implied_speed_kn", SPEED_BINS),
                            ("distance", "distance_nm", DIST_BINS)):
        if col not in v:
            continue
        cnt, edges = np.histogram(v[col].dropna(), bins=bins)
        pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "count": cnt}).to_csv(
            out / f"voyage_{name}_hist{tag}.csv", index=False)

    (v.groupby("year")
      .agg(voyages=("distance_nm", "size"),
           vessels=("imo", "nunique"),
           median_distance_nm=("distance_nm", "median"),
           median_speed_kn=("implied_speed_kn", "median"),
           total_sea_hours=("sea_hours", "sum"))
      .to_csv(out / f"voyages_by_year{tag}.csv"))

    return {
        "voyages": int(len(v)),
        "vessels": int(v["imo"].nunique()),
        "median_distance_nm": float(v["distance_nm"].median()),
        "median_speed_kn": float(v["implied_speed_kn"].median()),
        "countries": int(pd.concat([v["dep_country"], v["arr_country"]]).nunique()),
    }


def reduce_mrv(work: Path, out: Path, name: str, tag: str) -> dict:
    path = work / name
    if not path.exists():
        print(f"  !! {path.name} missing")
        return {}
    m = pd.read_csv(path)
    print(f"  mrv ship-years: {len(m):,}")
    g = (m.groupby("grp")
           .agg(n=("ratio", "size"), median_ratio=("ratio", "median"),
                q25=("ratio", lambda x: x.quantile(.25)),
                q75=("ratio", lambda x: x.quantile(.75)),
                svc_kn=("svc_kn", "median"), one_ended=("one_ended", "median"),
                median_dwt=("dwt", "median"))
           .sort_values("median_ratio"))
    g[g["n"] >= 20].to_csv(out / f"mrv_by_shiptype{tag}.csv")
    if "dwt_decile" in m:
        (m.groupby("dwt_decile")
           .agg(n=("ratio", "size"), median_dwt=("dwt", "median"),
                median_ratio=("ratio", "median"))
           .to_csv(out / f"mrv_by_dwt_decile{tag}.csv"))
    cnt, edges = np.histogram(m["ratio"].clip(0, 3), bins=np.arange(0, 3.05, 0.05))
    pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "count": cnt}).to_csv(
        out / f"mrv_ratio_hist{tag}.csv", index=False)
    return {
        "ship_years": int(len(m)), "ships": int(m["imo"].nunique()),
        "median_ratio": float(m["ratio"].median()),
        "geometric_mean_ratio": float(np.exp(np.log(m["ratio"]).mean())),
        "total_est_mt": float(m["est_co2_t"].sum() / 1e6),
        "total_reported_mt": float(m["reported_co2_t"].sum() / 1e6),
    }


def main():
    work, out, jobs, no_tar = parse_args(sys.argv[1:], Path(sys.argv[0]).name)
    out.mkdir(parents=True, exist_ok=True)
    print(f"work {work}\nout  {out}\n")

    written: set[str] = set()
    facts = {}

    print("copying result tables")
    for name in COPY:
        src = work / name
        if src.exists():
            shutil.copy(src, out / name)
            written.add(name)
            print(f"  {name}")
        else:
            print(f"  !! {name} missing")

    for name, tag in (("voyages_routed.csv.gz", ""),
                      ("voyages_final.csv.gz", "_gc")):
        stem = name.split(".")[0]
        mrv_name = (f"mrv_validation_{stem}.csv"
                    if (work / f"mrv_validation_{stem}.csv").exists()
                    else "mrv_validation.csv")
        print(f"\nreducing MRV for {name} (from {mrv_name})")
        facts[f"mrv{tag or '_routed'}"] = reduce_mrv(work, out, mrv_name, tag)
        print(f"summarising {name}")
        facts[f"voyages{tag or '_routed'}"] = summarise_voyages(
            work, out, name, tag)
    facts["mrv"] = facts.get("mrv_routed", {})
    facts["voyages"] = facts.get("voyages_routed", {})

    rp = work / "route_distances.csv"
    if rp.exists():
        rd = pd.read_csv(rp, usecols=["gc_nm", "routed_nm", "ratio", "status"])
        ok = rd[(rd["status"] == "ok") & rd["ratio"].notna()]
        cnt, edges = np.histogram(ok["ratio"].clip(1.0, 5.0),
                                  bins=ROUTE_RATIO_BINS)
        pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "count": cnt}).to_csv(
            out / "route_ratio_hist.csv", index=False)
        facts["routing"] = {
            "pairs": int(len(rd)), "routed": int(len(ok)),
            "median_ratio": float(ok["ratio"].median()),
            "p95_ratio": float(ok["ratio"].quantile(.95)),
            "shorter_than_gc": int((rd["status"] == "shorter_than_gc").sum()),
        }
        print(f"\nrouting: {len(ok):,} of {len(rd):,} pairs, "
              f"median ratio {ok['ratio'].median():.2f}")

    print("\nextraction provenance")
    ids = work / "vessel_ids.csv"
    if ids.exists():
        d = pd.read_csv(ids, dtype=str)
        facts["extraction"] = {
            "imos_requested": int(len(d)),
            "imos_matched": int(d["vessel_id"].notna().sum()
                                - (d["vessel_id"] == "").sum()),
        }
    shards = sorted(work.glob("port_visits_*.csv.gz"))
    facts["extraction"] = facts.get("extraction", {}) | {
        "shards": len(shards),
        "shard_bytes": sum(p.stat().st_size for p in shards),
    }
    ck = work / "checkpoint.json"
    if ck.exists():
        c = json.loads(ck.read_text())
        facts["extraction"]["port_visits"] = int(c.get("written", 0))
        facts["extraction"]["vessels_pulled"] = len(c.get("done_imos", []))
    for k, v in facts["extraction"].items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    (out / "facts.json").write_text(json.dumps(facts, indent=2))

    cjk, other = {}, {}
    for f in sorted(out.iterdir()):
        if f.suffix not in (".csv", ".txt", ".json"):
            continue
        try:
            txt = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            other[f.name] = "not valid UTF-8"
            continue
        c = {ch for ch in txt if "\u4e00" <= ch <= "\u9fff"}
        o = {ch for ch in txt if ord(ch) > 127} - c
        if c:
            cjk[f.name] = "".join(sorted(c))[:40]
        if o:
            other[f.name] = "".join(sorted(o))[:40]
    if other:
        print("\nnon-ASCII characters (check they are names, not commentary):")
        for k, v in other.items():
            print(f"  {k}: {v}")
    if cjk:
        print("\nCJK TEXT IN OUTPUT FILES -- this is script commentary, not data:")
        for k, v in cjk.items():
            print(f"  {k}: {v}")
        sys.exit("Fix the script that writes those files and re-export.")
    written.add("facts.json")
    written.add("MANIFEST.txt")

    for p in out.iterdir():
        if (p.name.startswith(("mrv_", "voyage_", "voyages_", "route_"))
                and p.suffix == ".csv"):
            written.add(p.name)

    stale = sorted(p.name for p in out.iterdir() if p.name not in written)
    if stale:
        print(f"\nremoving {len(stale)} file(s) left by an earlier export:")
        for n in stale:
            print(f"  {n}")
            (out / n).unlink()

    lines = ["# derived tables manifest",
             "# derived tables for the state-exposure analysis", ""]
    total = 0
    for p in sorted(out.iterdir()):
        if p.name == "MANIFEST.txt":
            continue
        n = sum(1 for _ in p.open("rb")) if p.suffix in (".csv", ".txt") else 0
        total += p.stat().st_size
        lines.append(f"{p.name:<36}{p.stat().st_size:>10,} B  "
                     f"{n:>8,} lines  {sha(p)[:16]}")
    lines += ["", f"total {total/1e6:.1f} MB"]
    (out / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-6:]))
    print(f"\nbundle ready: {out}")

    if no_tar:
        return

    tar_path = out.parent / "derived_tables.tar.gz"
    def anonymise(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        """tar records the owner's name and uid by default, so an archive published from a shared cluster says who ran it"""
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = int(ti.mtime)
        return ti

    with tarfile.open(tar_path, "w:gz") as tf:
        for p in sorted(out.iterdir()):
            tf.add(p, arcname=f"results/{p.name}", filter=anonymise)
    size = tar_path.stat().st_size
    print(f"\narchive: {tar_path}  ({size/1e6:.1f} MB)")
    print("\ndownload that one file, then locally:")
    print(f"  tar -xzf derived_tables.tar.gz -C /path/to/repo/results")
    print(f"  ./build.sh")
    print("\nverify it arrived intact:")
    print(f"  sha256sum derived_tables.tar.gz")
    print(f"  {sha(tar_path)}")


if __name__ == "__main__":
    main()
