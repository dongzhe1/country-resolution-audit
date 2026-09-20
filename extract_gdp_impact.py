#!/usr/bin/env python3
"""Extract the assessment's published GDP effects from MEPC 82/INF.8/Add.2."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

SCHEMES = (
    "SIDS and LDCs",
    "Developing economies, LDCs and SIDS",
    "All economies",
)
FIRST_SCHEME = SCHEMES[0]
TRIPLE = re.compile(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
SCENARIO = re.compile(r"Scenario\s+(\d+)\s*$")
HEAD_NONE = "Impact on GDP, no disbursement"
HEAD_WITH = "revenue disbursement scheme"
EXPECTED_ROWS = 111


ALIASES = {
    "Rest of LDCs in Africa": "Rest of LDC in Africa",
    "Korea, Republic of": "Republic of Korea",
}


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def resolution(name: str) -> str:
    return "aggregate" if name.startswith("Rest of") else "individual"


def clean(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()


def gdp_annex_region(lines):
    start = next(i for i, l in enumerate(lines) if HEAD_NONE in l)
    end = next(i for i, l in enumerate(lines[start:], start)
               if l.lstrip().startswith("Annex 15."))
    return lines[start:end]


FURNITURE = ("MEPC", "July 2024", "Comprehensive impact", "Task 3",
             "GTAP", "Economy", "Annex", "Scenario", "ENGLISH ONLY")


def is_furniture(text: str) -> bool:
    t = text.strip()
    return (not t) or t.isdigit() or any(f in t for f in FURNITURE)


def read_block(block, scen):
    parts, schemes, triples = [], [], []
    for ln in block:
        mv = TRIPLE.search(ln)
        hits = [(ln.rfind(s), s) for s in SCHEMES if s in ln]
        pos, scheme = max(hits) if hits else (None, None)
        cut = pos if pos is not None else (mv.start() if mv else len(ln))
        left = ln[:cut]
        if not is_furniture(left):
            parts.append(clean(left))
        if scheme:
            schemes.append(scheme)
        if mv:
            triples.append(mv.groups())
    name = canonical(clean(" ".join(parts)))
    if not name or len(schemes) != len(triples):
        return [], (name, len(schemes), len(triples))
    return [{
        "scenario": scen, "gtap_region": name, "resolution": resolution(name),
        "disbursement": s, "gdp_2030": v[0], "gdp_2040": v[1],
        "gdp_2050": v[2]} for s, v in zip(schemes, triples)], None


def parse(lines):
    rows = []
    anomalies = []
    scen = None
    mode = None
    block: list[str] = []

    def flush():
        nonlocal block
        if block:
            got, bad = read_block(block, scen)
            rows.extend(got)
            if bad:
                anomalies.append(bad)
        block = []

    for ln in lines:
        if HEAD_NONE in ln:
            mode = "none"
        elif HEAD_WITH in ln:
            mode = "with"
        m = SCENARIO.search(ln)
        if m:
            if m.group(1) != scen:
                flush()
                scen = m.group(1)
            continue
        if scen is None or mode is None:
            continue

        if mode == "none":
            mv = TRIPLE.search(ln)
            name = canonical(clean(ln[:mv.start()])) if mv else ""
            if mv and name:
                rows.append({
                    "scenario": scen, "gtap_region": name,
                    "resolution": resolution(name), "disbursement": "none",
                    "gdp_2030": mv.group(1), "gdp_2040": mv.group(2),
                    "gdp_2050": mv.group(3)})
            continue


        if FIRST_SCHEME in ln:
            flush()
        if block or FIRST_SCHEME in ln:
            block.append(ln)
    flush()
    return rows, anomalies


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <add2.txt> <out_dir>")
    src, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    lines = gdp_annex_region(src.read_text(encoding="utf-8").splitlines())
    rows, anomalies = parse(lines)

    groups: dict[tuple[str, str], int] = {}
    for r in rows:
        k = (r["scenario"], r["disbursement"])
        groups[k] = groups.get(k, 0) + 1

    print(f"{len(rows):,} rows from {src.name}\n")
    print(f"  {'scenario':>8}  {'disbursement':<36} {'rows':>5}")
    bad = []
    for k in sorted(groups, key=lambda x: (int(x[0]), x[1])):
        n = groups[k]
        flag = "" if n == EXPECTED_ROWS else f"  <- expected {EXPECTED_ROWS}"
        if n != EXPECTED_ROWS:
            bad.append((k, n))
        print(f"  {k[0]:>8}  {k[1]:<36} {n:>5}{flag}")

    names = {r["gtap_region"] for r in rows}
    aggs = sorted(n for n in names if n.startswith("Rest of"))
    print(f"\n  distinct economies: {len(names)}   of them aggregate rows: {len(aggs)}")
    for a in aggs:
        print(f"    {a}")

    path = out / "gtap_gdp_impact.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "scenario", "gtap_region", "resolution", "disbursement",
            "gdp_2030", "gdp_2040", "gdp_2050"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (
            int(r["scenario"]), r["disbursement"], r["gtap_region"])))
    print(f"\nwrote {path}")
    if bad:
        print("\nNOT CLEAN -- these groups do not have the expected row count:")
        for k, n in bad:
            print(f"  scenario {k[0]}, {k[1]}: {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
