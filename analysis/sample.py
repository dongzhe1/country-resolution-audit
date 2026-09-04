#!/usr/bin/env python3
"""Who is in the country-level sample."""

from __future__ import annotations

import csv
from pathlib import Path


def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference()

EXTRA_TERRITORIES = {"ASM"}


def m49_codes() -> set[str]:
    return {r["iso3"] for r in csv.DictReader(open(REF / "country_status.csv"))
            if r["iso3"]}


def territories(results) -> set[str]:
    """Codes to exclude: flagged dependencies plus the audited additions."""
    out = set(EXTRA_TERRITORIES)
    for r in csv.DictReader(open(Path(results) / "resolution_gap.csv")):
        if r.get("dependency") == "True":
            out.add(r["iso3"])
    return out


def universe(results, m49: set[str] | None = None) -> set[str]:
    """The resolution universe: every state the assessment could have resolved."""
    if m49 is None:
        m49 = m49_codes()
    return m49 - territories(results)


def in_sample(row, m49: set[str] | None = None, terr: set[str] | None = None) -> bool:
    """True if this row of resolution_gap.csv belongs in the exposure sample."""
    if m49 is None:
        m49 = m49_codes()
    iso = row.get("iso3") or row.get("country")
    if iso not in m49:
        return False
    if row.get("dependency") == "True" or iso in EXTRA_TERRITORIES:
        return False
    return True


def report(rows) -> None:
    m49 = m49_codes()
    out = [r for r in rows if not in_sample(r, m49)]
    off = sorted(r["iso3"] for r in out if r["iso3"] not in m49)
    dep = sorted(r["iso3"] for r in out if r["iso3"] in m49)
    print(f"country sample: {len(rows) - len(out)} of {len(rows)} entities")
    print(f"  not in the UN M49 list: {off or 'none'}")
    print(f"  dependent territories:  {len(dep)}")
