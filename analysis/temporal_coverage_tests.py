#!/usr/bin/env python3
"""Is the gap stable over time, and is our record equally complete everywhere?

    python temporal_coverage_tests.py /path/to/results_dir
"""

from __future__ import annotations

def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    from pathlib import Path as _P
    here = _P(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


import csv
import math
import statistics
import sys
from pathlib import Path

from facts import emit


EQUIV_SHARE = 0.05


from ais_visibility import diff_ci, welch, t_sf2, t_ppf, welch_df


def spearman(pairs):
    """Rank correlation over (x, y) pairs, averaging ties."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    rx, ry = rank(xs), rank(ys)
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** 0.5) * (sum((b - my) ** 2 for b in ry) ** 0.5)
    return num / den if den else float("nan")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    res = Path(sys.argv[1]).expanduser().resolve()
    for f in ("state_coverage.csv", "state_exposure_by_year.csv"):
        if not (res / f).exists():
            sys.exit(f"{res/f} not found -- copy it back from the extraction host "
                     f"($GFW/{f}, written by the voyage stage)")

    flags = {r["iso3"]: r for r in csv.DictReader(open(res / "resolution_gap.csv"))}
    ind = {r["country"]: r for r in csv.DictReader(open(res / "country_indicators.csv"))}

    cov = []
    for r in csv.DictReader(open(res / "state_coverage.csv")):
        f = flags.get(r["country"])
        if f is None or f["dependency"] == "True":
            continue
        try:
            share = float(r["gappy_co2_share"])
        except (TypeError, ValueError):
            continue
        if not (0.0 <= share <= 1.0):
            continue
        cov.append({"iso": r["country"], "share": share,
                    "resolved": f["resolved"] == "True",
                    "sids": f["sids"] == "True", "ldc": f["ldc"] == "True",
                    "income": ind.get(r["country"], {}).get("income_group", "")})
    print(f"A. COVERAGE  ({len(cov)} sovereign states)")
    print(f"   share of a state's CO2 carried by vessels that go silent "
          f"mid-history\n")
    print(f"   {'group':<26} {'n':>4} {'mean':>7} {'median':>8}")

    def grp(label, sel):
        s = [c["share"] for c in cov if sel(c)]
        if s:
            print(f"   {label:<26} {len(s):>4} {statistics.mean(s):>7.1%} "
                  f"{statistics.median(s):>8.1%}")
        return s

    unres = grp("no row of its own", lambda c: not c["resolved"])
    resd = grp("has its own row", lambda c: c["resolved"])
    sids = grp("SIDS", lambda c: c["sids"])
    ldc = grp("LDC", lambda c: c["ldc"])
    rest = grp("neither SIDS nor LDC", lambda c: not c["sids"] and not c["ldc"])
    print()
    for g in ("Low income", "Lower middle income", "Upper middle income", "High income"):
        grp(g, lambda c, g=g: c["income"] == g)

    print("\n   difference, interval, and whether it clears the margin")
    print(f"     {'comparison':<24} {'diff':>8} {'90% CI':>20} {'p':>7}  verdict")
    eqv = {}
    for label, a, b, key in (("unresolved vs resolved", unres, resd, "unresolved"),
                             ("SIDS vs rest", sids, rest, "sids"),
                             ("LDC vs rest", ldc, rest, "ldc")):
        t, p = welch(a, b)
        d, lo, hi = diff_ci(a, b)
        inside = -EQUIV_SHARE < lo and hi < EQUIV_SHARE
        eqv[key] = inside
        v = ("equivalent within %.0f pp" % (100 * EQUIV_SHARE) if inside
             else "NOT equivalent -- interval reaches the margin")
        print(f"     {label:<24} {d:>+8.3f} {'[%+.3f, %+.3f]' % (lo, hi):>20} "
              f"{p:>7.3f}  {v}")

    by_year = {}
    for r in csv.DictReader(open(res / "state_exposure_by_year.csv")):
        by_year.setdefault(int(r["year"]), {})[r["country"]] = float(r["co2_t"])
    years = sorted(by_year)
    complete = years[:-1]
    print(f"\nB. TIME  (years {years[0]}-{years[-1]}; {years[-1]} is partial "
          f"and is excluded below)")

    periods = [("2018-2019", [y for y in complete if y <= 2019]),
               ("2020-2021", [y for y in complete if 2020 <= y <= 2021]),
               ("2022-2025", [y for y in complete if y >= 2022])]
    tot = {}
    for name, ys in periods:
        acc = {}
        for y in ys:
            for c, v in by_year[y].items():
                acc[c] = acc.get(c, 0.0) + v
        tot[name] = acc

    print("\n   rank correlation of state exposure between periods")
    rho_pairs = {}
    names = [n for n, _ in periods]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = tot[names[i]], tot[names[j]]
            common = sorted(set(a) & set(b))
            rho = spearman([(a[c], b[c]) for c in common])
            rho_pairs[(i + 1, j + 1)] = rho
            print(f"     {names[i]} vs {names[j]}   rho = {rho:.4f}   "
                  f"({len(common)} states)")

    print("\n   unresolved / resolved median intensity, by period")
    print(f"     {'period':<12} {'unres':>8} {'res':>8} {'ratio':>7}")
    ratios = {}
    for name, ys in periods:
        acc, n = tot[name], len(ys)
        u, r_ = [], []
        for c, v in acc.items():
            f = flags.get(c)
            k = ind.get(c)
            if f is None or k is None or f["dependency"] == "True":
                continue
            try:
                gdp = float(k["gdp_usd"])
            except (TypeError, ValueError):
                continue
            if gdp <= 0:
                continue
            inten = (v / n) / (gdp / 1e6)
            (r_ if f["resolved"] == "True" else u).append(inten)
        if u and r_:
            mu, mr = statistics.median(u), statistics.median(r_)
            ratios[{"2018-2019": "early", "2020-2021": "mid",
                    "2022-2025": "late"}[name]] = mu / mr
            print(f"     {name:<12} {mu:>8.1f} {mr:>8.1f} {mu/mr:>7.2f}")
    emit(res, "temporal", {
        "states": len(cov),
        "gappy_median": round(statistics.median(c["share"] for c in cov), 3),
        "gappy_p_unresolved": round(welch(unres, resd)[1], 3),
        "gappy_p_sids": round(welch(sids, rest)[1], 3),
        "gappy_p_ldc": round(welch(ldc, rest)[1], 3),
        **{f"rho_{i}{j}": round(rr, 4) for (i, j), rr in rho_pairs.items()},
        **{f"ratio_{k}": round(v, 2) for k, v in ratios.items()},
        "equiv_margin_pp": round(100 * EQUIV_SHARE),
        "gappy_equivalent_unresolved": eqv["unresolved"],
        "gappy_equivalent_sids": eqv["sids"],
        "gappy_equivalent_ldc": eqv["ldc"],
    })

    print("\n   A ratio that holds across periods means the finding is not an "
          "artefact\n   of one stretch of years; GDP is held at its single "
          "vintage throughout,\n   so this varies the numerator only.")


if __name__ == "__main__":
    main()
