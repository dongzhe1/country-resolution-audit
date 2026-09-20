#!/usr/bin/env python3
"""Does our own record see poor and small states as well as it sees rich ones?"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

from facts import emit

def _find_reference():
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference()

from sample import in_sample, m49_codes

_M49 = m49_codes()


EQUIV_RESID = 0.5


def load(results: Path):
    teu = {r["iso3"]: float(r["container_teu"])
           for r in csv.DictReader(open(REF / "port_throughput_teu.csv"))
           if float(r["container_teu"]) > 0}
    ind = {r["country"]: r for r in csv.DictReader(open(results / "country_indicators.csv"))}
    rows = []
    for r in csv.DictReader(open(results / "resolution_gap.csv")):
        iso = r["iso3"]
        if not in_sample(r, _M49) or iso not in teu:
            continue
        c = ind.get(iso)
        try:
            co2 = float(r["co2_t"])
        except (TypeError, ValueError):
            continue
        if co2 <= 0 or c is None:
            continue
        rows.append({"iso": iso, "co2": co2, "teu": teu[iso],
                     "resolved": r["resolved"] == "True",
                     "sids": r["sids"] == "True", "ldc": r["ldc"] == "True",
                     "income": c.get("income_group", ""),
                     "name": c.get("name", iso)})
    return rows


def ols(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    b = sxy / sxx
    a = my - b * mx
    res = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    ss_res = sum(r * r for r in res)
    ss_tot = sum((v - my) ** 2 for v in y)
    return a, b, res, 1 - ss_res / ss_tot


def _betacf(a, b, x, itmax=200, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf2(t, df):
    if df <= 0 or t != t:
        return float("nan")
    return betai(0.5 * df, 0.5, df / (df + t * t))


def t_ppf(p, df):
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf2(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def welch_df(va, na, vb, nb):
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    return num / den if den > 0 else float("nan")


def diff_ci(a, b, alpha=0.10):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    d = ma - mb
    df = welch_df(va, len(a), vb, len(b))
    z = t_ppf(alpha, df)
    return d, d - z * se, d + z * se


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    return t, t_sf2(t, welch_df(va, len(a), vb, len(b)))


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    results = Path(sys.argv[1]).expanduser().resolve()
    rows = load(results)
    print(f"sovereign states with both an exposure figure and published "
          f"container throughput: {len(rows)}")

    x = [math.log(r["teu"]) for r in rows]
    y = [math.log(r["co2"]) for r in rows]
    a, b, res, r2 = ols(x, y)
    for r, e in zip(rows, res):
        r["resid"] = e
    print(f"log attributed CO2 on log container TEU: slope {b:.3f}, R2 {r2:.3f}")
    print("  (scatter is expected: throughput counts boxes, our measure counts "
          "fuel burned)\n")

    print("MEAN RESIDUAL BY GROUP -- the audit")
    print(f"  {'group':<26} {'n':>4} {'mean resid':>11} {'median':>9}")

    def show(label, sel):
        s = [r["resid"] for r in rows if sel(r)]
        if s:
            print(f"  {label:<26} {len(s):>4} {statistics.mean(s):>11.3f} "
                  f"{statistics.median(s):>9.3f}")
        return s

    groups = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    for g in groups:
        show(g, lambda r, g=g: r["income"] == g)
    print()
    unres = show("no row of its own", lambda r: not r["resolved"])
    res_ = show("has its own row", lambda r: r["resolved"])
    sids = show("SIDS", lambda r: r["sids"])
    ldc = show("LDC", lambda r: r["ldc"])
    rest = show("neither SIDS nor LDC", lambda r: not r["sids"] and not r["ldc"])

    print("\nIS THE DISAGREEMENT PATTERNED?")
    print(f"  {'comparison':<24} {'difference':>11} {'90% CI':>20} {'p':>7}  verdict")
    eq = {}
    for label, s1, s2, key in (("unresolved vs resolved", unres, res_, "unresolved"),
                               ("SIDS vs rest", sids, rest, "sids"),
                               ("LDC vs rest", ldc, rest, "ldc")):
        t, p = welch(s1, s2)
        d, lo, hi = diff_ci(s1, s2)
        inside = -EQUIV_RESID < lo and hi < EQUIV_RESID
        eq[key] = (d, lo, hi, inside)
        verdict = ("equivalent within +/-%.1f" % EQUIV_RESID if inside
                   else "NOT equivalent -- interval reaches the margin")
        print(f"  {label:<24} {d:>+11.3f} {'[%+.3f, %+.3f]' % (lo, hi):>20} "
              f"{p:>7.3f}  {verdict}")


    allf = {r["iso3"]: r for r in csv.DictReader(open(results / "resolution_gap.csv"))
            if r["dependency"] != "True"}
    have = {r["iso"] for r in rows}
    print("\nWHERE THE YARDSTICK IS MISSING")
    cover = {}
    for lab, sel in (("all sovereign", lambda r: True),
                     ("no row of its own", lambda r: r["resolved"] != "True"),
                     ("SIDS", lambda r: r["sids"] == "True"),
                     ("LDC", lambda r: r["ldc"] == "True")):
        grp = [i for i, r in allf.items() if sel(r)]
        n = sum(1 for i in grp if i in have)
        cover[lab] = 100 * n / max(len(grp), 1)
        print(f"  {lab:<24} throughput published for {n:>3}/{len(grp):<3} "
              f"({cover[lab]:.0f}%)")

    print("\n  A negative mean residual for a group means our record attributes "
          "less\n  activity to it than its published throughput would predict. "
          "That is the\n  direction that would undermine the paper's claim.")

    worst = sorted(rows, key=lambda r: r["resid"])[:6]
    print("\nmost under-attributed relative to throughput")
    for r in worst:
        print(f"  {r['iso']}  {r['name'][:28]:<28} resid {r['resid']:+.2f}")

    pu, ps, pl = welch(unres, res_)[1], welch(sids, rest)[1], welch(ldc, rest)[1]
    emit(results, "visibility", {
        "throughput_all_pct": round(cover["all sovereign"]),
        "throughput_unresolved_pct": round(cover["no row of its own"]),
        "throughput_sids_pct": round(cover["SIDS"]),
        "throughput_ldc_pct": round(cover["LDC"]),
        "states": len(rows), "slope": round(b, 3), "r2": round(r2, 3),
        "p_unresolved": round(pu, 3), "p_sids": round(ps, 3), "p_ldc": round(pl, 3),
        "equiv_margin": EQUIV_RESID,
        "diff_unresolved": round(eq["unresolved"][0], 3),
        "ci_unresolved_lo": round(eq["unresolved"][1], 3),
        "ci_unresolved_hi": round(eq["unresolved"][2], 3),
        "equivalent_unresolved": eq["unresolved"][3],
        "equivalent_sids": eq["sids"][3],
        "equivalent_ldc": eq["ldc"][3],
    })

    out = results / "ais_visibility.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "name", "income_group", "resolved", "sids", "ldc",
                    "co2_t", "container_teu", "log_residual"])
        for r in sorted(rows, key=lambda x: x["resid"]):
            w.writerow([r["iso"], r["name"], r["income"], r["resolved"], r["sids"],
                        r["ldc"], f"{r['co2']:.0f}", f"{r['teu']:.0f}",
                        f"{r['resid']:.4f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
