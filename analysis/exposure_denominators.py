#!/usr/bin/env python3
"""Does the exposure gap survive its own units and its own denominator?"""

from __future__ import annotations

import csv
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


def coverage_years(results: Path) -> tuple[float, str]:
    rows = list(csv.DictReader(open(results / "voyages_by_year.csv")))
    v = {int(r["year"]): float(r["voyages"]) for r in rows}
    years = sorted(v)
    full = years[:-1]
    mean_full = statistics.mean(v[y] for y in full)
    frac = v[years[-1]] / mean_full
    return len(full) + frac, (f"{years[0]}-{years[-1]}, {len(full)} complete years "
                              f"plus {frac:.2f} of {years[-1]}")


def load(results: Path):
    ppp = {}
    for r in csv.DictReader(open(REF / "gdp_ppp.csv")):
        ppp[r["iso3"]] = float(r["gdp_ppp_intl"])
    ind = {r["country"]: r for r in csv.DictReader(open(results / "country_indicators.csv"))}
    out = []
    for r in csv.DictReader(open(results / "resolution_gap.csv")):
        if not in_sample(r, _M49):
            continue
        iso, c = r["iso3"], ind.get(r["iso3"])
        if c is None:
            continue
        try:
            gdp = float(c["gdp_usd"])
            co2 = float(r["co2_t"])
        except (ValueError, KeyError):
            continue
        if gdp <= 0 or co2 <= 0:
            continue
        try:
            pop = float(c["population"])
        except (ValueError, KeyError):
            continue
        if pop <= 0:
            continue
        out.append({"iso": iso, "co2": co2, "gdp": gdp, "pop": pop, "ppp": ppp.get(iso),
                    "resolved": r["resolved"] == "True",
                    "sids": r["sids"] == "True", "ldc": r["ldc"] == "True",
                    "income": c.get("income_group", "")})
    return out


def pctl(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def spearman(a, b):
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    results = Path(sys.argv[1]).expanduser().resolve()
    span, desc = coverage_years(results)
    rows = load(results)


    base = 0
    ind = {r["country"]: r for r in csv.DictReader(open(results / "country_indicators.csv"))}
    for r in csv.DictReader(open(results / "resolution_gap.csv")):
        c = ind.get(r["iso3"])
        if not in_sample(r, _M49) or c is None:
            continue
        try:
            if float(c["gdp_usd"]) > 0 and float(c["population"]) > 0\
                    and float(r["co2_t"]) > 0:
                base += 1
        except (ValueError, KeyError):
            pass
    print(f"coastal sovereign states with exposure and indicators: {base}")
    print(f"observation window: {desc}  ->  {span:.2f} years")
    print(f"sovereign states with market GDP and exposure: {len(rows)}")
    print(f"  of which a purchasing-power figure is published for: "
          f"{sum(1 for r in rows if r.get('ppp') and r['ppp'] > 0)}")


    for r in rows:
        r["annual"] = r["co2"] / span
        r["mkt"] = r["annual"] / (r["gdp"] / 1e6)
        r["pp"] = (r["annual"] / (r["ppp"] / 1e6)
                   if r.get("ppp") and r["ppp"] > 0 else None)
    has_ppp = [r for r in rows if r["pp"] is not None]

    print("\nHEADLINE COMPARISONS, BOTH DENOMINATORS  (t CO2 per M$ per year)")
    print(f"  {'':<34} {'market':>10} {'PPP':>10} {'PPP/market':>11}")

    def line(label, sel):
        s = [r for r in rows if sel(r)]
        sp = [r for r in has_ppp if sel(r)]
        if not s:
            return None, None
        a = statistics.median(r["mkt"] for r in s)
        b = statistics.median(r["pp"] for r in sp) if sp else float("nan")
        print(f"  {label:<34} {a:>10.1f} {b:>10.1f} {b/a:>11.2f}")
        return a, b

    ur_m, ur_p = line("no row of its own", lambda r: not r["resolved"])
    re_m, re_p = line("has its own row", lambda r: r["resolved"])


    def med(src, sel, key):
        return statistics.median(r[key] for r in src if sel(r))
    ur_m_pp = med(has_ppp, lambda r: not r["resolved"], "mkt")
    re_m_pp = med(has_ppp, lambda r: r["resolved"], "mkt")
    print(f"  {'  same states, market denominator':<34} "
          f"{ur_m_pp/re_m_pp:>10.2f}   (n={len(has_ppp)})")
    print(f"  {'RATIO unresolved / resolved':<34} {ur_m/re_m:>10.2f} {ur_p/re_p:>10.2f}")

    print()
    for g in ("Low income", "Lower middle income", "Upper middle income", "High income"):
        line(g, lambda r, g=g: r["income"] == g)

    print()
    for label, sel in (("SIDS", lambda r: r["sids"]), ("LDC", lambda r: r["ldc"])):
        for name, key, src in (("market", "mkt", rows), ("PPP", "pp", has_ppp)):
            v = [r[key] for r in src if sel(r)]
            print(f"  {label} spread, {name:<6} n={len(v):<3} "
                  f"p90/p10 {pctl(v,0.9)/pctl(v,0.1):>6.1f}   "
                  f"max/min {max(v)/min(v):>7.1f}")

    print("\nTHE SAME CONTRAST UNDER DENOMINATORS THAT ARE NOT OUTPUT")
    print(f"  {'denominator':<26} {'unresolved':>11} {'resolved':>10} {'ratio':>7}  n")
    teu = {}
    for r in csv.DictReader(open(REF / "port_throughput_teu.csv")):
        v = float(r["container_teu"])
        if v > 0:
            teu[r["iso3"]] = v
    alt = [("output, market rates", lambda x: x["mkt"]),
           ("output, PPP", lambda x: x["pp"]),
           ("population", lambda x: x["annual"] / x["pop"]),
           ("container throughput", lambda x: x["annual"] / teu[x["iso"]]
            if x["iso"] in teu else None),
           ("none (absolute)", lambda x: x["annual"])]
    denom = {}
    for label, fn in alt:
        a = [v for v in (fn(x) for x in rows if not x["resolved"]) if v is not None]
        b = [v for v in (fn(x) for x in rows if x["resolved"]) if v is not None]
        if len(a) < 5 or len(b) < 5:
            continue
        ratio = statistics.median(a) / statistics.median(b)
        denom[label] = ratio
        print(f"  {label:<26} {statistics.median(a):>11.3g} "
              f"{statistics.median(b):>10.3g} {ratio:>7.2f}  {len(a)}/{len(b)}")
    print("\n  A gap that appears only when output is the denominator is a\n"
          "  statement about output, not about shipping burden.")

    rho = spearman([r["mkt"] for r in has_ppp], [r["pp"] for r in has_ppp])
    print(f"\n  rank correlation between the two orderings: {rho:.4f}")
    print("  A high correlation means the choice of denominator does not decide\n"
          "  which states look exposed; the group contrasts above say whether it\n"
          "  decides the size of the gap.")


    import csv as _csv
    yr = {r["country"]: int(r["gdp_year"])
          for r in _csv.DictReader(open(REF / "country_indicators_wdi.csv"))
          if r["gdp_year"].isdigit()}
    print("\nDENOMINATOR CURRENCY  (gdp vintage; reported, not adopted)")
    print(f"  {'sample':<34} {'ratio':>7} {'n':>5}")
    vint = {}
    for cut in (0, 2022, 2024, 2025):
        sub = [r for r in rows if cut == 0 or yr.get(r["iso"], 0) >= cut]
        u = [r["mkt"] for r in sub if not r["resolved"]]
        e = [r["mkt"] for r in sub if r["resolved"]]
        if len(u) < 5 or len(e) < 5:
            continue
        v = statistics.median(u) / statistics.median(e)
        vint[cut] = round(v, 2)
        lab = "every state with the data" if cut == 0 else f"GDP year {cut} or later"
        print(f"  {lab:<34} {v:>7.2f} {len(sub):>5}")
    print("  The reported figure is the first row. The rest is what a stricter\n"
          "  currency rule would give, and it is given because it moves the\n"
          "  ratio up rather than down.")

    sids = [r for r in rows if r["sids"]]
    sids_ppp = [r for r in has_ppp if r["sids"]]
    emit(results, "denominators", {


        "span_years": round(span, 2), "span_years_exact": round(span, 6),
        "states": len(has_ppp),
        "states_base": base,
        "ratio_market": round(ur_m / re_m, 2), "ratio_ppp": round(ur_p / re_p, 2),
        "ratio_market_same_states": round(ur_m_pp / re_m_pp, 2),
        "ratio_market_gdp_2024": vint.get(2024, float("nan")),
        "ratio_market_gdp_2025": vint.get(2025, float("nan")),
        "states_ppp": len(has_ppp),
        "rho_market_ppp": round(rho, 4),
        "sids_p90p10_market": round(pctl([r["mkt"] for r in sids], 0.9)
                                    / pctl([r["mkt"] for r in sids], 0.1), 1),
        "sids_p90p10_ppp": round(pctl([r["pp"] for r in sids_ppp], 0.9)
                                 / pctl([r["pp"] for r in sids_ppp], 0.1), 1),
        "unres_market": round(ur_m, 1), "res_market": round(re_m, 1),
        "ratio_per_capita": round(denom.get("population", float("nan")), 2),
        "ratio_per_teu": round(denom.get("container throughput", float("nan")), 2),
        "ratio_absolute": round(denom.get("none (absolute)", float("nan")), 3),
    })

    out = results / "exposure_denominators.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "resolved", "sids", "ldc", "income_group",
                    "annual_co2_t", "t_per_musd_market_yr", "t_per_musd_ppp_yr"])
        for r in sorted(rows, key=lambda x: -x["mkt"]):
            w.writerow([r["iso"], r["resolved"], r["sids"], r["ldc"], r["income"],
                        f"{r['annual']:.0f}", f"{r['mkt']:.3f}",
                         "" if r["pp"] is None else f"{r['pp']:.3f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
