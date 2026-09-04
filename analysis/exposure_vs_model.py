#!/usr/bin/env python3
"""How much of the model's own answer does our physical measure explain?

    python exposure_vs_model.py /path/to/results_dir
"""

from __future__ import annotations

import csv
import math
import random
import statistics
import sys
from pathlib import Path

from facts import emit


def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


REF = _find_reference()
SCENARIOS_111 = None


def ols_hc1(X, y, names):
    """OLS with heteroskedasticity-consistent (HC1) standard errors."""
    n, k = len(y), len(X[0])
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]

    M = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(XtX)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(k):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    inv = [row[k:] for row in M]

    b = [sum(inv[a][c] * Xty[c] for c in range(k)) for a in range(k)]
    resid = [y[i] - sum(X[i][a] * b[a] for a in range(k)) for i in range(n)]
    meat = [[sum(X[i][a] * X[i][c] * resid[i] ** 2 for i in range(n)) for c in range(k)]
            for a in range(k)]
    tmp = [[sum(inv[a][m] * meat[m][c] for m in range(k)) for c in range(k)] for a in range(k)]
    cov = [[sum(tmp[a][m] * inv[m][c] for m in range(k)) for c in range(k)] for a in range(k)]
    scale = n / max(n - k, 1)
    se = [math.sqrt(max(cov[a][a] * scale, 0.0)) for a in range(k)]

    my = statistics.mean(y)
    ss_res = sum(r * r for r in resid)
    ss_tot = sum((v - my) ** 2 for v in y)
    return b, se, 1 - ss_res / ss_tot, names


def cv_r2(rows, cols, folds=10, repeats=20, seed=0):
    """Repeated k-fold cross-validated R^2, and mean absolute error."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    r2s, maes = [], []
    for _ in range(repeats):
        rng.shuffle(idx)
        sse = sst = abserr = 0.0
        n = 0
        for f in range(folds):
            test = set(idx[f::folds])
            tr = [rows[i] for i in idx if i not in test]
            te = [rows[i] for i in idx if i in test]
            if len(tr) < len(cols) + 3 or not te:
                continue
            X = [[1.0] + [r[c] for c in cols] for r in tr]
            y = [r[2] for r in tr]
            out = ols_hc1(X, y, ["const"] + [str(c) for c in cols])
            if out is None:
                continue
            b = out[0]
            ybar = sum(y) / len(y)
            for r in te:
                pred = b[0] + sum(b[k + 1] * r[c] for k, c in enumerate(cols))
                sse += (r[2] - pred) ** 2
                sst += (r[2] - ybar) ** 2
                abserr += abs(r[2] - pred)
                n += 1
        if sst > 0:
            r2s.append(1 - sse / sst)
            maes.append(abserr / n)
    if not r2s:
        return (float("nan"),) * 4
    r2s.sort()
    lo = r2s[int(0.025 * len(r2s))]
    hi = r2s[min(len(r2s) - 1, int(0.975 * len(r2s)))]
    return statistics.median(r2s), lo, hi, statistics.median(maes)


def num(x) -> str:
    """A signed number that is safe in LaTeX text and math alike."""
    return r"\ensuremath{%+.3f}" % x


def paired_cv_bootstrap(rows, cols_a, cols_b, draws=2000, folds=10, seed=1):
    """Bootstrap the DIFFERENCE in CV R^2 between two models, over countries."""
    rng = random.Random(seed)
    n = len(rows)
    diffs = []
    for _ in range(draws):
        samp = [rows[rng.randrange(n)] for _ in range(n)]
        a = cv_r2(samp, cols_a, folds=folds, repeats=1, seed=rng.randrange(1 << 30))
        b = cv_r2(samp, cols_b, folds=folds, repeats=1, seed=rng.randrange(1 << 30))
        if a[0] == a[0] and b[0] == b[0]:
            diffs.append(a[0] - b[0])
    if not diffs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    share = sum(1 for d in diffs if d > 0) / len(diffs)
    return statistics.median(diffs), lo, hi, share


def spearman(pairs):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) ** .5) * (sum((b - my) ** 2 for b in ry) ** .5)
    return num / den if den else float("nan")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    res = Path(sys.argv[1]).expanduser().resolve()

    reg = {r["gtap_region"]: r["iso3"]
           for r in csv.DictReader(open(REF / "gtap_regions_mepc82.csv"))
           if r["resolution"] == "individual" and r["iso3"]}
    impact = list(csv.DictReader(open(REF / "gtap_gdp_impact.csv")))
    ind = {r["country"]: r for r in csv.DictReader(open(res / "country_indicators.csv"))}
    expo = {r["iso3"]: r for r in csv.DictReader(open(res / "resolution_gap.csv"))}

    counts = {}
    for r in impact:
        counts[r["scenario"]] = counts.get(r["scenario"], 0) + 1
    scenarios = sorted((s for s, n in counts.items() if n == 111), key=int)
    print(f"scenarios reporting the assessment's 111 rows: {', '.join(scenarios)}")
    print("(the others carry a different, larger row set and are left alone)\n")

    print(f"  {'scen':>5} {'n':>4} {'rho':>7} {'slope':>9} {'se':>8} {'t':>7} {'R2':>7}")
    slopes = []
    for s in scenarios:
        rows = []
        for r in impact:
            if r["scenario"] != s:
                continue
            iso = reg.get(r["gtap_region"])
            if iso is None or iso not in expo or iso not in ind:
                continue
            try:
                gdp_eff = float(r["gdp_2050"])
                inten = float(expo[iso]["t_per_musd"])
                pc = float(ind[iso]["gdp_usd"]) / float(ind[iso]["population"])
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if inten <= 0 or pc <= 0:
                continue
            rows.append((math.log(inten), math.log(pc), gdp_eff))
        if len(rows) < 20:
            continue
        y = [r[2] for r in rows]
        X = [[1.0, r[0]] for r in rows]
        out = ols_hc1(X, y, ["const", "log exposure"])
        if out is None:
            continue
        b, se, r2, _ = out
        rho = spearman([(r[0], r[2]) for r in rows])
        t = b[1] / se[1] if se[1] else float("nan")
        slopes.append(b[1])
        if s == "21":
            rho_headline, t_headline = rho, t
        print(f"  {s:>5} {len(rows):>4} {rho:>7.3f} {b[1]:>+9.4f} {se[1]:>8.4f} "
              f"{t:>+7.2f} {r2:>7.3f}")

    print(f"\n  median slope across scenarios: {statistics.median(slopes):+.4f}"
          if slopes else "\n  no scenario produced a usable fit")

    s0 = scenarios[0]
    rows = []
    for r in impact:
        if r["scenario"] != s0:
            continue
        iso = reg.get(r["gtap_region"])
        if iso is None or iso not in expo or iso not in ind:
            continue
        try:
            gdp_eff = float(r["gdp_2050"])
            inten = float(expo[iso]["t_per_musd"])
            pc = float(ind[iso]["gdp_usd"]) / float(ind[iso]["population"])
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        if inten <= 0 or pc <= 0:
            continue
        rows.append((math.log(inten), math.log(pc), gdp_eff))
    y = [r[2] for r in rows]
    print(f"\nSCENARIO {s0}: what each variable is worth  (n={len(rows)})")
    r2_by_spec = {}
    for label, cols, names in (
            ("income per head alone", [1], ["const", "log GDPpc"]),
            ("exposure alone", [0], ["const", "log exposure"]),
            ("both", [0, 1], ["const", "log exposure", "log GDPpc"])):
        X = [[1.0] + [r[c] for c in cols] for r in rows]
        b, se, r2, nm = ols_hc1(X, y, names)
        terms = "  ".join(f"{n} {v:+.4f} (t {v/s:+.1f})"
                          for n, v, s in zip(nm[1:], b[1:], se[1:]))
        r2_by_spec[label] = r2
        print(f"  {label:<24} R2 {r2:5.3f}   {terms}")


    cat = []
    for r in impact:
        if r["scenario"] != s0:
            continue
        iso = reg.get(r["gtap_region"])
        if iso is None or iso not in expo:
            continue
        try:
            gdp_eff = float(r["gdp_2050"])
        except (TypeError, ValueError):
            continue
        cat.append(iso)
    sid = {r["iso3"]: (r["sids"] == "True", r["ldc"] == "True")
           for r in csv.DictReader(open(res / "resolution_gap.csv"))}
    rows2 = []
    for (lx, lp, eff), iso in zip(rows, cat):
        sd, ld = sid.get(iso, (False, False))
        rows2.append((lx, lp, eff, float(sd), float(ld)))

    print(f"\nOUT-OF-SAMPLE, 10-fold x 20 repeats  (n={len(rows2)})")
    print(f"  {'model':<26} {'CV R2':>8} {'95% band':>18} {'MAE':>8}")
    cv = {}
    for label, cols in (("income per head", [1]),
                        ("SIDS and LDC flags", [3, 4]),
                        ("measured exposure", [0]),
                        ("exposure + income", [0, 1]),
                        ("exposure + categories", [0, 3, 4])):
        m, lo, hi, mae = cv_r2(rows2, cols)
        cv[label] = m
        print(f"  {label:<26} {m:>8.3f} {'[%+.3f, %+.3f]' % (lo, hi):>18} "
              f"{mae:>8.4f}")
    print("  A negative CV R2 means the model predicts a held-out country worse\n"
          "  than the training mean does. The band is over fold assignments,\n"
          "  not over countries, so it is NOT a sampling interval -- see below.")

    print(f"\nPAIRED BOOTSTRAP OVER COUNTRIES (2,000 draws, 10-fold)")
    print(f"  {'difference in CV R2':<34} {'median':>8} {'95% CI':>20} "
          f"{'P(>0)':>7}")
    boot = {}
    for label, ca, cb in (("exposure - income per head", [0], [1]),
                          ("exposure - SIDS/LDC flags", [0], [3, 4]),
                          ("exposure+income - income", [0, 1], [1])):
        d, lo, hi, share = paired_cv_bootstrap(rows2, ca, cb)
        boot[label] = (d, lo, hi, share)
        print(f"  {label:<34} {d:>8.3f} "
              f"{'[%+.3f, %+.3f]' % (lo, hi):>20} {share:>7.1%}")

    emit(res, "vs_model", {
        "scenarios": len(scenarios), "n": len(rows),
        "slope_median": round(statistics.median(slopes), 4),
        "rho": f"{rho_headline:.2f}",
        "r2_exposure": round(r2_by_spec["exposure alone"], 3),
        "r2_income": round(r2_by_spec["income per head alone"], 3),
        "boot_exp_inc_mid": num(boot["exposure+income - income"][0]),
        "boot_exp_inc_lo": num(boot["exposure+income - income"][1]),
        "boot_exp_inc_hi": num(boot["exposure+income - income"][2]),
        "boot_exp_inc_p": round(boot["exposure+income - income"][3], 3),
        "boot_exp_alone_mid": num(boot["exposure - income per head"][0]),
        "boot_exp_alone_lo": num(boot["exposure - income per head"][1]),
        "boot_exp_alone_hi": num(boot["exposure - income per head"][2]),
        "boot_exp_alone_p": round(boot["exposure - income per head"][3], 3),
        "boot_exp_cat_mid": num(boot["exposure - SIDS/LDC flags"][0]),
        "boot_exp_cat_lo": num(boot["exposure - SIDS/LDC flags"][1]),
        "boot_exp_cat_hi": num(boot["exposure - SIDS/LDC flags"][2]),
        "boot_exp_cat_p": round(boot["exposure - SIDS/LDC flags"][3], 3),
        "r2_both": round(r2_by_spec["both"], 3),
        "t_exposure": round(t_headline, 1),
        "cv_exposure": round(cv["measured exposure"], 3),
        "cv_income": round(cv["income per head"], 3),
        "cv_categories": round(cv["SIDS and LDC flags"], 3),
        "cv_both": round(cv["exposure + income"], 3),
    })

    print("\n  Read the R2 as the share of the assessment's own country-level\n"
          "  answer that is recoverable from ship movements alone. It is the\n"
          "  number that decides whether this analysis may speak about incidence\n"
          "  for the states the assessment does not resolve, or only about\n"
          "  resolution itself.")


if __name__ == "__main__":
    main()
