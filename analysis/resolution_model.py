#!/usr/bin/env python3
"""What decides whether a state gets its own row in the impact assessment?

    python resolution_model.py /path/to/results_dir
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

from facts import emit


def _find_reference():
    """The reference tables, wherever this file sits relative to them."""
    here = Path(__file__).resolve()
    for d in [here.parent] + list(here.parents):
        cand = d / "reference"
        if cand.is_dir():
            return cand
    raise SystemExit("cannot find reference/ above " + str(here.parent))


FIRTH = True
MAX_ITER = 500
TOL = 1e-11


REF = _find_reference()


def load(results: Path, keep_dependencies=False):
    resolved = {r["iso3"] for r in csv.DictReader(open(REF / "gtap_regions_mepc82.csv"))
                if r["resolution"] == "individual" and r["iso3"]}
    status = {r["iso3"]: r for r in csv.DictReader(open(REF / "country_status.csv"))}
    ind = {r["country"]: r for r in csv.DictReader(open(results / "country_indicators.csv"))}
    from sample import universe
    keep = universe(results)
    rows, dropped = [], 0
    for iso, c in ind.items():
        st = status.get(iso)
        if st is None:
            dropped += 1
            continue
        r = {"resolved": "True" if iso in resolved else "False",
             "sids": "True" if st["sids"] == "TRUE" else "False",
             "ldc": "True" if st["ldc"] == "TRUE" else "False"}
        try:
            gdp, pop = float(c["gdp_usd"]), float(c["population"])
        except (ValueError, KeyError):
            dropped += 1
            continue
        if gdp <= 0 or pop <= 0:
            dropped += 1
            continue
        if iso not in keep and not keep_dependencies:
            dropped += 1
            continue
        rows.append({
            "iso": iso,
            "resolved": r["resolved"] == "True",
            "sids": r["sids"] == "True",
            "ldc": r["ldc"] == "True",
            "log_gdp": math.log(gdp),
            "log_gdppc": math.log(gdp / pop),
            "log_pop": math.log(pop),
            "gdp": gdp,
            "income_group": c.get("income_group", ""),
            "landlocked": st["landlocked"] == "TRUE",
        })
    return rows, dropped


def firth_fit(X: np.ndarray, y: np.ndarray):
    """Firth-penalized logistic regression by modified IRLS."""
    n, k = X.shape
    b = np.zeros(k)
    for _ in range(MAX_ITER):
        eta = np.clip(X @ b, -500, 500)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        XtWX = X.T @ (X * w[:, None])
        try:
            cov = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            sys.exit("design matrix is singular -- see the collinearity note in "
                     "the docstring; you probably passed log GDP, log GDPpc and "
                     "log population together")
        if FIRTH:
            h = np.einsum("ij,jk,ik->i", X * w[:, None], cov, X)
            grad = X.T @ (y - p + h * (0.5 - p))
        else:
            grad = X.T @ (y - p)
        step = cov @ grad
        for _ in range(30):
            eta_new = np.clip(X @ (b + step), -500, 500)
            if np.all(np.isfinite(eta_new)):
                break
            step /= 2.0
        b = b + step
        if np.max(np.abs(step)) < TOL:
            break
    eta = np.clip(X @ b, -500, 500)
    p = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(np.clip(p, 1e-300, 1)) +
                      (1 - y) * np.log(np.clip(1 - p, 1e-300, 1))))
    w = p * (1 - p)
    XtWX = X.T @ (X * w[:, None])
    sign, logdet = np.linalg.slogdet(XtWX)
    pll = ll + 0.5 * logdet if sign > 0 else ll
    se = np.sqrt(np.diag(np.linalg.inv(XtWX)))
    return b, se, pll


def pfmt(p: float) -> str:
    """A p-value as it should appear in prose."""
    return r"\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"

def constrained_pll(X, y, j, value, start=None):
    """Penalised log-likelihood with coefficient j held at `value`."""
    keep = [c for c in range(X.shape[1]) if c != j]
    off = X[:, j] * value
    Xr = X[:, keep]
    b = np.zeros(Xr.shape[1]) if start is None else np.asarray(start, float)[keep]
    for _ in range(MAX_ITER):
        eta = np.clip(Xr @ b + off, -500, 500)
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = pr * (1 - pr)
        try:
            cov_full = np.linalg.inv(X.T @ (X * w[:, None]))
            cov_red = np.linalg.inv(Xr.T @ (Xr * w[:, None]))
        except np.linalg.LinAlgError:
            return -np.inf
        h = np.einsum("ij,jk,ik->i", X * w[:, None], cov_full, X)
        step = cov_red @ (Xr.T @ (y - pr + h * (0.5 - pr)))
        big = np.max(np.abs(step))
        b = b + (2.0 / big if big > 2.0 else 1.0) * step
        if big < TOL:
            break
    eta = np.clip(Xr @ b + off, -500, 500)
    pr = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(np.clip(pr, 1e-300, 1))
                      + (1 - y) * np.log(np.clip(1 - pr, 1e-300, 1))))
    w = pr * (1 - pr)
    sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
    return ll + 0.5 * logdet if sign > 0 else ll


def constrained_pll_multi(X, y, drop, start=None):
    """Penalised log-likelihood with several coefficients held at zero."""
    keep = [c for c in range(X.shape[1]) if c not in drop]
    Xr = X[:, keep]
    b = np.zeros(Xr.shape[1])
    for _ in range(MAX_ITER):
        eta = np.clip(Xr @ b, -500, 500)
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = pr * (1 - pr)
        try:
            cov_full = np.linalg.inv(X.T @ (X * w[:, None]))
            cov_red = np.linalg.inv(Xr.T @ (Xr * w[:, None]))
        except np.linalg.LinAlgError:
            return -np.inf
        h = np.einsum("ij,jk,ik->i", X * w[:, None], cov_full, X)
        step = cov_red @ (Xr.T @ (y - pr + h * (0.5 - pr)))
        big = np.max(np.abs(step))
        b = b + (2.0 / big if big > 2.0 else 1.0) * step
        if big < TOL:
            break
    eta = np.clip(Xr @ b, -500, 500)
    pr = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(np.clip(pr, 1e-300, 1))
                      + (1 - y) * np.log(np.clip(1 - pr, 1e-300, 1))))
    w = pr * (1 - pr)
    sign, logdet = np.linalg.slogdet(X.T @ (X * w[:, None]))
    return ll + 0.5 * logdet if sign > 0 else ll


def profile_ci(X, y, j, bhat, pll_hat, bhat_all, level=1.920729):
    """Penalized profile-likelihood interval for coefficient j."""
    def pll_at(value):
        return constrained_pll(X, y, j, value, bhat_all)

    out = []
    for direction in (-1, 1):
        lo, hi = 0.0, 1.0
        while pll_at(bhat + direction * hi) > pll_hat - level:
            hi *= 2.0
            if hi > 64:
                out.append(direction * np.inf)
                break
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                if pll_at(bhat + direction * mid) > pll_hat - level:
                    lo = mid
                else:
                    hi = mid
            out.append(bhat + direction * (lo + hi) / 2)
    return out[0], out[1]


def chi2_sf_1df(x: float) -> float:
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def norm_sf2(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def fit_and_report(rows, terms, label, quiet=False):
    y = np.array([float(r["resolved"]) for r in rows])
    X = np.column_stack([np.ones(len(rows))] +
                        [np.array([float(r[t]) for r in rows]) for t in terms])
    b, se, pll = firth_fit(X, y)
    if not quiet:
        print(f"\n  [{label}]")
        print(f"    {'term':<12} {'beta':>8} {'95% profile CI':>20} "
              f"{'LR p':>8} {'Wald p':>8}")
    lrp, ci = {}, {}
    for j, t in enumerate(terms, start=1):
        lr = 2.0 * (pll - constrained_pll(X, y, j, 0.0))
        lrp[t] = chi2_sf_1df(lr)
        ci[t] = profile_ci(X, y, j, b[j], pll, b)
        if not quiet:
            print(f"    {t:<12} {b[j]:>+8.3f} "
                  f"{'[%+.3f, %+.3f]' % ci[t]:>20} "
                  f"{lrp[t]:>8.3f} {norm_sf2(b[j]/se[j]):>8.3f}")
    return b, se, pll, lrp, ci


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_dep = "--with-dependencies" in sys.argv
    if len(args) != 1:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir> "
                 f"[--with-dependencies]")
    results = Path(args[0]).expanduser().resolve()
    rows, dropped = load(results, keep_dependencies=keep_dep)
    if keep_dep:
        print("INCLUDING dependent territories -- diagnostic only, not the "
              "specification the write-up reports")
    n = len(rows)
    nres = sum(r["resolved"] for r in rows)
    what = "entities" if keep_dep else "sovereign states"
    print(f"{what} with a resolution flag and indicators: {n} "
          f"({nres} resolved, {n - nres} not); {dropped} dropped "
          f"(missing data{'' if keep_dep else ', or a dependent territory'})")

    ident = max(abs(r["log_gdp"] - r["log_gdppc"] - r["log_pop"]) for r in rows)
    assert ident < 1e-9, f"log GDP != log GDPpc + log pop (max dev {ident})"

    print("\nSPECIFICATION LADDER")
    fit_and_report(rows, ["log_gdppc"], "income per head alone")
    fit_and_report(rows, ["log_gdp"], "economic size alone")
    fit_and_report(rows, ["log_gdp", "log_gdppc"], "size and income together")
    b, se, _, lrp, ci = fit_and_report(
        rows, ["log_gdp", "log_gdppc", "sids", "ldc"], "size, income, category")

    print("\nTHE SAME MODEL, THREE PARAMETERISATIONS")
    for terms, label in ((["log_gdp", "log_gdppc", "sids", "ldc"], "GDP + GDP/head"),
                         (["log_gdp", "log_pop", "sids", "ldc"], "GDP + population"),
                         (["log_gdppc", "log_pop", "sids", "ldc"], "GDP/head + population")):
        fit_and_report(rows, terms, label)

    print("\nWHAT EACH VARIABLE IS WORTH, INDEPENDENT OF PARAMETERISATION")
    y = np.array([float(r["resolved"]) for r in rows])
    full_terms = ["log_gdp", "log_gdppc", "sids", "ldc"]
    Xf = np.column_stack([np.ones(len(rows))] +
                         [np.array([float(r[t]) for r in rows]) for t in full_terms])
    _, _, pll_full = firth_fit(Xf, y)
    print(f"  {'dropped':<22} {'2*dLL':>8} {'p':>8}")
    for drop, label in (([1, 2], "size and income"), ([3], "small island"),
                        ([4], "least developed"), ([1], "log GDP (given GDP/head)"),
                        ([2], "GDP/head (given GDP)")):
        pll_r = constrained_pll_multi(Xf, y, drop)
        lr = 2.0 * (pll_full - pll_r)
        pv = (chi2_sf_1df(lr) if len(drop) == 1
              else math.exp(-max(lr, 0.0) / 2.0))
        print(f"  {label:<22} {lr:>8.1f} {pv:>8.3f}")
    print("  Dropping size and income together costs far more likelihood than\n"
          "  dropping either alone, which is what 'size dominates' rests on.")
    dev = {}
    for drop, key in (([1], "lr_gdp_given_pc"), ([2], "lr_pc_given_gdp"),
                      ([3], "lr_sids"), ([4], "lr_ldc"), ([1, 2], "lr_scale_both")):
        dev[key] = round(2.0 * (pll_full - constrained_pll_multi(Xf, y, drop)), 1)

    print("\nPREDICTED PROBABILITY OF AN INDIVIDUAL ROW")
    g = np.array(sorted(r["log_gdp"] for r in rows))
    med_pc = float(np.median([r["log_gdppc"] for r in rows]))
    print(f"    at the median income per head, not SIDS, not LDC:")
    for q in (10, 25, 50, 75, 90):
        lg = float(np.percentile(g, q))
        eta = b[0] + b[1] * lg + b[2] * med_pc
        print(f"      GDP p{q:<2} (${math.exp(lg)/1e9:8.1f}bn)   "
              f"P = {1/(1+math.exp(-eta)):.2f}")
    lg_half = (-b[0] - b[2] * med_pc) / b[1]
    print(f"    P = 0.50 at GDP = ${math.exp(lg_half)/1e9:.0f}bn")
    for name, j in (("SIDS", 3), ("LDC", 4)):
        print(f"    holding size and income fixed, {name} multiplies the odds "
              f"of a row by {math.exp(b[j]):.2f}")
    for name, j in (("SIDS", 3), ("LDC", 4)):
        print(f"    a {name} needs to be {math.exp(-b[j]/b[1]):.0f}x larger to "
              f"reach the same probability as a state that is neither")

    emit(results, "resolution_model", {
        "n": n, "resolved": nres, "unresolved": n - nres,
        "beta_gdp": round(b[1], 3), "beta_gdppc": round(b[2], 3),
        "p_gdppc_wald": pfmt(norm_sf2(b[2] / se[2])),
        "p_gdppc_lr": pfmt(lrp["log_gdppc"]),
        "p_gdp_lr": pfmt(lrp["log_gdp"]),
        "p_sids_lr": pfmt(lrp["sids"]), "p_ldc_lr": pfmt(lrp["ldc"]),
        "odds_sids": round(math.exp(b[3]), 2), "odds_ldc": round(math.exp(b[4]), 2),
        "gdp_bn_at_half": round(math.exp(lg_half) / 1e9),
        "size_penalty_sids": round(math.exp(-b[3] / b[1])),
        "size_penalty_ldc": round(math.exp(-b[4] / b[1])),
        "or_sids_lo": round(math.exp(ci["sids"][0]), 2),
        "or_sids_hi": round(math.exp(ci["sids"][1]), 2),
        "or_ldc_lo": round(math.exp(ci["ldc"][0]), 3),
        "or_ldc_hi": round(math.exp(ci["ldc"][1]), 2),
        "beta_gdp_lo": round(ci["log_gdp"][0], 3),
        "beta_gdp_hi": round(ci["log_gdp"][1], 3),
        "beta_gdppc_lo": round(ci["log_gdppc"][0], 3),
        "beta_gdppc_hi": round(ci["log_gdppc"][1], 3),
        "beta_sids": round(b[3], 3), "beta_ldc": round(b[4], 3),
        "beta_sids_lo": round(ci["sids"][0], 3), "beta_sids_hi": round(ci["sids"][1], 3),
        "beta_ldc_lo": round(ci["ldc"][0], 3), "beta_ldc_hi": round(ci["ldc"][1], 3),
        "or_gdp": round(math.exp(b[1]), 2), "or_gdppc": round(math.exp(b[2]), 2),
        "intercept": round(b[0], 2),
        **dev,
        "p_at_gdp_p10": round(1/(1+math.exp(-(b[0]+b[1]*float(np.percentile(g,10))+b[2]*med_pc))), 2),
        "p_at_gdp_p90": round(1/(1+math.exp(-(b[0]+b[1]*float(np.percentile(g,90))+b[2]*med_pc))), 2),
        "gdp_bn_p10": round(math.exp(float(np.percentile(g,10)))/1e9, 1),
        "gdp_bn_p90": round(math.exp(float(np.percentile(g,90)))/1e9),
    })

    B = 600
    rng = np.random.default_rng(0)
    lgg = np.linspace(np.percentile(g, 2), np.percentile(g, 98), 60)
    cats = {"neither": (0.0, 0.0), "sids": (1.0, 0.0), "ldc": (0.0, 1.0)}
    draws = {k: [] for k in cats}
    Xall = np.column_stack([np.ones(len(rows)),
                            np.array([r["log_gdp"] for r in rows]),
                            np.array([r["log_gdppc"] for r in rows]),
                            np.array([float(r["sids"]) for r in rows]),
                            np.array([float(r["ldc"]) for r in rows])])
    yall = np.array([float(r["resolved"]) for r in rows])
    for _ in range(B):
        idx = rng.integers(0, len(rows), len(rows))
        try:
            bb, _, _ = firth_fit(Xall[idx], yall[idx])
        except SystemExit:
            continue
        for k, (sd, ld) in cats.items():
            eta = bb[0] + bb[1] * lgg + bb[2] * med_pc + bb[3] * sd + bb[4] * ld
            draws[k].append(1 / (1 + np.exp(-np.clip(eta, -500, 500))))
    curve = results / "resolution_curve.csv"
    with open(curve, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["log_gdp", "category", "fit", "lo", "hi"])
        for k, (sd, ld) in cats.items():
            eta = b[0] + b[1] * lgg + b[2] * med_pc + b[3] * sd + b[4] * ld
            fit = 1 / (1 + np.exp(-np.clip(eta, -500, 500)))
            arr = np.array(draws[k])
            lo = np.percentile(arr, 2.5, axis=0)
            hi = np.percentile(arr, 97.5, axis=0)
            for i in range(len(lgg)):
                w.writerow([f"{lgg[i]:.6f}", k, f"{fit[i]:.4f}",
                            f"{lo[i]:.4f}", f"{hi[i]:.4f}"])
    print(f"wrote {curve}  ({len(draws['neither'])} bootstrap fits)")

    out = results / "resolution_model.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["term", "beta", "se", "wald_p", "lr_p", "odds_ratio"])
        for j, t in enumerate(["log_gdp", "log_gdppc", "sids", "ldc"], start=1):
            w.writerow([t, f"{b[j]:.6f}", f"{se[j]:.6f}",
                        f"{norm_sf2(b[j]/se[j]):.6f}", f"{lrp[t]:.6f}",
                        f"{math.exp(b[j]):.6f}"])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
