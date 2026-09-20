#!/usr/bin/env python3
"""Check the Firth fit against an independent optimisation of the same objective."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from resolution_model import firth_fit, load


def penalised_ll(X: np.ndarray, y: np.ndarray, b: np.ndarray) -> float:
    eta = np.clip(X @ b, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    ll = float(np.sum(y * np.log(np.clip(p, 1e-300, 1))
                      + (1 - y) * np.log(np.clip(1 - p, 1e-300, 1))))
    w = p * (1 - p)
    info = X.T @ (X * w[:, None])
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        return -np.inf
    return ll + 0.5 * logdet


def nelder_mead(f, x0, steps, iters=20000, tol=1e-12):
    n = len(x0)
    simplex = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += steps[i]
        simplex.append(p)
    val = [-f(p) for p in simplex]
    for _ in range(iters):
        order = np.argsort(val)
        simplex = [simplex[i] for i in order]
        val = [val[i] for i in order]
        if abs(val[-1] - val[0]) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = -f(xr)
        if fr < val[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = -f(xe)
            simplex[-1], val[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < val[-2]:
            simplex[-1], val[-1] = xr, fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = -f(xc)
            if fc < val[-1]:
                simplex[-1], val[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    val[i] = -f(simplex[i])
    return simplex[int(np.argmin(val))]


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    rows = load(Path(sys.argv[1]).expanduser().resolve())[0]

    terms = ["log_gdp", "log_gdppc", "sids", "ldc"]
    X = np.column_stack([np.ones(len(rows))]
                        + [[float(r[t]) for r in rows] for t in terms])
    y = np.array([float(r["resolved"]) for r in rows])

    b_irls, _se, pll_irls = firth_fit(X, y)
    print(f"n = {len(rows)},  terms = {', '.join(terms)}\n")

    b_nm = nelder_mead(lambda b: penalised_ll(X, y, b),
                       np.zeros(X.shape[1]), [0.5] * X.shape[1])
    pll_nm = penalised_ll(X, y, b_nm)

    print(f"  {'coefficient':<16}{'IRLS':>12}{'Nelder-Mead':>14}{'diff':>12}")
    worst = 0.0
    for name, a, c in zip(["(intercept)"] + list(terms), b_irls, b_nm):
        d = abs(a - c)
        worst = max(worst, d)
        print(f"  {name:<16}{a:>12.6f}{c:>14.6f}{d:>12.2e}")
    print(f"\n  penalised log-likelihood  {pll_irls:.8f}  vs  {pll_nm:.8f}"
          f"   diff {abs(pll_irls - pll_nm):.2e}")
    print(f"  largest coefficient difference: {worst:.2e}")

    ok = worst < 1e-3 and abs(pll_irls - pll_nm) < 1e-6
    print("\n" + ("VERDICT: the two agree. The IRLS reaches the maximum of the\n"
                  "  penalised log-likelihood as written. This does not check that\n"
                  "  the penalty is Firth's -- for that, run R logistf on the same\n"
                  "  data."
                  if ok else
                  "VERDICT: THEY DISAGREE. One of the two is wrong; do not report\n"
                  "  the fit until this is resolved."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
