"""Section 6.6: the one-factor Gaussian copula. Default count distributions, the large pool limit,
tranche expected losses against correlation, and a simulation check.

Run on its own:  python code/ch06/copula_tranches.py   (writes code/ch06/out/copula_tranches.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch06 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401
import hazard_cds

NAMES = 100
TR = [(0.0, 0.03), (0.03, 0.07), (0.07, 0.10), (0.10, 0.15), (0.15, 0.30), (0.30, 1.0)]
TN = ["Eq", "MzA", "MzB", "SnA", "SnB", "Sup"]


def pd5():
    if "hzPdFive" not in RAW:
        hazard_cds.main()
    return RAW["hzPdFive"] / 100


def main():
    p = pd5(); R = hazard_cds.REC
    put("cpN", NAMES, "{:.0f}"); pct("cpEL", (1 - R) * p, 2)
    # distributions
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    k = np.arange(NAMES + 1)
    for rho, col, ls in [(0.0, GREY, ":"), (0.1, BLUE, "-"), (0.3, RED, "--"), (0.6, ROSE, "-.")]:
        pmf = L.pool_pmf(NAMES, p, rho)
        ax.plot(k[:41], pmf[:41], color=col, ls=ls, marker="o", ms=2, label=f"$\\rho={rho}$")
        nm = {0.0: "Zero", 0.1: "One", 0.3: "Three", 0.6: "Six"}[rho]
        pct(f"cpPZero{nm}", pmf[0], 1); pct(f"cpPTail{nm}", pmf[16:].sum(), 2)
    ax.set_xlabel("defaults among 100 names within five years"); ax.set_ylabel("probability")
    ax.legend(loc="upper right")
    save(fig, "fig08_default_counts")
    # tranche expected loss against correlation
    rhos = np.round(np.arange(0.0, 0.91, 0.05), 2)
    el = {nm: [] for nm in TN}
    for rho in rhos:
        for (a, d), nm in zip(TR, TN):
            el[nm].append(L.tranche_el_pool(NAMES, p, rho, R, a, d))
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    sty = [(BLUE, "-"), (RED, "--"), (GREY, "-."), (SKY, ":"), (ROSE, "-"), ("black", "--")]
    for ((a, d), nm), (c, ls) in zip(zip(TR, TN), sty):
        ax.plot(rhos, 100 * np.array(el[nm]), color=c, ls=ls, label=f"{100*a:.0f} to {100*d:.0f}%")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 100)
    ax.set_xlabel(r"factor correlation $\rho$"); ax.set_ylabel("expected loss (% of tranche)")
    ax.legend(loc="lower right", ncol=2, fontsize=7.5)
    save(fig, "fig09_tranche_el")
    for rho, rn in [(0.1, "One"), (0.3, "Three"), (0.6, "Six")]:
        i = int(np.where(np.isclose(rhos, rho))[0][0]); tot = 0.0
        for (a, d), nm in zip(TR, TN):
            v = el[nm][i]; tot += v * (d - a)
            put(f"tr{nm}{rn}", 100 * v, "{:.2f}" if v >= 0.0001 else "{:.4f}")
        RAW[f"trSum{rn}"] = tot
        lhp = [L.tranche_el_lhp(p, rho, R, a, d) for a, d in TR]
        for nm, v in zip(TN, lhp):
            put(f"lh{nm}{rn}", 100 * v, "{:.2f}" if v >= 0.0001 else "{:.4f}")
    # simulation check at rho = 0.3
    scen = 200000
    m, se = L.copula_mc(NAMES, p, 0.3, R, TR, scen, rng_for(6))
    i = int(np.where(np.isclose(rhos, 0.3))[0][0])
    z = [(mm - el[nm][i]) / s for mm, s, nm in zip(m, se, TN) if s > 0]
    put("cpMcScen", scen, "{:,.0f}"); put("cpMcMaxZ", max(abs(v) for v in z), "{:.1f}")
    put("cpMcEq", 100 * m[0], "{:.2f}"); put("cpMcEqSE", 100 * se[0], "{:.2f}")
    put("cpMcSnB", 100 * m[4], "{:.3f}"); put("cpMcSnBSE", 100 * se[4], "{:.3f}")
    # the six errors share the factor draws, so a second stream shows whether a large one is chance
    m2, se2 = L.copula_mc(NAMES, p, 0.3, R, TR, scen, rng_for(8))
    z2 = [(mm - el[nm][i]) / s for mm, s, nm in zip(m2, se2, TN) if s > 0]
    put("cpMcMaxZAlt", max(abs(v) for v in z2), "{:.1f}")
    put("cpMcSignNeg", sum(v < 0 for v in z), "{:.0f}")
    # the tranche with the largest gap, with enough digits for the gap to be checked from the text
    zs = [((mm - el[nm][i]) / s if s > 0 else 0.0) for mm, s, nm in zip(m, se, TN)]
    k = int(np.argmax(np.abs(zs))); a_, d_ = TR[k]
    N["cpMcMaxName"] = f"{100 * a_:.0f} to {100 * d_:.0f}\\%"
    put("cpMcMaxEst", 100 * m[k], "{:.4f}"); put("cpMcMaxSE", 100 * se[k], "{:.4f}")
    put("cpMcMaxTab", 100 * el[TN[k]][i], "{:.4f}")
    return finish("copula_tranches")


if __name__ == "__main__":
    main()
