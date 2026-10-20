"""Sections 3.5 and 3.6: Ito sums against Stratonovich sums, the Ito isometry, and a pathwise check
of Ito's lemma.

Run on its own:  python code/ch03/ito_integral.py   (writes code/ch03/out/ito_integral.json)
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
from numbers_ch03 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401


def ito_vs_strat(rng):
    t, W, hist = L.levy_bm(17, 1.0, rng, m=1, record=True)
    WT = W[0, -1]
    ks = np.arange(2, 17)
    ito = []; strat = []
    for k in ks:
        i, s = L.ito_strat_ww(hist[k + 1][0])
        ito.append(i); strat.append(s)
    ito, strat = np.array(ito), np.array(strat)
    put("itoWT", WT, "{:.4f}")
    put("itoLimit", (WT ** 2 - 1) / 2, "{:.4f}"); put("stratLimit", WT ** 2 / 2, "{:.4f}")
    put("itoSumBig", ito[-1], "{:.4f}"); put("stratSumBig", strat[-1], "{:.4f}")
    put("itoSumSmall", ito[2], "{:.4f}"); put("stratSumSmall", strat[2], "{:.4f}"); put("itoSmallN", 2 ** ks[2], "{:d}")
    put("itoBigN", 2 ** ks[-1], "{:,d}")
    # across paths: n = 500 coarse steps on a fine grid of 1,000
    M = 20000; I = []; D = []; IT = []; IE = []
    for _ in range(4):
        tt, Wf = L.bm_paths(1000, M // 4, 1.0, rng)
        i, s = L.ito_strat_ww(Wf); I.append(i); D.append(s - i)
        Wc = Wf[:, ::2]; tc = tt[::2]
        IT.append(L.ito_sum(np.broadcast_to(tc, Wc.shape), Wc))
        IE.append(L.ito_sum(np.exp(Wc), Wc))
    I, D, IT, IE = map(np.concatenate, (I, D, IT, IE))
    put("itoMean", I.mean(), "{:.4f}"); put("itoVar", I.var(), "{:.4f}")
    put("stratGapMean", D.mean(), "{:.4f}"); put("stratGapSD", D.std(), "{:.4f}")
    put("isoTVar", IT.var(), "{:.4f}"); put("isoTExact", 1 / 3, "{:.4f}")
    put("isoEVar", np.mean(IE ** 2), "{:.3f}"); put("isoEExact", (math.exp(2) - 1) / 2, "{:.3f}")
    put("isoESE", np.std(IE ** 2) / math.sqrt(M), "{:.3f}")
    put("itoPaths", M, "{:,d}")
    RAW["isoEz"] = float((np.mean(IE ** 2) - (math.exp(2) - 1) / 2) / (np.std(IE ** 2) / math.sqrt(M)))

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    n = 2.0 ** ks
    ax[0].semilogx(n, ito, "o-", color=BLUE, ms=3, lw=1, label="left point (Itô)")
    ax[0].semilogx(n, strat, "s-", color=RED, ms=3, lw=1, label="midpoint (Stratonovich)")
    ax[0].axhline((WT ** 2 - 1) / 2, color=BLUE, ls="--", lw=0.8)
    ax[0].axhline(WT ** 2 / 2, color=RED, ls="--", lw=0.8)
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("intervals $n$")
    ax[0].set_ylabel(r"sum for $\int_0^1 W\,dW$"); ax[0].set_title("(a) Two sums, two limits")
    ax[0].legend(fontsize=7.2, loc="lower right")
    y = np.linspace(-0.499, 3, 600)
    dens = 2 * stats.chi2.pdf(2 * y + 1, 1)
    ax[1].hist(I, bins=np.linspace(-0.5, 3, 71), density=True, color=PALE, ec=SKY, lw=0.3, label="Itô sums, $n=500$")
    ax[1].plot(y, dens, color=RED, lw=1, label=r"density of $(W_1^2-1)/2$")
    ax[1].set_ylim(0, 2.2); ax[1].set_xlabel("value"); ax[1].set_title(r"(b) $\int_0^1 W\,dW$ across paths")
    ax[1].legend(fontsize=7.2)
    fig.tight_layout(); save(fig, "fig05_ito_strat")


def ito_lemma_check(rng):
    ns = 2 ** np.arange(4, 13); M = 4000
    rms_ito = {"cube": [], "exp": []}; rms_naive = {"cube": [], "exp": []}
    fs = {"cube": (lambda w: w ** 3, lambda w: 3 * w ** 2, lambda w: 6 * w),
          "exp": (np.exp, np.exp, np.exp)}
    for n in ns:
        t, W = L.bm_paths(int(n), M, 1.0, rng); h = 1.0 / n
        for nm, (f, f1, f2) in fs.items():
            lhs = f(W[:, -1]) - f(W[:, 0])
            first = L.ito_sum(f1(W), W)
            second = 0.5 * np.sum(f2(W[:, :-1]) * h, axis=1)
            rms_ito[nm].append(math.sqrt(np.mean((lhs - first - second) ** 2)))
            rms_naive[nm].append(math.sqrt(np.mean((lhs - first) ** 2)))
    for nm, lab in [("cube", "Cube"), ("exp", "Exp")]:
        put("lemmaSlope" + lab, L.ols_slope(ns, rms_ito[nm]), "{:.2f}")
        put("lemmaRms" + lab, rms_ito[nm][-1], "{:.4f}"); put("naiveRms" + lab, rms_naive[nm][-1], "{:.3f}")
    put("lemmaN", ns[-1], "{:,d}"); put("lemmaPaths", M, "{:,d}")
    # the naive chain rule misses exactly 3 int W dt for f = w^3
    t, W = L.bm_paths(4096, 2000, 1.0, rng); h = 1 / 4096
    naive_gap = W[:, -1] ** 3 - L.ito_sum(3 * W ** 2, W)
    drift = 3 * np.sum(W[:, :-1] * h, axis=1)
    put("naiveGapCorr", np.corrcoef(naive_gap, drift)[0, 1], "{:.5f}")
    put("naiveGapRms", math.sqrt(np.mean((naive_gap - drift) ** 2)), "{:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].loglog(ns, rms_ito["cube"], "o-", color=BLUE, ms=3, lw=1, label=r"Itô's lemma, $W^3$")
    ax[0].loglog(ns, rms_ito["exp"], "s-", color=SKY, ms=3, lw=1, label=r"Itô's lemma, $e^W$")
    ax[0].loglog(ns, rms_naive["cube"], "o--", color=RED, ms=3, lw=1, label=r"chain rule, $W^3$")
    ax[0].loglog(ns, rms_naive["exp"], "s--", color=ROSE, ms=3, lw=1, label=r"chain rule, $e^W$")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("intervals $n$"); ax[0].set_ylabel("root mean square residual")
    ax[0].set_title("(a) Residual of each rule"); ax[0].legend(fontsize=6.8, loc="lower left")
    ax[1].plot(drift, naive_gap, ".", color=BLUE, ms=1.5, alpha=0.5)
    lim = [min(drift.min(), naive_gap.min()), max(drift.max(), naive_gap.max())]
    ax[1].plot(lim, lim, color=RED, lw=0.9)
    ax[1].set_xlabel(r"$3\int_0^1 W_t\,dt$"); ax[1].set_ylabel(r"$W_1^3 - \int_0^1 3W_t^2\,dW_t$")
    ax[1].set_title("(b) What the chain rule misses")
    fig.tight_layout(); save(fig, "fig06_ito_lemma")


def main():
    rng = rng_for(3)
    ito_vs_strat(rng); ito_lemma_check(rng)
    return finish("ito_integral")


if __name__ == "__main__":
    main()
