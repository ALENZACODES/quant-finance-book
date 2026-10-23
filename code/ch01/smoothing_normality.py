"""Sections 1.11 and 1.12: smoothing and unsmoothing, variance ratios, Jarque-Bera size and power.

Run on its own:  python code/ch01/smoothing_normality.py   (writes code/ch01/out/smoothing_normality.json)
run_all.py runs every script of the chapter, in a fixed order with one random generator, and writes
chapters/ch01-numbers.tex.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from numbers_ch01 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, SEED  # noqa: F401


# =============================================================== 1.12 smoothing and autocorrelation
def smoothing(rng):
    alpha, n = 0.6, 240
    true = rng.normal(0.008, 0.04, n)
    obs = L.smooth(true, alpha)
    s_true, s_obs = true.std(ddof=1), obs.std(ddof=1)
    pct("smTrueSD", s_true); pct("smObsSD", s_obs)
    pct("smTheoryRatio", np.sqrt((1 - alpha) / (1 + alpha)), 1)
    put("smRatioSample", s_obs / s_true, "{:.3f}"); put("smRatioTheory", np.sqrt((1 - alpha) / (1 + alpha)), "{:.3f}")
    rho1 = L.acf(obs, 1)[1]; put("smRhoOne", rho1, "{:.3f}")
    dw = L.durbin_watson(obs - obs.mean()); put("smDW", dw, "{:.3f}"); put("smDWapprox", 2 * (1 - rho1), "{:.3f}")
    un = L.unsmooth(obs, alpha); put("smUnSD", 100 * un.std(ddof=1), "{:.2f}")
    put("smUnRho", L.acf(un, 1)[1], "{:.3f}")
    # annualised
    pct("smAnnObs", s_obs * np.sqrt(12), 1); pct("smAnnTrue", s_true * np.sqrt(12), 1)
    pct("smAnnModel", 0.04 * np.sqrt(12), 1)
    # measured beta and correlation of a smoothed asset against a serially independent market
    put("smBetaFactor", 1 - alpha, "{:.1f}"); put("smCorrFactor", np.sqrt(1 - alpha ** 2), "{:.1f}")
    put("smAlpha", alpha, "{:.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    m = np.arange(n)
    ax[0].plot(m[:120], 100 * true[:120], color=GREY, lw=0.9, label="economic return $R_t$")
    ax[0].plot(m[:120], 100 * obs[:120], color=BLUE, lw=1.4, label=r"reported return $R^{*}_t$, $\phi=0.6$")
    ax[0].set_xlabel("month"); ax[0].set_ylabel("return (%)"); ax[0].legend(fontsize=7.5, loc="lower left")
    ax[0].set_title("(a) Smoothing hides volatility")
    k = np.arange(1, 9)
    ax[1].bar(k - 0.18, L.acf(true, 8)[1:], width=0.36, color=GREY, label="$R_t$")
    ax[1].bar(k + 0.18, L.acf(obs, 8)[1:], width=0.36, color=BLUE, label="$R^{*}_t$")
    ax[1].plot(k, alpha ** k, color=RED, marker="o", ms=3, lw=1, label=r"theory $\phi^k$")
    band = 1.96 / np.sqrt(n); ax[1].axhspan(-band, band, color=PALE, zorder=0)
    ax[1].set_xlabel("lag $k$"); ax[1].set_title("(b) Autocorrelation"); ax[1].legend(fontsize=7.5)
    fig.tight_layout(); save(fig, "fig11_smoothing")

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    qs = np.arange(1, 37)
    for rho, col in [(-0.2, SKY), (0.0, BLUE), (0.3, RED), (0.6, GREY)]:
        ax.plot(qs, [L.variance_ratio_ar1(rho, q) for q in qs], color=col, label=f"$\\rho={rho}$")
    ax.set_xlabel("horizon $q$ (periods)"); ax.set_ylabel(r"variance ratio $VR(q)$")
    ax.legend(ncol=2); fig.tight_layout(); save(fig, "fig12_variance_ratio")
    put("vrTwelveSix", L.variance_ratio_ar1(0.6, 12), "{:.3f}")
    put("vrTwelveSixSqrt", np.sqrt(L.variance_ratio_ar1(0.6, 12)), "{:.3f}")
    put("vrLimitSix", (1 + 0.6) / (1 - 0.6), "{:.1f}")


# =============================================================== 1.14 JB size and power
def jb_size_power(rng):
    ns = [20, 30, 50, 100, 250, 500, 1000]
    reps = 20000
    crit = stats.chi2.ppf(0.95, 2); put("jbCrit", crit, "{:.3f}")
    # the listing's own two samples, drawn from a fresh generator with the chapter seed
    r2 = np.random.default_rng(101)
    for nm, x in [("Norm", r2.standard_normal(500)), ("T", r2.standard_t(5, 500))]:
        d0 = x - x.mean(); m2 = (d0 ** 2).mean()
        s0, k0 = (d0 ** 3).mean() / m2 ** 1.5, (d0 ** 4).mean() / m2 ** 2 - 3
        jb0 = len(x) / 6 * (s0 ** 2 + k0 ** 2 / 4)
        put(f"jbSnip{nm}", jb0, "{:.2f}"); put(f"jbSnip{nm}P", stats.chi2.sf(jb0, 2), "{:.4f}")
    size, power = [], []
    for n in ns:
        for dist, store in [("norm", size), ("t5", power)]:
            x = rng.standard_normal((reps, n)) if dist == "norm" else rng.standard_t(5, (reps, n))
            d = x - x.mean(1, keepdims=True)
            m2 = (d ** 2).mean(1); s = (d ** 3).mean(1) / m2 ** 1.5; k = (d ** 4).mean(1) / m2 ** 2 - 3
            jb = n / 6 * (s ** 2 + k ** 2 / 4)
            store.append(np.mean(jb > crit))
    for n, sz, pw in zip(ns, size, power):
        word = {20: "Twenty", 30: "Thirty", 50: "Fifty", 100: "Hundred", 250: "TwoFifty",
                500: "FiveHundred", 1000: "Thousand"}[n]
        pct("jbSize" + word, sz, 1); pct("jbPow" + word, pw, 1)
    RAW["jbSize"] = dict(zip(ns, size)); RAW["jbPower"] = dict(zip(ns, power))
    # n = 60, five years of monthly data, from its own stream so the grid above is unchanged
    r60 = np.random.default_rng([101, 60])
    for dist, nm in [("norm", "jbSizeSixty"), ("t5", "jbPowSixty")]:
        x = r60.standard_normal((reps, 60)) if dist == "norm" else r60.standard_t(5, (reps, 60))
        d = x - x.mean(1, keepdims=True)
        m2 = (d ** 2).mean(1); s = (d ** 3).mean(1) / m2 ** 1.5; k = (d ** 4).mean(1) / m2 ** 2 - 3
        pct(nm, np.mean(60 / 6 * (s ** 2 + k ** 2 / 4) > crit), 1)
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.semilogx(ns, 100 * np.array(size), color=BLUE, marker="o", ms=3.5, label="rejection rate, normal data (size)")
    ax.semilogx(ns, 100 * np.array(power), color=RED, marker="s", ms=3.5, label="rejection rate, Student $t_5$ data (power)")
    ax.axhline(5, color=GREY, ls=":", lw=1)
    ax.set_xlabel("sample size $n$"); ax.set_ylabel("rejections at 5% level (%)")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.legend(loc="center right")
    fig.tight_layout(); save(fig, "fig13_jb_size_power")


def main(rng=None):
    if rng is None:   # standalone: replay the draws made before this script in run_all.py
        import distributions
        rng = np.random.default_rng(SEED)
        distributions.main(rng)
    smoothing(rng); jb_size_power(rng)
    return finish('smoothing_normality')


if __name__ == "__main__":
    main()
