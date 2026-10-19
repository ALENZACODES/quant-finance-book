"""Sections 1.9, 1.10 and 1.13: lognormal laws, moments, put writing, tail bounds and correlation.

Run on its own:  python code/ch01/distributions.py   (writes code/ch01/out/distributions.json)
run_all.py runs every script of the chapter, in a fixed order with one random generator, and writes
chapters/ch01-numbers.tex.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from numbers_ch01 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, SEED  # noqa: F401


# =============================================================== 1.9 lognormal
def lognormal():
    mu, sig = 0.06, 0.18
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.linspace(0.01, 6, 1200)
    for h, col in [(1, BLUE), (5, RED), (20, GREY)]:
        dist = stats.lognorm(s=sig * np.sqrt(h), scale=np.exp(mu * h))
        ax.plot(x, dist.pdf(x), color=col, label=f"{h} year horizon")
        m, sd, sk, ek = L.lognormal_moments(mu * h, sig * np.sqrt(h))
        nm = {1: "One", 5: "Five", 20: "Twenty"}[h]
        put("lnMean" + nm, m, "{:.3f}"); put("lnMedian" + nm, np.exp(mu * h), "{:.3f}")
        put("lnMode" + nm, np.exp(mu * h - sig ** 2 * h), "{:.3f}"); put("lnSkew" + nm, sk, "{:.2f}")
        put("lnExk" + nm, ek, "{:.2f}")
    ax.set_xlim(0, 6); ax.set_xlabel("terminal wealth per unit invested, $1+R$"); ax.set_ylabel("density")
    ax.legend(); fig.tight_layout(); save(fig, "fig06_lognormal")


# =============================================================== 1.10 moments gallery
def moments_gallery():
    x = np.linspace(-6, 6, 2001)
    nu = 5
    t5 = stats.t(df=nu, scale=np.sqrt((nu - 2) / nu))
    lap = stats.laplace(scale=1 / np.sqrt(2))
    uni = stats.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3))
    nor = stats.norm()
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for dist, lab, col, ls in [(uni, "uniform (platykurtic)", GREY, "-"), (nor, "normal (mesokurtic)", BLUE, "-"),
                               (lap, "Laplace (leptokurtic)", SKY, "--"), (t5, "Student $t_5$ (leptokurtic)", RED, "-")]:
        ax[0].plot(x, dist.pdf(x), color=col, ls=ls, label=lab)
        ax[1].semilogy(x, dist.pdf(x), color=col, ls=ls)
    ax[0].set_xlim(-4, 4); ax[0].set_title("(a) Unit variance densities"); ax[0].legend(fontsize=7, loc="upper left")
    ax[1].set_xlim(0, 6); ax[1].set_ylim(1e-6, 1); ax[1].set_title("(b) Right tail, log scale")
    for a in ax: a.set_xlabel("standardised return")
    fig.tight_layout(); save(fig, "fig07_kurtosis")
    put("exkTfive", 6 / (nu - 4), "{:.0f}")
    # tail probabilities beyond 4 sd
    put("tailNormFour", 2 * nor.sf(4) * 1e5, "{:.1f}")   # per 100,000
    put("tailTFour", 2 * t5.sf(4) * 1e5, "{:.0f}")


# =============================================================== nonlinearity: put writing
def put_writing(rng):
    S0, K, Tm, r, sig_true, sig_imp, mu = 100.0, 95.0, 1 / 12, 0.03, 0.16, 0.18, 0.07
    p = L.bs_put(S0, K, Tm, r, sig_imp)
    put("putPrem", p, "{:.3f}")
    n = 200_000
    z = rng.standard_normal(n)
    ST = S0 * np.exp((mu - 0.5 * sig_true ** 2) * Tm + sig_true * np.sqrt(Tm) * z)
    R = ((K + p) * np.exp(r * Tm) - np.maximum(K - ST, 0)) / K - 1
    s = L.sample_moments(R)
    pct("pwMean", s["mean"], 3); pct("pwSD", s["sd"], 3); put("pwSkew", s["skew"]); put("pwExk", s["exkurt"])
    pct("pwProbLoss", np.mean(R < 0), 1)
    sample = R[:240]
    jb, pv = L.jarque_bera(sample); put("pwJB", jb, "{:.1f}"); RAW["pwJBp"] = pv
    s240 = L.sample_moments(sample); put("pwSkewS", s240["skew"]); put("pwExkS", s240["exkurt"])
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax[0].hist(100 * R, bins=160, density=True, color=PALE, edgecolor=BLUE, linewidth=0.3)
    xx = np.linspace(100 * R.min(), 100 * R.max(), 400)
    ax[0].plot(xx, stats.norm(100 * s["mean"], 100 * s["sd"]).pdf(xx), color=RED, lw=1.1, label="normal, same mean and sd")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-4, 5e1)
    ax[0].set_xlim(-12, 3); ax[0].set_xlabel("monthly return (%)"); ax[0].set_ylabel("density (log scale)")
    ax[0].set_title("(a) Cash-secured put writing"); ax[0].legend(fontsize=7.5, loc="upper left")
    q = np.sort(sample); th = stats.norm.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q))
    ax[1].plot(th, (q - q.mean()) / q.std(ddof=1), "o", ms=2.2, color=BLUE)
    ax[1].plot([-3, 3], [-3, 3], color=RED, lw=1)
    ax[1].set_xlabel("normal quantile"); ax[1].set_ylabel("standardised sample quantile")
    ax[1].set_title("(b) Q-Q plot, 240 months")
    fig.tight_layout(); save(fig, "fig08_putwriting")


# =============================================================== 1.11 correlation
def correlation(rng):
    mkt = np.array([2.0, -1.5, 3.0, 0.5, -2.5, 1.5])
    fnd = np.array([1.2, -0.4, 2.1, 0.8, -1.0, 1.1])
    cov = np.cov(fnd, mkt, ddof=1)[0, 1]; var = np.var(mkt, ddof=1)
    put("exCov", cov, "{:.4f}"); put("exVarM", var, "{:.4f}"); put("exBeta", cov / var, "{:.4f}")
    put("exRho", np.corrcoef(fnd, mkt)[0, 1], "{:.4f}")
    put("exSpear", stats.spearmanr(fnd, mkt)[0], "{:.4f}")
    put("exMeanM", mkt.mean(), "{:.4f}"); put("exMeanF", fnd.mean(), "{:.4f}")
    put("exSdF", fnd.std(ddof=1), "{:.4f}"); put("exSdM", mkt.std(ddof=1), "{:.4f}")
    RAW["exMkt"] = mkt.tolist(); RAW["exFnd"] = fnd.tolist()

    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.3))
    n = 300
    for a, rho in zip(ax[:2], [-0.7, 0.7]):
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        a.plot(z[:, 0], z[:, 1], "o", ms=1.8, color=BLUE, alpha=0.7)
        a.set_title(f"$\\rho={rho:+.1f}$"); a.set_xlim(-3.5, 3.5); a.set_ylim(-3.5, 3.5)
    xs = np.sort(rng.uniform(0, 3, 60)); ys = np.exp(2.2 * xs)
    ax[2].plot(xs, ys, "o", ms=2.2, color=RED)
    pr = np.corrcoef(xs, ys)[0, 1]; sr = stats.spearmanr(xs, ys)[0]
    ax[2].set_title(f"Pearson {pr:.2f}, Spearman {sr:.2f}")
    put("convexPearson", pr); put("convexSpearman", sr)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); save(fig, "fig09_correlation")

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    nn = np.arange(1, 51)
    for rho, col in [(0.0, BLUE), (0.2, SKY), (0.5, RED), (0.8, GREY)]:
        ax.plot(nn, 100 * L.portfolio_sd_equal(0.20, rho, nn), color=col, label=f"$\\rho={rho}$")
        ax.axhline(100 * 0.20 * np.sqrt(rho), color=col, lw=0.6, ls=":")
    ax.set_xlabel("number of assets, $N$"); ax.set_ylabel("portfolio volatility (%)")
    ax.legend(ncol=2); fig.tight_layout(); save(fig, "fig10_diversification")
    pct("divTenTwo", L.portfolio_sd_equal(0.20, 0.2, 10), 2); pct("divFloorTwo", 0.2 * np.sqrt(0.2), 2)


# =============================================================== chebyshev table
def chebyshev():
    for k, nm in [(1, "One"), (2, "Two"), (3, "Three")]:
        pct("cheb" + nm, min(1.0, 1 / k ** 2), 1)
        pct("normOut" + nm, 2 * stats.norm.sf(k), 2)
        # two fat-tailed laws scaled to unit variance: beyond one sd they have LESS mass than the normal
        pct("lapOut" + nm, 2 * stats.laplace(scale=1 / np.sqrt(2)).sf(k), 2)
        pct("tOut" + nm, 2 * stats.t(5, scale=np.sqrt(3 / 5)).sf(k), 2)


def main(rng=None):
    rng = rng if rng is not None else np.random.default_rng(SEED)
    lognormal(); moments_gallery(); put_writing(rng); correlation(rng); chebyshev()
    return finish('distributions')


if __name__ == "__main__":
    main()
