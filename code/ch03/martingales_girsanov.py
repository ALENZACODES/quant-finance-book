"""Sections 3.8 and 3.9: martingales, optional stopping and exit problems, and Girsanov's theorem
used as a change of drift and for importance sampling.

Run on its own:  python code/ch03/martingales_girsanov.py   (writes code/ch03/out/martingales_girsanov.json)
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats, integrate
from numbers_ch03 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401


def martingales(rng):
    M = 200000
    W1 = rng.standard_normal(M)
    put("mgSq", np.mean(W1 ** 2 - 1), "{:.4f}"); put("mgSqSE", np.std(W1 ** 2 - 1) / math.sqrt(M), "{:.4f}")
    Z = np.exp(W1 - 0.5); put("mgExp", Z.mean(), "{:.4f}"); put("mgExpSE", Z.std() / math.sqrt(M), "{:.4f}")
    Z3 = np.exp(3 * W1 - 4.5); put("mgExpThree", Z3.mean(), "{:.3f}"); put("mgExpThreeSE", Z3.std() / math.sqrt(M), "{:.3f}")
    put("mgExpThreeSD", math.sqrt(math.exp(9) - 1), "{:.1f}")
    put("mgExpThreeTrueSE", math.sqrt(math.exp(9) - 1) / math.sqrt(M), "{:.3f}")
    pct("mgExpThreeShare", np.mean(Z3 > 1), 1)
    put("mgPaths", M, "{:,d}")
    put("mgSERatio", math.sqrt(math.exp(9) - 1) / math.sqrt(M) / (Z3.std() / math.sqrt(M)), "{:.1f}")
    # repeat the experiment: how often does the sample standard error mislead?
    reps, true_se = 400, math.sqrt(math.exp(9) - 1) / math.sqrt(M)
    means, ses = np.empty(reps), np.empty(reps); rep_rng = rng_for(99)   # own stream: leaves the other draws alone
    for j in range(reps):
        z = np.exp(3 * rep_rng.standard_normal(M) - 4.5)
        means[j], ses[j] = z.mean(), z.std(ddof=1) / math.sqrt(M)
    put("mgReps", reps, "{:d}")
    pct("mgSEHalfShare", np.mean(ses < 0.5 * true_se), 0)
    pct("mgRejShare", np.mean(np.abs(means - 1) > 1.96 * ses), 0)
    RAW["mgRepMeanZ"] = float((means.mean() - 1) / (means.std(ddof=1) / math.sqrt(reps)))


def exits(rng):
    a = b = 0.10; sig = 0.20
    mus = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    mc_p, mc_t = [], []
    for mu in mus:
        p, t = L.exit_mc(mu, sig, a, b, 8000, 1e-3, rng); mc_p.append(p); mc_t.append(t)
    mc_p, mc_t = np.array(mc_p), np.array(mc_t)
    fp = np.array([L.exit_prob_up(m, sig, a, b) for m in mus]); ft = np.array([L.exit_time_mean(m, sig, a, b) for m in mus])
    RAW["exitMus"] = mus.tolist(); RAW["exitMCp"] = mc_p.tolist(); RAW["exitMCt"] = mc_t.tolist()
    RAW["exitZ"] = float(np.max(np.abs(mc_p - fp) / np.sqrt(fp * (1 - fp) / 8000)))
    pct("exitPup", fp[4], 2); pct("exitPupMC", mc_p[4], 2); put("exitTau", ft[4], "{:.4f}"); put("exitTauMC", mc_t[4], "{:.4f}")
    put("exitTauZero", ft[3], "{:.4f}"); put("exitTauZeroMC", mc_t[3], "{:.4f}"); put("exitPaths", 8000, "{:,d}")
    mm = np.linspace(-0.4, 0.4, 161)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].plot(mm, [100 * L.exit_prob_up(m, sig, a, b) for m in mm], color=RED, lw=1.1, label="formula")
    ax[0].plot(mus, 100 * mc_p, "o", color=BLUE, ms=3.5, label="simulation")
    ax[0].set_xlabel(r"drift $\mu$"); ax[0].set_ylabel(r"$P(\mathrm{hit}\ a\ \mathrm{first})$ (%)")
    ax[0].set_title(r"(a) Exit at the upper level"); ax[0].legend(fontsize=7.4)
    ax[1].plot(mm, [L.exit_time_mean(m, sig, a, b) for m in mm], color=RED, lw=1.1, label="formula")
    ax[1].plot(mus, mc_t, "o", color=BLUE, ms=3.5, label="simulation")
    ax[1].set_xlabel(r"drift $\mu$"); ax[1].set_ylabel(r"$\mathbb{E}[\tau]$ (years)")
    ax[1].set_title("(b) Expected exit time"); ax[1].legend(fontsize=7.4)
    fig.tight_layout(); save(fig, "fig12_exit")


def girsanov(rng):
    # change of drift: E[f(W_T + mu T)] = E[f(W_T) exp(mu W_T - mu^2 T / 2)], f(x) = max(x - 0.5, 0)
    T, mu, M = 1.0, 0.4, 400000
    f = lambda x: np.maximum(x - 0.5, 0)
    W = rng.standard_normal(M) * math.sqrt(T)
    direct = f(W + mu * T); weighted = f(W) * np.exp(mu * W - 0.5 * mu ** 2 * T)
    exact = integrate.quad(lambda x: f(x) * stats.norm.pdf(x, mu * T, math.sqrt(T)), 0.5, 12)[0]
    put("girExact", exact, "{:.4f}"); put("girDirect", direct.mean(), "{:.4f}"); put("girWeighted", weighted.mean(), "{:.4f}")
    put("girDirectSE", direct.std() / math.sqrt(M), "{:.4f}"); put("girWeightedSE", weighted.std() / math.sqrt(M), "{:.4f}")
    zw = (weighted.mean() - exact) / (weighted.std() / math.sqrt(M))
    put("girWeightedZ", abs(zw), "{:.1f}"); put("girWeightedOdds", 1 / (2 * stats.norm.sf(abs(zw))), "{:.0f}")
    put("girWeightedOddsOne", 1 / stats.norm.sf(abs(zw)), "{:.0f}"); RAW["girWeightedSigned"] = float(zw)
    put("girPaths", M, "{:,d}")
    # importance sampling for p = P(W_1 > c)
    c = 3.0; p = stats.norm.sf(c)
    put("isP", 1e4 * p, "{:.2f}")
    sizes = np.array([1000, 3000, 10000, 30000, 100000, 300000])
    plain, isv, ps, iss = [], [], [], []
    for m in sizes:
        x = rng.standard_normal(m); e1 = (x > c).astype(float)
        y = rng.standard_normal(m) + c; e2 = (y > c) * np.exp(-c * y + 0.5 * c ** 2)
        plain.append(e1.mean()); ps.append(e1.std() / math.sqrt(m)); isv.append(e2.mean()); iss.append(e2.std() / math.sqrt(m))
    plain, isv, ps, iss = map(np.array, (plain, isv, ps, iss))
    var_plain = p * (1 - p)
    var_is = math.exp(c ** 2) * stats.norm.sf(2 * c) - p ** 2       # E_Q[(1 Z)^2] - p^2, exact
    put("isRatio", var_plain / var_is, "{:,.0f}")
    put("isPlainBig", 1e4 * plain[-1], "{:.2f}"); put("isPlainBigSE", 1e4 * ps[-1], "{:.2f}")
    put("isBig", 1e4 * isv[-1], "{:.2f}"); put("isBigSE", 1e4 * iss[-1], "{:.3f}")
    put("isSmall", 1e4 * isv[0], "{:.2f}"); put("isSmallSE", 1e4 * iss[0], "{:.2f}")
    put("isPlainSmall", 1e4 * plain[0], "{:.1f}")
    put("isPlainSmallSE", 1e4 * ps[0], "{:.1f}")
    put("isPlainSmallTrueSE", 1e4 * math.sqrt(var_plain / sizes[0]), "{:.1f}")
    RAW["isVarIS"] = var_is; RAW["isEmpRatio"] = float((ps[-1] / iss[-1]) ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    x = np.linspace(-4, 7, 600)
    ax[0].plot(x, stats.norm.pdf(x), color=BLUE, lw=1.1, label=r"law of $W_1$ under $P$")
    ax[0].plot(x, stats.norm.pdf(x, c), color=RED, lw=1.1, label=r"law of $W_1$ under $Q$")
    xx = x[x > c]; ax[0].fill_between(xx, 0, stats.norm.pdf(xx), color=BLUE, alpha=0.5, lw=0)
    ax[0].axvline(c, color=GREY, ls=":", lw=0.8); ax[0].set_xlabel("$x$"); ax[0].legend(fontsize=7.0, loc="upper right")
    ax[0].set_ylim(0, 0.55); ax[0].set_title("(a) Moving the mass to the event")
    ax[1].errorbar(sizes, 1e4 * plain, yerr=2e4 * ps, fmt="o", color=BLUE, ms=3, lw=0.8, capsize=2, label="plain Monte Carlo")
    ax[1].errorbar(sizes * 1.15, 1e4 * isv, yerr=2e4 * iss, fmt="s", color=RED, ms=3, lw=0.8, capsize=2, label="importance sampling")
    ax[1].axhline(1e4 * p, color=GREY, ls="--", lw=0.8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("sample size"); ax[1].set_ylabel(r"estimate of $P(W_1>3)\times 10^4$")
    ax[1].set_title("(b) Estimates with two standard errors"); ax[1].legend(fontsize=7.0)
    fig.tight_layout(); save(fig, "fig11_girsanov")


def main():
    rng = rng_for(5)
    martingales(rng); exits(rng); girsanov(rng)
    return finish("martingales_girsanov")


if __name__ == "__main__":
    main()
