"""Section 3.2: simple random walks, scaling towards Brownian motion and the reflection principle.

Run on its own:  python code/ch03/random_walks.py   (writes code/ch03/out/random_walks.json)
run_all.py runs every script of the chapter and writes chapters/ch03-numbers.tex.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
from numbers_ch03 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401


def scaling(rng):
    # (a) scaled walks S_[nt] / sqrt(n) for three n; (b) exact Kolmogorov distance to the normal
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    for n, col in [(16, GREY), (256, RED), (4096, BLUE)]:
        S = L.random_walk(n, 1, rng)[0]
        ax[0].plot(np.linspace(0, 1, n + 1), S / math.sqrt(n), color=col, lw=0.9 if n > 100 else 1.2,
                   drawstyle="steps-post" if n == 16 else "default", label=f"$n={n}$")
    ax[0].set_xlabel("$t$"); ax[0].set_ylabel(r"$S_{\lfloor nt\rfloor}/\sqrt{n}$")
    ax[0].set_title("(a) Rescaled random walks"); ax[0].legend(loc="lower left", fontsize=7.5)
    ns = np.unique(np.round(np.logspace(1, 4, 25)).astype(int))
    ks = []
    for n in ns:
        k = np.arange(n + 1); x = (2 * k - n) / math.sqrt(n)
        F = np.cumsum(stats.binom.pmf(k, n, 0.5))
        # the CDF jumps at each x: compare both one-sided limits with Phi
        Phi = stats.norm.cdf(x)
        ks.append(max(np.max(np.abs(F - Phi)), np.max(np.abs(np.r_[0, F[:-1]] - Phi))))
    ks = np.array(ks)
    slope = L.ols_slope(ns, ks)
    ax[1].loglog(ns, ks, "o", color=BLUE, ms=3, label="exact distance")
    ax[1].loglog(ns, 0.4 / np.sqrt(ns), color=RED, lw=1, label=r"$0.4/\sqrt{n}$")
    ax[1].set_xlabel("number of steps $n$"); ax[1].set_ylabel(r"$\sup_x |F_n(x)-\Phi(x)|$")
    ax[1].set_title("(b) Distance to the normal law"); ax[1].legend(fontsize=7.5)
    fig.tight_layout(); save(fig, "fig01_rw_scaling")
    put("rwKolSlope", slope, "{:.3f}")
    for n, nm in [(10, "Ten"), (100, "Hundred"), (1000, "Thousand")]:
        k = np.arange(n + 1); x = (2 * k - n) / math.sqrt(n)
        F = np.cumsum(stats.binom.pmf(k, n, 0.5))
        put("rwCdf" + nm, F[x <= 1.0][-1], "{:.4f}")
    put("normCdfOne", stats.norm.cdf(1.0), "{:.4f}")
    # moments of S_n by simulation, a sanity check quoted in the text
    S = L.random_walk(400, 20000, rng)
    put("rwVarFourHundred", S[:, -1].var(), "{:.1f}")
    put("rwCovCheck", np.mean(S[:, 100] * S[:, 300]), "{:.1f}")


def reflection(rng):
    n, a = 20, 4
    put("reflN", n, "{:d}"); put("reflA", a, "{:d}")
    exact = L.srw_max_tail(n, a); pct("reflExact", exact, 2)
    pct("reflNaive", stats.binom.sf((n + a) / 2 - 1, n, 0.5), 2)   # P(S_n >= a)
    mc = np.mean([np.mean(L.random_walk(n, 50000, rng).max(axis=1) >= a) for _ in range(8)])
    pct("reflMC", mc, 2)
    pct("reflMCse", math.sqrt(mc * (1 - mc) / 400000), 2)
    RAW["reflEnumerate"] = L.srw_max_tail_enumerate(n, a)
    RAW["reflSixteen"] = L.srw_max_tail(16, 3)

    # Brownian motion: P(max W >= 1) on [0, 1], monitored at n dates
    aa, T = 1.0, 1.0
    cont = L.bm_max_tail(aa, T); pct("bmMaxCont", cont, 2)
    ns = [8, 16, 32, 64, 128, 256, 512, 1024]
    mcv, se, bgk = [], [], []
    for n in ns:
        hits = 0
        for _ in range(8):                       # 8 chunks of 5,000 paths keep memory small
            _, W = L.bm_paths(n, 5000, T, rng); hits += int(np.sum(W.max(axis=1) >= aa))
        p = hits / 40000; mcv.append(p); se.append(math.sqrt(p * (1 - p) / 40000))
        bgk.append(L.bm_max_tail_discrete(aa, T, n))
    mcv, se, bgk = map(np.array, (mcv, se, bgk))
    for n, nm in [(16, "Sixteen"), (1024, "Big")]:
        i = ns.index(n); pct("bmMaxMC" + nm, mcv[i], 2); pct("bmMaxBGK" + nm, bgk[i], 2)
    put("bgkBeta", L.BGK_BETA, "{:.4f}")
    zs = (np.abs(mcv - bgk) / se)[np.array(ns) >= 128]; RAW["bmMaxZ"] = float(np.max(zs))
    put("bmMaxZmax", RAW["bmMaxZ"], "{:.2f}"); pct("bmMaxSeBig", se[ns.index(1024)], 2)
    put("bmMaxZAt", [n for n in ns if n >= 128][int(np.argmax(zs))], "{:,.0f}")   # the grid with the largest gap
    RAW["bmMaxMC"] = mcv.tolist(); RAW["bmMaxBGK"] = bgk.tolist(); RAW["bmMaxNs"] = ns

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    # (a) one path that hits the level, and its reflection after the hitting time
    while True:
        t, W = L.bm_paths(1000, 1, 1.0, rng); W = W[0]
        hit = np.argmax(W >= 0.8)
        if W.max() >= 0.8 and 150 < hit < 650 and W[-1] > 0.85:
            break
    R = W.copy(); R[hit:] = 2 * 0.8 - W[hit:]
    ax[0].plot(t, W, color=BLUE, lw=0.9, label="path $W$")
    ax[0].plot(t[hit:], R[hit:], color=RED, lw=0.9, label="reflected after $\\tau_a$")
    ax[0].axhline(0.8, color=GREY, ls="--", lw=0.8); ax[0].axvline(t[hit], color=GREY, ls=":", lw=0.8)
    ax[0].text(0.02, 0.84, "$a$", color=GREY, fontsize=8.5); ax[0].text(t[hit] + 0.01, 1.45, r"$\tau_a$", color=GREY, fontsize=8.5)
    ax[0].set_xlabel("$t$"); ax[0].set_title("(a) The reflection principle"); ax[0].legend(loc="upper left", fontsize=7.2); ax[0].set_ylim(top=max(1.7, W.max() + 0.25))
    ax[1].errorbar(ns, 100 * mcv, yerr=200 * se, fmt="o", color=BLUE, ms=3, lw=0.8, label="simulated, $n$ dates")
    ax[1].semilogx(ns, 100 * bgk, color=RED, lw=1, label="corrected formula")
    ax[1].axhline(100 * cont, color=GREY, ls="--", lw=0.9, label="continuous maximum")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("monitoring dates $n$")
    ax[1].set_ylabel(r"$P(\max W \geq 1)$ (%)"); ax[1].set_title("(b) Discrete against continuous")
    ax[1].legend(loc="lower right", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig02_reflection")


def main():
    rng = rng_for(1)
    scaling(rng); reflection(rng)
    return finish("random_walks")


if __name__ == "__main__":
    main()
