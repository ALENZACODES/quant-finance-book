"""Sections 3.3 and 3.4: Levy's construction of Brownian motion, its covariance, and quadratic and
total variation along refining partitions.

Run on its own:  python code/ch03/brownian_qv.py   (writes code/ch03/out/brownian_qv.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch03 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401


def construction(rng):
    t, W, hist = L.levy_bm(10, 1.0, rng, m=1, record=True)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    cols = {0: GREY, 1: SKY, 3: ROSE, 10: BLUE}
    for lev, col in cols.items():
        h = hist[lev][0]
        ax.plot(np.linspace(0, 1, len(h)), h, color=col, lw=1.2 if lev < 10 else 0.8,
                marker="o" if lev <= 3 else None, ms=2.5, label=f"level {lev}: {len(h) - 1} intervals")
    ax.set_xlabel("$t$"); ax.set_ylabel("$W_t$"); ax.legend(fontsize=7.4, loc="best")
    ax.set_title("Lévy's construction: refining the same path by midpoints")
    fig.tight_layout(); save(fig, "fig03_levy")
    # distributional checks of the construction
    m = 0
    acc = np.zeros(4); M = 40000
    for _ in range(4):
        _, Wm = L.levy_bm(8, 1.0, rng, m=M // 4)
        acc += [np.sum(Wm[:, 128] ** 2), np.sum(Wm[:, 64] * Wm[:, 192]), np.sum((Wm[:, 256] - Wm[:, 128]) * Wm[:, 128]),
                np.sum(Wm[:, 256] ** 2)]
        m += Wm.shape[0]
    acc /= m
    put("levyVarHalf", acc[0], "{:.3f}"); put("levyCovQ", acc[1], "{:.3f}")
    put("levyIncCov", acc[2], "{:.3f}"); put("levyVarOne", acc[3], "{:.3f}")
    put("levyPaths", M, "{:,d}")
    RAW["levyBridgeVar"] = 0.5 / 4          # conditional variance of W(1/2) given W(1/4), W(3/4)


def variation(rng):
    levels = 16
    t, W, hist = L.levy_bm(levels, 1.0, rng, m=1, record=True)
    ks = np.arange(1, levels + 1)
    qv = np.array([L.quadratic_variation(hist[k][0]) for k in ks])
    tv = np.array([L.total_variation(hist[k][0]) for k in ks])
    n = 2.0 ** ks
    for k, nm in [(4, "Four"), (8, "Eight"), (12, "Twelve"), (16, "Sixteen")]:
        put("qvPath" + nm, qv[k - 1], "{:.4f}"); put("tvPath" + nm, tv[k - 1], "{:.1f}")
    put("tvMeanSixteen", math.sqrt(2 * 2 ** 16 / math.pi), "{:.1f}")
    # Monte Carlo: E(QV_n - T)^2 = 2 T^2 / n
    nn = 2 ** np.arange(2, 12); M = 20000
    msd = []
    for k in nn:
        e = []
        for _ in range(4):
            _, Wm = L.bm_paths(int(k), M // 4, 1.0, rng)
            e.append(L.quadratic_variation(Wm) - 1.0)
        e = np.concatenate(e); msd.append(np.mean(e ** 2))
    msd = np.array(msd)
    RAW["qvMSE"] = msd.tolist(); RAW["qvNs"] = nn.tolist()
    put("qvMSEratio", np.max(np.abs(msd * nn / 2 - 1)), "{:.3f}")
    put("qvSlope", L.ols_slope(nn, np.sqrt(msd)), "{:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].semilogx(n, qv, "o-", color=BLUE, ms=3, lw=1, label="quadratic variation")
    ax[0].axhline(1.0, color=GREY, ls="--", lw=0.8)
    ax[0].set_xscale("log", base=2); ax[0].set_ylim(0, 2.0); ax[0].set_xlabel("intervals $n$ in $[0,1]$")
    ax[0].set_ylabel("quadratic variation", color=BLUE); ax[0].set_title("(a) One path, refined")
    a2 = ax[0].twinx(); a2.semilogx(n, tv, "s-", color=RED, ms=3, lw=1); a2.set_yscale("log")
    a2.set_ylabel("total variation", color=RED); a2.grid(False)
    a2.spines["right"].set_visible(True)
    ax[1].loglog(nn, np.sqrt(msd), "o", color=BLUE, ms=3, label="simulated")
    ax[1].loglog(nn, np.sqrt(2 / nn), color=RED, lw=1, label=r"$\sqrt{2/n}$")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("intervals $n$"); ax[1].set_ylabel(r"root mean square of $Q_n-1$")
    ax[1].set_title(r"(b) $L^2$ convergence"); ax[1].legend(fontsize=7.5)
    fig.tight_layout(); save(fig, "fig04_variation")


def main():
    rng = rng_for(2)
    construction(rng); variation(rng)
    return finish("brownian_qv")


if __name__ == "__main__":
    main()
