"""Section 6.4: the Merton model. Spread term structures, the short-maturity limit, and backing out
asset value and volatility from equity.

Run on its own:  python code/ch06/merton_model.py   (writes code/ch06/out/merton_model.json)
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize_scalar
from numbers_ch06 import N, RAW, sci, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

SIGV, R = 0.25, 0.03


def curves():
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    T = np.linspace(0.02, 20, 800)
    for Lv, col, ls in [(0.5, GREY, ":"), (0.7, BLUE, "-"), (0.9, RED, "--"), (1.1, ROSE, "-.")]:
        s = L.merton_spread_lev(Lv, SIGV, T)
        ax.plot(T, 1e4 * s, color=col, ls=ls, label=f"$\\ell={Lv}$")
    ax.set_ylim(0, 1500); ax.set_xlabel("maturity $T$ (years)"); ax.set_ylabel("credit spread (bp)")
    ax.legend(loc="upper right")
    save(fig, "fig06_merton_spreads")
    for Lv, nm in [(0.7, "Seven"), (0.9, "Nine")]:
        for Tm, tn in [(1 / 12, "Q"), (1, "One"), (5, "Five"), (10, "Ten")]:
            v = 1e4 * float(L.merton_spread_lev(Lv, SIGV, Tm))
            if tn == "Q" and v < 0.01:
                sci(f"mtS{nm}{tn}", v, 1)
            else:
                put(f"mtS{nm}{tn}", v, "{:.1f}")
        s = L.merton_spread_lev(Lv, SIGV, T); i = int(np.argmax(s))
        # refine the grid maximum, so the printed peak is the true one and not the nearest grid point
        opt = minimize_scalar(lambda t: -float(L.merton_spread_lev(Lv, SIGV, t)), bounds=(T[max(i - 1, 0)], T[i + 1]),
                              method="bounded", options={"xatol": 1e-8})
        put(f"mtPeakT{nm}", opt.x, "{:.1f}"); put(f"mtPeak{nm}", -1e4 * opt.fun, "{:.0f}")
        RAW[f"mtPeakTGrid{nm}"] = float(T[i])
    pct("mtSigV", SIGV, 0); pct("mtR", R, 0)


def calibrate():
    E, sigE, D, T, mu = 40.0, 0.60, 80.0, 5.0, 0.07
    V, sV = L.merton_calibrate(E, sigE, D, R, T)
    m = L.merton(V, D, R, sV, T)
    put("mcV", V); pct("mcSigV", sV, 1); put("mcSpread", 1e4 * m["spread"], "{:.0f}"); pct("mcPdQ", m["pdQ"], 1)
    dd = (math.log(V / D) + (mu - 0.5 * sV ** 2) * T) / (sV * math.sqrt(T))
    put("mcDD", dd); pct("mcPdP", L.ncdf(-dd), 1); pct("mcMu", mu, 0)
    RAW["mcResE"] = m["E"] - E; RAW["mcResVol"] = L.ncdf(m["d1"]) * sV * V - sigE * E
    put("mcLev", D * math.exp(-R * T) / V, "{:.3f}")


def main():
    curves(); calibrate()
    return finish("merton_model")


if __name__ == "__main__":
    main()
