"""Section 5.6: continuous dividend yields, discrete cash dividends, Black-76 and Garman-Kohlhagen.

Run on its own:  python code/ch05/dividends_black.py   (writes code/ch05/out/dividends_black.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch05 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

S0, R, SIG, T = 100.0, 0.05, 0.20, 1.0
DIVS = [(0.25, 2.0), (0.75, 2.0)]
NPATH = 1_000_000


def yields():
    q = 0.03
    put("divYieldCall", float(L.bs(S0, 100, R, SIG, T, "call", q)), "{:.4f}")
    put("divYieldPut", float(L.bs(S0, 100, R, SIG, T, "put", q)), "{:.4f}")
    put("divYieldDelta", float(L.greeks(S0, 100, R, SIG, T, "call", q)["delta"]), "{:.4f}")
    put("divYieldFwd", S0 * math.exp((R - q) * T), "{:.4f}")


def black_gk():
    F, K, r, s, T_ = 70.0, 72.0, 0.04, 0.35, 0.5
    c, p = L.black76(F, K, r, s, T_), L.black76(F, K, r, s, T_, "put")
    put("blackCall", c, "{:.4f}"); put("blackPut", p, "{:.4f}")
    RAW["blackParityGap"] = float(c - p - math.exp(-r * T_) * (F - K))
    RAW["blackAsYield"] = float(L.bs(F, K, r, s, T_, "call", q=r))
    # currency option: Canadian dollars per US dollar
    Sx, Kx, rd, rf, sx, Tx = 1.38, 1.40, 0.025, 0.040, 0.07, 0.25
    put("gkCall", 1e6 * float(L.gk(Sx, Kx, rd, rf, sx, Tx)), "{:,.0f}")
    put("gkPut", 1e6 * float(L.gk(Sx, Kx, rd, rf, sx, Tx, "put")), "{:,.0f}")
    put("gkFwd", Sx * math.exp((rd - rf) * Tx), "{:.4f}")
    put("gkDelta", float(L.greeks(Sx, Kx, rd, sx, Tx, "call", rf)["delta"]), "{:.4f}")
    put("gkHedge", 1e6 * RAW["gkDelta"], "{:,.0f}")
    RAW["gkParityGap"] = float(L.gk(Sx, Kx, rd, rf, sx, Tx) - L.gk(Sx, Kx, rd, rf, sx, Tx, "put")
                               - math.exp(-rd * Tx) * (Sx * math.exp((rd - rf) * Tx) - Kx))


def drop_model_mc(Ks, rng):
    """Stock follows GBM with volatility SIG and drops by each cash dividend on its ex date.
    Returns discounted call prices and standard errors for each strike (antithetic pairs)."""
    acc = np.zeros(len(Ks)); acc2 = np.zeros(len(Ks)); m_tot = 0; floored = 0
    for _ in range(NPATH // 200_000):
        m = 100_000
        z = rng.standard_normal((3, m)); times = [0.0] + [t for t, _ in DIVS] + [T]
        pay = []
        for sign in (1, -1):
            S = np.full(m, S0)
            for i in range(3):
                dt = times[i + 1] - times[i]
                S = S * np.exp((R - 0.5 * SIG ** 2) * dt + SIG * math.sqrt(dt) * sign * z[i])
                if i < 2:
                    floored += int(np.sum(S < DIVS[i][1]))   # convention: a stock worth less than the dividend goes to zero and stays there
                    S = np.maximum(S - DIVS[i][1], 0.0)
            pay.append(np.maximum(S[None, :] - Ks[:, None], 0))
        Y = math.exp(-R * T) * 0.5 * (pay[0] + pay[1])
        acc += Y.sum(1); acc2 += (Y ** 2).sum(1); m_tot += m
    mean = acc / m_tot
    RAW["dropFloored"] = floored; RAW["dropFloorPaths"] = 2 * m_tot
    assert floored == 0   # the text says no path fell below a dividend with these inputs
    return mean, np.sqrt((acc2 / m_tot - mean ** 2) / (m_tot - 1))


def discrete():
    pv = sum(D * math.exp(-R * t) for t, D in DIVS)
    put("divPV", pv, "{:.4f}")
    esc = L.escrowed(S0, 100, R, SIG, T, DIVS); put("escCall", esc, "{:.4f}")
    sadj = SIG * S0 / (S0 - pv); put("sigAdj", 100 * sadj, "{:.2f}")
    adj = float(L.bs(S0 - pv, 100, R, sadj, T)); put("adjCall", adj, "{:.4f}")
    Ks = np.linspace(70, 130, 25)
    mc, se = drop_model_mc(Ks, rng_for(1))
    i = int(np.argmin(abs(Ks - 100)))
    put("dropCall", mc[i], "{:.4f}"); put("dropSE", se[i], "{:.4f}")
    put("noDivCall", float(L.bs(S0, 100, R, SIG, T)), "{:.4f}")
    escK = np.array([float(L.escrowed(S0, k, R, SIG, T, DIVS)) for k in Ks])
    adjK = np.array([float(L.bs(S0 - pv, k, R, sadj, T)) for k in Ks])
    RAW["dropMinusEscMax"] = float((mc - escK).max()); RAW["dropMinusAdjMaxAbs"] = float(abs(mc - adjK).max())
    put("dropEscGap", mc[i] - escK[i], "{:.3f}"); put("dropAdjGap", mc[i] - adjK[i], "{:.3f}")
    j70, j130 = 0, len(Ks) - 1
    put("dropAdjGapLow", mc[j70] - adjK[j70], "{:.3f}"); put("dropAdjGapHigh", mc[j130] - adjK[j130], "{:.3f}")
    RAW["dropSEmax"] = float(se.max())
    put("dropPaths", NPATH, "{:,d}"); put("dropPairs", NPATH // 2, "{:,d}")
    # an upper bound, so round up: the text says no error exceeds this
    gmax = max(float(abs(mc - escK).max()), RAW["dropMinusAdjMaxAbs"]); RAW["dropGapMaxRaw"] = gmax
    put("dropGapMax", math.ceil(gmax * 100) / 100, "{:.2f}")
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.axhline(0, color=GREY, lw=0.8)
    ax.fill_between(Ks, -2 * se, 2 * se, color=PALE, label=r"$\pm 2$ standard errors")
    ax.plot(Ks, mc - escK, color=RED, marker="o", ms=3, label="spot drop minus escrowed")
    ax.plot(Ks, mc - adjK, color=BLUE, marker="s", ms=3, label="spot drop minus escrowed, adjusted volatility")
    ax.set_xlabel("strike $K$"); ax.set_ylabel("price difference")
    ax.set_ylim(-0.2, 0.26)
    # legend below the axes, where no curve can run through it
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5, frameon=False); fig.tight_layout(); save(fig, "fig04_dividends")


def main():
    yields(); black_gk(); discrete()
    return finish("dividends_black")


if __name__ == "__main__":
    main()
