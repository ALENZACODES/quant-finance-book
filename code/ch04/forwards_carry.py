"""Section 4.4 and 4.5: forward prices by cash and carry, the value of a forward, the no-arbitrage band
with frictions, covered interest parity, and futures against forwards when interest rates move.

Run on its own:  python code/ch04/forwards_carry.py   (writes code/ch04/out/forwards_carry.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch04 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

KAPPA, THETA, R0, SIGR, SIGS = 0.3, 0.03, 0.03, 0.01, 0.20


def carry_examples():
    # equity index with a dividend yield
    F = L.forward_price(100, 0.04, 0.5, q=0.025); put("fwdIndex", F, "{:.4f}")
    # value of that forward three months later with the index at 103
    St, t = 103.0, 0.25
    Ft = L.forward_price(St, 0.04, 0.5 - t, q=0.025)
    put("fwdIndexLater", Ft, "{:.4f}")
    put("fwdValueLater", (Ft - F) * math.exp(-0.04 * (0.5 - t)), "{:.4f}")
    put("fwdValueAlt", St * math.exp(-0.025 * (0.5 - t)) - F * math.exp(-0.04 * (0.5 - t)), "{:.4f}")
    # commodity: implied convenience yield from a quoted forward
    S, F1, r, u, T = 80.0, 78.0, 0.04, 0.01, 0.5
    y = r + u - math.log(F1 / S) / T
    pct("convYield", y, 2); put("convNoY", L.forward_price(S, r, T, u=u), "{:.4f}")
    # frictions band
    lo, hi = L.forward_band(99.95, 100.05, 0.038, 0.044, 1.0)
    put("bandLo", lo, "{:.4f}"); put("bandHi", hi, "{:.4f}"); put("bandMid", L.forward_price(100, 0.041, 1.0), "{:.4f}")
    quote = 104.80
    put("bandQuote", quote, "{:.2f}")
    put("bandProfit", quote - hi, "{:.4f}")          # cash and carry per unit at expiry
    Ts = np.linspace(0, 2, 201)
    band = np.array([L.forward_band(99.95, 100.05, 0.038, 0.044, T) for T in Ts])
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.fill_between(Ts, band[:, 0], band[:, 1], color=PALE, lw=0, label="no-arbitrage band")
    ax.plot(Ts, band[:, 1], color=RED, lw=1.2, label="cash and carry bound")
    ax.plot(Ts, band[:, 0], color=BLUE, lw=1.2, label="reverse cash and carry bound")
    ax.plot(Ts, 100 * np.exp(0.041 * Ts), color=GREY, lw=0.9, ls="--", label="frictionless forward at 4.1%")
    ax.plot([1.0], [quote], "o", ms=4, color=RED)
    ax.annotate("quote 104.80", (1.0, quote), xytext=(1.12, quote + 1.4), fontsize=7.6, color=RED,
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6))
    ax.set_xlabel("maturity (years)"); ax.set_ylabel("forward price")
    ax.set_title("Forward prices with a spot spread and unequal borrowing and lending rates")
    ax.legend(loc="upper left", fontsize=7.4); ax.set_xlim(0, 2)
    fig.tight_layout(); save(fig, "fig03_band")


def cip():
    S, i_d, i_f, tau = 1.3700, 0.0275, 0.0400, 1.0
    F = L.cip_forward(S, i_d, i_f, tau); put("cipFwd", F, "{:.4f}")
    put("cipPoints", 1e4 * (F - S), "{:.0f}")
    quote = 1.3600; put("cipQuote", quote, "{:.4f}")
    # the quote is above parity, so sell USD forward: borrow CAD, buy 1,000,000 USD spot, invest the USD,
    # and deliver the proceeds under the forward
    usd = 1_000_000; owe_cad = usd * S * (1 + i_d * tau); usd_end = usd * (1 + i_f * tau)
    profit_cad = usd_end * quote - owe_cad
    put("cipProfit", profit_cad, "{:,.0f}"); put("cipOweCAD", owe_cad, "{:,.0f}")
    put("cipUSDEnd", usd_end, "{:,.0f}"); put("cipReceive", usd_end * quote, "{:,.0f}")
    put("cipBorrow", usd * S, "{:,.0f}")


def fut_fwd(rng):
    Ts = np.linspace(0.05, 10, 200); rhos = [-0.5, 0.0, 0.5]
    for T, nm in [(1, "One"), (5, "Five"), (10, "Ten")]:
        for rho, rn in zip(rhos, ["Neg", "Zero", "Pos"]):
            put(f"ff{nm}{rn}", 1e4 * L.fut_fwd_logratio(KAPPA, SIGR, SIGS, rho, T), "{:.1f}")
    mc = {}; paths = 120000; put("ffPaths", paths, "{:,.0f}")
    for T in [1, 5, 10]:
        for rho in [-0.5, 0.5]:
            fut, fwd, se = L.fut_fwd_mc(KAPPA, THETA, R0, SIGR, SIGS, rho, T, 12 * T, paths, rng)
            mc[(T, rho)] = (1e4 * math.log(fut / fwd), 1e4 * se)
    put("ffMCFivePos", mc[(5, 0.5)][0], "{:.1f}"); put("ffMCFivePosSE", mc[(5, 0.5)][1], "{:.1f}")
    put("ffMCFivePosZ", abs(mc[(5, 0.5)][0] - 1e4 * L.fut_fwd_logratio(KAPPA, SIGR, SIGS, 0.5, 5)) / mc[(5, 0.5)][1], "{:.1f}")
    put("ffMCFiveNeg", mc[(5, -0.5)][0], "{:.1f}"); put("ffMCFiveNegSE", mc[(5, -0.5)][1], "{:.1f}")
    RAW["ffMCz"] = float(max(abs(m - 1e4 * L.fut_fwd_logratio(KAPPA, SIGR, SIGS, rho, T)) / s
                             for (T, rho), (m, s) in mc.items()))
    # forward under constant rates equals futures: the variance term vanishes with sig_r = 0
    RAW["ffDeterministic"] = float(L.fut_fwd_logratio(KAPPA, 0.0, SIGS, 0.5, 5))
    # for which correlation are the two equal at T = 5?
    k = KAPPA; T = 5.0
    intB = (T - L.vasicek_B(k, T)) / k; intB2 = (T - 2 * L.vasicek_B(k, T) + L.vasicek_B(2 * k, T)) / k ** 2
    put("ffRhoStar", -SIGR * intB2 / (SIGS * intB), "{:.3f}")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for rho, col, ls in zip(rhos, [BLUE, GREY, RED], ["-", "--", "-"]):
        ax.plot(Ts, 1e4 * L.fut_fwd_logratio(KAPPA, SIGR, SIGS, rho, Ts), color=col, ls=ls, lw=1.3,
                label=f"$\\rho={rho:+.1f}$" if rho else "$\\rho=0$")
    for (T, rho), (m, s) in mc.items():
        ax.errorbar([T], [m], yerr=[2 * s], fmt="o", ms=3.5, color=RED if rho > 0 else BLUE, mfc="white", capsize=2)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("maturity $T$ (years)"); ax.set_ylabel("ln(futures / forward), basis points")
    ax.set_title("Futures less forward with a Vasicek short rate (dots: simulation)")
    ax.legend(loc="upper left", fontsize=7.8); ax.set_xlim(0, 10.2)
    fig.tight_layout(); save(fig, "fig04_futfwd")


def main():
    carry_examples(); cip(); fut_fwd(rng_for(3))
    return finish("forwards_carry")


if __name__ == "__main__":
    main()
