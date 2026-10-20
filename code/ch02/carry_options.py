"""Section 2.7: carried interest as a portfolio of call options, with a Monte Carlo check.

Run on its own:  python code/ch02/carry_options.py   (writes code/ch02/out/carry_options.json)
run_all.py runs every script of the chapter and writes chapters/ch02-numbers.tex.
"""
from __future__ import annotations
import numpy as np
from numbers_ch02 import N, put, pct, word, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, SEED  # noqa: F401


# ============================================================ 2.7 carry as options
def carry_options(rng):
    K, T, h, r, cr = 100.0, 5.0, 0.08, 0.03, 0.20
    H = K * (1 + h) ** T
    sig = np.linspace(0.05, 0.6, 56)
    full = [L.carry_option_value(100, K, H, T, r, s) for s in sig]
    hard = [L.carry_option_value(100, K, H, T, r, s, hard=True) for s in sig]
    nohu = [cr * float(L.bs_call(100, K, T, r, s)) for s in sig]
    for s, nm in [(0.15, "Fifteen"), (0.30, "Thirty"), (0.45, "FortyFive")]:
        put("carryFull" + nm, L.carry_option_value(100, K, H, T, r, s))
        put("carryHard" + nm, L.carry_option_value(100, K, H, T, r, s, hard=True))
    put("carryNoHurdleThirty", cr * float(L.bs_call(100, K, T, r, 0.30)))
    # Monte Carlo check at sigma = 30%
    n = 500_000                                   # pairs of antithetic draws
    z = rng.standard_normal(n)
    pays = []
    for zz in (z, -z):
        VT = 100 * np.exp((r - 0.5 * 0.09) * T + 0.3 * np.sqrt(T) * zz)
        pays.append(np.exp(-r * T) * L.carry_payoff(VT, K, H))
    pay = 0.5 * (pays[0] + pays[1])
    put("carryMCThirty", pay.mean()); put("carryMCse", pay.std() / np.sqrt(n), "{:.3f}")
    put("carryMCn", 2 * n, "{:,.0f}")
    put("carryMCz", (pay.mean() - L.carry_option_value(100, K, H, T, r, 0.30)) / (pay.std() / np.sqrt(n)))
    # sensitivity to volatility (per point of volatility) at two portfolio values, sigma = 15%
    def vega(V0, s, e=1e-5):
        return (L.carry_option_value(V0, K, H, T, r, s + e) - L.carry_option_value(V0, K, H, T, r, s - e)) / (2 * e) / 100
    put("carryVegaLow", vega(100, 0.15), "{:.3f}"); put("carryVegaHigh", vega(200, 0.15), "{:.3f}")
    put("carryVegaHighThirty", vega(200, 0.30), "{:.3f}")
    put("carryVstar", (H - cr * K) / (1 - cr))
    # V0 above cost: value creation by the GP
    put("carryFullThirtyRich", L.carry_option_value(115, K, H, T, r, 0.30))
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8))
    ax[0].plot(100 * sig, full, color=BLUE, label="hurdle, full catch-up")
    ax[0].plot(100 * sig, hard, color=RED, ls="-.", label="hard hurdle")
    ax[0].plot(100 * sig, nohu, color=GREY, ls=":", label="no hurdle")
    mc_s = [0.1, 0.2, 0.3, 0.4, 0.5]
    mc_v = []
    for s in mc_s:
        VT = 100 * np.exp((r - 0.5 * s * s) * T + s * np.sqrt(T) * z[:100_000])
        mc_v.append(np.exp(-r * T) * L.carry_payoff(VT, K, H).mean())
    ax[0].plot(100 * np.array(mc_s), mc_v, "o", ms=3.5, color=BLUE, mfc="white", label="Monte Carlo")
    ax[0].set_xlabel("volatility of fund value (% a year)"); ax[0].set_ylabel("value of carry")
    ax[0].set_title("(a) Carry value rises with risk"); ax[0].legend(loc="upper left", fontsize=7.2)
    V0 = np.linspace(50, 220, 300)
    for s, col, ls in [(0.15, SKY, "--"), (0.30, BLUE, "-"), (0.45, RED, "-.")]:
        val = [L.carry_option_value(v, K, H, T, r, s) for v in V0]
        ax[1].plot(V0, val, color=col, ls=ls, label=f"$\\sigma$ = {int(100*s)}%")
    ax[1].plot(V0, L.carry_payoff(V0 * np.exp(r * T), K, H) * np.exp(-r * T), color=GREY, lw=0.8, ls=":",
               label="no volatility")
    ax[1].set_xlabel("current value of the portfolio, $V_0$"); ax[1].set_ylabel("value of carry")
    ax[1].set_title("(b) Value against portfolio value"); ax[1].legend(loc="upper left", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig04_carry_option")


def main():
    carry_options(np.random.default_rng(SEED))
    return finish('carry_options')


if __name__ == "__main__":
    main()
