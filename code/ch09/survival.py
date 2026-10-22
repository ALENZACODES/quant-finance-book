"""Section 9.4: funds that close after a deep loss. The log value of each fund is a Brownian motion with
drift; a fund closes the first time it has lost a given fraction of its starting value. Checks the
survival probability and the survivors' mean against a simulation monitored monthly, with and without
the continuity correction of Broadie, Glasserman and Kou. Writes fig12_closure.png."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, put, pct, finish

MU, SIG, LOSS, YEARS = 0.0, 0.15, 0.40, 10   # no skill; 15% volatility; close after losing 40%
N_FUNDS = 20_000


def main():
    rng = R.rng_for(12)
    b = -math.log(1 - LOSS); dt = 1 / 12
    pct("clSig", SIG, 0); pct("clLoss", LOSS, 0); put("clYears", YEARS, "{:.0f}"); put("clB", b, "{:.3f}")
    put("clFunds", N_FUNDS, "{:,.0f}")
    steps = YEARS * 12
    X = np.cumsum(MU * dt + SIG * math.sqrt(dt) * rng.standard_normal((N_FUNDS, steps)), axis=1)
    alive = np.minimum.accumulate(X, axis=1) > -b
    yrs = np.arange(1, YEARS + 1)
    sim_s, sim_m, th_s, th_m, bgk_s, bgk_m = [], [], [], [], [], []
    bb = b + L.BGK * SIG * math.sqrt(dt)
    for T in yrs:
        k = alive[:, 12 * T - 1]
        sim_s.append(k.mean()); sim_m.append(X[k, 12 * T - 1].mean() / T)
        th_s.append(L.survival_barrier(MU, SIG, b, T)); th_m.append(L.survivor_mean_barrier(MU, SIG, b, T) / T)
        bgk_s.append(L.survival_barrier(MU, SIG, bb, T)); bgk_m.append(L.survivor_mean_barrier(MU, SIG, bb, T) / T)
    sim_s, sim_m, th_s, th_m, bgk_s, bgk_m = map(np.array, (sim_s, sim_m, th_s, th_m, bgk_s, bgk_m))
    for T, tag in ((5, "Five"), (10, "Ten")):
        i = T - 1
        pct(f"clSurv{tag}", th_s[i], 1); pct(f"clSurvSim{tag}", sim_s[i], 1); pct(f"clSurvBgk{tag}", bgk_s[i], 1)
        pct(f"clMean{tag}", th_m[i], 2); pct(f"clMeanSim{tag}", sim_m[i], 2); pct(f"clMeanBgk{tag}", bgk_m[i], 2)
    put("clSurvSe", 100 * math.sqrt(sim_s[-1] * (1 - sim_s[-1]) / N_FUNDS), "{:.1f}")
    put("clMaxGapBgk", 100 * np.max(np.abs(bgk_s - sim_s)), "{:.1f}")
    # the survivors' arithmetic return: the log mean plus half the variance, as a simple return per year
    pct("clDrift", MU, 0)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ax[0].plot(yrs, 100 * th_s, color=GREY, ls="--", lw=0.8, label="formula, continuous")
    ax[0].plot(yrs, 100 * bgk_s, color=BLUE, label="formula, corrected")
    ax[0].plot(yrs, 100 * sim_s, "o", ms=3, color=RED, label="simulation, monthly")
    ax[0].set_xlabel("years"); ax[0].set_ylabel("funds still open, %"); ax[0].set_ylim(0, 100)
    ax[0].legend(loc="lower left")
    ax[1].plot(yrs, 100 * th_m, color=GREY, ls="--", lw=0.8)
    ax[1].plot(yrs, 100 * bgk_m, color=BLUE)
    ax[1].plot(yrs, 100 * sim_m, "o", ms=3, color=RED)
    ax[1].axhline(100 * MU, color=GREY, lw=0.6)
    ax[1].set_xlabel("years"); ax[1].set_ylabel("survivors' mean log return, % a year")
    fig.tight_layout(); R.save(fig, "fig12_closure"); plt.close(fig)
    return finish("survival")


if __name__ == "__main__":
    print(main()["N"])
