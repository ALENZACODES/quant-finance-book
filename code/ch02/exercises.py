"""Section 2.16: numbers in the answers to the exercises.

Run on its own:  python code/ch02/exercises.py   (writes code/ch02/out/exercises.json)
run_all.py runs every script of the chapter and writes chapters/ch02-numbers.tex.
"""
from __future__ import annotations
import numpy as np
from numbers_ch02 import N, put, pct, word, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, SEED  # noqa: F401


# ============================================================ exercises
def exercises():
    # 2.1 fees on committed vs invested
    put("exOneA", 0.02 * 200 * 5 + 0.015 * 120 * 5, "{:.1f}")
    # 2.3 single period with V = 150, K = 100, T = 4, h = 8%
    K, T, V = 100.0, 4.0, 150.0
    H = K * 1.08 ** T
    r = L.european_waterfall([0], [K], [T], [V]); put("exThreeH", H); put("exThreeGP", r.gp[0])
    put("exThreeGPHard", L.european_waterfall([0], [K], [T], [V], hard=True).gp[0])
    put("exThreeVstar", (H - 0.2 * K) / 0.8)
    # 2.4 hedge fund: 100, +20%, -15%, +12%, 1.5% and 20%
    hf = L.hedge_fund_fees([0.20, -0.15, 0.12], mgmt=0.015)
    for t in range(3):
        put("exFourInc" + word(t + 1), hf["inc"][t]); put("exFourClose" + word(t + 1), hf["close"][t])
    # 2.6 annuity
    put("exSixPV", L.fee_annuity_pv(500, 0.08, 0.015, 0.15, 0.08, 5))
    put("exOneB", 200 - (0.02 * 200 * 5 + 0.015 * 120 * 5), "{:.0f}")
    # 2.7 vega peak at sigma 40%, r 3%
    put("exSevenRatio", np.exp(0.5 * 0.16 - 0.03), "{:.4f}")
    put("exSevenVegaPeak", 0.2 * float(L.bs_call_vega(100 * np.exp(0.05), 100, 1, 0.03, 0.4)) / 100, "{:.3f}")
    put("exSevenVegaDown", 0.2 * float(L.bs_call_vega(85, 100, 1, 0.03, 0.4)) / 100, "{:.3f}")


def main():
    exercises()
    return finish('exercises')


if __name__ == "__main__":
    main()
