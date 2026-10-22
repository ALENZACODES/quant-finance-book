"""Answers to the exercises of Chapter 11, computed from the library so that the text and the code agree."""
import math
import numpy as np
import numbers_ch11 as R
from numbers_ch11 import L, put, pct, finish


def main():
    # 11.1 a whole fund waterfall with two distributions
    w = L.european_waterfall([0, 3, 6], [100, 0, 0], [0, 60, 120], 0.08, 0.20, 1.0)
    put("yaRocThree", w["roc"][1], "{:.1f}"); put("yaRocSix", w["roc"][2], "{:.1f}")
    put("yaPref", w["pref"][2], "{:.2f}"); put("yaCatch", w["catch"][2], "{:.2f}")
    put("yaRest", 120 - w["roc"][2] - w["pref"][2] - w["catch"][2], "{:.2f}")
    put("yaGp", w["gp_total"], "{:.2f}"); put("yaLp", w["lp_total"], "{:.2f}")
    # the waterfall listing's fund: 50 at year 3 and 140 at year 6, a profit of 90
    wl = L.european_waterfall([0, 3, 6], [100, 0, 0], [0, 50, 140], 0.08, 0.20, 1.0)
    put("wlGp", wl["gp_total"], "{:.2f}"); put("wlLp", wl["lp_total"], "{:.2f}"); put("wlProfit", 90, "{:.0f}")
    bal3 = 100 * 1.08 ** 3 - 60; put("yaBalThree", bal3, "{:.2f}"); put("yaBalSix", bal3 * 1.08 ** 3, "{:.2f}")
    put("yaLpIrr", 100 * L.irr([0, 3, 6], [-100, 60 - 0, 120 - w["catch"][2] - w["gp_split"][2]]), "{:.2f}")
    # 11.2 deal by deal and the clawback
    r = 0.08; put("ybCarryEarly", 10.0, "{:.0f}"); put("ybFinal", 4.0, "{:.0f}"); put("ybClaw", 6.0, "{:.0f}")
    put("ybPv", 10 * (math.exp(-2 * r) - math.exp(-5 * r)), "{:.3f}")
    # 11.3 implied duration
    put("ycDur", math.log(1.8) / math.log(1.15), "{:.2f}")
    # 11.4 PME and direct alpha
    put("ydPme", 160 / 130, "{:.3f}"); pct("ydAlpha", math.log(160 / 130) / 4, 2); pct("ydIrr", (1.6) ** 0.25 - 1, 2)
    # 11.5 beta bias
    put("yePme", L.pme_beta_mean(0.0, 1.5, 0.07, 0.16, 5, 0.04), "{:.3f}")
    put("yePmeLow", L.pme_beta_mean(0.0, 0.5, 0.07, 0.16, 5, 0.04), "{:.3f}")
    # 11.6 leverage
    k = (1 + 0.07 * 0.75) ** 5; put("yfK", k, "{:.3f}"); put("yfMoic", L.lbo_moic(1.8, 0.6, 0.07, 0.25, 5), "{:.2f}")
    put("yfMoicLow", L.lbo_moic(1.2, 0.6, 0.07, 0.25, 5), "{:.2f}")
    # 11.7 bridge
    g, m, d = L.shapley_bridge(40, 9, 55, 10, 200, 120)
    put("ygG", g, "{:.1f}"); put("ygM", m, "{:.1f}"); put("ygD", d, "{:.0f}"); put("ygTot", g + m + d, "{:.0f}")
    # 11.8 value driver
    put("yhV", 10 * L.ev_nopat_multiple(0.08, 0.03, 0.15), "{:.1f}"); put("yhVflat", 10 / 0.08, "{:.1f}")
    put("yhVhigh", 10 * L.ev_nopat_multiple(0.08, 0.05, 0.15), "{:.1f}")
    # 11.9 Miles and Ezzell against continuous rebalancing
    pct("yiMe", L.miles_ezzell_wacc(0.09, 0.06, 0.25, 0.40), 3); pct("yiHp", 0.09 - 0.40 * 0.06 * 0.25, 3)
    # 11.10 unlevering
    put("yjHam", L.unlever_beta(1.3, 0.5, 0.25, rule="hamada"), "{:.3f}")
    put("yjHp", L.unlever_beta(1.3, 0.5, 0.25, rule="harris_pringle"), "{:.3f}")
    # 11.11 normalisation
    e = 8.0 + (0.9 - 0.35) - 0.3; put("ykE", e, "{:.2f}"); put("ykV", 7 * e, "{:.2f}")
    # 11.12 combined discounts
    dloc = 1 - 1 / 1.3; pct("ylDloc", dloc, 1); pct("ylComb", L.combined_discount(dloc, 0.25), 1)
    put("ylStake", 0.15 * 60 * (1 - L.combined_discount(dloc, 0.25)), "{:.2f}")
    # 11.13 the put and the lookback
    pct("ymPut", L.chaffe_dlom(2, 0.40, r=0.04), 1); pct("ymLs", L.longstaff_bound(2, 0.40), 1)
    # 11.14 growth that destroys value
    put("ynLow", L.ev_nopat_multiple(0.09, 0.02, 0.07), "{:.2f}"); put("ynHigh", L.ev_nopat_multiple(0.09, 0.04, 0.07), "{:.2f}")
    put("ynFlat", 1 / 0.09, "{:.2f}")
    return finish("exercises")


if __name__ == "__main__":
    main()
