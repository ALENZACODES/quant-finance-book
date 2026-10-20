"""Answers to the Chapter 6 exercises.

Run on its own:  python code/ch06/exercises.py   (writes code/ch06/out/exercises.json)
"""
from __future__ import annotations
import math
from scipy import stats
from numbers_ch06 import N, RAW, put, pct, save, finish, L  # noqa: F401


def main():
    # 6.1
    p0 = L.bond_price(0.05, 0.04, 5); D = L.mod_duration(0.05, 0.04, 5); C = L.convexity(0.05, 0.04, 5)
    put("xsOneP", p0, "{:.4f}"); put("xsOneD", D, "{:.4f}"); put("xsOneC", C, "{:.3f}")
    put("xsOneDur", p0 * (1 - D * 0.005), "{:.4f}"); put("xsOneQd", p0 * (1 - D * 0.005 + 0.5 * C * 0.005 ** 2), "{:.4f}")
    put("xsOneEx", L.bond_price(0.055, 0.04, 5), "{:.4f}")
    # 6.2
    put("xsTwoD", L.par_duration(0.04, 2), "{:.6f}"); put("xsTwoNum", L.mod_duration(0.04, 0.04, 2), "{:.6f}")
    # 6.3
    a1, a2 = L.barbell(0.04, 6, 1, 10, L=50.0); put("xsThreeA", a1, "{:.3f}"); put("xsThreeB", a2, "{:.3f}")
    put("xsThreePV", 50 * math.exp(-0.24), "{:.3f}")
    # 6.4
    put("xsFour", 1e4 * L.holee_adjustment(0.01, 3, 3.25), "{:.3f}")
    # 6.5
    m = L.merton(120, 100, 0.03, 0.20, 3)
    put("xsFiveE", m["E"], "{:.3f}"); put("xsFiveDebt", m["debt"], "{:.3f}"); put("xsFiveS", 1e4 * m["spread"], "{:.1f}")
    pct("xsFivePd", m["pdQ"], 2)
    # 6.6
    lam = 0.015 / 0.6; pct("xsSixL", lam, 2); pct("xsSixS", math.exp(-5 * lam), 2)
    # 6.7
    q = stats.norm.cdf((stats.norm.ppf(0.02) + math.sqrt(0.2) * stats.norm.ppf(0.99)) / math.sqrt(0.8))
    pct("xsSeven", q, 2); RAW["xsSevenCheck"] = float(L.lhp_cdf(q, 0.02, 0.2)) - 0.99
    # 6.8
    d = dict(loan=7e6, value=10e6, noi=6e5, rate=7.0, amort=0, termYrs=3, capRate=6, exitCapShift=0, noiGrowth=0,
             fees=0, sponsorScore=3)
    mm = L.screener_metrics(d)
    put("xsEightLtv", mm["ltv"], "{:.0f}"); put("xsEightDy", mm["dy"], "{:.3f}"); put("xsEightDscr", mm["dscr"], "{:.3f}")
    # 6.9
    B = L.balloon(1e6, 4.0, 25, 60); k1 = L.annual_debt_service(1.0, 6.0, 25)
    put("xsNineB", B, "{:,.0f}"); put("xsNineK", 100 * k1, "{:.3f}"); put("xsNineMax", 9e4 / (1.25 * k1), "{:,.0f}")
    put("xsNineCover", 9e4 / (1.25 * k1) / B, "{:.3f}")
    return finish("exercises")


if __name__ == "__main__":
    main()
