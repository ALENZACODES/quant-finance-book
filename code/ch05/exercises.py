"""Answers to the Chapter 5 exercises.

Run on its own:  python code/ch05/exercises.py   (writes code/ch05/out/exercises.json)
"""
from __future__ import annotations
import math
from numbers_ch05 import N, RAW, put, finish, L  # noqa: F401


def main():
    # 5.1 prices and parity
    S, K, r, s, T = 50.0, 55.0, 0.04, 0.25, 0.5
    c, p = float(L.bs(S, K, r, s, T)), float(L.bs(S, K, r, s, T, "put"))
    d1, d2 = L.d12(S, K, r, 0.0, s, T)
    put("xvOneDOne", d1, "{:.4f}"); put("xvOneDTwo", d2, "{:.4f}")
    put("xvOneCall", c, "{:.4f}"); put("xvOnePut", p, "{:.4f}"); put("xvOneParity", S - K * math.exp(-r * T), "{:.4f}")
    # 5.2 delta-gamma approximation for a one unit rise
    g = L.greeks(S, K, r, s, T)
    put("xvTwoDelta", g["delta"], "{:.4f}"); put("xvTwoGamma", g["gamma"], "{:.4f}")
    put("xvTwoApprox", c + g["delta"] + 0.5 * g["gamma"], "{:.4f}"); put("xvTwoExact", float(L.bs(S + 1, K, r, s, T)), "{:.4f}")
    put("xvTwoDeltaOnly", c + g["delta"], "{:.4f}")
    # 5.3 where gamma peaks
    put("xvThreeS", K * math.exp(-(r + 1.5 * s ** 2) * T), "{:.2f}")
    # 5.4 Black-76 at the money
    F, K4, r4, s4, T4 = 80.0, 80.0, 0.03, 0.30, 0.25
    put("xvFourCall", L.black76(F, K4, r4, s4, T4), "{:.4f}")
    put("xvFourApprox", math.exp(-r4 * T4) * F * s4 * math.sqrt(T4) / math.sqrt(2 * math.pi), "{:.4f}")
    # 5.5 currency put on one million US dollars
    put("xvFivePut", 1e6 * float(L.gk(1.36, 1.36, 0.03, 0.045, 0.08, 0.5, "put")), "{:,.0f}")
    put("xvFiveFwd", 1.36 * math.exp((0.03 - 0.045) * 0.5), "{:.4f}")
    # 5.7 explicit stability
    s7, dx = 0.30, 0.01
    dt = dx ** 2 / s7 ** 2; put("xvSevenDt", dt, "{:.5f}"); put("xvSevenSteps", math.ceil(1 / dt), "{:d}")
    # 5.8 implied volatility by Newton
    sig, it, path = L.iv_newton(8.0, 100, 105, 0.03, 0.75)
    put("xvEightSig", 100 * sig, "{:.4f}"); put("xvEightIter", it, "{:d}")
    put("xvEightStart", 100 * path[0] / math.sqrt(0.75), "{:.2f}"); put("xvEightFirst", 100 * path[1] / math.sqrt(0.75), "{:.4f}")
    RAW["xvEightCheck"] = float(L.bs(100, 105, 0.03, sig, 0.75)) - 8.0
    # 5.9 rebalancing needed for a target hedging error
    sd = L.hedge_sd_theory(100, 100, 0.05, 0.2, 1.0, 52, mu=0.10)
    put("xvNineSd", sd, "{:.3f}"); put("xvNineN", math.ceil(52 * (sd / 0.25) ** 2), "{:d}")
    put("xvNinePerDay", RAW["xvNineN"] / 252, "{:.1f}")
    put("xvNineCheck", L.hedge_sd_theory(100, 100, 0.05, 0.2, 1.0, math.ceil(52 * (sd / 0.25) ** 2), mu=0.10), "{:.3f}")
    # 5.10 seller overcharging volatility
    put("xvTenGap", float(L.bs(100, 100, 0.05, 0.25, 1) - L.bs(100, 100, 0.05, 0.20, 1)), "{:.4f}")
    return finish("exercises")


if __name__ == "__main__":
    main()
