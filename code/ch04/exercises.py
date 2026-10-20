"""Answers to the exercises of Chapter 4. Every number quoted in the answers is computed here.

Run on its own:  python code/ch04/exercises.py   (writes code/ch04/out/exercises.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch04 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401


def main():
    # 4.1 bootstrap with annual coupons
    D1 = 1 / 1.03; D2 = (1 - 0.035 * D1) / 1.035
    c = L.bootstrap({1.0: 0.03}, {2.0: 0.035}, m=1)
    assert abs(c.D(2.0) - D2) < 1e-12
    put("exOneDOne", D1, "{:.6f}"); put("exOneDTwo", D2, "{:.6f}"); pct("exOneFwd", D1 / D2 - 1, 3)
    # 4.2 forward with a dividend yield
    put("exTwoF", L.forward_price(50, 0.05, 0.75, q=0.02), "{:.4f}")
    # 4.3 value of a long forward
    tau = 4 / 12
    put("exThreeV", 55 * math.exp(-0.02 * tau) - 52 * math.exp(-0.05 * tau), "{:.4f}")
    # 4.4 covered interest parity
    F = L.cip_forward(1.35, 0.03, 0.05, 0.5); put("exFourF", F, "{:.4f}")
    usd = 1_000_000; owe_cad = usd * 1.35 * (1 + 0.03 * 0.5); usd_end = usd * (1 + 0.05 * 0.5)
    put("exFourProfit", usd_end * 1.34 - owe_cad, "{:,.0f}"); put("exFourOwe", owe_cad, "{:,.0f}"); put("exFourGet", usd_end * 1.34, "{:,.0f}")
    # 4.5 two-state market
    psi = L.state_prices([[1, 110], [1, 90]], [1, 100])
    put("exFivePsiU", psi[0], "{:.2f}"); put("exFivePsiD", psi[1], "{:.2f}")
    put("exFiveCall", psi @ np.array([10, 0]), "{:.2f}"); put("exFivePut", psi @ np.array([0, 15]), "{:.2f}")
    # 4.6 implied convenience yield
    pct("exSixY", 0.04 + 0.02 - math.log(66 / 70), 2)
    # 4.7 one-period put
    q = (1.02 - 0.9) / 0.2; put("exSevenQ", q, "{:.2f}")
    put("exSevenP", (1 - q) * 5 / 1.02, "{:.4f}"); put("exSevenDelta", (0 - 5) / (55 - 45), "{:.2f}")
    # 4.8 two-step American put, K = 52
    params = (1.1, 0.9, q, 1 / 1.02)
    put("exEightAm", L.tree_american(50, 52, 2, params, "put"), "{:.4f}")
    put("exEightEu", L.tree_european(50, 52, 2, params, "put"), "{:.4f}")
    cont_d = (q * max(52 - 49.5, 0) + (1 - q) * max(52 - 40.5, 0)) / 1.02
    put("exEightContD", cont_d, "{:.4f}"); put("exEightExD", 52 - 45, "{:.2f}")
    cont_u = (q * max(52 - 60.5, 0) + (1 - q) * max(52 - 49.5, 0)) / 1.02
    put("exEightContU", cont_u, "{:.4f}")
    # 4.9 roll decomposition
    put("exNineEx", 100 * math.log(74 / 72), "{:.4f}"); put("exNineSpot", 100 * math.log(73 / 70), "{:.4f}")
    put("exNineRoll", 100 * (math.log(74 / 72) - math.log(73 / 70)), "{:.4f}")
    put("exNineBasisIn", 100 * math.log(72 / 70), "{:.4f}"); put("exNineBasisOut", 100 * math.log(74 / 73), "{:.4f}")
    return finish("exercises")


if __name__ == "__main__":
    main()
