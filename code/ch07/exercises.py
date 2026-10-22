"""Answers to the Chapter 7 exercises. No random numbers."""
import math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L


def main():
    # 7.1 two assets, minimum variance
    s1, s2, rho = 0.20, 0.10, 0.3
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    v = w1**2 * s1**2 + (1 - w1)**2 * s2**2 + 2 * w1 * (1 - w1) * rho * s1 * s2
    R.put("xrOneW", 100 * w1); R.put("xrOneVol", 100 * math.sqrt(v))
    # 7.2 three uncorrelated assets, tangency
    mu = np.array([0.04, 0.06, 0.08]); sd = np.array([0.10, 0.15, 0.20]); S = np.diag(sd**2)
    w = L.tangency(mu, S); sr = math.sqrt(L.sharpe2(mu, S))
    for i, nm in enumerate(["A", "B", "C"]):
        R.put(f"xrTwo{nm}", 100 * w[i], "{:.1f}")
    R.put("xrTwoSR", sr, "{:.3f}"); R.put("xrTwoMu", 100 * w @ mu); R.put("xrTwoVol", 100 * math.sqrt(w @ S @ w))
    # 7.3 betas on that tangency portfolio
    beta = S @ w / (w @ S @ w)
    for i, nm in enumerate(["A", "B", "C"]):
        R.put(f"xrThree{nm}", beta[i], "{:.3f}")
    R.RAW["xrThreeErr"] = float(np.max(np.abs(beta * (w @ mu) - mu)))
    # 7.4 years to detect alpha
    R.put("xrFour", L.detection_months(0.4 / math.sqrt(12), 0.45 / math.sqrt(12)) / 12, "{:.1f}")
    # 7.5 smoothing
    phi, b, prem = 0.5, 1.2, 0.06
    R.put("xrFiveB", (1 - phi) * b, "{:.2f}"); R.put("xrFiveA", 100 * phi * b * prem, "{:.1f}")
    R.put("xrFiveD", b * (1 - phi**3), "{:.2f}")
    # 7.6 Brinson, two sectors
    a, s, i = L.brinson([0.7, 0.3], [0.6, 0.4], [0.12, 0.01], [0.10, 0.02])
    R.put("xrSixAlloc", 100 * a.sum()); R.put("xrSixSel", 100 * s.sum()); R.put("xrSixInt", 100 * i.sum())
    R.put("xrSixActive", 100 * (0.7 * 0.12 + 0.3 * 0.01 - 0.6 * 0.10 - 0.4 * 0.02))
    R.put("xrSixRb", 100 * (0.6 * 0.10 + 0.4 * 0.02))
    # 7.7 Carino
    Rp, Rb = np.array([0.02, 0.03]), np.array([0.01, 0.01]); k, P, B = L.carino(Rp, Rb)
    R.put("xrSevenP", 100 * P, "{:.2f}"); R.put("xrSevenB", 100 * B, "{:.2f}"); R.put("xrSevenSum", 100 * (Rp - Rb).sum(), "{:.2f}")
    R.put("xrSevenKone", k[0], "{:.4f}"); R.put("xrSevenKtwo", k[1], "{:.4f}")
    R.put("xrSevenOne", 100 * k[0] * (Rp - Rb)[0], "{:.4f}"); R.put("xrSevenTwo", 100 * k[1] * (Rp - Rb)[1], "{:.4f}")
    R.RAW["xrSevenErr"] = float(abs((k * (Rp - Rb)).sum() - (P - B)))
    # 7.8 Black-Litterman, two uncorrelated assets, one view on the first
    S2 = np.diag([0.04, 0.09]); wm = np.array([0.5, 0.5]); delta, tau = 2.5, 0.05
    pi = delta * S2 @ wm; P = np.array([[1.0, 0.0]]); q = np.array([0.03]); Om = np.array([[tau * 0.04]])
    mu8 = L.bl_posterior(pi, S2, P, q, Om, tau); w8 = np.linalg.solve(delta * S2, mu8)
    R.put("xrEightPiA", 100 * pi[0], "{:.1f}"); R.put("xrEightPiB", 100 * pi[1], "{:.2f}")
    R.put("xrEightMuA", 100 * mu8[0], "{:.2f}"); R.put("xrEightWA", 100 * w8[0], "{:.1f}"); R.put("xrEightWB", 100 * w8[1], "{:.1f}")
    # 7.9 equal risk contributions, two assets
    s1, s2, rho = 0.20, 0.05, 0.2; S9 = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w9 = L.erc(S9); R.put("xrNineW", 100 * w9[0], "{:.0f}"); R.RAW["xrNineErr"] = float(abs(w9[0] - s2 / (s1 + s2)))
    R.put("xrNineVol", 100 * math.sqrt(w9 @ S9 @ w9))
    return R.finish("exercises")


if __name__ == "__main__":
    main()
