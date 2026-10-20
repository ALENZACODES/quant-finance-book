"""Tests for implied_vol.py."""
import math
import numpy as np
from conftest import close
import ch05lib as L


def test_inversion(RAW):
    assert RAW["ivGridWorst"] < 1e-12 and RAW["ivNonMonotone"] == 0
    assert RAW["ivNewtonMax"] <= 8 and RAW["ivExErr"] < 1e-14
    close(RAW["ivExStepC"], 0.3, 1e-6)


def test_newton_monotone_random():
    rng = np.random.default_rng(11)
    for _ in range(300):
        T = rng.uniform(0.02, 3); s = rng.uniform(0.03, 1.2); K = 100 * math.exp(rng.uniform(-1, 1) * s * math.sqrt(T))
        kind = "call" if rng.random() < 0.5 else "put"
        p = float(L.bs(100, K, 0.02, s, T, kind))
        sig, it, path = L.iv_newton(p, 100, K, 0.02, T, kind)
        assert abs(sig - s) < 1e-10
        d = np.diff(path[:-1])
        assert np.all(d >= -1e-15) or np.all(d <= 1e-15)
        # every Newton iterate stays on the start's side of the root
        v = s * math.sqrt(T); assert all((x - v) * (path[0] - v) >= -1e-12 for x in path)


def test_bounds():
    lo, hi = L.iv_bounds(100, 90, 0.05, 1.0)
    assert lo == 100 - 90 * math.exp(-0.05) and hi == 100


def test_smiles(RAW):
    assert RAW["skewMerShort"] > RAW["skewMerMid"] > RAW["skewMerLong"] > 0
    assert RAW["skewHesShort"] > 0 and RAW["skewHesLong"] > 0
    assert RAW["atmMerShort"] < RAW["atmMerLong"] < RAW["merTotalVol"]


def test_density(RAW):
    close(RAW["densInt"], 1.0, 2e-3); close(RAW["densMean"], RAW["densFwd"], 2e-2)
    # the raw second difference integrates to the discount factor, the scaled one to one
    close(RAW["densIntRaw"], math.exp(-0.05 * 0.5), 1e-4); close(RAW["densDisc"], math.exp(-0.05 * 0.5), 1e-12)
    assert RAW["densMinNeg"] > -1e-6 and RAW["callIncr"] == 0 and RAW["callConvexViol"] == 0
    assert RAW["tailHes"] > RAW["tailLogn"] and RAW["rtailHes"] < RAW["rtailLogn"]


def test_heston_checks(RAW):
    assert abs(RAW["hesMC"] - RAW["hesFourier"]) < 3.5 * RAW["hesMCSE"]
    assert abs(RAW["lewisBSGap"]) < 1e-12
    ratio = RAW["hesSmallXiGapA"] / RAW["hesSmallXiGapB"]; assert 8 < ratio < 12.5
    p = L.heston(100, 95, 0.05, 0.5, 0.04, 2, 0.04, 0.5, -0.7, "put")
    c = L.heston(100, 95, 0.05, 0.5, 0.04, 2, 0.04, 0.5, -0.7, "call")
    close(c - p, 100 - 95 * math.exp(-0.025), 1e-10)
    # Merton with no jumps is Black-Scholes
    close(L.merton_jd(100, 105, 0.05, 0.2, 1, 0.0, -0.1, 0.1), float(L.bs(100, 105, 0.05, 0.2, 1)), 1e-12)


def test_brenner_rule_needs_forward_strike(RAW):
    # struck at the forward the rule is close to 20%; struck at spot it is far off
    close(RAW["bsApprox"], 19.967, 1e-3)
    close(RAW["bsApproxStrike"], 100 * math.exp(0.05), 1e-9)
    assert 26.0 < RAW["bsApproxSpot"] < 26.4


def test_heston_tails_exact(RAW, NUM):
    # 5.116: the printed tails come from the characteristic function at the exact cut points
    import implied_vol as I
    T = 0.5; F = I.S0 * math.exp(I.R * T); h = 1e-3
    for K, name, lower in ((0.8 * F, "tailHes", True), (1.2 * F, "rtailHes", False)):
        p = L.heston_cdf(I.S0, K, I.R, T, **I.HESTON)
        # Q(S_T > K) = -e^{rT} dC/dK, by a central difference of the pricer itself
        dC = (L.heston(I.S0, K + h, I.R, T, **I.HESTON) - L.heston(I.S0, K - h, I.R, T, **I.HESTON)) / (2 * h)
        close(1 - p, -math.exp(I.R * T) * dC, 1e-6)
        close(RAW[name], 100 * (p if lower else 1 - p), 1e-9)
        assert NUM[name] == f"{RAW[name]:.2f}"
        close(RAW[name + "Grid"], RAW[name], 0.05)   # the old grid sum drifts only in the second decimal
    assert NUM["tailHes"] == "8.34" and NUM["rtailHes"] == "3.47"


def test_newton_at_the_forward():
    """y = 0 is outside the convergence theorem; the code starts at zero with the limiting slope."""
    r, T = 0.02, 0.75
    K = 100 * math.exp(r * T)                                   # strike at the forward
    for s in (0.05, 0.2, 0.6, 1.2):
        for kind in ("call", "put"):
            p = float(L.bs(100, K, r, s, T, kind))
            sig, it, path = L.iv_newton(p, 100, K, r, T, kind)
            assert abs(sig - s) < 1e-10 and path[0] == 0.0
            assert np.all(np.diff(path[:-1]) >= -1e-15)          # the price is concave in v, so the iterates rise
