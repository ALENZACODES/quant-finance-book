"""Tests for hedging_pnl.py."""
import math
import numpy as np
from conftest import close
import ch05lib as L

W = ["Thirteen", "FiftyTwo", "Daily", "Quarter"]


def test_error_matches_leading_order(RAW):
    for w in W:
        assert 0.9 < RAW[f"hedgeSd{w}"] / RAW[f"hedgeTh{w}"] < 1.03, w
        assert abs(RAW[f"hedgeMean{w}"]) < 3.5 * RAW[f"hedgeMeanSE{w}"] + 1e-12, w
    assert abs(RAW["hedgeSlope"] + 0.5) < 0.03
    assert RAW["hedgeCorrGTDaily"] > RAW["hedgeCorrGTFiftyTwo"] > 0.95


def test_wrong_volatility(RAW):
    assert abs(RAW["wrongImpMean"] - RAW["volGap"]) < 4 * RAW["wrongImpSE"]
    assert abs(RAW["wrongRealMean"] - RAW["volGap"]) < 0.02
    assert RAW["wrongRealSd"] < 0.5 * RAW["wrongImpSd"]
    assert abs(RAW["underMean"] - RAW["underGap"]) < 4 * RAW["underSE"]
    assert RAW["volGap"] - RAW["wrongImpMeanDrift"] > 5 * RAW["wrongImpSEDrift"]
    assert RAW["gtStepCorr"] > 0.99


def test_costs(RAW):
    assert abs(RAW["costSlope"] - 0.5) < 0.06
    close(RAW["costInit"], 0.002 * RAW["deltaC"] * 100, 1e-12)
    assert abs(RAW["lelandExInit"]) < 3.5 * RAW["lelandSE"]
    close(RAW["lelandMarkup"], RAW["costRebalFiftyTwo"], 0.05 * RAW["costRebalFiftyTwo"])


def test_gamma_moment_closed_form():
    rng = np.random.default_rng(5); t, T = 0.4, 1.0
    St = 100 * np.exp((0.1 - 0.02) * t + 0.2 * math.sqrt(t) * rng.standard_normal(1_000_000))
    G = L.greeks(St, 100, 0.05, 0.2, T - t)["gamma"]; m = np.mean((G * St ** 2) ** 2)
    se = np.std((G * St ** 2) ** 2) / 1000
    assert abs(m - L.gamma2_moment(100, 100, 0.05, 0.2, t, T, 0.1)) < 4 * se


def test_perfect_hedge_limit():
    pnl, _ = L.hedge_sim(100, 100, 0.05, 0.2, 0.2, 1.0, 2000, 2000, np.random.default_rng(1))
    assert pnl.std() < 0.2
