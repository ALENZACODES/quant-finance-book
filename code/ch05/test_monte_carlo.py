"""Tests for monte_carlo.py: unbiasedness, variance reduction, rates, coverage, delta estimators."""
import math
import numpy as np
from conftest import close
import ch05lib as L


def test_estimates_unbiased(RAW):
    for k in ("Plain", "Anti", "CV", "Both"):
        assert abs(RAW[f"mcZ{k}"]) < 3.5, k


def test_variance_reduction(RAW):
    close(RAW["vrMCAnti"], RAW["vrAntiTheory"], 0.05 * RAW["vrAntiTheory"])
    close(RAW["vrMCCV"], RAW["vrCVTheory"], 0.05 * RAW["vrCVTheory"])
    assert RAW["vrMCBoth"] > max(RAW["vrMCAnti"], RAW["vrMCCV"])
    close(RAW["vrCVTheory"], 1 / (1 - RAW["mcRho"] ** 2), 1e-12)
    close(RAW["vrAntiTheory"], 1 / (1 + RAW["mcCorrAnti"]), 1e-9)
    assert RAW["mcCorrAnti"] < 0          # monotone payoff: antithetic pairs are negatively correlated


def test_rates_and_coverage(RAW):
    for k in ("Plain", "Anti", "CV", "Both"):
        assert abs(RAW[f"mcSlope{k}"] + 0.5) < 0.07, k
        assert 88 <= RAW[f"mcCover{k}"] <= 99.5, k


def test_delta_estimators(RAW):
    d = RAW["deltaExact"]
    assert abs(RAW["deltaPW"] - d) < 3.5 * RAW["deltaPWSE"] and abs(RAW["deltaLR"] - d) < 3.5 * RAW["deltaLRSE"]
    assert RAW["deltaLRSE"] > RAW["deltaPWSE"]


def test_snippet(RAW):
    close(RAW["snipFormula"], RAW["callBase"], 1e-4)
    assert RAW["snipZ"] < 3.5


def test_antithetic_even_payoff_is_worse():
    rng = np.random.default_rng(3); z = rng.standard_normal(200_000)
    f = z ** 2
    pair = 0.5 * (f[:100_000] + f[:100_000])          # f(z) = f(-z): pairs repeat the same draw
    assert pair.var() * 2 > 1.9 * f.var()
