"""Checks for carry_options.py: option decomposition, Black-Scholes values and the vega claims."""
import numpy as np
from scipy import integrate, stats
import ch02lib as L
from conftest import close

K, T, r, H = 100.0, 5.0, 0.03, 100 * 1.08 ** 5


def lognormal_expectation(f, V0, s):
    m = np.log(V0) + (r - 0.5 * s * s) * T; sd = s * np.sqrt(T)
    g = lambda z: f(np.exp(m + sd * z)) * stats.norm.pdf(z)
    return np.exp(-r * T) * integrate.quad(g, -12, 12, limit=400, points=[(np.log(H) - m) / sd])[0]


def test_values_by_quadrature(RAW):
    for s, nm in [(0.15, "Fifteen"), (0.30, "Thirty"), (0.45, "FortyFive")]:
        close(RAW["carryFull" + nm], lognormal_expectation(lambda v: L.carry_payoff(v, K, H), 100, s), 1e-6)
        close(RAW["carryHard" + nm], lognormal_expectation(lambda v: L.carry_payoff(v, K, H, hard=True), 100, s), 1e-6)


def test_monte_carlo_within_three_se(RAW):
    assert abs(RAW["carryMCThirty"] - RAW["carryFullThirty"]) < 3 * RAW["carryMCse"]
    assert abs(RAW["carryMCz"]) < 1.0       # text says within one standard error


def test_ordering_and_vega_signs(RAW):
    assert RAW["carryHardThirty"] < RAW["carryFullThirty"] < RAW["carryNoHurdleThirty"]
    assert RAW["carryFullFifteen"] < RAW["carryFullThirty"] < RAW["carryFullFortyFive"]
    assert RAW["carryVegaLow"] > 0 and RAW["carryVegaHigh"] < 0 and RAW["carryVegaHighThirty"] > 0
    assert RAW["carryFullThirtyRich"] > RAW["carryFullThirty"]


def test_vega_formula():
    """Equation (2.9) against a finite difference."""
    for V0 in (80, 100, 150, 200):
        for s in (0.15, 0.3):
            vstar = (H - 20) / 0.8
            d1 = lambda X: (np.log(V0 / X) + (r + 0.5 * s * s) * T) / (s * np.sqrt(T))
            formula = np.sqrt(T) * V0 * (stats.norm.pdf(d1(H)) - 0.8 * stats.norm.pdf(d1(vstar)))
            e = 1e-6
            fd = (L.carry_option_value(V0, K, H, T, r, s + e) - L.carry_option_value(V0, K, H, T, r, s - e)) / (2 * e)
            close(formula, fd, 1e-4)
