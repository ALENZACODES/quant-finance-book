"""Tests for closed_form.py: the formula, put-call parity, the Greeks and the gamma-theta identity."""
import math
import numpy as np
from conftest import close
import ch05lib as L


def test_base_prices(RAW):
    close(RAW["callBase"], 10.450583572185565, 1e-12)
    close(RAW["parityGap"], 0.0, 1e-12)
    close(RAW["putBase"], RAW["callBase"] - RAW["parityRHS"], 1e-12)


def test_greeks_against_differences(RAW):
    for k, tol in [("Delta", 1e-8), ("Gamma", 1e-6), ("Vega", 1e-4), ("Theta", 1e-5), ("Rho", 1e-3)]:
        assert RAW[f"fdErr{k}"] < tol, k


def test_greeks_random_points():
    rng = np.random.default_rng(7)
    for _ in range(200):
        S, K = rng.uniform(50, 150), rng.uniform(50, 150); r, q = rng.uniform(-0.01, 0.1), rng.uniform(0, 0.05)
        s, T = rng.uniform(0.05, 0.8), rng.uniform(0.05, 3)
        for kind in ("call", "put"):
            g = L.greeks(S, K, r, s, T, kind, q); h = 1e-4 * S
            f = lambda x: float(L.bs(x, K, r, s, T, kind, q))
            close(float(g["delta"]), (f(S + h) - f(S - h)) / (2 * h), 1e-6)
            close(float(g["gamma"]), (f(S + h) - 2 * f(S) + f(S - h)) / h ** 2, 1e-4)
            fs = lambda x: float(L.bs(S, K, r, x, T, kind, q))
            close(float(g["vega"]), (fs(s + 1e-5) - fs(s - 1e-5)) / 2e-5, 1e-4 * max(1, float(g["vega"])))
            # the Black-Scholes equation holds pointwise
            V = f(S)
            res = g["theta"] + 0.5 * s ** 2 * S ** 2 * g["gamma"] + (r - q) * S * g["delta"] - r * V
            assert abs(res) < 1e-9 * max(1, S)


def test_identities(RAW, NUM):
    assert abs(RAW["phiIdentity"]) < 1e-14 and abs(RAW["pdeResidual"]) < 1e-12
    close(RAW["deltaP"], RAW["deltaC"] - 1, 1e-12)
    close(RAW["thetaC"], RAW["thetaGammaPart"] + RAW["thetaCarryPart"], 1e-12)
    close(RAW["gammaPeakS"], RAW["gammaPeakNumeric"], 1e-3)
    assert RAW["callDeepLower"] < RAW["callDeep"] < 150


def test_bounds_and_monotonicity():
    s = np.linspace(0.01, 2, 200); c = L.bs(100, 110, 0.05, s, 1.0)
    assert np.all(np.diff(c) > 0) and c[0] >= 100 - 110 * math.exp(-0.05) - 1e-12 and c[-1] < 100
    K = np.linspace(50, 150, 201); c = L.bs(100, K, 0.05, 0.2, 1.0)
    assert np.all(np.diff(c) < 0) and np.all(np.diff(c, 2) > 0)
