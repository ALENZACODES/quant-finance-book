"""Tests for no_arbitrage.py: state prices, arbitrage detection and LP duality."""
import numpy as np
from conftest import close
import ch04lib as L


def test_state_prices_price_everything(RAW):
    psi = np.array([RAW["psiUp"], RAW["psiMid"], RAW["psiDown"]])
    assert (psi > 0).all()
    close(psi.sum(), 1 / 1.02, 1e-4)
    close(psi @ np.array([120, 100, 80.0]), 100, 1e-3)
    close(psi @ np.array([20, 0, 0.0]), 7.0, 1e-3)


def test_arbitrage_flags(RAW):
    assert RAW["threeAssetNoArb"] == 1.0 and RAW["fourAssetNoArb"] == 0.0 and RAW["lowCallNoArb"] == 0.0


def test_arbitrage_portfolio_is_an_arbitrage():
    A = np.array([[1.02, 120, 20, 0], [1.02, 100, 0, 10], [1.02, 80, 0, 30.0]]); p = np.array([1, 100, 7, 13.0])
    ok, th = L.arbitrage_check(A, p)
    assert not ok
    pay = A @ th
    assert (pay >= -1e-9).all() and p @ th <= 1e-9 and (pay.max() > 1e-6 or p @ th < -1e-6)


def test_duality_and_bounds(RAW):
    assert RAW["dualGap"] < 1e-13 and RAW["callInside"] == 1.0
    close(RAW["boundLo"], 100 - 100 / 1.02, 1e-4)                 # max(0, S - K/R) for K = 100
    close(RAW["boundHi"], 0.55 * 20 / 1.02, 1e-4)
    close(RAW["qUpMin"], 10, 1e-9); close(RAW["qUpMax"], 55, 1e-9)


def test_random_markets_ftap():
    rng = np.random.default_rng(3)
    for _ in range(200):
        S, Nn = 4, 3
        A = rng.uniform(0.5, 2, (S, Nn)); psi = rng.uniform(0.05, 0.4, S)
        p = A.T @ psi
        ok, _ = L.arbitrage_check(A, p); assert ok
        # push one price below the cheapest it could be: an arbitrage must appear
        p2 = p.copy(); p2[0] = -0.1
        ok2, th = L.arbitrage_check(A, p2); assert not ok2
