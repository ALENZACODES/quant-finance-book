"""Tests for dividends_black.py."""
import math
from conftest import close
import ch05lib as L


def test_parity_and_equivalences(RAW):
    assert abs(RAW["blackParityGap"]) < 1e-12 and abs(RAW["gkParityGap"]) < 1e-12
    close(RAW["blackCall"], RAW["blackAsYield"], 1e-12)
    close(RAW["divYieldFwd"], 100 * math.exp(0.02), 1e-9)
    close(RAW["gkFwd"], 1.38 * math.exp(-0.015 * 0.25), 1e-12)


def test_dividend_models(RAW):
    se = RAW["dropSE"]
    assert RAW["escCall"] < RAW["dropCall"] < RAW["adjCall"]
    assert RAW["dropEscGap"] > 5 * se and -RAW["dropAdjGap"] > 5 * se
    assert RAW["dropCall"] < RAW["noDivCall"]
    close(RAW["escCall"], float(L.bs(100 - RAW["divPV"], 100, 0.05, 0.2, 1.0)), 1e-12)


def test_continuous_yield_limits():
    # a yield equal to the interest rate makes the forward equal to spot
    close(float(L.bs(100, 100, 0.05, 0.2, 1, "call", 0.05)), float(L.bs(100, 100, 0.05, 0.2, 1, "put", 0.05)), 1e-12)
