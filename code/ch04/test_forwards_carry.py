"""Tests for forwards_carry.py: carry formulas, the band, parity and futures against forwards."""
import math
import numpy as np
from scipy import integrate
from conftest import close
import ch04lib as L


def test_carry_and_value(RAW):
    close(RAW["fwdIndex"], 100 * math.exp(0.015 * 0.5), 1e-9)
    close(RAW["fwdValueLater"], RAW["fwdValueAlt"], 1e-9)
    # implied convenience yield reproduces the quote
    close(L.forward_price(80, 0.04, 0.5, u=0.01, y=RAW["convYield"] / 100), 78.0, 1e-9)


def test_band_contains_frictionless_and_quote_outside(RAW):
    assert RAW["bandLo"] < RAW["bandMid"] < RAW["bandHi"] < RAW["bandQuote"]
    close(RAW["bandProfit"], RAW["bandQuote"] - 100.05 * math.exp(0.044), 1e-9)


def test_cip(RAW):
    close(RAW["cipFwd"], 1.37 * 1.0275 / 1.04, 1e-9)
    close(RAW["cipProfit"], 1_040_000 * 1.36 - 1_370_000 * 1.0275, 1e-6)
    assert RAW["cipProfit"] > 0


def test_fut_fwd_closed_form_by_quadrature():
    k, sr, ss, rho, T = 0.3, 0.01, 0.2, 0.5, 5.0
    B = lambda u: (1 - math.exp(-k * (T - u))) / k
    v = sr ** 2 * integrate.quad(lambda u: B(u) ** 2, 0, T)[0]
    c = rho * ss * sr * integrate.quad(B, 0, T)[0]
    close(L.fut_fwd_logratio(k, sr, ss, rho, T), v + c, 1e-12)


def test_fut_fwd_numbers(RAW):
    assert RAW["ffMCz"] < 3.5
    assert RAW["ffDeterministic"] == 0.0
    assert RAW["ffTenPos"] > RAW["ffFivePos"] > RAW["ffOnePos"] > 0 > RAW["ffFiveNeg"]
    close(L.fut_fwd_logratio(0.3, 0.01, 0.2, RAW["ffRhoStar"], 5.0), 0.0, 1e-6)


def test_simulated_futures_within_one_se(RAW):
    # 4.058: both simulated log ratios lie within one standard error of the formula
    for mc, se, f in (("ffMCFivePos", "ffMCFivePosSE", "ffFivePos"), ("ffMCFiveNeg", "ffMCFiveNegSE", "ffFiveNeg")):
        assert abs(RAW[mc] - RAW[f]) < RAW[se], (mc, RAW[mc], RAW[f], RAW[se])


def test_band_with_lending_fee():
    # a lending fee lowers only the reverse cash and carry bound, and acts like extra income on the asset
    lo0, hi0 = L.forward_band(99.95, 100.05, 0.038, 0.044, 2.0)
    lo1, hi1 = L.forward_band(99.95, 100.05, 0.038, 0.044, 2.0, fee=0.01)
    assert hi1 == hi0 and lo1 < lo0
    assert abs(lo1 - L.forward_band(99.95, 100.05, 0.038, 0.044, 2.0, q=0.01)[0]) < 1e-12
