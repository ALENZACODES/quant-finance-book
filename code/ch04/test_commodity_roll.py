"""Tests for commodity_roll.py: the Schwartz futures formula and the roll decomposition."""
import math
import numpy as np
from conftest import close
import ch04lib as L
import commodity_roll as C


def test_futures_is_risk_neutral_mean():
    rng = np.random.default_rng(11)
    k, s, a = 1.2, 0.35, math.log(70); x0 = math.log(90); tau = 0.75
    e = math.exp(-k * tau)
    X = a + (x0 - a) * e + s * math.sqrt((1 - e * e) / (2 * k)) * rng.standard_normal(400000)
    mc = np.exp(X); F = float(L.schwartz_futures(x0, tau, k, a, s))
    close(mc.mean(), F, 4 * mc.std() / math.sqrt(4e5))
    close(float(L.schwartz_futures(x0, 0.0, k, a, s)), 90.0, 1e-12)


def test_term_structure_shapes(RAW):
    assert RAW["schOneLow"] > 50 and RAW["schOneHigh"] < 100
    tau = np.linspace(0, 1e-4, 2)
    for S, sign in [(60, 1), (90, -1)]:
        F = L.schwartz_futures(math.log(S), tau, 1.2, math.log(70), 0.35)
        assert sign * (F[1] - F[0]) > 0
    # flat at the short end exactly at schSFlat
    F = L.schwartz_futures(math.log(RAW["schSFlat"]), np.array([0, 1e-6]), 1.2, math.log(70), 0.35)
    assert abs(F[1] - F[0]) < 1e-8


def test_identity_and_expected_roll(RAW):
    assert RAW["rollIdentityErr"] < 1e-12
    assert abs(RAW["rollMCz"]) < 3
    close(RAW["rollTheory"], 100 * 12 * C.expected_roll(), 1e-9)
    assert RAW["rollBasisCorr"] < -0.5


def test_collateralised_wealth_is_exact():
    """The wealth recursion V' = V (e^{rh} + G'/G - 1), checked by stepping a portfolio by hand, and the
    factorisation used in the text that splits it into r h + g plus a small correction."""
    rng = np.random.default_rng(5)
    g = 0.08 * rng.standard_normal(480); r, h = 0.03, 1 / 12
    V = 1.0
    for gk in g:
        bills = V * math.exp(r * h); contracts = V / 1.0; V = bills + contracts * (math.exp(gk) - 1.0)
    close(math.log(V), float(C.wealth_log_return(g, h, r)), 1e-10)
    corr = np.log(1 - (1 - np.exp(-g)) * (1 - math.exp(-r * h)))
    close(float(C.wealth_log_return(g, h, r)), float(np.sum(r * h + g) + corr.sum()), 1e-10)
    # each correction term is close to r h (e^{-g} - 1)
    assert np.max(np.abs(corr - r * h * (np.exp(-g) - 1))) < 1e-4


def test_gap_and_median_path(RAW):
    assert 0 < RAW["rollGapMedBp"] < RAW["rollGapMaxBp"] < 20
    close(RAW["rollWealthMean"] - RAW["rollApproxMean"], 0.0, 0.05)  # percent units
    # the figure path is the median path, so its roll is near the cross-sectional median, not in a tail
    assert RAW["rollMCLo"] < RAW["rollRollAnn"] < RAW["rollMCHi"]
    assert abs(RAW["rollRollAnn"] - RAW["rollTheory"]) < 0.5 * RAW["rollMCSd"]
