"""Independent checks of the Chapter 1 numbers produced by distributions.py (pass (b) of the production process).

Run from the repository root:  python -m pytest -q code/ch01
The R cross-check (code/ch01/crosscheck.R) must have been run first; run_all.py writes numbers.json.
"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
import numpy_financial as npf
import pytest
from scipy import stats
from statsmodels.stats.stattools import jarque_bera as sm_jb, durbin_watson as sm_dw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import ch01lib as L  # noqa: E402

RAW = json.loads((HERE / "out" / "numbers.json").read_text())["RAW"]
rng = np.random.default_rng(7)


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def test_lognormal_moments_by_simulation():
    mu, s = 0.06 * 5, 0.18 * np.sqrt(5)
    x = np.exp(rng.normal(mu, s, 2_000_000))
    m, sd, sk, ek = L.lognormal_moments(mu, s)
    close(x.mean(), m, 3e-3); close(x.std(), sd, 3e-3)
    close(stats.skew(x), sk, 0.05)
    # scipy's closed-form agrees exactly
    d = stats.lognorm(s=s, scale=np.exp(mu)).stats(moments="mvsk")
    close(float(d[2]), sk, 1e-9); close(float(d[3]), ek, 1e-9)


def test_kurtosis_gallery():
    nu = 5
    assert stats.t(nu).stats(moments="k") == pytest.approx(6.0)
    assert stats.laplace.stats(moments="k") == pytest.approx(3.0)
    assert stats.uniform.stats(moments="k") == pytest.approx(-1.2)
    assert RAW["tailTFour"] / RAW["tailNormFour"] > 50


def test_put_premium_and_parity():
    p = L.bs_put(100, 95, 1 / 12, 0.03, 0.18)
    close(RAW["putPrem"], p, 1e-12)
    z = rng.standard_normal(2_000_000)
    ST = 100 * np.exp((0.03 - 0.5 * 0.18 ** 2) / 12 + 0.18 * np.sqrt(1 / 12) * z)
    close(np.exp(-0.03 / 12) * np.maximum(95 - ST, 0).mean(), p, 5e-3)


def test_beta_example():
    m, f = np.array(RAW["exMkt"]), np.array(RAW["exFnd"])
    b = np.polyfit(m, f, 1)[0]
    close(RAW["exBeta"], b, 1e-9)
    close(RAW["exBeta"], RAW["exRho"] * RAW["exSdF"] / RAW["exSdM"], 1e-6)


def test_diversification_formula_by_simulation():
    n, rho, s = 10, 0.2, 0.2
    C = s ** 2 * (rho * np.ones((n, n)) + (1 - rho) * np.eye(n))
    w = np.ones(n) / n
    close(np.sqrt(w @ C @ w), L.portfolio_sd_equal(s, rho, n), 1e-12)


def test_fat_tail_mass_against_normal():
    """Fat tails put less mass beyond one sd and more beyond two and three (the claim in the text)."""
    lap = stats.laplace(scale=1 / np.sqrt(2)); t5 = stats.t(5, scale=np.sqrt(3 / 5))
    close(lap.var(), 1, 1e-12); close(t5.var(), 1, 1e-12)
    x = rng.laplace(0, 1 / np.sqrt(2), 2_000_000)
    close(100 * np.mean(np.abs(x) > 1), RAW["lapOutOne"], 0.15)
    for k, nm in ((1, "One"), (2, "Two"), (3, "Three")):
        close(RAW["lapOut" + nm], 100 * 2 * lap.sf(k), 1e-9); close(RAW["tOut" + nm], 100 * 2 * t5.sf(k), 1e-9)
    assert RAW["lapOutOne"] < RAW["normOutOne"] and RAW["tOutOne"] < RAW["normOutOne"]
    for nm in ("Two", "Three"):
        assert RAW["lapOut" + nm] > RAW["normOut" + nm] and RAW["tOut" + nm] > RAW["normOut" + nm]
        assert RAW["lapOut" + nm] < RAW["cheb" + nm] and RAW["tOut" + nm] < RAW["cheb" + nm]


def test_chebyshev_and_normal_table():
    close(RAW["normOutTwo"], 100 * 2 * stats.norm.sf(2), 1e-9)
