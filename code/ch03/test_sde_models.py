"""Checks of the Chapter 3 numbers produced by sde_models.py."""
import math, subprocess, sys
import numpy as np
from scipy import stats
from conftest import close, HERE
import ch03lib as L


def test_gbm_moments_against_scipy(RAW):
    for sig in (0.25, 0.5):
        d = stats.lognorm(s=sig * math.sqrt(30), scale=100 * math.exp((0.08 - sig ** 2 / 2) * 30))
        m, v, med, mode = L.gbm_moments(100, 0.08, sig, 30)
        close(m, d.mean(), 1e-8 * m); close(v, d.var(), 1e-8 * v); close(med, d.median(), 1e-9 * med)
    close(RAW["gbmMedianHigh"], 100 * math.exp(-0.045 * 30), 1e-9)
    assert RAW["gbmGrowthHigh"] < 0 < RAW["gbmGrowthLow"]


def test_gbm_paths_vanish_when_growth_negative():
    rng = np.random.default_rng(51)
    t, W = L.bm_paths(2000, 2000, 400.0, rng)
    S = L.gbm_exact(1.0, 0.08, 0.5, t[-1], W[:, -1])
    assert np.median(S) < 1e-6
    p = stats.norm.cdf((math.log(1e-3) - (0.08 - 0.125) * 400) / (0.5 * 20))
    close(np.mean(S < 1e-3), p, 4 * math.sqrt(p * (1 - p) / 2000))


def test_scheme_orders(RAW):
    close(RAW["strongEuler"], 0.5, 0.08); close(RAW["strongMilstein"], 1.0, 0.1)
    close(RAW["weakSlopeMean"], 1.0, 0.03); close(RAW["weakSlopeSecond"], 1.0, 0.05)
    rng = np.random.default_rng(52)
    dW = rng.standard_normal((200000, 12)) * math.sqrt(1 / 12)
    e, m = L.gbm_schemes(100, 0.08, 0.4, 1.0, dW)
    close(e.mean(), L.euler_gbm_mean(100, 0.08, 1.0, 12), 4 * e.std() / math.sqrt(len(e)))
    close(np.mean(e ** 2), L.euler_gbm_second(100, 0.08, 0.4, 1.0, 12), 4 * np.std(e ** 2) / math.sqrt(len(e)))


def test_ou_moments_and_fit(RAW):
    rng = np.random.default_rng(53)
    X = L.ou_exact(0.10, 1.2, 0.03, 0.02, 0.05, 40, 100000, rng)
    m, v = L.ou_moments(0.10, 1.2, 0.03, 0.02, 2.0)
    close(X[:, -1].mean(), m, 4 * math.sqrt(v / 1e5)); close(X[:, -1].var(), v, 4 * v * math.sqrt(2 / 1e5))
    x = L.ou_exact(0.03, 1.2, 0.03, 0.02, 1 / 12, 60000, 1, rng)[0]
    f = L.ou_fit(x, 1 / 12)
    close(f["kappa"], 1.2, 0.06); close(f["theta"], 0.03, 0.002); close(f["sigma"], 0.02, 0.0005)
    close(RAW["ouHalfLife"], math.log(2) / 1.2, 1e-12)
    # Kendall's approximation to the small sample bias of phi: simulated bias within 3 standard errors
    se = math.sqrt((1 - RAW["ouPhiMonthly"] ** 2) / 600) / math.sqrt(400)
    close(RAW["ouPhiBias"], RAW["ouKendall"], 3 * se)


def test_listing_output(RAW):
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "ou_fit.py")], capture_output=True, text=True).stdout.strip()
    assert out == RAW["snippetOut"]
    assert out.startswith(f"kappa_hat = {RAW['snipKappa']:.3f}")


def test_yield_case(RAW):
    close(RAW["caseADF"], RAW["caseDFt"], 1e-9)
    assert RAW["caseADF"] > RAW["caseADFcrit"]                  # a random walk is not rejected
    assert RAW["caseBiasRatio"] > 1.5
    close(RAW["caseHalf"], math.log(2) / RAW["caseKappa"], 1e-12)
    assert RAW["caseHalfLo"] < RAW["caseHalf"] < RAW["caseHalfHi"]
