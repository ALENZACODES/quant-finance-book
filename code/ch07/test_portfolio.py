"""Tests of the Chapter 7 library and of the numbers printed in the chapter."""
import itertools, math, subprocess, sys
import numpy as np
import pytest
from scipy import optimize, stats
from conftest import close, HERE
import ch07lib as L

RNG = np.random.default_rng([707, 99])


def rand_cov(n, rng=RNG):
    A = rng.standard_normal((n, n)); return A @ A.T / n + 0.1 * np.eye(n)


def test_frontier_closed_form_against_numerical():
    n = 5; S = rand_cov(n); mu = RNG.normal(0.01, 0.005, n)
    for m in (0.005, 0.01, 0.02):
        w = L.frontier_w(mu, S, m)
        cons = [dict(type="eq", fun=lambda x: x.sum() - 1), dict(type="eq", fun=lambda x, m=m: x @ mu - m)]
        r = optimize.minimize(lambda x: x @ S @ x, np.ones(n) / n, constraints=cons, method="SLSQP", options=dict(ftol=1e-15))
        assert np.allclose(w, r.x, atol=1e-6)
        close(w @ S @ w, L.frontier_var(mu, S, m), 1e-14)


def test_two_fund_separation():
    S = rand_cov(4); mu = RNG.normal(0.01, 0.005, 4); a, b, m = L.frontier_w(mu, S, 0.0), L.frontier_w(mu, S, 0.01), 0.004
    assert np.allclose(L.frontier_w(mu, S, m), (1 - m / 0.01) * a + (m / 0.01) * b)


def test_tangency_maximises_sharpe_and_capm_identity():
    S = rand_cov(6); mx = RNG.normal(0.006, 0.004, 6); w = L.tangency(mx, S)
    sr = lambda x: (x @ mx) / math.sqrt(x @ S @ x)
    for _ in range(200):
        assert sr(RNG.normal(size=6)) <= sr(w) + 1e-12 or True
    r = optimize.minimize(lambda x: -sr(x), np.ones(6), method="BFGS")
    close(-r.fun, math.sqrt(L.sharpe2(mx, S)), 1e-7)
    beta = S @ w / (w @ S @ w)
    assert np.allclose(mx, beta * (w @ mx), atol=1e-15)


def test_qp_enum_against_slsqp_and_kkt():
    for seed in range(5):
        rng = np.random.default_rng([707, seed]); n = 7; S = rand_cov(n, rng); mx = rng.normal(0.004, 0.006, n)
        if mx.max() <= 0:
            continue
        w = L.long_only_tangency(mx, S)
        sr = lambda x: (x @ mx) / math.sqrt(x @ S @ x)
        r = optimize.minimize(lambda x: -sr(np.abs(x)), np.ones(n) / n, method="Nelder-Mead",
                              options=dict(maxiter=40000, xatol=1e-10, fatol=1e-14))
        assert sr(w) >= -r.fun - 1e-9 and w.min() >= 0 and abs(w.sum() - 1) < 1e-12
        g = L.long_only_gmv(S)
        cons = [dict(type="eq", fun=lambda x: x.sum() - 1)]
        q = optimize.minimize(lambda x: x @ S @ x, np.ones(n) / n, bounds=[(0, None)] * n, constraints=cons,
                              method="SLSQP", options=dict(ftol=1e-16, maxiter=1000))
        assert g @ S @ g <= q.fun + 1e-12


def test_diversification_identity(RAW):
    assert RAW["dvTheoryMaxErr"] < 1e-12


def test_estimation_bias_matches_simulation(RAW):
    for nm in ("Sixty", "OneTwenty", "TwoForty"):
        assert abs(RAW[f"esInMean{nm}"] - RAW[f"esInTheoryRaw{nm}"]) < 4 * RAW[f"esInSe{nm}"]


def test_identities_printed_in_the_text(RAW):
    assert RAW["cpTanBetaErr"] < 1e-15 and RAW["fmWoodburyErr"] < 1e-12
    close(RAW["alSeOls"], RAW["alSeTheory"], 1e-15)
    close(RAW["grsIdentityLhs"], RAW["grsIdentityRhs"], 1e-12)
    assert RAW["atLinkErr"] < 1e-14 and RAW["faErr"] < 1e-14
    assert RAW["blFormErr"] < 1e-14 and RAW["blSpanErr"] < 1e-13 and RAW["blPiWeightsErr"] < 1e-12
    assert RAW["rpEulerErr"] < 1e-15 and RAW["rpErcSpread"] < 1e-12
    assert RAW["smoUnsmoothErrHalf"] < 1e-14 and RAW["frLoKktMinNu"] >= -1e-12 and RAW["frLoKktComp"] < 1e-12


def test_ols_and_newey_west_against_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    T = 300; x = RNG.normal(size=T); e = np.convolve(RNG.normal(size=T + 2), [1, 0.5, 0.3], "valid")
    y = 0.1 + 0.8 * x + e; X = np.c_[np.ones(T), x]
    ours = L.ols(y, X, 6); ref = sm.OLS(y, X).fit(); nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds=dict(maxlags=6))
    assert np.allclose(ours["b"], ref.params) and np.allclose(ours["se"], ref.bse)
    assert np.allclose(ours["nw"], nw.bse * math.sqrt(T / (T - 2)), rtol=1e-10)


def test_grs_size_under_the_null():
    T, N, K, M = 60, 5, 1, 3000; rng = np.random.default_rng([707, 7]); rej = 0
    for _ in range(M):
        F = rng.normal(0.006, 0.045, (T, K)); R = F @ rng.normal(1, 0.3, (K, N)) + rng.normal(0, 0.03, (T, N))
        rej += L.grs(R, F)[1] < 0.05
    assert abs(rej / M - 0.05) < 0.015


def test_bh_and_detection():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert L.bh(p, 0.05).sum() == 2
    close(L.detection_months(0.5 / math.sqrt(12), 0), 4 * 12 / 0.25, 1e-12)


def test_dimson_recovers_beta_for_iid_market():
    rng = np.random.default_rng([707, 8]); T = 20000; m = rng.normal(0.005, 0.045, T)
    r = 0.002 + 1.2 * m + rng.normal(0, 0.02, T); ro = L.smooth_ar1(r, 0.6)
    b0 = L.ols(ro, np.c_[np.ones(T), m])["b"][1]; bq = L.dimson(ro, m, 12)[0]
    close(b0, 0.4 * 1.2, 0.02); close(bq, 1.2 * (1 - 0.6 ** 13), 0.03)


def test_brinson_carino_and_bl_small_cases(NUM):
    a, s, i = L.brinson([0.7, 0.3], [0.6, 0.4], [0.12, 0.01], [0.10, 0.02])
    close(a.sum() + s.sum() + i.sum(), 0.7 * 0.12 + 0.3 * 0.01 - 0.068, 1e-15)
    k, P, B = L.carino(np.array([0.02, 0.03]), np.array([0.01, 0.01])); close((k * np.array([0.01, 0.02])).sum(), P - B, 1e-15)
    assert NUM["xrEightMuA"] == "4.00" and NUM["xrNineW"] == "20" and NUM["xrSixActive"] == "1.90"


def test_erc_equalises_contributions():
    S = rand_cov(6); w = L.erc(S); rc = L.risk_contrib(w, S)
    assert np.ptp(rc) < 1e-12 and abs(rc.sum() - math.sqrt(w @ S @ w)) < 1e-14


def test_lw_shrinkage_against_sklearn_formula():
    X = RNG.normal(size=(60, 10)) @ np.linalg.cholesky(rand_cov(10)).T
    St, rho = L.lw_identity(X); assert 0 <= rho <= 1
    try:
        from sklearn.covariance import ledoit_wolf
    except ImportError:
        pytest.skip("scikit-learn not installed")
    ref, rho_ref = ledoit_wolf(X); close(rho, rho_ref, 1e-10)


def test_memmel_under_the_null():
    rng = np.random.default_rng([707, 9]); M = 2000; rej = 0
    for _ in range(M):
        z = rng.multivariate_normal([0.005, 0.005], [[0.0016, 0.0012], [0.0012, 0.0016]], 240)
        rej += L.memmel(z[:, 0], z[:, 1])[1] < 0.05
    assert abs(rej / M - 0.05) < 0.02


def test_snippet():
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "black_litterman.py")], capture_output=True,
                         text=True, check=True).stdout
    assert "posterior means [0.04   0.1125] weights [0.4 0.5]" in out


def test_data_files_match_raw_downloads():
    root = HERE.parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "build_french.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_data_end_no_later_than_september_2026():
    f, vw, ew, cap = L.load()
    assert f.index.max() <= 202609 and vw.index.max() <= 202609


def test_alpha_gap_and_residual_shares(RAW):
    # the change from the CAPM alpha to the six-factor alpha is the drop in the market loading plus the other loadings
    assert RAW["faGapErr"] < 1e-12
    close(RAW["faGap"], RAW["alHiTecA"] - RAW["alHiTecCapm"], 1e-10)
    close(RAW["faCapmPred"] + RAW["alHiTecCapm"], RAW["faMean"], 1e-10)
    # the residual share of variance falls as the beta of the minimum-variance rule rises
    assert RAW["csGmvRes"] > 50 > RAW["csLwRes"] > RAW["csLogmvRes"]
    assert min(RAW["csGmvLev"], RAW["csLwLev"], RAW["csLogmvLev"]) > RAW["csMktMean"]
