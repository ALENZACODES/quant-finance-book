"""Tests of the Chapter 12 library against closed forms, finite differences, an independent library
(scikit-learn, used only for comparison and skipped if it is not installed) and identities."""
import math
import numpy as np
import pytest
from scipy.stats import norm
from conftest import close
import ch12lib as L


def data(n=80, p=12, seed=1):
    rng = np.random.default_rng([seed, 99])
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(L.toeplitz_corr(p, 0.5)).T
    y = X[:, 0] - 0.5 * X[:, 3] + rng.standard_normal(n)
    return X - X.mean(0), y - y.mean()


# ---------------------------------------------------------------- 12.1
def test_ridge_matches_normal_equations_and_p_bigger_than_n():
    for n, p in [(80, 12), (20, 50)]:
        X, y = data(n, p)
        for lam in (0.1, 3.0, 100.0):
            b = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
            assert np.allclose(L.ridge(X, y, lam), b, atol=1e-10)


def test_ridge_shrinkage_and_df():
    X, y = data()
    d = np.linalg.svd(X, compute_uv=False)
    close(L.ridge_df(X, 2.0), np.trace(X @ np.linalg.solve(X.T @ X + 2 * np.eye(12), X.T)), 1e-10)
    assert L.ridge_df(X, 0.0) == pytest.approx(12) and L.ridge_df(X, 1e9) < 1e-5
    assert np.all(np.diff([np.linalg.norm(L.ridge(X, y, l)) for l in (0.1, 1, 10, 100)]) < 0)


def test_ridge_mse_derivative_negative_at_zero():
    """the proposition: for fixed X, d MSE / d lambda at 0 is -2 beta' (X'X)^-2 X'X ... < 0; checked numerically."""
    X, _ = data(60, 8)
    beta = np.linspace(1, -1, 8); s2 = 1.0
    def mse(lam):
        A = np.linalg.inv(X.T @ X + lam * np.eye(8))
        bias = A @ X.T @ X @ beta - beta
        return bias @ bias + s2 * np.trace(A @ X.T @ X @ A)
    assert mse(1e-4) < mse(0.0) and (mse(1e-6) - mse(0)) / 1e-6 < 0


def test_lasso_orthonormal_soft_threshold():
    n, p = 50, 5
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((n, p)))
    X = Q * math.sqrt(n); y = np.random.default_rng(4).standard_normal(n)
    z = X.T @ y / n
    assert np.allclose(L.lasso(X, y, 0.1), L.soft(z, 0.1), atol=1e-12)


def test_lasso_kkt_and_zero_above_lambda_max():
    X, y = data()
    lm = L.lasso_lam_max(X, y)
    assert np.all(L.lasso(X, y, 1.0001 * lm) == 0) and np.any(L.lasso(X, y, 0.99 * lm) != 0)
    for lam in (0.01, 0.1, 0.3):
        assert L.kkt_gap(X, y, L.lasso(X, y, lam), lam) < 1e-8


def test_lasso_against_sklearn():
    sk = pytest.importorskip("sklearn.linear_model")
    X, y = data()
    for lam in (0.02, 0.2):
        m = sk.Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y)
        assert np.allclose(L.lasso(X, y, lam), m.coef_, atol=1e-7)


def test_bias_variance_numbers(RAW):
    assert RAW["bvOlsBias"] < 0.05 * RAW["bvOlsVar"]           # least squares is unbiased up to simulation noise
    assert RAW["bvRidgeErr"] < RAW["bvOlsErr"] and RAW["bvLassoErr"] < RAW["bvRidgeErr"]
    assert RAW["bvRidgeSmall"] < RAW["bvOlsErr"] and RAW["bvKkt"] < 1e-8


# ---------------------------------------------------------------- 12.2
def test_purged_training_set():
    n, h = 100, 12
    te = np.arange(40, 50)
    tr = L.purged_train(n, te, h, 5)
    for t in tr:                                               # no shared return with any test label
        assert t + h < te[0] + 1 or t + 1 > te[-1] + h
    assert set(range(29)) <= set(tr) and 29 not in tr and 61 not in tr and 66 in tr and 65 not in tr


def test_overlap_correlation(RAW):
    for nm, k in [("One", 1), ("Six", 6)]:
        close(RAW[f"kfCorr{nm}"], (12 - k) / 12, 0.01)


def test_leak_ordering(RAW):
    assert RAW["kfShufMean"] > 0 > RAW["kfPurgeMean"] and RAW["kfSelOut"] > 0 > RAW["kfSelIn"]


# ---------------------------------------------------------------- 12.3
def test_tree_matches_sklearn():
    sk = pytest.importorskip("sklearn.tree")
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(300, 4)); y = np.sin(4 * X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.standard_normal(300)
    Xt = rng.uniform(size=(500, 4))
    for d in (2, 4, 6):
        a = L.Tree(d, 5).fit(X, y).predict(Xt)
        b = sk.DecisionTreeRegressor(max_depth=d, min_samples_leaf=5).fit(X, y).predict(Xt)
        assert np.allclose(a, b, atol=1e-12)


def test_split_gain_identity():
    """the gain S_L^2/n_L + S_R^2/n_R - S^2/n is the fall in the sum of squares."""
    rng = np.random.default_rng(6); y = rng.standard_normal(20); k = 7
    sse = lambda v: np.sum((v - v.mean()) ** 2)
    gain = y[:k].sum() ** 2 / k + y[k:].sum() ** 2 / (20 - k) - y.sum() ** 2 / 20
    close(sse(y) - sse(y[:k]) - sse(y[k:]), gain, 1e-12)


def test_boosting_linear_smoother():
    """boosting a fixed linear smoother S gives (I - (I - nu S)^M) y."""
    rng = np.random.default_rng(7); n = 30; y = rng.standard_normal(n)
    A = rng.standard_normal((n, 4)); S = A @ np.linalg.solve(A.T @ A, A.T)   # a projection
    nu, M = 0.3, 12; F = np.zeros(n)
    for _ in range(M):
        F = F + nu * S @ (y - F)
    assert np.allclose(F, (np.eye(n) - np.linalg.matrix_power(np.eye(n) - nu * S, M)) @ y)


def test_forest_numbers(RAW):
    close(RAW["tbVarRatio"], 1.0, 0.1)
    assert RAW["tbBoost"] < RAW["tbBag"] < RAW["tbTree"] and RAW["tbBoostOne"] > RAW["tbBoostBest"]


# ---------------------------------------------------------------- 12.4
def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(8)
    params = L.mlp_init([3, 5, 4, 1], rng); X = rng.standard_normal((7, 3)); y = rng.standard_normal(7)
    loss, acts, dout = L.mse_loss(params, X, y)
    grads, gin = L.mlp_backward(params, acts, dout)
    h = 1e-6
    for l, (W, b) in enumerate(params):
        for idx in [(0, 0), (1, 0), (W.shape[0] - 1, W.shape[1] - 1)]:
            Wp = W.copy(); Wp[idx] += h; Wm = W.copy(); Wm[idx] -= h
            pp = list(params); pp[l] = (Wp, b); pm = list(params); pm[l] = (Wm, b)
            close(grads[l][0][idx], (L.mse_loss(pp, X, y)[0] - L.mse_loss(pm, X, y)[0]) / (2 * h), 1e-8)
        bp = b.copy(); bp[0] += h; bm = b.copy(); bm[0] -= h
        pp = list(params); pp[l] = (W, bp); pm = list(params); pm[l] = (W, bm)
        close(grads[l][1][0], (L.mse_loss(pp, X, y)[0] - L.mse_loss(pm, X, y)[0]) / (2 * h), 1e-8)
    Xp = X.copy(); Xp[2, 1] += h; Xm = X.copy(); Xm[2, 1] -= h
    close(gin[2, 1], (L.mse_loss(params, Xp, y)[0] - L.mse_loss(params, Xm, y)[0]) / (2 * h), 1e-8)


def test_call_from_total_vol():
    close(L.call_from_total_vol(np.array([1.1]), np.array([0.2]))[0], float(L.bs_call(110, 100, 0.25, 0.4)) / 100, 1e-14)


def test_network_beats_polynomial(RAW):
    assert RAW["nnRmse"] < RAW["nnPolyRmse"] and RAW["nnDeltaRmse"] < 0.02


# ---------------------------------------------------------------- 12.5
def test_r2os_and_clark_west_identities():
    rng = np.random.default_rng(9); r = rng.standard_normal(200); b = np.full(200, 0.1); f = 0.3 * rng.standard_normal(200)
    mspe = lambda g: np.mean((r - g) ** 2)
    close(L.r2_os(r, f, b), 1 - mspe(f) / mspe(b), 1e-14)
    d = (r - b) ** 2 - ((r - f) ** 2 - (b - f) ** 2)
    close(d.mean(), 2 * np.mean((r - b) * (f - b)), 1e-12)          # the adjusted difference is a cross product


def test_campbell_thompson_gain_by_simulation():
    """a timer with weights proportional to the conditional mean earns (S^2 + R^2) / (S^2 (1 - R^2)) times the constant weight."""
    rng = np.random.default_rng(10); n = 2_000_000; S, R2 = 0.12, 0.02
    x = math.sqrt(R2) * rng.standard_normal(n); e = math.sqrt(1 - R2) * rng.standard_normal(n); r = S + x + e
    w = (S + x) / (1 - R2)
    close(np.mean(w * r) / (S ** 2) - 1, L.ct_gain(S, R2), 0.03)


@pytest.mark.needs_data
def test_equity_premium_results(RAW):
    assert RAW["eqOosN"] == 732 and RAW["eqRtwoOls"] < 0 and RAW["eqUniPos"] <= 3


# ---------------------------------------------------------------- 12.6
def test_learned_hedge_is_close_to_delta(RAW):
    assert RAW["hdDeltaGap"] < 0.01 and abs(RAW["hdStdLearn"] / RAW["hdStdBs"] - 1) < 0.03
    close(RAW["hdPriceLearn"], RAW["hdBs"], 0.03)


def test_bands_cut_the_price(RAW):
    assert RAW["hdPriceWw"] < RAW["hdPriceCost"] < RAW["hdPriceBs"] and RAW["hdPriceWw"] < RAW["hdPriceWwOne"]
    assert RAW["hdTurnWw"] < RAW["hdTurnBs"]


def test_gains_have_mean_zero_for_any_hedge():
    rng = np.random.default_rng(11)
    paths = L.gbm_paths(400000, 5, 1.0, 100, 0.0, 0.3, rng)
    D = np.tanh(np.log(paths[:, :-1] / 100))                        # any hedge that uses only the past
    g = np.sum(D * np.diff(paths, axis=1), axis=1)
    assert abs(g.mean()) < 4 * g.std() / math.sqrt(len(g))


def test_cost_hedge_gradient():
    rng = np.random.default_rng(12)
    paths = L.gbm_paths(300, 6, 0.25, 100, 0.0, 0.2, rng); H = np.maximum(paths[:, -1] - 100, 0)
    Phi, G = L.hedge_design(paths, 100, 0.25, np.linspace(-0.5, 0.5, 3), 0.2)
    th = rng.standard_normal(7) * 0.1
    _, g = L.cost_objective(th, Phi, G, H, paths, 0.002, 1.0)
    for k in range(7):
        e = np.zeros(7); e[k] = 1e-6
        num = (L.cost_objective(th + e, Phi, G, H, paths, 0.002, 1.0)[0] - L.cost_objective(th - e, Phi, G, H, paths, 0.002, 1.0)[0]) / 2e-6
        close(g[k], num, 1e-5 * max(1, abs(num)))


# ---------------------------------------------------------------- 12.7
def test_expected_max_bound_and_approximation(RAW):
    for nm, n in [("Ten", 10), ("Hundred", 100), ("Thousand", 1000)]:
        assert RAW[f"ovMax{nm}"] < math.sqrt(2 * math.log(n))
        close(RAW[f"ovMax{nm}"], RAW[f"ovSim{nm}"], 0.02)
    assert RAW["ovApproxErr"] < 0.05


def test_holm_controls_fwer_under_the_null():
    rng = np.random.default_rng(13); p = rng.uniform(size=(20000, 10))
    rej = np.array([L.holm(q, 0.1).any() for q in p])
    assert rej.mean() < 0.1 + 3 * math.sqrt(0.1 * 0.9 / 20000)


def test_holm_between_bonferroni_and_bh():
    p = np.array([0.001, 0.008, 0.012, 0.04, 0.2])
    assert (p <= 0.01).sum() <= L.holm(p, 0.05).sum() <= L.bh(p, 0.05).sum()


def test_psr_dsr(RAW):
    se = L.sharpe_se(0.1, 500)
    close(L.psr(0.1, 500), norm.cdf(0.1 / se), 1e-14)
    d, bench = L.dsr(0.1, 500, 10, 0.01)
    close(bench, 0.1 * L.expected_max_approx(10), 1e-14); close(d, L.psr(0.1, 500, bench=bench), 1e-14)
    assert RAW["ovEqDsr"] < RAW["ovEqPsr"]


def test_pbo_on_noise(RAW):
    assert 0.3 < RAW["ovPbo"] < 0.8 and abs(RAW["ovBestOos"]) < 4 * RAW["ovBestOosSe"]


def test_snippet_matches_library():
    import subprocess, sys
    from pathlib import Path
    out = subprocess.run([sys.executable, str(Path(__file__).parent / "snippets" / "lasso_cd.py")],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    rng = np.random.default_rng(12022)
    X = rng.standard_normal((200, 6)); X -= X.mean(axis=0)
    y = X @ [1.0, -0.5, 0, 0, 0.25, 0] + rng.standard_normal(200); y -= y.mean()
    for line, lam in zip(out, (0.01, 0.1, 0.3)):
        got = np.array(line.split("[")[1].rstrip("]").split(), float)
        assert np.allclose(got, np.round(L.lasso(X, y, lam), 3), atol=1.5e-3)


def test_round2_printed_values(RAW, NUM):
    """The deflated Sharpe ratio reproduces from the printed inputs, least squares and its floored
    version are one portfolio, and the new macros agree with their raw values."""
    sr, sd, bench, T, n = (RAW[k] for k in ("ovEqSr", "ovEqSrSd", "ovEqBench", "ovEqT", "ovEqModels"))
    close(bench, sd * L.expected_max_approx(int(n)), 1e-12)
    close(RAW["ovEqDsr"], L.psr(sr, int(T), RAW["ovEqSkew"], RAW["ovEqKurt"], bench=bench), 1e-12)
    assert RAW["ovEqDistinct"] == RAW["ovEqModels"] - 1
    assert NUM["ovEqSrSd"] == f"{sd:.4f}" and NUM["ovEqBench"] == f"{bench:.4f}"
    assert NUM["kfSelT"] == f"{RAW['kfSelT']:.0f}" and RAW["eqNfirst"] < RAW["eqNlast"]
    assert NUM["eqOlsGainDecade"] == "2010s" and NUM["nnDeltaMaxV"] == "0.50" and RAW["nnDeltaMaxSign"] == -1
