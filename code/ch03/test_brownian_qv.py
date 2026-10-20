"""Checks of the Chapter 3 numbers produced by brownian_qv.py."""
import math
import numpy as np
from conftest import close
import ch03lib as L


def test_levy_construction_covariance(RAW):
    M = RAW["levyPaths"]
    se = lambda v: 4 * v * math.sqrt(2 / M)
    close(RAW["levyVarHalf"], 0.5, se(0.5)); close(RAW["levyVarOne"], 1.0, se(1.0))
    close(RAW["levyCovQ"], 0.25, 4 * math.sqrt(0.25 * 0.75 + 0.25 ** 2) / math.sqrt(M))
    close(RAW["levyIncCov"], 0.0, 4 * 0.5 / math.sqrt(M))
    rng = np.random.default_rng(31)
    t, W = L.levy_bm(6, 2.0, rng, m=20000)
    C = np.cov(W[:, 1:].T)
    target = np.minimum.outer(t[1:], t[1:])
    assert np.max(np.abs(C - target)) < 0.08


def test_bridge_midpoint_law(RAW):
    # Gaussian conditioning with covariance min(s, t): mean is the average, variance h / 4
    s, u = 0.3, 0.9; m = (s + u) / 2
    C = np.minimum.outer([s, m, u], [s, m, u])
    w = np.linalg.solve(C[np.ix_([0, 2], [0, 2])], C[[0, 2], 1])
    close(w[0], 0.5, 1e-12); close(w[1], 0.5, 1e-12)
    close(C[1, 1] - C[1, [0, 2]] @ w, (u - s) / 4, 1e-12)
    close(RAW["levyBridgeVar"], 0.125, 1e-15)


def test_quadratic_and_total_variation(RAW):
    close(RAW["qvPathSixteen"], 1.0, 4 * math.sqrt(2 / 2 ** 16))
    assert RAW["qvMSEratio"] < 0.06
    close(RAW["qvSlope"], -0.5, 0.02)
    close(RAW["tvMeanSixteen"], math.sqrt(2 * 2 ** 16 / math.pi), 1e-9)
    close(RAW["tvPathSixteen"], RAW["tvMeanSixteen"], 4 * math.sqrt(1 - 2 / math.pi))
    assert RAW["tvPathSixteen"] > RAW["tvPathTwelve"] > RAW["tvPathEight"]


def test_total_variation_bound_used_in_proof():
    # sum (dW)^2 <= max|dW| * sum |dW| on any path
    rng = np.random.default_rng(32)
    _, W = L.bm_paths(1000, 50, 1.0, rng)
    d = np.abs(np.diff(W, axis=1))
    assert np.all(np.sum(d ** 2, 1) <= d.max(1) * d.sum(1) + 1e-15)
