"""Checks of the Chapter 3 numbers produced by ito_integral.py."""
import math
import numpy as np
from conftest import close
import ch03lib as L


def test_discrete_identity_behind_int_w_dw():
    rng = np.random.default_rng(41)
    _, W = L.bm_paths(500, 100, 1.0, rng)
    left = L.ito_sum(W, W)
    close(np.max(np.abs(left - 0.5 * (W[:, -1] ** 2 - L.quadratic_variation(W)))), 0.0, 1e-12)


def test_ito_and_stratonovich_limits(RAW):
    close(RAW["itoLimit"], (RAW["itoWT"] ** 2 - 1) / 2, 1e-12)
    close(RAW["stratLimit"], RAW["itoWT"] ** 2 / 2, 1e-12)
    assert abs(RAW["itoSumBig"] - RAW["itoLimit"]) < 0.02
    assert abs(RAW["stratSumBig"] - RAW["stratLimit"]) < 0.02
    M = RAW["itoPaths"]
    close(RAW["itoMean"], 0.0, 4 * math.sqrt(0.5 / M))
    close(RAW["itoVar"], 0.5 * (1 - 1 / 500), 4 * math.sqrt(3.75 / M))
    close(RAW["stratGapMean"], 0.5, 4 * RAW["stratGapSD"] / math.sqrt(M))


def test_ito_isometry(RAW):
    M = RAW["itoPaths"]
    close(RAW["isoTVar"], 1 / 3, 4 * (1 / 3) * math.sqrt(2 / M))
    assert abs(RAW["isoEz"]) < 4


def test_ito_lemma_rates(RAW):
    close(RAW["lemmaSlopeCube"], -0.5, 0.06); close(RAW["lemmaSlopeExp"], -0.5, 0.06)
    assert RAW["naiveRmsCube"] > 20 * RAW["lemmaRmsCube"]
    assert RAW["naiveGapCorr"] > 0.999
