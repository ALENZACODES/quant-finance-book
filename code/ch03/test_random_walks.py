"""Checks of the Chapter 3 numbers produced by random_walks.py."""
import math
import numpy as np
from scipy import stats, integrate
from conftest import close
import ch03lib as L


def test_reflection_srw_exact_against_enumeration(RAW):
    for n, a in [(8, 1), (10, 2), (12, 5), (15, 3), (16, 3)]:
        close(L.srw_max_tail(n, a), L.srw_max_tail_enumerate(n, a), 1e-12)
    close(RAW["reflExact"], 100 * RAW["reflEnumerate"], 1e-10)
    assert abs(RAW["reflMC"] - RAW["reflExact"]) < 4 * RAW["reflMCse"]
    # the maximum is at least as large as the endpoint, so the tail of the maximum is larger
    assert RAW["reflExact"] > RAW["reflNaive"]


def test_random_walk_moments(RAW):
    close(RAW["rwVarFourHundred"], 400, 4 * 400 * math.sqrt(2 / 20000))
    close(RAW["rwCovCheck"], 100, 4 * math.sqrt(100 * 300 + 2 * 100 ** 2) / math.sqrt(20000))
    close(RAW["rwKolSlope"], -0.5, 0.03)
    close(RAW["normCdfOne"], stats.norm.cdf(1), 1e-4)


def test_brownian_maximum(RAW):
    close(RAW["bmMaxCont"], 100 * 2 * stats.norm.sf(1), 1e-10)
    assert RAW["bmMaxZ"] < 4
    # the quoted standard error is the binomial one for 40,000 paths, and the quoted bound is the largest |z| from 128 dates up
    p = RAW["bmMaxMC"][-1]
    close(RAW["bmMaxSeBig"], 100 * np.sqrt(p * (1 - p) / 40000), 1e-10)
    close(RAW["bmMaxZmax"], RAW["bmMaxZ"], 1e-12)
    assert RAW["bmMaxZ"] < 2
    assert max(RAW["bmMaxMC"]) < RAW["bmMaxCont"] / 100          # discrete monitoring misses crossings
    assert np.all(np.diff(RAW["bmMaxBGK"]) > 0)
    close(L.BGK_BETA, 0.5826, 1e-4)


def test_first_passage_density_integrates_to_reflection():
    for a, T in [(1.0, 1.0), (0.5, 2.0), (2.0, 3.0)]:
        val = integrate.quad(L.first_passage_pdf, 0, T, args=(a,))[0]
        close(val, L.bm_max_tail(a, T), 1e-8)
