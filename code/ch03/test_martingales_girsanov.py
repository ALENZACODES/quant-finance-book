"""Checks of the Chapter 3 numbers produced by martingales_girsanov.py."""
import math
import numpy as np
from scipy import stats
from conftest import close
import ch03lib as L


def test_martingale_means(RAW):
    assert abs(RAW["mgSq"]) < 4 * RAW["mgSqSE"]
    assert abs(RAW["mgExp"] - 1) < 4 * RAW["mgExpSE"]
    # heavy tails: the sample standard error understates the true one
    assert RAW["mgExpThreeTrueSE"] > RAW["mgExpThreeSE"]
    assert RAW["mgRejShare"] > 10 and RAW["mgSEHalfShare"] > 10     # the nominal 5% test misleads
    assert abs(RAW["mgRepMeanZ"]) < 4                                 # but the estimator is unbiased
    assert abs(RAW["mgExpThree"] - 1) < 4 * RAW["mgExpThreeTrueSE"]
    close(RAW["mgExpThreeShare"], 100 * stats.norm.sf(1.5), 4 * 100 * math.sqrt(0.067 / 2e5))


def test_exit_formulas():
    close(L.exit_prob_up(1e-9, 0.2, 0.1, 0.3), 0.75, 1e-6)
    close(L.exit_time_mean(1e-7, 0.2, 0.1, 0.3), 0.1 * 0.3 / 0.04, 1e-5)
    # symmetry: drift -mu with levels swapped
    close(L.exit_prob_up(-0.1, 0.2, 0.1, 0.3), 1 - L.exit_prob_up(0.1, 0.2, 0.3, 0.1), 1e-12)
    # the exponential martingale exp(-2 mu X / sigma^2) has the same value at both levels on average
    mu, s, a, b = 0.07, 0.25, 0.15, 0.05; p = L.exit_prob_up(mu, s, a, b); k = 2 * mu / s ** 2
    close(p * math.exp(-k * a) + (1 - p) * math.exp(k * b), 1.0, 1e-12)


def test_exit_simulation(RAW):
    assert RAW["exitZ"] < 4
    ft = [L.exit_time_mean(m, 0.2, 0.1, 0.1) for m in RAW["exitMus"]]
    assert np.max(np.abs(np.array(RAW["exitMCt"]) - ft)) < 0.006


def test_girsanov(RAW):
    assert abs(RAW["girDirect"] - RAW["girExact"]) < 4 * RAW["girDirectSE"]
    assert abs(RAW["girWeighted"] - RAW["girExact"]) < 4 * RAW["girWeightedSE"]
    close(RAW["isP"], 1e4 * stats.norm.sf(3), 1e-9)
    assert abs(RAW["isBig"] - RAW["isP"]) < 4 * RAW["isBigSE"]
    assert 0.7 < RAW["isEmpRatio"] / RAW["isRatio"] < 1.4
    rng = np.random.default_rng(61)
    W = rng.standard_normal(400000)
    w = L.girsanov_weight(0.7, W, 1.0)
    close(w.mean(), 1.0, 4 * w.std() / math.sqrt(len(W)))
    close(np.mean(w * (W + 0.7)), 0.0, 4 * np.std(w * (W + 0.7)) / math.sqrt(len(W)))   # W + theta t has mean 0 under Q


def test_plain_small_sample_text(RAW):
    # the text says no plain draw of the 1,000 exceeds 3, so the estimate and its sample SE are zero
    assert RAW["isPlainSmall"] == 0 and RAW["isPlainSmallSE"] == 0
    assert RAW["isPlainSmallTrueSE"] > 5


def test_girsanov_odds(RAW):
    # 3.120: the miss is above the exact value; two-sided odds about half the one-sided odds
    from scipy import stats
    z = RAW["girWeightedSigned"]
    assert z > 0 and RAW["girWeighted"] > RAW["girExact"]
    close(RAW["girWeightedOdds"], 1 / (2 * stats.norm.sf(z)), 1e-9)
    close(RAW["girWeightedOddsOne"], 1 / stats.norm.sf(z), 1e-9)
