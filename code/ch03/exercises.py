"""Section 3.13: the numerical answers to the exercises, each computed from its closed form and, where
the exercise is about a simulation statement, checked by Monte Carlo in test_crosschecks.py.

Run on its own:  python code/ch03/exercises.py   (writes code/ch03/out/exercises.json)
"""
from __future__ import annotations
import math
from scipy import stats
from numbers_ch03 import N, RAW, put, pct, word, save, finish, rng_for, L  # noqa: F401


def main():
    put("exOneVar", 1 + 2 + 2 * 1, "{:.0f}"); put("exOneCov", 2 - 1, "{:.0f}")
    put("exTwo", 3 * 2 ** 2, "{:.0f}")
    S0, mu, sig, T = 100.0, 0.08, 0.25, 2.0
    m, v, med, mode = L.gbm_moments(S0, mu, sig, T)
    put("exThreeMean", m, "{:.2f}"); put("exThreeMedian", med, "{:.2f}")
    pct("exThreeLoss", stats.norm.cdf(-(mu - 0.5 * sig ** 2) * math.sqrt(T) / sig), 1)
    put("exFourHalf", math.log(2) / 0.5, "{:.3f}"); put("exFourSd", 100 * 0.02 / math.sqrt(2 * 0.5), "{:.2f}")
    put("exFourAcf", math.exp(-0.5), "{:.4f}")
    put("exFive", 3 ** 3 / 3, "{:.0f}")
    pct("exSixP", L.exit_prob_up(0.05, 0.2, 0.1, 0.2), 2); put("exSixTau", L.exit_time_mean(0.05, 0.2, 0.1, 0.2), "{:.4f}")
    pct("exSixPZero", L.exit_prob_up(0.0, 0.2, 0.1, 0.2), 2)
    put("exSixTauZero", L.exit_time_mean(0.0, 0.2, 0.1, 0.2), "{:.2f}")
    put("exSeven", 0.5 * 2, "{:.0f}")
    put("exEight", 1 ** 2 + 2, "{:.0f}")
    put("exEightB", (1 + 0.1 * 2) ** 2 + 0.3 ** 2 * 2, "{:.2f}")
    return finish("exercises")


if __name__ == "__main__":
    main()
