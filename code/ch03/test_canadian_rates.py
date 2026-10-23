"""Checks of the Canadian rate examples (canadian_rates.py) and of the case-study sample end."""
import math
import numpy as np
import pandas as pd
from scipy import stats
from conftest import close
import canadian_rates as C


def test_sample_ends_with_last_complete_month():
    y = C.load()
    assert y.index[-1] <= pd.Timestamp("2026-09-30") and y.index[-1].month == 9 and y.index[-1].year == 2026


def test_case_study_has_210_month_ends(RAW, NUM):
    assert RAW["caseN"] == 210 and NUM["caseEnd"] == "September 2026" and NUM["caseStart"] == "April 2009"


def test_realised_variance_by_hand(RAW):
    y = C.load(); d = y[(y.index > "2024-12-30") & (y.index <= "2025-12-31")]
    q = sum((b - a) ** 2 for a, b in zip(d.values[:-1], d.values[1:]))
    close(RAW["canQvTwentyFive"], q, 1e-12)
    rv = C.realised(y)
    close(RAW["canRvDailyMed"], float(np.median(rv["daily"])), 1e-12)


def test_reflection_model_and_ratio(RAW):
    s = RAW["canSigma"]
    close(RAW["canHitModel"], 200 * stats.norm.sf(0.5 / s), 1e-9)
    close(RAW["canHitModel"], 2 * RAW["canEndModel"], 1e-9)
    assert RAW["canHitMonthEmp"] < RAW["canHitEmp"]          # sparse monitoring misses crossings


def test_convexity_drift_against_lognormal_mean(RAW):
    s = RAW["canSigma"] / 100
    # E[exp(-5 s W_1)] = exp(12.5 s^2); to first order the drift is 12.5 s^2
    close(RAW["canConvex"] / 100, math.expm1(12.5 * s ** 2), 1e-6)
    rng = np.random.default_rng(71); z = rng.standard_normal(2_000_000)
    close(np.mean(np.exp(-5 * s * z)) - 1, RAW["canConvex"] / 100, 4e-5)


def test_kendall_correction(RAW):
    n = RAW["caseN"] - 1
    close(RAW["casePhiKendall"], RAW["casePhi"] + (1 + 3 * RAW["casePhi"]) / n, 1e-4)


def test_either_level_probability(RAW):
    """The two-sided touch probability from the series agrees with a direct simulation and exceeds one side."""
    import math
    import numpy as np
    rng = np.random.default_rng(303)
    b = 0.5 / RAW["canSigma"]
    n, m = 20_000, 2_000
    W = np.cumsum(rng.standard_normal((n, m)) * math.sqrt(1 / m), axis=1)
    sim = np.mean(np.abs(W).max(axis=1) >= b)
    assert RAW["canHitEitherModel"] > RAW["canHitModel"]
    assert abs(RAW["canHitEitherModel"] / 100 - sim) < 0.02   # the discrete grid misses a few touches


def test_carry_and_convexity_drift_of_an_aging_zero(RAW):
    # generator of f(t, y) = exp(-(T - t) y) under dy = sigma dW, by finite differences at T - t = 5:
    # (f_t + sigma^2 f_yy / 2) / f = y + 12.5 sigma^2, the carry plus the convexity term of Example 3.1
    y, s, h = RAW["canYieldEndCc"] / 100, RAW["canSigma"] / 100, 1e-5
    f = lambda tau, yy: math.exp(-tau * yy)
    ft = (f(5 - h, y) - f(5 + h, y)) / (2 * h)
    fyy = (f(5, y + h) - 2 * f(5, y) + f(5, y - h)) / h ** 2
    close((ft + 0.5 * s ** 2 * fyy) / f(5, y), y + 12.5 * s ** 2, 1e-6)
    close(RAW["canYieldEndCc"] / 100, 2 * math.log1p(RAW["canYieldEnd"] / 200), 1e-12)
    close(RAW["canCarryBp"], 1e4 * RAW["canYieldEndCc"] / 100, 1e-9)
    close(RAW["canConvexShare"] / 100, 12.5 * s ** 2 / (y + 12.5 * s ** 2), 1e-9)
    y_end = C.load()
    assert abs(y_end.iloc[-1] - RAW["canYieldEnd"]) < 1e-12
