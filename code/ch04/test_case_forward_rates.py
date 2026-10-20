"""Tests for case_forward_rates.py on the Government of Canada data."""
import numpy as np
import pandas as pd
from conftest import close
import case_forward_rates as C


def test_sample_and_signs(RAW):
    assert RAW["caseMonths"] >= 300
    close(RAW["fbOneN"], RAW["caseMonths"] - 12, 0); close(RAW["fbTwoN"], RAW["caseMonths"] - 24, 0)
    assert 0 < RAW["fbOneRSq"] < 1
    close(RAW["fbOneTOne"], (RAW["fbOneSlope"] - 1) / RAW["fbOneSE"], 0.02)


def test_regression_by_hand(RAW):
    me = C.month_end_forwards()
    d, res = C.fb_regression(me, "f12", 12)
    x = d["x"].values; y = d["y"].values
    b = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
    close(b, res.params[1], 1e-10)
    # Newey-West by hand with Bartlett weights
    X = np.column_stack([np.ones_like(x), x]); e = y - X @ res.params; n = len(y)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, 13):
        w = 1 - l / 13; G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None]); S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X); V = XtXi @ S @ XtXi
    close(np.sqrt(V[1, 1]), res.bse[1], 1e-10)


def test_forwards_are_bootstrapped_consistently():
    me = C.month_end_forwards()
    # forward rates above zero and within ten points of the spot rate
    assert (me["f12"] > -0.01).all() and ((me["f12"] - me["y1"]).abs() < 0.1).all()


def test_confidence_intervals_and_sample_end(RAW, NUM):
    for nm in ["One", "Two"]:
        b, se = RAW[f"fb{nm}Slope"], RAW[f"fb{nm}SE"]
        assert abs(RAW[f"fb{nm}Lo"] - (b - 1.96 * se)) < 1e-12 and abs(RAW[f"fb{nm}Hi"] - (b + 1.96 * se)) < 1e-12
    assert NUM["fbOneLastDate"] == "September 2026" and NUM["fbEnd"] == "September 2026"
    assert RAW["fbOneN"] == RAW["caseMonths"] - 12 and RAW["fbTwoN"] == RAW["caseMonths"] - 24
