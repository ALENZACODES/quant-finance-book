"""Independent checks of the Chapter 1 numbers produced by rates_returns.py (pass (b) of the production process).

Run from the repository root:  python -m pytest -q code/ch01
The R cross-check (code/ch01/crosscheck.R) must have been run first; run_all.py writes numbers.json.
"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
import numpy_financial as npf
import pytest
from scipy import stats
from statsmodels.stats.stattools import jarque_bera as sm_jb, durbin_watson as sm_dw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import ch01lib as L  # noqa: E402

RAW = json.loads((HERE / "out" / "numbers.json").read_text())["RAW"]
rng = np.random.default_rng(7)


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def test_compounding_closed_forms():
    close(RAW["earA"], 8.0, 1e-12)
    close(RAW["earD"], 100 * ((1 + 0.08 / 12) ** 12 - 1), 1e-10)
    close(RAW["earF"], 100 * (np.exp(0.08) - 1), 1e-10)
    close(RAW["ccEquivEight"], 100 * np.log(1.08), 1e-10)
    # quarterly compounding closes most of the annual-to-continuous gap (claim in Example 1.1)
    assert (RAW["earC"] - RAW["earA"]) / (RAW["earF"] - RAW["earA"]) > 0.7


def test_log_vs_simple():
    R = np.linspace(-0.9, 2, 1000)
    assert np.all(np.log1p(R) <= R + 1e-15)
    close(RAW["upLog"] + RAW["downLog"], 0.0, 1e-9)
    close(RAW["udApprox"], RAW["udMean"] - RAW["udHalfVar"], 1e-9)
    close(RAW["udApprox"], -100 / 288, 1e-9); close(RAW["udApproxSample"], 100 / 12 - 100 / 288 * 50, 1e-9)


def test_notional():
    close(RAW["simpleFull"], 2.40, 1e-9)
    close(RAW["simplePart"], 100 * (0.02 / 0.25 + 0.004), 1e-9)
    close(RAW["simplePartLoss"], 100 * (-0.02 / 0.25 + 0.004), 1e-9)
    close(RAW["logPart"], 100 * np.log(1.084), 1e-9)
