"""Independent checks of the Chapter 1 numbers produced by irr_multiples.py (pass (b) of the production process).

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


def test_fund_irr_three_ways():
    t, c, d, nav = L.fund_example()
    cf = d - c
    close(RAW["fundIRR"], 100 * npf.irr(cf), 1e-8)
    r_check = {row["name"]: float(row["value"]) for row in csv.DictReader(open(HERE / "out" / "r_check.csv"))}
    close(RAW["fundIRR"], r_check["fundIRR"], 1e-6)
    # the printed snippet uses flows rounded to cents and must agree to two decimals
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "irr_newton.py")], capture_output=True, text=True).stdout
    assert out.strip() == "IRR of Fund A: 11.5298%"
    assert f"{RAW['fundIRR']:.2f}" == "11.53"
    # fund identities
    close(c.sum(), 97.0, 1e-12)
    close(d.sum(), 20 * 1.15 ** 5 + 22 * 1.15 ** 5 + 20 * 1.15 ** 5 + 15 * 1.15 ** 5 + 10 * 1.15 ** 5 - 8, 1e-9)
    assert RAW["fundIRR"] < 15.0


def test_interim_irr_shape():
    it = {int(k): v for k, v in RAW["interim"].items()}
    assert it[1] < it[2] < 0 < it[3] < it[4] < it[5] < it[6]
    for s in (7, 8, 9):
        assert abs(it[s] - it[9]) < 0.0010          # "within ten basis points"


def test_multiple_and_no_roots():
    roots = L.irr_all([-100, 230, -132])
    close(roots[0], 0.10, 1e-10); close(roots[1], 0.20, 1e-10)
    assert RAW["noneCount"] == 0
    assert 300 ** 2 - 4 * 100 * 250 < 0
    close(RAW["noneMinNPV"], 10.0, 1e-3)              # min of 100-300x+250x^2 at x=0.6


def test_crossover_and_aggregation():
    cr = RAW["crossRate"] / 100
    close(L.npv(cr, [-100, 0, 0, 200]), L.npv(cr, [-100, 130]), 1e-9)
    close(RAW["irrLong"], 100 * (2 ** (1 / 3) - 1), 1e-8)
    # combined IRR: 121x^2 + 150x - 200 = 0
    x = (-150 + np.sqrt(150 ** 2 + 4 * 121 * 200)) / (2 * 121)
    close(RAW["irrXY"], 100 * (1 / x - 1), 1e-8)


def test_mirr_against_numpy_financial():
    close(RAW["mirrMulti"], 100 * npf.mirr([-100, 230, -132], 0.08, 0.12), 1e-9)
    t, c, d, nav = L.fund_example()
    close(RAW["mirrFundTwelve"], 100 * npf.mirr(d - c, 0.08, 0.12), 1e-9)
    assert RAW["mirrFundEight"] < RAW["fundIRR"] and RAW["mirrFundTwelve"] < RAW["fundIRR"]


def test_twr_dwr():
    close(RAW["twrTotal"], 8.0, 1e-9)
    y = (-200 + np.sqrt(200 ** 2 + 4 * 100 * 288)) / 200
    close(RAW["dwrPer"], 100 * (y - 1), 1e-9)


def test_multiples_and_pme():
    t, c, d, nav = L.fund_example()
    I = L.index_levels()
    close(RAW["tvpiFive"], (d[:6].sum() + nav[5]) / c[:6].sum(), 1e-9)
    # Kaplan-Schoar by discounting instead of compounding (identical ratio)
    num = np.sum(d / I) + nav[-1] / I[-1]
    den = np.sum(c / I)
    close(RAW["ksPME"], num / den, 1e-12)
    assert RAW["shadowMin"] < 0 and len(RAW["lnRoots"]) == 2


def test_jcurve_claims():
    t, c, d, nav = L.fund_example()
    cum = np.cumsum(d - c)
    assert int(np.argmin(cum)) == 4 and cum[6] < 0 < cum[7]


def test_fee_drag_is_the_whole_gap():
    """With the fee set to zero every deal and the fund earn exactly 15%, so the gap is all fee drag."""
    t, c, d, nav = L.fund_example(fee=0.0)
    close(L.irr(d - c), 0.15, 1e-10)
    close(RAW["fundIRRNoFee"], 15.0, 1e-8)
    close(RAW["feeDrag"], 15.0 - RAW["fundIRR"], 1e-8)


def test_mirr_rates_either_side_of_irr():
    t, c, d, nav = L.fund_example()
    close(RAW["mirrFundTwelveTwelve"], 100 * npf.mirr(d - c, 0.12, 0.12), 1e-9)
    # 8% is below the IRR and 12% above it, so the MIRR at 12%/12% lies between the IRR and 12%
    assert 8.0 < RAW["fundIRR"] < 12.0
    assert RAW["fundIRR"] < RAW["mirrFundTwelveTwelve"] < 12.0
