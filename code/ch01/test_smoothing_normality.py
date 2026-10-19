"""Independent checks of the Chapter 1 numbers produced by smoothing_normality.py (pass (b) of the production process).

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


def test_jb_matches_statsmodels_and_R():
    x = rng.standard_t(5, 400)
    jb, _ = L.jarque_bera(x)
    close(jb, float(sm_jb(x)[0]), 1e-8)
    r_check = {row["name"]: float(row["value"]) for row in csv.DictReader(open(HERE / "out" / "r_check.csv"))}
    for k in ["gocDsd", "gocDskew", "gocDexk", "gocJB", "exBeta", "exRho", "exSpear", "irrXY", "mirrMulti",
              "dwrPer", "crossRate"]:
        close(RAW[k], r_check[k], 1e-6 * max(1, abs(r_check[k])))
    close(RAW["jbCrit"], 5.991, 1e-3)


def test_jb_size_monotone_and_conservative():
    sz = {int(k): v for k, v in RAW["jbSize"].items()}
    assert all(v < 0.05 for v in sz.values())
    pw = {int(k): v for k, v in RAW["jbPower"].items()}
    assert pw[20] < pw[100] < pw[250]


def test_smoothing_theory_by_simulation():
    a, sig = 0.6, 0.04
    r = rng.normal(0, sig, 400_000)
    rs = L.smooth(r, a)
    close(rs.std() / sig, np.sqrt((1 - a) / (1 + a)), 0.005)
    ac = L.acf(rs, 3)
    for k in (1, 2, 3):
        close(ac[k], a ** k, 0.01)
    close(L.unsmooth(rs, a).std(), sig, 2e-4)
    e = rs - rs.mean()
    close(L.durbin_watson(e), float(sm_dw(e)), 1e-12)


def test_variance_ratio_by_simulation():
    rho, q = 0.6, 12
    n = 600_000
    e = rng.standard_normal(n)
    x = np.empty(n); x[0] = e[0] / np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    agg = x[: n // q * q].reshape(-1, q).sum(1)
    close(agg.var() / (q * x.var()), L.variance_ratio_ar1(rho, q), 0.08)
    close(L.variance_ratio_ar1(rho, 5000), (1 + rho) / (1 - rho), 0.01)


def test_smoothing_scales_beta_and_correlation():
    """Under R*_t = (1-a) R_t + a R*_{t-1} with a serially independent market, the measured beta is
    (1-a) times the true beta and the measured correlation sqrt(1-a^2) times the true one."""
    a = RAW["smAlpha"]; rng = np.random.default_rng(101)
    n = 400_000; m = rng.normal(0.0, 0.04, n); r = 0.9 * m + rng.normal(0.0, 0.03, n)
    rs = L.smooth(r, a)
    beta_true = np.cov(r, m)[0, 1] / m.var(ddof=1); beta_obs = np.cov(rs, m)[0, 1] / m.var(ddof=1)
    corr_true, corr_obs = np.corrcoef(r, m)[0, 1], np.corrcoef(rs, m)[0, 1]
    assert abs(beta_obs / beta_true - (1 - a)) < 0.01 and abs(corr_obs / corr_true - np.sqrt(1 - a * a)) < 0.01
    assert RAW["smBetaFactor"] == 1 - a and abs(RAW["smCorrFactor"] - np.sqrt(1 - a * a)) < 1e-12


def test_jb_listing_matches_text():
    import subprocess, sys
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "jb_test.py")], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert out == [f"normal: JB = {RAW['jbSnipNorm']:.2f}, p = {RAW['jbSnipNormP']:.4f}",
                   f"t5: JB = {RAW['jbSnipT']:.2f}, p = {RAW['jbSnipTP']:.4f}"]
