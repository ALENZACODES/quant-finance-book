"""Tests of the listings in code/ch12/snippets and of the numbers printed beside the new figures."""
import math, re, shutil, subprocess, sys
import numpy as np
import pytest
from scipy import stats
from conftest import close, HERE
import ch12lib as L

ROOT = HERE.parents[1]
SN = HERE / "snippets"
sys.path.insert(0, str(SN))


def run(*cmd):
    return subprocess.run(list(cmd), cwd=ROOT, capture_output=True, text=True, check=True).stdout


def floats(text):
    return [float(v) for v in re.findall(r"-?\d+\.\d+(?:e[-+]\d+)?", text)]


def test_ridge_svd_listing():
    out = run(sys.executable, str(SN / "ridge_svd.py"))
    assert re.findall(r"df (\d+\.\d+)", out) == ["6.00", "4.76", "1.72"]
    from ridge_svd import ridge_svd
    rng = np.random.default_rng(0); X = rng.standard_normal((30, 4)); y = rng.standard_normal(30)
    for lam in (0.0, 3.0):
        b, sh, df = ridge_svd(X, y, lam)
        close(np.abs(b - np.linalg.solve(X.T @ X + lam * np.eye(4), X.T @ y)).max(), 0, 1e-12)
        close(df, np.trace(X @ np.linalg.solve(X.T @ X + lam * np.eye(4), X.T)), 1e-10)
    # rank-deficient at lambda = 0: the tolerance drops the null direction, giving the minimum-norm fit
    Xd = np.column_stack([X, X[:, 0] + X[:, 1]])
    b, sh, df = ridge_svd(Xd, y, 0.0)
    close(df, 4.0, 1e-12); close(np.abs(b - np.linalg.pinv(Xd) @ y).max(), 0, 1e-10)
    with pytest.raises(ValueError):
        ridge_svd(X, y, -1.0)


def test_purged_kfold_listing_matches_library(NUM):
    from purged_kfold import purged_kfold
    n, k, h, e = (int(NUM[x].replace(",", "")) for x in ("kfN", "kfK", "kfH", "kfEmb"))
    for (train, test), lib in zip(purged_kfold(n, k, h, e), L.kfold_blocks(n, k)):
        assert np.array_equal(test, lib)
        assert np.array_equal(train, L.purged_train(n, lib, h, e))
    sizes = [(len(te), len(tr)) for tr, te in purged_kfold(n, k, h, e)]
    assert {s[0] for s in sizes} == {46, 47} and min(s[1] for s in sizes) == 387 and max(s[1] for s in sizes) == 410
    assert run(sys.executable, str(SN / "purged_kfold.py")).strip().endswith("labels excluded 28")
    # Exercise 12.4 with the library: six test labels, h = 12, no embargo
    assert 100 - len(L.purged_train(100, np.arange(40, 46), 12, 0)) == 28


def test_best_split_listing():
    from best_split import best_split, sse
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = np.round(rng.uniform(size=40), 1); y = rng.standard_normal(40)    # many ties
        t, g = best_split(x, y)
        u = np.unique(x); cuts = (u[:-1] + u[1:]) / 2
        brute = [sse(y) - sse(y[x <= c]) - sse(y[x > c]) for c in cuts]
        close(g, max(brute), 1e-10); close(t, cuts[int(np.argmax(brute))], 1e-12)
    v = floats(run(sys.executable, str(SN / "best_split.py")))
    assert v[0] == v[2] and v[1] == v[3]


def test_backprop_listing():
    out = run(sys.executable, str(SN / "backprop.py"))
    assert "y hat 0.9242, dL/dw2 0.4271, dL/dw1 1.4537" in out
    a = math.tanh(0.5); yh = 2 * a
    assert (f"{yh:.4f}", f"{yh * a:.4f}", f"{yh * 2 * (1 - a * a):.4f}") == ("0.9242", "0.4271", "1.4537")
    assert floats(out)[-1] < 1e-8


@pytest.mark.needs_data
@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_clark_west_r_listing(NUM):
    out = run("Rscript", str(SN / "clark_west.R"))
    n = int(re.match(r"(\d+) forecasts", out).group(1)); r2, t, p = floats(out)
    assert (str(n), f"{r2:.2f}", f"{t:.2f}", f"{p:.3f}") == \
        (NUM["eqOosN"], NUM["eqRtwoTbl"], NUM["eqCwTbl"], NUM["eqPTbl"])


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="Octave not installed")
def test_expected_max_octave_listing(NUM):
    v = floats(run("octave-cli", "-q", str(SN / "expected_max.m")).replace("Exercise 12.9", ""))
    tab = [v[i:i + 3] for i in range(0, 9, 3)]
    for row, tg in zip(tab, ("Ten", "Hundred", "Thousand")):
        assert [f"{x:.3f}" for x in row] == [NUM[f"ovMax{tg}"], NUM[f"ovApprox{tg}"], NUM[f"ovBound{tg}"]]
    close(v[9], float(NUM["ovApproxHundred"]), 5e-4); close(v[10], v[9] / math.sqrt(5), 5e-3)


def test_new_figure_numbers(RAW, NUM):
    # cross-validation against the oracle: the chosen penalty never beats the oracle on average
    assert RAW["cvRidgeErrCv"] >= RAW["cvRidgeErrOracle"] and RAW["cvLassoErrCv"] >= RAW["cvLassoErrOracle"]
    assert RAW["cvRidgeLo"] <= RAW["cvRidgeMed"] <= RAW["cvRidgeHi"]
    # boosting shrinkage: rounds to fit half of a component at nu = 0.1
    for s, tg in ((0.05, "Small"), (0.9, "Big")):
        k = int(NUM[f"skHalf{tg}"])                     # the first whole round with half fitted
        assert 1 - (1 - 0.1 * s) ** k >= 0.5 > 1 - (1 - 0.1 * s) ** (k - 1)
    # Campbell and Thompson at the sample Sharpe ratio and at 0.05
    close(RAW["ctLowHalf"], 100 * L.ct_gain(0.05, 0.005), 1e-12)
    assert NUM["eqCtHalf"] == f"{100 * L.ct_gain(RAW['eqS'], 0.005):.0f}"
    # deflated Sharpe ratio: the curve passes through the chapter's value at the number of models
    assert NUM["dsAtModels"] == NUM["ovEqDsr"]
    assert RAW["dsTwo"] > 0.95 > RAW["dsHundred"] and int(NUM["dsCross"]) <= int(NUM["ovEqModels"])
    # hedging table: every markup is the price less the Black-Scholes value
    for nm in ("Bs", "Zero", "Cost", "WwOne", "Ww", "Flat"):
        close(RAW[f"hdMarkup{nm}"], RAW[f"hdPrice{nm}"] - RAW["hdBs"], 1e-12)
    assert RAW["hpTurnBand"] < RAW["hpTurnDelta"] and RAW["hpTrades"] <= RAW["hpDays"]
