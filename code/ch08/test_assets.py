"""Tests of the listings in code/ch08/snippets and of the numbers printed beside the later figures."""
import math, re, shutil, subprocess, sys
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from conftest import close, HERE
import ch08lib as L

ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "snippets"))
from hist_var_es import hist_var_es  # noqa: E402
import lr_tests  # noqa: E402


def floats(text):
    return [float(v) for v in re.findall(r"-?\d+\.\d+", text)]


def test_hist_var_es_listing(RAW):
    r, _ = L.load_daily(); x = r.values
    for a, tag in ((0.99, "N"), (0.975, "S")):
        v, e = hist_var_es(x, a)
        assert v == L.var_hist(x, a)
        close(e, RAW[f"esExact{tag}"], 1e-14)
        close(100 * (e - L.es_hist(x, a)), RAW[f"esGap{tag}"], 1e-12)
        assert e >= L.es_hist(x, a)                     # the boundary loss is the smallest of the k
    # whole n(1 - a): exact ES is the mean of the k largest; otherwise the boundary gets a fraction
    y = np.arange(1.0, 201.0)
    assert hist_var_es(-y, 0.95) == (191.0, np.mean(y[-10:]))
    v, e = hist_var_es(-y, 0.9525)                       # n(1 - a) = 9.5
    assert v == 191.0 and math.isclose(e, (y[-9:].sum() + 0.5 * 191) / 9.5)
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "hist_var_es.py")], capture_output=True,
                         text=True, check=True).stdout
    assert floats(out) == [0.90, 0.05, 0.05, 0.85, 0.03, 0.0433]


def test_lr_listing_matches_library_and_case(RAW):
    h = pd.read_csv(HERE / "out" / "case_hits.csv")
    for col, tag in (("nor", "Nor"), ("hs", "Hs"), ("hsl", "Hsl"), ("ew", "Ew"), ("gn", "Gn"), ("gt", "Gt"),
                     ("fhs", "Fhs")):
        hits = h[col].values
        lk, pk = lr_tests.kupiec(hits, 0.01); lc, pc, n = lr_tests.christoffersen(hits)
        close(lk, L.kupiec(int(hits.sum()), len(hits), 0.01)[0], 1e-9); close(pk, RAW[f"ca{tag}Pk"], 1e-12)
        close(lc, L.christoffersen(hits)[0], 1e-9); close(pc, RAW[f"ca{tag}Pc"], 1e-12)
    # n11 = 0: the statistic is finite and equals the closed form with pi11 = 0
    hits = np.zeros(250, int); hits[[40, 90, 160, 220]] = 1
    lc, pc, n = lr_tests.christoffersen(hits)
    assert n[1, 1] == 0 and math.isfinite(lc)
    p01 = n[0, 1] / (n[0, 0] + n[0, 1]); p1 = 4 / 249
    l1 = n[0, 0] * math.log(1 - p01) + n[0, 1] * math.log(p01)
    l0 = (n[0, 0] + n[1, 0]) * math.log(1 - p1) + 4 * math.log(p1)
    close(lc, -2 * (l0 - l1), 1e-12); close(lc, L.christoffersen(hits)[0], 1e-12)
    # no exceptions at all
    assert lr_tests.christoffersen(np.zeros(250, int))[0] == 0.0
    close(lr_tests.kupiec(np.zeros(250, int), 0.01)[0], -2 * 250 * math.log(0.99), 1e-12)
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "lr_tests.py")], capture_output=True,
                         text=True, check=True).stdout
    assert floats(out) == [1.957, 0.162, 3.154, 0.076, 0.131, 0.718]


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_z2_listing(RAW):
    out = subprocess.run(["octave-cli", "--no-gui", str(HERE / "snippets" / "es_null.m")], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    m1, v, sd, crit = floats(out)
    close(m1, 0.025, 1e-6)
    pm1, pm2 = L.z2_moments("t", 0.975, nu=5); close(v, pm2 - pm1 ** 2, 1e-6)
    assert f"{sd:.4f}" == f"{RAW['ztSdAn']:.4f}" and f"{crit:.4f}" == f"{RAW['ztCritAn']:.4f}"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_garch_listing(NUM):
    out = subprocess.run(["Rscript", str(HERE / "snippets" / "garch_mle.R")], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    a, b, p, lr, ll = floats(out)
    half = int(re.search(r"half life (\d+)", out).group(1))
    assert (f"{a:.3f}", f"{b:.3f}", f"{p:.4f}", str(half), f"{lr:.1f}") == \
        (NUM["gaAlphaN"], NUM["gaBetaN"], NUM["gaPersN"], NUM["gaHalfN"], NUM["gaLrVolN"])
    assert f"{-ll:,.1f}" == NUM["gaLogLikN"].replace("\\ensuremath{-}", "")
    assert "Nelder-Mead 0, BFGS 0" in out                 # both optimisers report convergence


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_unsmoothing_listing(NUM):
    out = subprocess.run(["Rscript", str(HERE / "snippets" / "unsmooth.R")], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    ph, cor, vt, vs, vu, b0, bs = floats(out)
    assert (f"{ph:.3f}", f"{cor:.4f}", f"{vt:.1f}", f"{vs:.1f}", f"{vu:.1f}", f"{b0:.2f}", f"{bs:.2f}") == \
        tuple(NUM[k] for k in ("smPhiHat", "smCorrU", "smVolT", "smVolS", "smVolU", "smBetaZero", "smBetaSum"))


def test_loans_band(RAW):
    from tails import tail_measures
    lo = 1 - math.sqrt(0.99); close(RAW["saBandLo"], 100 * lo, 1e-12)
    assert lo < RAW["saBadMin"] <= lo + 1e-5 and RAW["saBadMax"] == 0.01   # (lo, 1%], grid step 1e-5
    assert RAW["saEsWorst"] <= 1e-9                                          # ES never above the sum
    for q in (lo * 0.999, 0.0101):                                           # outside the band: no violation
        v1 = tail_measures([0, 100], [1 - q, q], 0.99)[0]
        v2 = tail_measures([0, 100, 200], [(1 - q) ** 2, 2 * q * (1 - q), q * q], 0.99)[0]
        assert v2 <= 2 * v1
    one = tail_measures([0, 100], [0.991, 0.009], 0.99)
    two = tail_measures([0, 100, 200], [0.991 ** 2, 2 * 0.009 * 0.991, 0.009 ** 2], 0.99)
    assert one == (0.0, pytest.approx(RAW["saEsOne"])) and two[0] == 100 and two[1] == pytest.approx(RAW["saEsTwo"])


def test_cornish_fisher_turning_point(RAW):
    r, _ = L.load_daily(); _, _, s, k = L.moments(r.values)
    # d z~/dz = 1 + z s/3 + (z^2 - 1) k/8 - (6 z^2 - 5) s^2/36: its upper root gives the last level
    # at which the Cornish-Fisher VaR still falls as the level rises
    A, B, C = k / 8 - s * s / 6, s / 3, 1 - k / 8 + 5 * s * s / 36
    roots = np.sort(np.roots([A, B, C]).real)
    a_turn = 1 - stats.norm.cdf(roots[0])
    assert abs(100 * a_turn - RAW["cfDownHi"]) < 0.6
    close(-L.cf_z(stats.norm.ppf(0.001), s, k), -RAW["cfZnine"], 1e-12)


def test_cluster_and_qq_numbers(RAW):
    assert RAW["acSqAbove"] == RAW["acLags"] == RAW["acAbsAbove"] == 100
    nu = RAW["gaNu"]; sc = math.sqrt((nu - 2) / nu)
    close(stats.t(nu, scale=sc).var(), 1.0, 1e-12)                          # the reference has variance one
    close(RAW["gaTQ"], sc * RAW["gaTQRaw"], 1e-12)
    assert RAW["gaResQThou"] < RAW["gaTQThou"] < RAW["gaNQThou"]
    assert -RAW["gaNQThou"] < RAW["gaResQTop"] < RAW["gaTQTop"]          # right tail lighter than the t


def test_threshold_and_ratio_numbers(RAW):
    assert RAW["evXiMin"] < RAW["evXi"] < RAW["evXiMax"] and RAW["evXiMaxQ"] > 95
    close(RAW["prYearsCheck"], 1.0, 1e-12)
    th = RAW["prSRm"]; close(RAW["prSR"], math.sqrt(12) * th, 2e-3)
    assert RAW["ddWinAboveNinety"] <= RAW["ddWinN"]


def test_crisis_numbers(RAW, NUM):
    v = pd.read_csv(HERE / "out" / "case_var.csv", index_col=0, parse_dates=True)
    h = pd.read_csv(HERE / "out" / "case_hits.csv", index_col=0, parse_dates=True)
    w = h.loc["2007-01-01":"2009-12-31"]
    assert (RAW["crHs"], RAW["crFhs"], RAW["crDays"]) == (w.hs.sum(), w.fhs.sum(), len(w))
    assert RAW["crDropLoss"] == RAW["crHsMax"]                      # the day that left set the peak
    i = v.index.get_loc(pd.Timestamp(NUM["crHsDrop"]))
    win = v.loss.iloc[i - 251:i - 1]                                   # the window of the day before the drop
    assert -np.sort(-win.values)[2] == v.hs.iloc[i - 1]               # third largest of 250 at 99%


def test_listing_input_checks():
    for bad in ([], [0.01, float("nan")]):
        with pytest.raises(ValueError):
            hist_var_es(bad, 0.99)
    for a in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            hist_var_es([-0.01, 0.02], a)
    with pytest.raises(ValueError):
        hist_var_es([-0.01, 0.02], 1 - 1e-12)          # no observation in the tail
    for bad in ([], [0, 2, 1], [0.5, 1]):
        with pytest.raises(ValueError):
            lr_tests.kupiec(bad, 0.01)
    with pytest.raises(ValueError):
        lr_tests.christoffersen([0, 1, 2])


def test_round3_printed_values(RAW, NUM):
    # 8.034: the printed shape and standard error reproduce the printed tail index and its band
    xi, se = float(NUM["evXi"]), float(NUM["evXiSe"])
    assert f"{1 / xi:.2f}" == NUM["evAlphaTail"]
    assert f"{1 / (xi + 2 * se):.1f}" == NUM["evAlphaLo"] and f"{1 / (xi - 2 * se):.1f}" == NUM["evAlphaHi"]
    # 8.036: the 159 exceedances and the standard error of 0.10 belong to the 99th percentile
    loss = -L.load_daily()[0].values
    nu = int(np.sum(loss > np.quantile(loss, 0.99)))
    assert str(nu) == NUM["evNuTop"] and NUM["evXiMaxQ"] != "99"
    # 8.019: the loss quantile is minus the return quantile
    close(RAW["cfLossNine"], -RAW["cfZnine"], 1e-12)
    # 8.077: the Sharpe ratios reproduce from the printed excess return volatilities
    f = L.load_monthly(); rf = f["rf"].values; r = f["mkt_rf"].values + rf
    for x, tg in ((r, "T"), (L.smooth(r, 0.6), "S")):
        ex = x - rf
        close(12 * ex.mean() / (RAW[f"smExVol{tg}"] / 100), RAW[f"smSR{tg}"], 1e-12)   # pct holds percent
        assert f"{float(NUM[f'smExMean{tg}']) / float(NUM[f'smExVol{tg}']):.2f}" == NUM[f"smSR{tg}"]


def test_round4_printed_values(RAW, NUM):
    """The rescaled t quantile at 0.1% follows from the unrounded nu printed beside it, and
    the smallest Z2 is about twice the largest reference critical value."""
    nu = float(NUM["gaNuFine"])
    q = math.sqrt((nu - 2) / nu) * stats.t.ppf(0.001, nu)
    assert f"{q:.2f}" == NUM["gaTQThou"].replace("\\ensuremath{-}", "-")
    crit = max(float(NUM[k]) for k in ("caCritNor", "caCritT", "caCritEmp"))
    assert 1.8 < float(NUM["caFhsEs"]) / crit < 2.2
    src = (HERE / "case_backtest.py").read_text()
    assert "sn = math.sqrt(hn[t])" in src            # filtered HS scales by the volatility


def test_quantile_standard_error(RAW):
    """The standard error quoted for the simulated 95th percentile of Z2 is the asymptotic one,
    sqrt(p(1-p)/n) over the normal density at the percentile."""
    from scipy import stats
    n = 20_000; dens = stats.norm.pdf(stats.norm.ppf(0.95)) / RAW["ztSd"]
    assert abs(RAW["ztCritSeAn"] - math.sqrt(0.05 * 0.95 / n) / dens) < 1e-12
