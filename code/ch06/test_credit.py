"""Merton, hazard rates, the copula, private credit and the case study."""
import math
import numpy as np
from scipy import stats
from conftest import close, HERE
import ch06lib as L


def test_merton_parity_and_limit():
    m = L.merton(100, 70, 0.03, 0.25, 5)
    close(m["E"] + m["debt"], 100, 1e-12)
    close(float(L.merton_spread_lev(70 * math.exp(-0.15) / 100, 0.25, 5)), m["spread"], 1e-12)
    s = [float(L.merton_spread_lev(0.7, 0.25, T)) for T in (0.3, 0.2, 0.1)]
    assert s[0] > s[1] > s[2] > 0
    assert abs(float(L.merton_spread_lev(0.7, 0.25, 0.01))) < 1e-12


def test_merton_calibration(RAW):
    assert abs(RAW["mcResE"]) < 1e-8 and abs(RAW["mcResVol"]) < 1e-8


def test_credit_triangle_exact_with_continuous_premium():
    lam, R, r = 0.02, 0.4, 0.03
    prem = lambda T: (1 - math.exp(-(r + lam) * T)) / (r + lam)
    prot = lambda T: (1 - R) * lam * prem(T)
    close(prot(5) / prem(5), lam * (1 - R), 1e-15)


def test_bootstrap_reprices(RAW):
    assert RAW["hzReprice"] < 1e-12
    assert abs(RAW["hzMcPdZ"]) < 4 and abs(RAW["hzMcProtZ"]) < 4


def test_default_time_law():
    knots, lams = [1, 3, np.inf], [0.01, 0.03, 0.05]
    tau = L.default_times(400000, knots, lams, np.random.default_rng(1))
    for t in (0.5, 2, 6):
        p = 1 - float(L.survival(t, knots, lams)); se = math.sqrt(p * (1 - p) / 400000)
        assert abs(np.mean(tau <= t) - p) < 4 * se


def test_tranches_add_up(RAW):
    el = 0.6 * RAW["hzPdFive"] / 100
    for k in ("One", "Three", "Six"):
        close(RAW[f"trSum{k}"], el, 1e-12)


def test_large_pool_tranches_add_up():
    """Tranches that tile [0, 1] carry the whole expected loss, including far-tail senior pieces."""
    tr = [(0.0, 0.03), (0.03, 0.07), (0.07, 0.10), (0.10, 0.15), (0.15, 0.30), (0.30, 1.0)]
    for p, rho in ((0.0575, 0.1), (0.0575, 0.3), (0.02, 0.05), (0.1, 0.8)):
        tot = sum((d - a) * L.tranche_el_lhp(p, rho, 0.4, a, d) for a, d in tr)
        close(tot, 0.6 * p, 1e-9)


def test_snippet(RAW):
    import subprocess, sys
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "lhp_tranche.py")], capture_output=True,
                         text=True, check=True).stdout.splitlines()
    vals = [float(l.split(":")[1].strip().rstrip("%")) for l in out]
    p = RAW["hzPdFive"] / 100
    assert round(p, 5) == 0.05753     # the value the listing and the text print
    names = ["EqThree", "MzAThree", "MzBThree", "SnAThree", "SnBThree", "SupThree"]
    for (a, d), v, k in zip([(0, .03), (.03, .07), (.07, .10), (.10, .15), (.15, .30), (.30, 1)], vals, names):
        close(v, 100 * L.tranche_el_lhp(round(p, 5), 0.3, 0.4, a, d), 1e-3)
        # 6.099: the listing reproduces the limit column of the table at printed precision
        assert abs(v - RAW[f"lh{k}"]) < 0.006, (k, v, RAW[f"lh{k}"])


def test_pool_pmf_sums_to_one_and_mean():
    for rho in (0.0, 0.2, 0.7):
        pmf = L.pool_pmf(50, 0.05, rho)
        close(pmf.sum(), 1, 1e-10); close(np.sum(np.arange(51) * pmf), 2.5, 1e-9)


def test_large_pool_limit():
    p, rho = 0.05, 0.3
    for x in (0.02, 0.05, 0.1):
        cdf_n = np.cumsum(L.pool_pmf(2000, p, rho))[int(x * 2000)]
        assert abs(cdf_n - float(L.lhp_cdf(x, p, rho))) < 0.01


def test_copula_simulation(RAW):
    assert RAW["cpMcMaxZ"] < 4


def test_screener_identities(RAW):
    for k in ("Cal", "Tor", "Ott"):
        assert abs(RAW[f"pc{k}IdOne"]) < 1e-12 and abs(RAW[f"pc{k}IdTwo"]) < 1e-12


def test_slice_el_is_a_put_spread():
    rng = np.random.default_rng(3); n = 400000
    V0, g, sig, T, c = 100.0, 0.02, 0.15, 5, 0.1
    VT = V0 * (1 + g) ** T * np.exp(-0.5 * sig ** 2 * T + sig * math.sqrt(T) * rng.standard_normal(n))
    lo, hi = 54.0, 67.0
    rec = (1 - c) * VT
    loss = np.minimum(np.maximum(hi - rec, 0), hi - lo)
    x = L.slice_el(lo, hi, V0, g, sig, T, c)
    assert abs(loss.mean() - x) < 4 * loss.std() / math.sqrt(n)


def test_refinance_reduces_to_threshold(RAW):
    """With income and value both proportional to the shock, failure is a threshold on the shock."""
    assert abs(RAW["csMcElZ"]) < 4 and RAW["csChi"] < 0.005
    close(RAW["csMcMeanFail"], RAW["csMeanFail"], 4 * RAW["csMcMeanFailSE"])
    assert RAW["csFails"] == 0 and RAW["csCoverMin"] > 1


def test_exercise_answers(RAW):
    assert abs(RAW["xsOneQd"] - RAW["xsOneEx"]) < abs(RAW["xsOneDur"] - RAW["xsOneEx"]) / 10
    close(RAW["xsTwoD"], RAW["xsTwoNum"], 1e-12)
    close(RAW["xsThreeA"] * math.exp(-0.04) + RAW["xsThreeB"] * math.exp(-0.4), RAW["xsThreePV"], 1e-9)
    assert abs(RAW["xsSevenCheck"]) < 1e-10
    close(RAW["xsEightLtv"] * RAW["xsEightDy"] / 100, 6.0, 1e-12)


def test_merton_peak_is_refined(RAW):
    """the printed peak is a true maximum, not the nearest grid point."""
    for nm, lv in (("Seven", 0.7), ("Nine", 0.9)):
        t = RAW[f"mtPeakT{nm}"]; f = lambda x: float(L.merton_spread_lev(lv, 0.25, x))
        assert f(t) >= f(t - 1e-3) and f(t) >= f(t + 1e-3)
        assert abs(t - RAW[f"mtPeakTGrid{nm}"]) <= 0.03
    assert f"{RAW['mtPeakTSeven']:.1f}" == "6.4"


def test_screen_inputs_are_printed(RAW):
    """the score depends on the sponsor rating, which the text now states."""
    assert RAW["csSponsor"] == 3


def test_printed_gaps_reproduce(RAW):
    """the standard-error gaps in the text follow from the values as printed."""
    m, se, ex = round(RAW["hzMcPd"], 3), round(RAW["hzMcPdSE"], 3), round(RAW["hzPdFiveX"], 3)
    assert f"{abs(m - ex) / se:.1f}" == f"{abs(RAW['hzMcPdZ']):.1f}"
    m, se, ex = round(RAW["csMcEl"], 3), round(RAW["csMcElSE"], 3), round(RAW["csEl"], 3)
    assert f"{abs(m - ex) / se:.1f}" == f"{abs(RAW['csMcElZ']):.1f}"


def test_senior_claim(RAW):
    """the senior claim's loss is a small part of the pool's, its two units agree, and its error is small."""
    assert 0 < RAW["csSenEl"] < RAW["csMcEl"]
    assert abs(RAW["csSenElOwn"] - RAW["csSenEl"] / 0.99) < 1e-12
    assert RAW["csSenElSE"] < 0.02 and RAW["csSenScen"] >= 2_000_000
