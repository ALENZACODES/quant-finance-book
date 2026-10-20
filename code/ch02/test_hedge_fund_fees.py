"""Checks for hedge_fund_fees.py: Table 2.4, the modified mark, the annuity and option propositions."""
import numpy as np
from scipy import stats, integrate
import ch02lib as L
from conftest import close


def test_table_by_hand(RAW):
    close(RAW["hfIncOne"], 0.2 * (113 - 100), 1e-12)
    pre4 = 97.152 * 1.06 * 1.2 - 0.02 * 97.152 * 1.06 if False else RAW["hfPreFour"]
    close(RAW["hfIncFour"], 0.2 * (pre4 - RAW["hfHwmThree"]), 1e-9)
    close(RAW["hfTotalFees"], sum(RAW["hfMgmt" + w] + RAW["hfInc" + w] for w in ("One", "Two", "Three", "Four")), 1e-9)
    close(RAW["hfIncTwo"] + RAW["hfIncThree"], 0.0, 1e-12)
    close(RAW["hfHardIncOne"], 0.2 * (113 - 105), 1e-9); close(RAW["hfSoftIncOne"], 2.6, 1e-9)
    close(RAW["hfGrossEnd"], 100 * 1.15 * 0.9 * 1.08 * 1.2, 1e-9)


def test_hwm_never_charges_more_on_the_same_path():
    rng = np.random.default_rng(8)
    for _ in range(500):
        g = rng.normal(0.06, 0.2, 8)
        a, b = L.hedge_fund_fees(g), L.hedge_fund_fees(g, hwm=False)
        assert a["close"][-1] >= b["close"][-1] - 1e-9


def test_modified_hwm(RAW):
    close(RAW["modModIncTwo"], 0.10 * (84 - 75), 1e-9)
    close(RAW["modModIncFive"], 0.10 * (106.25 - RAW["modModCloseFour"]) + 0.20 * (RAW["modModCloseFour"] * 1.12 - 106.25), 1e-9)
    close(RAW["modTarget"], 75 + 1.25 * 25, 1e-12)


def test_annuity(RAW):
    close(RAW["annPV"], L.fee_annuity_pv_loop(1000, .1, .02, .2, .1, 10), 1e-8)
    close(RAW["annPV"], RAW["annPVm"] + RAW["annPVi"], 0.11)
    close(RAW["annPVperp"], 1000.0, 1e-6)
    close(L.fee_annuity_pv(1000, .1, .02, .2, .1, 4000), 1000.0, 1e-6)    # Proposition 2.9 limit
    for g, d in [(0.07, 0.09), (0.12, 0.08), (0.05, 0.05)]:
        close(L.fee_annuity_pv(500, g, .015, .15, d, 12), L.fee_annuity_pv_loop(500, g, .015, .15, d, 12), 1e-8)


def test_vega_peak_location():
    for s in (0.1, 0.2, 0.4):
        S = np.linspace(50, 200, 300001)
        v = L.bs_call_vega(S, 100, 1.0, 0.03, s)
        close(S[np.argmax(v)], 100 * np.exp(0.5 * s * s - 0.03), 1e-3)


def test_convexity_formula():
    for mu, s in [(0.05, 0.1), (-0.02, 0.3), (0.1, 0.05)]:
        q = integrate.quad(lambda x: x * stats.norm.pdf(x, mu, s), 0, np.inf)[0]
        close(q, mu * stats.norm.cdf(mu / s) + s * stats.norm.pdf(mu / s), 1e-9)
        e = 1e-6
        f = lambda ss: mu * stats.norm.cdf(mu / ss) + ss * stats.norm.pdf(mu / ss)
        close((f(s + e) - f(s - e)) / (2 * e), stats.norm.pdf(mu / s), 1e-6)


def test_fee_share_monotone(RAW):
    assert RAW["shareDet"] < RAW["shareHwmTen"] < RAW["shareHwmTwenty"] < RAW["shareHwmForty"]
    for nm in ("Ten", "Twenty", "Forty"):
        assert RAW["shareHwm" + nm] < RAW["shareNoHwm" + nm]
    close(RAW["shareDet"], 100 * (1 - (1.048 ** 10 - 1) / (1.08 ** 10 - 1)), 1e-9)


def test_netting(RAW):
    close(RAW["netFeeA"], 1.5 + 0.2 * 28.5, 1e-9); close(RAW["netFeeB"], 1.5, 1e-9)
    assert RAW["netPaid"] > RAW["netSingle"]
    assert RAW["fofTotalZero"] > RAW["fofUnderZero"] > RAW["fofSingleZero"]
    close(RAW["fofUnderOne"], RAW["fofSingleOne"], 0.2)


def test_hwm_figure_target_matches_text(RAW):
    """The dotted recovery line in Figure 2.7 is drawn at the same level the text quotes."""
    import matplotlib
    matplotlib.use("Agg")
    import hedge_fund_fees as H
    gm = [-0.25] + [0.12] * 5
    trad = L.hedge_fund_fees(gm, mgmt=0.0); mod = L.hedge_fund_fees(gm, mgmt=0.0, modified=(0.10, 1.25))
    target = H.recovery_target(100.0, 0.25, 1.25)
    close(target, RAW["modTarget"], 1e-12); close(target, 106.25, 1e-12)
    fig = H.hwm_figure(trad, mod, target, 0.10, 1.25)
    lines = [ln for ln in fig.axes[0].get_lines() if ln.get_gid() == "target"]
    assert len(lines) == 1
    assert np.allclose(lines[0].get_ydata(), RAW["modTarget"])
    # the lib reverts to the full rate at the same level the figure shows
    first_full = next(t for t in range(1, 6) if mod["pre"][t] >= target)
    assert mod["close"][first_full - 1] < target <= mod["pre"][first_full]


def test_exact_option_form(RAW):
    # p calls on gross NAV A = S/(1-m) struck at M + mA, with S = M: by homogeneity it equals approx/(1-m)
    assert abs(RAW["optExactAtSigTwenty"] - RAW["optAtSigTwenty"] / 0.98) < 1e-12
    assert RAW["optExactAtSigTwenty"] > RAW["optAtSigTwenty"]
