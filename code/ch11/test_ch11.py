"""Tests of the Chapter 11 library and of the numbers quoted in the text."""
import math, runpy, subprocess, sys
import numpy as np
import pytest
from conftest import close, HERE
import ch11lib as L


def test_waterfall_gives_carry_share_after_catch_up():
    rng = np.random.default_rng(31)
    for _ in range(200):
        n = 10; C = np.zeros(n); C[:4] = rng.uniform(5, 30, 4)
        X = np.zeros(n); X[3:] = rng.uniform(0, 40, n - 3)
        w = L.european_waterfall(np.arange(n), C, X, 0.08, 0.20, 1.0)
        tiers = sum(w[k].sum() for k in ("roc", "pref", "catch", "lp_split", "gp_split"))
        close(tiers, X.sum(), 1e-9)                                   # every dollar is allocated
        close(w["gp_total"] + w["lp_total"], X.sum(), 1e-9)
        if w["lp_split"].sum() > 0:                                    # past the catch-up: exactly the carry share
            close(w["gp_total"], 0.20 * (X.sum() - C.sum()), 1e-9)
        assert w["gp_total"] <= 0.20 * max(X.sum() - C.sum(), 0) + 1e-9


def test_catch_up_formula():
    w = L.european_waterfall([0, 6], [100, 0], [0, 200], 0.08, 0.2, 1.0)
    P = 100 * 1.08 ** 6 - 100
    close(w["catch"][1], 0.2 * P / 0.8, 1e-9)
    w = L.european_waterfall([0, 6], [100, 0], [0, 200], 0.08, 0.2, 0.8)     # partial catch-up rate
    y = (0.2 * P) / (0.8 - 0.2); close(w["catch"][1], 0.8 * y, 1e-9)


def test_deal_by_deal_accrues_ahead_of_whole_fund():
    rng = np.random.default_rng(7)
    for _ in range(100):
        K = np.full(6, 10.0); si = np.sort(rng.integers(0, 4, 6)).astype(float); so = si + rng.integers(1, 6, 6)
        P = K * rng.lognormal(0.1, 0.6, 6)
        d, _, A, final = L.deal_carry_paths(K, si, P, so)
        # whole fund carry accrued to each date, ending at the final whole fund carry
        E = np.array([0.2 * max(P[so <= x].sum() - K[si <= x].sum(), 0.0) for x in d])
        assert np.all(A[:-1] >= E[:-1] - 1e-12)
        close(A[-1], final, 1e-12); close(E[-1], final, 1e-12)
        assert L.pv_of_path(d, A, 0.08) >= L.pv_of_path(d, E, 0.08) - 1e-12


def test_pme_and_alpha_agree_in_sign():
    rng = np.random.default_rng(11)
    t = np.arange(8.0); I = dict(zip(t, 100 * np.exp(np.cumsum(np.r_[0, rng.normal(0.05, 0.15, 7)]))))
    for _ in range(300):
        C = np.r_[rng.uniform(10, 40, 3), np.zeros(5)]; D = np.r_[np.zeros(3), rng.uniform(0, 60, 5)]
        pme = L.ks_pme(t, C, D, 5.0, I.get, 7.0); a = L.direct_alpha(t, C, D, 5.0, I.get, 7.0)
        assert (pme > 1) == (a > 0)


def test_snippet_matches_library(RAW):
    out = subprocess.run([sys.executable, str(HERE / "snippets" / "pme_alpha.py")], capture_output=True, text=True, check=True).stdout
    assert f"KS-PME {RAW['pkSnipPme']:.4f}" in out
    assert f"direct alpha {RAW['pkSnipAlpha']:.2f}%" in out


def test_duration_bounds():
    for irr in (0.03, 0.1, 0.25):
        t = np.array([0.0, 1.0, 2.0, 5.0, 7.0]); c = np.array([-50, -50, 0, 0, 0.0])
        x = np.array([0, 0, 40, 60, 0.0]); x[4] = -sum((c + x)[:4] * (1 + irr) ** (-t[:4])) * (1 + irr) ** 7
        close(L.irr(t, c + x), irr, 1e-10)
        tvpi = x.sum() / -c.sum(); dur = math.log(tvpi) / math.log1p(irr)
        lo, hi = L.duration_bounds([0, 1], [2, 5, 7]); assert lo - 1e-12 <= dur <= hi + 1e-12


def test_beta_pme_formula_by_simulation():
    rng = np.random.default_rng(5); T, mu, s = 5, 0.07, 0.16
    x = rng.normal((mu - s * s / 2) * T, s * math.sqrt(T), 400_000)
    for b in (0.5, 1.5, 2.0):
        for r in (0.0, 0.04):
            sim = L.levered_index_payoff(b, x, r, s, T) * np.exp(-x); se = sim.std() / math.sqrt(len(x))
            assert abs(sim.mean() - L.pme_beta_mean(0, b, mu, s, T, r)) < 4 * se


def test_levered_index_payoff_is_the_rebalanced_portfolio():
    """Discrete daily rebalancing of a position beta financed at r converges to the closed form."""
    rng = np.random.default_rng(8); b, r, mu, s, T, n = 1.5, 0.04, 0.07, 0.16, 1.0, 20_000
    dt = T / n; z = rng.standard_normal(n)
    lr = (mu - s * s / 2) * dt + s * math.sqrt(dt) * z
    v = np.prod(1 + b * np.expm1(lr) + (1 - b) * r * dt)
    assert abs(v / float(L.levered_index_payoff(b, lr.sum(), r, s, T)) - 1) < 2e-3


def test_leverage_proposition_against_schedule(RAW):
    close(L.lbo_moic(RAW["lxUb"], RAW["lxLamB"] / 100, 0.08, 0.25, 5), RAW["lxMoic"], 1e-9)
    close(RAW["lxMcBeat"], RAW["lxMcUAbove"], 1e-12); assert RAW["lxMcMismatch"] == 0
    k = 1.06 ** 5
    for U in (0.8, 1.2, 1.5, 2.5):
        for lam in (0.2, 0.5, 0.7):
            m = L.lbo_moic(U, lam, 0.08, 0.25, 5)
            assert (m > U) == (U > k); close(m - U, lam * (U - k) / (1 - lam), 1e-12)


def test_bridge_adds_up(RAW):
    close(RAW["lxBrG"] + RAW["lxBrM"] + RAW["lxBrD"] - RAW["lxBrF"], RAW["lxGain"], 1e-9)
    close(0.5 * (RAW["lxGfirst"] + RAW["lxMfirst"]), RAW["lxBrG"], 1e-9)


def test_value_driver_derivative():
    W, roc = 0.08, 0.12; f = lambda g: L.ev_nopat_multiple(W, g, roc); h = 1e-6
    for g in (0.0, 0.02, 0.04):
        fd = (f(g + h) - f(g - h)) / (2 * h); close(fd, (1 - W / roc) / (W - g) ** 2, 1e-5)
    close(L.ev_nopat_multiple(W, 0.03, W), 1 / W, 1e-12)
    assert L.ev_nopat_multiple(0.09, 0.04, 0.07) < L.ev_nopat_multiple(0.09, 0.02, 0.07)


def test_miles_ezzell_and_apv(RAW):
    rU, rD, tau, Lv = 0.0854, 0.0674, 0.25, 0.3
    W = L.miles_ezzell_wacc(rU, rD, tau, Lv)
    fcf = [10, 11, 12, 12.5, 13]; v_w, _ = L.dcf_value(fcf, W, 0.02, nopat_next=16, roc=0.1)
    close(L.apv_rebalanced(fcf, 16 * (1 - 0.02 / 0.1), 0.02, rU, rD, tau, Lv), v_w, 1e-9)
    re_ = L.lever_cost_equity(rU, rD, Lv / (1 - Lv), tau, "miles_ezzell")
    close((1 - Lv) * re_ + Lv * rD * (1 - tau), W, 1e-14)
    close(RAW["vlApvGap"], 0.0, 1e-9)


def test_beta_rules_round_trip():
    for rule in ("hamada", "harris_pringle"):
        bu = L.unlever_beta(1.2, 0.4, 0.25, rule=rule); close(L.relever_beta(bu, 0.4, 0.25, rule=rule), 1.2, 1e-14)


def test_longstaff_bound_against_simulation(RAW):
    assert RAW["dmMcMaxZ"] < 4
    close(L.longstaff_bound(1, 0.6), L.longstaff_bound(4, 0.3), 1e-14)
    assert L.longstaff_bound(2, 0.45) > L.chaffe_dlom(2, 0.45)
    # the bound grows like v/2
    assert abs(L.longstaff_bound(400, 1.0) / 200 - 1) < 0.05


def test_put_parity_and_combination():
    p = L.chaffe_dlom(2, 0.45, r=0.05); c = p + 1 - math.exp(-0.1)
    d1 = (0.05 + 0.45 ** 2 / 2) * math.sqrt(2) / 0.45; d2 = d1 - 0.45 * math.sqrt(2)
    from scipy.stats import norm
    close(c, norm.cdf(d1) - math.exp(-0.1) * norm.cdf(d2), 1e-12)
    close(L.combined_discount(0.2, 0.25), 0.4, 1e-14)


def test_text_numbers(RAW):
    close(RAW["wfGpShare"], 20.0, 1e-9); close(RAW["wfClaw"], 0.0, 1e-12)
    assert RAW["cvDurOver"] == 0 and RAW["cvDurRatioMax"] < 100
    assert RAW["vlTauD"] < RAW["vlShield"] and RAW["vlCapEarn"] > RAW["vlEv"]
    assert RAW["dmComb"] < RAW["dmSum"]
    assert RAW["pkAgree"] == 1.0 and RAW["lxMcWipe"] == 0.0


def test_round2_printed_values(RAW):
    import math
    # the beta boost of the simulated deals, (beta - 1)(mu - sigma^2) with r = 0
    assert abs(RAW["pkBetaBoost"] - 100 * 0.2 * (0.07 - 0.16 ** 2)) < 1e-9
    # with a beta of one the same funds do worse against the index
    assert RAW["pkMedBetaOne"] < RAW["pkMed"] and RAW["pkAboveBetaOne"] < RAW["pkAbove"]
    # the fake alpha is half the gap between the premium and the variance
    assert abs(RAW["pkFakeAlpha"] - 0.5 * (RAW["pkPrem"] - RAW["pkVar"])) < 1e-6
    # tornado: a lower cost of capital raises the value, and the terminal growth moves are asymmetric
    assert RAW["vlToWaccUp"] > 0 > RAW["vlToWaccDn"] and RAW["vlToGrowthUp"] > -RAW["vlToGrowthDn"]
    # the implied growth solves the trailing value driver at the unrounded medians
    w, r, g = RAW["mkWmed"] / 100, RAW["mkRocMed"] / 100, RAW["mkGstar"] / 100
    assert abs((1 + g) * (1 - g / r) / (w - g) - RAW["mkObsMed"]) < 1e-6
    assert abs(RAW["dmLongOverPut"] - RAW["dmLong"] / RAW["dmChaffe"]) < 1e-9


def test_finnerty_dlom():
    # below the share price at any variance, rising in the variance, near the simulated average-strike put
    vals = [L.finnerty_dlom(T, 0.4) for T in (0.25, 1, 4, 16, 64)]
    assert all(0 < a < b < 1 for a, b in zip(vals, vals[1:]))
    rng = np.random.default_rng(11)
    m, se = L.average_strike_mc(1.0, 0.3, 100_000, 200, rng)
    assert abs(L.finnerty_dlom(1.0, 0.3) - m) < 0.02 * m + 4 * se
    assert abs(L.finnerty_dlom(1.0, 0.3, q=0.02) - math.exp(-0.02) * L.finnerty_dlom(1.0, 0.3)) < 1e-15


def test_interest_cap():
    import lbo
    base = lbo.deal()
    p = dict(lbo.BASE); p["int_cap"] = 10.0          # a cap that never binds changes nothing
    loose = lbo.deal(p)
    assert abs(loose["moic"] - base["moic"]) < 1e-12
    p["int_cap"] = 0.0; p["carry_forward"] = False   # no deduction at all: the same as a zero tax shield
    none = lbo.deal(p)
    assert all(abs(r["tax"] - max(r["ebit"], 0) * p["tax"]) < 1e-12 for r in none["rows"])
    assert none["moic"] < base["moic"]
