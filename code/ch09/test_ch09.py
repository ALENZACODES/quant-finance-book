"""Tests of the Chapter 9 library and of the numbers quoted in the text."""
import math
import numpy as np
import pandas as pd
from scipy import integrate, stats
import ch09lib as L
from conftest import close


def test_par_bond_at_unchanged_yield_earns_its_coupon():
    for y in (0.01, 0.04, 0.07):
        close(float(L.bond_return(y, y)), y / 12, 1e-12)
    assert L.bond_return(0.04, 0.05) < 0 < L.bond_return(0.05, 0.04)


def test_month_end_takes_last_value():
    s = pd.Series([1.0, 2.0, np.nan, 3.0], index=pd.to_datetime(["2020-01-02", "2020-01-30", "2020-01-31", "2020-02-03"]))
    m = L.month_end(s)
    assert list(m.index) == [202001, 202002] and list(m.values) == [2.0, 3.0]


def test_straddle_identity_is_exact():
    rng = np.random.default_rng([9473, 99])
    x = 0.01 * rng.standard_normal(500); S = np.cumsum(x)
    pnl = np.sum(np.concatenate([[0], S[:-1]]) * x)
    close(pnl, 0.5 * (S[-1] ** 2 - np.sum(x * x)), 1e-14)


def test_sign_rule_formula_by_simulation():
    rng = np.random.default_rng([9473, 98]); rho, n = 0.3, 400000
    X = rng.standard_normal(n); Y = 2.0 * (rho * X + math.sqrt(1 - rho * rho) * rng.standard_normal(n))
    close(np.mean(np.sign(X) * Y), L.sign_rule_mean(rho, 2.0), 0.01)


def test_trend_rule_on_a_martingale_earns_nothing():
    rng = np.random.default_rng([9473, 97])
    r = pd.DataFrame(0.03 * rng.standard_normal((3000, 4)))
    sl, p = L.tsmom_returns(r, L.tsmom_weights(r), cost=0.0)
    assert abs(p.mean()) < 3 * p.std() / math.sqrt(len(p))


def test_combined_sharpe():
    close(L.combined_sharpe(0.2, 1, 0.5), 0.2, 1e-12)
    close(L.combined_sharpe(0.2, 10 ** 8, 0.25), 0.4, 1e-6)


def test_fees():
    x = np.full(24, 0.01); rf = np.zeros(24); m = np.array([200100 + k % 12 + 1 + 100 * (k // 12) for k in range(24)])
    close(L.fees_hwm(x, rf, 0.0, 0.0, m).sum(), x.sum(), 1e-12)
    net = L.fees_hwm(-x, rf, 0.0, 0.2, m)
    np.testing.assert_allclose(net, -x, atol=1e-12)        # no incentive fee below the high-water mark
    assert L.fees_hwm(x, rf, 0.02, 0.2, m).mean() < x.mean()


def test_survivor_mean_by_integration():
    mu, s, T, c = 0.005, 0.04, 24, 0.001
    se = s / math.sqrt(T)
    num = integrate.quad(lambda m: m * stats.norm.pdf(m, mu, se), c, 1)[0]
    close(L.survivor_mean(mu, s, T, c), num / stats.norm.sf(c, mu, se), 1e-10)


def test_call_mean_by_integration():
    close(L.call_mean(0.08, 0.15), integrate.quad(lambda x: x * stats.norm.pdf(x, 0.08, 0.15), 0, 3)[0], 1e-10)


def test_weight_for_share_inverts_risk_share():
    for a, b, c in ((0.01, 0.5, 0.02), (0.01, 0.5, -0.01), (0.02, 0.1, 0.0)):
        for s in (0.05, 0.3, 0.7):
            close(L.risk_share(L.weight_for_share(s, a, b, c), a, b, c), s, 1e-12)


def test_netting_inequality():
    rng = np.random.default_rng([9473, 96]); G = rng.normal(0.05, 0.2, (1000, 7))
    assert np.all(np.maximum(G, 0).sum(axis=1) >= np.maximum(G.sum(axis=1), 0) - 1e-15)


def test_numbers_in_the_text(RAW, NUM):
    assert NUM["tfFirst"] == "January 2001" and NUM["tfLast"] == "August 2026" and NUM["btLast"] == "September 2026"
    assert RAW["tfMonths"] == 308
    close(RAW["tfSRpred"], RAW["tfSR"], 0.05)               # the diversification formula against the program
    assert RAW["pmP"] <= 0.06
    assert RAW["slQuad"] > 0 and RAW["slQuadT"] > 2
    assert RAW["svMaxGap"] < 1.0                                 # basis points a month
    assert RAW["dbLive"] > RAW["dbBack"] > RAW["dbClean"] > RAW["dbTruth"] - 0.3
    assert abs(RAW["pvSim"] - RAW["pvTheory"]) < 3 * RAW["pvSimSe"]
    assert RAW["rpCorrB"] > RAW["rpCorrA"] and RAW["rpOutB"] > 0 > RAW["rpOutA"]
    close(RAW["ffSimPool"], RAW["ffPoolTen"], 0.02)
    close(RAW["ffSimSep"], RAW["ffSepTen"], 0.02)
    close(RAW["stLossShare"], RAW["stLossTheory"], 3.0)
    assert RAW["btCorrLate"] > RAW["btCorrEarly"]


def test_data_end_dates():
    root = L.ROOT / "data"
    for f in ("fred_markets_daily.csv", "crypto_daily.csv"):
        last = pd.read_csv(root / f, usecols=[0]).iloc[-1, 0]
        assert last <= "2026-09-30", (f, last)


def test_soft_hurdle_identity_and_means():
    R = np.linspace(-0.2, 0.4, 2001)
    soft = L.hurdle_fee(R, 0.2, 0.05, "soft")
    close(np.max(np.abs(soft - (np.maximum(R - 0.05, 0) - 0.8 * np.maximum(R - 0.05 / 0.8, 0)))), 0.0, 1e-15)
    assert np.all(L.hurdle_fee(R, 0.2, 0.05, "hard") <= soft + 1e-15) and np.all(soft <= L.hurdle_fee(R, 0.2, 0.05, "none") + 1e-15)
    for kind in ("none", "hard", "soft"):
        num = integrate.quad(lambda x: float(L.hurdle_fee(x, 0.2, 0.05, kind)) * stats.norm.pdf(x, 0.08, 0.12), -1, 1.5,
                             points=[0, 0.05, 0.0625], limit=200)[0]
        close(L.hurdle_fee_mean(0.08, 0.12, 0.2, 0.05, kind), num, 1e-9)


def test_high_water_mark_identity_on_random_paths():
    rng = np.random.default_rng([9473, 98])
    for _ in range(200):
        g = 0.2 * rng.standard_normal(12)
        fees, nav, mark = L.incentive_path(g, 0.2, hwm=True)
        close(fees.sum(), 0.25 * (mark - 1), 1e-12)


def test_fee_option_at_the_money_forward():
    for s in (0.05, 0.1, 0.3):
        close(L.fee_option(0.2, math.exp(0.03), s, 0.03, 1.0), 0.2 * (2 * stats.norm.cdf(s / 2) - 1), 1e-12)


def test_closure_formula_against_integration_with_drift():
    mu, s, b, T = 0.03, 0.2, 0.5, 4.0
    k = math.exp(-2 * mu * b / s ** 2); v = s * math.sqrt(T)
    dens = lambda x: (stats.norm.pdf((x - mu * T) / v) - k * stats.norm.pdf((x + 2 * b - mu * T) / v)) / v
    P = integrate.quad(dens, -b, np.inf)[0]
    close(L.survival_barrier(mu, s, b, T), P, 1e-9)
    close(L.survivor_mean_barrier(mu, s, b, T), integrate.quad(lambda x: x * dens(x), -b, np.inf)[0] / P, 1e-8)


def test_protocol_arithmetic():
    assert L.total_supply_sats() == 2_099_999_997_690_000
    assert L.block_subsidy_sats(0) == 5_000_000_000 and L.block_subsidy_sats(210_000) == 2_500_000_000
    assert L.block_subsidy_sats(33 * 210_000) == 0
    table = [1.0, 0.2045873, 0.0509779, 0.0131722, 0.0034552, 0.0009137]
    assert all(round(L.catch_up_nakamoto(0.1, z), 7) == t for z, t in enumerate(table))
    close(L.catch_up_ruin(0.25, 3), (1 / 3) ** 3, 1e-15)


def test_currency_carry_split_is_exact(RAW):
    c = L.load_cad_carry()
    close(float((c.excess - c.spot - c.carry - c.cross).abs().max()), 0.0, 1e-14)
    assert c.index[-1] <= 202609


def test_futures_return_has_the_cross_term():
    """r^F = C + (1 + C) r^S exactly; dropping the cross term C r^S leaves an error of that size."""
    rng = np.random.default_rng(11)
    S0 = 100.0; F0 = S0 / (1 + rng.uniform(-0.02, 0.02, 1000)); S1 = S0 * (1 + rng.normal(0, 0.05, 1000))
    C = (S0 - F0) / F0; rS = S1 / S0 - 1; rF = (S1 - F0) / F0
    assert np.allclose(rF, C + (1 + C) * rS, atol=1e-13)
    assert np.allclose(rF - (C + rS), C * rS, atol=1e-13)


def test_nakamoto_second_table_all_rows():
    paper = {0.10: 5, 0.15: 8, 0.20: 11, 0.25: 15, 0.30: 24, 0.35: 41, 0.40: 89, 0.45: 340}
    for q, z in paper.items():
        assert next(k for k in range(2000) if L.catch_up_nakamoto(q, k) < 0.001) == z


def test_round2_fee_split_and_loss_share(RAW):
    # the management and incentive parts add up to the whole fee drag
    assert abs(RAW["tfFeeMgmt"] + RAW["tfFeeInc"] - RAW["tfFeeExact"]) < 1e-10
    assert RAW["tfFeeInc"] > 0.2 * RAW["tfLeftMgmt"]       # more than a fifth: charged on the bill rate too
    # the 252 day loss probability sits just below the limit, and the small sample is within 3 standard errors
    assert RAW["stLossFinite"] < RAW["stLossTheory"]
    assert abs(RAW["stLossShare"] - RAW["stLossFinite"]) < 3 * RAW["stLossSe"]
    assert RAW["stLossTEquiv"] == 1.0                       # loses exactly when |t| < 1, path by path
    assert abs(RAW["stLossFinite"] - 68.17) < 0.005
    assert 2 <= RAW["ocRatioRecentMin"] < RAW["ocRatioRecentMax"] <= 4.05
