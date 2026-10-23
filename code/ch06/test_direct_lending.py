"""Direct lending (Section 6.8): the exact results against independent computations, the simulation
against the closed form, and the historical figures against the filings they come from."""
import json, math
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats
from conftest import close, HERE
import ch06lib as L
import direct_lending as D

ROOT = HERE.parents[1]


def test_unitranche_pieces_add_up(RAW):
    u = RAW["ut"]
    close(u["el"]["f"] + u["el"]["l"], u["el"]["u"], 1e-10)
    close(u["w"] * u["s_f"] + (1 - u["w"]) * u["s_l"], u["s_u"], 1e-12)
    assert u["ls"]["f"] < u["ls"]["u"] < u["ls"]["l"]


def test_slice_loss_by_quadrature():
    # E[(K - Y)^+] for lognormal Y by direct integration, for the first-out and last-out pieces
    V0, g, sig, T, cost = 100.0, 0.0, 0.35, 5, 0.20
    m = (1 - cost) * V0 * (1 + g) ** T; s = sig * math.sqrt(T); mu = math.log(m) - 0.5 * s * s
    put = lambda K: integrate.quad(lambda y: (K - y) * stats.lognorm.pdf(y, s, scale=math.exp(mu)), 0, K)[0]
    close(L.slice_el(0, 20, V0, g, sig, T, cost), put(20), 1e-7)
    close(L.slice_el(20, 50, V0, g, sig, T, cost), put(50) - put(20), 1e-7)


def test_breach_probability():
    # zero drift: reflection gives 2 Phi(log(b/x0) / (sigma sqrt T))
    close(D.breach_prob(1.0, 0.8, 0.0, 0.25, 5), 2 * stats.norm.cdf(math.log(0.8) / (0.25 * math.sqrt(5))), 1e-12)
    # against a fine simulation with drift, and the corrected monthly simulation of the chapter
    rng = np.random.default_rng(11)
    p = D.breach_prob(1.0, 0.8, 0.03, 0.25, 2)
    mc = D.breach_mc(1.0, 0.8, 0.03, 0.25, 2, 500, 40_000, rng, shift=True)
    assert abs(mc - p) < 0.01


def test_covenant_numbers(RAW):
    cv = RAW["cv"]
    assert abs(cv["mc"] - cv["p_cov"]) < 0.006 and cv["mc_raw"] < cv["p_cov"] - 0.03
    assert cv["p_cov"] > cv["p_lite"] and cv["rec_cov"] > cv["rec_lite"]


def test_bdc_fees():
    close(D.bdc_base_fee(1.0), 0.03, 1e-15)
    close(D.bdc_base_fee(2.0), 0.04, 1e-15)
    h, c = D.SOURCE["arcc_hurdle_q"], D.SOURCE["arcc_catchup_q"]
    close(h / (1 - D.SOURCE["arcc_inc"]), c, 1e-15)          # the catch-up ends where 20% of all is reached
    close(D.income_fee(c), 0.2 * c, 1e-15)
    assert D.income_fee(h) == 0 and D.income_fee(0.03) == pytest.approx(0.006)


def test_tender_queue_by_simulation():
    # an investor tendering everything each quarter among others who tender 15% of NAV
    c, f, left = 0.05, 0.15, 1.0
    for _ in range(6):
        left -= left * c / f
    close(D.tender_exit(c, f, 6), left, 1e-12)
    assert D.tender_exit(c, 0.04, 1) == 0.0


def test_desmoothing_inverts_smoothing():
    rng = np.random.default_rng(3)
    r = rng.normal(0.02, 0.05, 400); phi = 0.4
    s = np.empty_like(r); s[0] = r[0]
    for t in range(1, len(r)):
        s[t] = (1 - phi) * r[t] + phi * s[t - 1]
    close(np.max(np.abs(D.desmooth(s, phi) - r[1:])), 0, 1e-12)


def test_bdc_data_against_the_filings():
    d = D.load_bdc()
    assert d.quarter_end.max() <= pd.Timestamp("2026-09-30") and d.ticker.nunique() == 19
    # Ares Capital, 31 December 2025: investments at fair value of $29,485 million in the 10-K
    a = d[(d.ticker == "ARCC") & (d.quarter_end == "2025-12-31")].iloc[0]
    close(a.fv / 1e6, 29485, 1)
    # a fourth quarter is the annual figure less nine months, from the raw facts
    raw = json.loads((ROOT / "data" / "raw" / "sec_bdc_facts.json").read_text())["companies"]["ARCC"]["facts"]
    f = {(x["start"], x["end"]): x["val"] for x in raw["NetIncomeLoss"]}
    close(a.operations, f[("2025-01-01", "2025-12-31")] - f[("2025-01-01", "2025-09-30")], 1)


def test_panel_statistics(RAW):
    p = D.bdc_panel(D.load_bdc())
    rho, n = D.pooled_rho(p)
    close(rho, RAW["dlRhoRaw"], 1e-12)
    y = p["yield"].dropna()
    assert 0.05 < y.median() < 0.2 and 0.5 < RAW["dlPassRaw"][1] < 1.6
    pf = RAW["dlPf"]
    close(pf["vol_ds"] / pf["vol"], math.sqrt((1 + pf["phi"]) / (1 - pf["phi"])), 0.05)


@pytest.mark.skipif(not D.FF.exists(), reason="run python scripts/build_french.py --download")
def test_dimson_against_statsmodels(RAW):
    import statsmodels.api as sm
    p = D.bdc_panel(D.load_bdc()); roe = D.ew(p, "roe"); m = D.market_quarterly()
    j = pd.concat([roe, m, m.shift(1)], axis=1, keys=["r", "m0", "m1"], sort=True).dropna()
    fit = sm.OLS(j.r.values, sm.add_constant(j[["m0", "m1"]].values)).fit()
    np.testing.assert_allclose(fit.params, RAW["dlDimson"], atol=1e-12)


def test_yield_to_life_and_coverage(RAW):
    c, u, T = 0.09, 0.02, 3
    y = D.yield_to_life(c, u, T)
    close(c * (1 - math.exp(-y * T)) / y + math.exp(-y * T), 1 - u, 1e-12)
    first = c / (1 - math.exp(-c * T))
    assert abs((y - c) - first * u) < 5 * u * u and first > 1 / T
    # coverage: at the threshold the coverage is exactly 150%
    d = 1.12; Lx = D.coverage_threshold(d)
    close((1 + d) * (1 - Lx) / d, 1.5, 1e-12)
    # the breach probability by simulating the factor of the large pool
    ac = RAW["ac"]; rng = np.random.default_rng(5)
    m = rng.standard_normal(400_000)
    share = stats.norm.cdf((stats.norm.ppf(ac["p"]) - math.sqrt(ac["rho"]) * m) / math.sqrt(1 - ac["rho"]))
    mc = float((ac["lgd"] * share > D.coverage_threshold(1.5)).mean())
    assert abs(mc - D.coverage_breach_prob(1.5, ac["p"], ac["rho"], ac["lgd"])) < 0.001
    close(D.lhp_quantile(0.999, ac["p"], ac["rho"]), float(np.quantile(share, 0.999)), 0.01)
