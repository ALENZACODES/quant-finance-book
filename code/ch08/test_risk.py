"""Tests of the Chapter 8 library and numbers against independent implementations (arch, statsmodels,
scipy) and against simulation."""
import math
import numpy as np
import pytest
from scipy import stats
from conftest import close
import ch08lib as L
import numbers_ch08 as R


@pytest.fixture(scope="module")
def daily():
    r, _ = L.load_daily()
    return r.values


def test_normal_and_t_var_es_against_scipy():
    for a in (0.95, 0.99):
        close(L.var_normal(0.001, 0.02, a), -stats.norm.ppf(1 - a, 0.001, 0.02), 1e-12)
        # ES as the integral of VaR over the tail
        u = np.linspace(a, 1, 200001)[:-1]
        close(L.es_normal(0.0, 1.0, a), np.mean(-stats.norm.ppf(1 - u)), 2e-3)
        s = math.sqrt(3 / 5)
        close(L.es_t(0.0, 1.0, 5, a), np.mean(s * stats.t.ppf(u, 5)), 3e-3)
        close(L.var_t(0.0, 1.0, 5, a), s * stats.t.ppf(a, 5), 1e-12)


def test_historical_es_is_subadditive_on_data(daily):
    a, b = daily[:8000], daily[-8000:]
    close(0, 0, 0)
    assert L.es_hist(a + b, 0.99) <= L.es_hist(a, 0.99) + L.es_hist(b, 0.99) + 1e-12


def test_var_fails_subadditivity_for_two_loans(RAW):
    rng = np.random.default_rng([R.SEED, 9]); M = 2_000_000
    d1 = rng.random(M) < 0.009; d2 = rng.random(M) < 0.009
    v1 = np.quantile(100.0 * d1, 0.99); v12 = np.quantile(100.0 * d1 + 100.0 * d2, 0.99)
    assert v1 == 0 and v12 == 100
    close(100 * np.mean(d1 | d2), RAW["saBoth"], 0.02)


def test_cornish_fisher_is_exact_to_first_order_for_small_skew():
    # a gamma with large shape has small skew; CF should beat the normal quantile
    k = 400.0; g = stats.gamma(k); z = stats.norm.ppf(0.01)
    s, ek = 2 / math.sqrt(k), 6 / k
    exact = (g.ppf(0.01) - k) / math.sqrt(k)
    assert abs(L.cf_z(z, s, ek) - exact) < 0.1 * abs(z - exact)


def test_garch_matches_arch(daily):
    from arch import arch_model
    x = 100 * daily
    for d in ("normal", "t"):
        mine = L.garch_fit(x, d)
        ref = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1, dist=d).fit(disp="off").params
        close(mine["alpha"], ref["alpha[1]"], 0.005); close(mine["beta"], ref["beta[1]"], 0.005)
        close(mine["persistence"], ref["alpha[1]"] + ref["beta[1]"], 0.001)
        if d == "t":
            close(mine["nu"], ref["nu"], 0.2)


def test_garch_forecast_against_simulation():
    rng = np.random.default_rng([R.SEED, 8])
    om, a, b = 0.05, 0.1, 0.85; lr = om / (1 - a - b); h1 = 4.0
    M, K = 200_000, 20
    h = np.full(M, h1); acc = np.zeros(K)
    for k in range(K):
        acc[k] = h.mean(); e = np.sqrt(h) * rng.standard_normal(M); h = om + a * e * e + b * h
    close(acc[-1], float(L.garch_forecast(h1, lr, a + b, K)), 0.03)


def test_garch_kurtosis_by_simulation():
    rng = np.random.default_rng([R.SEED, 7])
    om, a, b = 0.05, 0.08, 0.88
    n = 2_000_000; h = om / (1 - a - b); e = np.empty(n); z = rng.standard_normal(n)
    for t in range(n):
        e[t] = math.sqrt(h) * z[t]; h = om + a * e[t] ** 2 + b * h
    close(stats.kurtosis(e, fisher=False), L.garch_kurtosis(a, b), 0.35)


def test_kupiec_and_christoffersen_against_hand_counts():
    lr, p = L.kupiec(8, 250, 0.01)
    by_hand = -2 * (242 * math.log(0.99) + 8 * math.log(0.01) - 242 * math.log(242 / 250) - 8 * math.log(8 / 250))
    close(lr, by_hand, 1e-10)
    hits = np.array([0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0])
    lr, p, c = L.christoffersen(hits)
    assert (c["n00"], c["n01"], c["n10"], c["n11"]) == (4, 3, 3, 3)


def test_kupiec_size_and_zone_probabilities(RAW):
    close(100 * L.zones()[0], RAW["znGreen"], 1e-9)
    close(L.kupiec_power(2500, 0.01, 0.01), 0.05, 0.03)


def test_hit_sequence_is_bernoulli_when_var_is_right():
    rng = np.random.default_rng([R.SEED, 6])
    om, a, b = 0.02, 0.08, 0.9; n = 60000; h = om / (1 - a - b); hits = np.empty(n, int)
    q = stats.norm.ppf(0.99)
    for t in range(n):
        e = math.sqrt(h) * rng.standard_normal(); hits[t] = -e > q * math.sqrt(h); h = om + a * e * e + b * h
    assert L.kupiec(int(hits.sum()), n, 0.01)[1] > 0.001 and L.christoffersen(hits)[1] > 0.001


def test_gpd_formulas_against_scipy():
    from scipy.stats import genpareto
    xi, beta, u, n, nu = 0.3, 0.5, 1.0, 10000, 400
    v, e = L.gpd_var_es(u, xi, beta, n, nu, 0.999)
    # P(L > v) = nu/n * survival of the GPD at v - u
    close(nu / n * genpareto.sf(v - u, xi, scale=beta), 0.001, 1e-12)
    close(e - v, genpareto.mean(xi, scale=beta + xi * (v - u)), 1e-10)


def test_gpd_fit_recovers_known_tail():
    rng = np.random.default_rng([R.SEED, 5])
    exc = stats.genpareto.rvs(0.25, scale=0.6, size=20000, random_state=rng)
    xi, b = L.gpd_fit(exc)
    close(xi, 0.25, 0.03); close(b, 0.6, 0.03)


def test_sharpe_standard_error_by_simulation():
    rng = np.random.default_rng([R.SEED, 4])
    T, M, mu = 120, 20000, 0.15
    E = rng.exponential(1.0, (M, T)); x = mu - (E - 1)        # skew -2, excess kurtosis 6
    sr = x.mean(1) / x.std(1, ddof=1)
    close(sr.std(), L.sharpe_se(mu, T, -2.0, 6.0), 0.012)
    assert abs(sr.std() - L.sharpe_se(mu, T)) > abs(sr.std() - L.sharpe_se(mu, T, -2.0, 6.0))


def test_ratios_against_definitions():
    rng = np.random.default_rng([R.SEED, 3]); x = rng.normal(0.01, 0.04, 100000)
    close(L.omega(x), 1 + x.mean() / np.mean(np.maximum(-x, 0)), 1e-10)
    close(L.mppm(np.zeros(5) + 0.01, np.zeros(5)), 12 * math.log(1.01), 1e-12)


def test_omega_identity_and_mppm_penalises_leverage():
    rng = np.random.default_rng([R.SEED, 2]); x = rng.normal(0.006, 0.04, 200000); rf = np.zeros_like(x)
    assert L.mppm(3 * x, rf) < L.mppm(x, rf) * 3


def test_drawdown_law_by_simulation():
    rng = np.random.default_rng([R.SEED, 1]); m, s = 0.0004, 0.01
    y = np.cumsum(rng.normal(m, s, 2_000_000)); d = np.maximum.accumulate(y) - y
    close(d[200000:].mean(), s * s / (2 * m), 0.012)


def test_smoothing_moments_and_unsmoothing():
    rng = np.random.default_rng([R.SEED, 11]); r = rng.normal(0.01, 0.05, 400000); phi = 0.6
    rs = L.smooth(r, phi)
    close(rs.std() / r.std(), math.sqrt((1 - phi) / (1 + phi)), 0.005)
    close(L.ar1(rs), phi, 0.005)
    close(np.max(np.abs(L.unsmooth(rs, phi) - r[1:])), 0.0, 1e-9)


def test_lo_factor_is_sqrt_q_without_autocorrelation():
    rng = np.random.default_rng([R.SEED, 12]); x = rng.normal(size=400000)
    close(L.lo_annual_factor(x, 12), math.sqrt(12), 0.02)


def test_case_numbers_are_consistent(RAW):
    for m in ("Nor", "Hs", "Hsl", "Ew", "Gn", "Gt", "Fhs"):
        close(RAW[f"ca{m}Rate"], 100 * RAW[f"ca{m}X"] / 14897, 0.006)
    assert RAW["caFhsX"] < RAW["caHsX"] < RAW["caNorX"]


def test_seed_is_not_a_date():
    s = str(R.SEED)
    assert not (len(s) in (4, 6, 8) and s[:2] in ("19", "20"))


def test_z2_null_matches_simulation():
    """The analytic CLT null for Z2 agrees with a direct simulation under normal and t laws."""
    from scipy import stats as st
    from ch08lib import z2_moments, z2_null
    a, T, M = 0.975, 1000, 4000
    rng = np.random.default_rng(4417)
    for law, nu in (("normal", None), ("t", 6.0)):
        m1, m2 = z2_moments(law, a, nu=nu)
        mean, sd, _ = z2_null(m1, m2, T, a)
        assert abs(m1 - (1 - a)) < 1e-8 and abs(mean) < 1e-6
        if law == "normal":
            z = rng.standard_normal((M, T)); q = st.norm.ppf(a); es = st.norm.pdf(q) / (1 - a)
        else:
            sc = np.sqrt((nu - 2) / nu); z = rng.standard_t(nu, (M, T)) * sc
            q = st.t.ppf(a, nu) * sc
            es = sc * st.t.pdf(st.t.ppf(a, nu), nu) * (nu + st.t.ppf(a, nu) ** 2) / ((nu - 1) * (1 - a))
        z2 = (z * (z > q) / es).mean(axis=1) / (1 - a) - 1
        assert abs(z2.mean()) < 4 * sd / np.sqrt(M)
        assert abs(z2.std() / sd - 1) < 0.06


def test_mppm_law_lognormal_closed_form():
    """For a law bounded above the floor, quadrature MPPM equals the lognormal closed form."""
    from scipy import stats as st
    from ch08lib import mppm_law
    mu, s, rho, dt = 0.008, 0.04, 3.0, 1 / 12
    pdf = lambda r: st.lognorm.pdf(1 + r, s, scale=np.exp(mu))
    got = mppm_law(pdf, -0.6, 1.0, rho=rho, dt=dt, floor=-0.99)
    exact = (mu + 0.5 * (1 - rho) * s ** 2) / dt
    assert abs(got - exact) < 1e-4


def test_round2_printed_inputs_reproduce(RAW):
    # the drawdown law from the printed drift and volatility gives the printed mean and probability
    mu, vol = round(RAW["ddLogMu"], 2), round(RAW["ddLogVol"], 2)
    mean = vol ** 2 / (2 * mu)
    assert round(mean, 1) == round(RAW["ddLawMean"], 1)
    assert round(100 * math.exp(-math.log(1.25) / (mean / 100)), 1) == round(RAW["ddLawTwenty"], 1)
    # the tail index is below four, and two standard errors either side straddle four
    assert RAW["evAlphaTail"] < 4 and RAW["evAlphaLo"] < 4 < RAW["evAlphaHi"]
