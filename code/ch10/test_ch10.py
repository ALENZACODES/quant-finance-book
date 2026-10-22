"""Tests of the Chapter 10 library and of the numbers quoted in the text."""
import math
import numpy as np
import pandas as pd
from scipy import integrate, optimize
import ch10lib as L
from conftest import close


def test_dcf_at_the_going_in_cap_equals_the_gordon_value():
    for r, g, c, T in ((0.075, 0.025, 0.1, 10), (0.09, 0.0, 0.0, 3), (0.06, 0.03, 0.15, 40)):
        cap = L.cap_rate(r, g, c)
        close(L.dcf_value(1.0, g, r, T, cap, c), L.gordon_value(1.0, r, g, c), 1e-12)


def test_irr_at_constant_cap_is_cash_yield_plus_growth():
    cap, g, T = 0.055, 0.02, 12
    t = np.arange(1, T + 1); f = np.concatenate([[-1.0], cap * (1 + g) ** (t - 1)]); f[-1] += cap * (1 + g) ** T / cap
    close(L.irr(f), cap + g, 1e-12)


def test_exit_cap_elasticity_by_finite_difference():
    r, g, T, c = 0.075, 0.025, 10, 0.1
    cap = L.cap_rate(r, g, c); h = 1e-6
    num = (math.log(L.dcf_value(1, g, r, T, cap * (1 + h), c)) - math.log(L.dcf_value(1, g, r, T, cap * (1 - h), c))) / (2 * h)
    close(num, L.exit_cap_elasticity(r, g, T, cap, c), 1e-8)
    close(L.exit_cap_elasticity(r, g, T, cap, c), -((1 + g) / (1 + r)) ** T, 1e-12)


def test_mortgage_constant_and_balance_by_simulating_the_loan():
    rate, yrs = 0.0625, 25
    k = L.mortgage_constant(rate, yrs); pay = k / 12; b = 1.0
    for month in range(1, yrs * 12 + 1):
        b = b * (1 + rate / 12) - pay
        if month == 120:
            close(b, L.balance(rate, yrs, 10), 1e-12)
    close(b, 0.0, 1e-12)
    close(L.mortgage_constant(0.0, 20), 1 / 20, 1e-15)


def test_max_loan_meets_both_tests():
    k = L.mortgage_constant(0.06, 25)
    loan, bind = L.max_loan(1.5e6, 25e6, k, 1.25, 0.65)
    assert 1.5e6 / (loan * k) >= 1.25 - 1e-12 and loan <= 0.65 * 25e6 + 1e-6
    close(L.dscr(1.5e6 / 25e6, loan / 25e6, k), 1.25, 1e-12) if bind == "DSCR" else None


def test_levered_volatility_scales_with_one_over_equity_share():
    rng = np.random.default_rng([3817, 99]); ra = 0.07 + 0.1 * rng.standard_normal(200000)
    re = L.levered_return(ra, 0.05, 0.6)
    close(re.std() / ra.std(), 1 / 0.4, 1e-12)


def test_smoothing_variance_and_autocorrelation():
    rng = np.random.default_rng([3817, 98]); a = 0.7
    x = rng.standard_normal(400000); s = L.smooth(x, a)[100:]
    close(s.var() / x.var(), (1 - a) / (1 + a), 0.01)
    for k in (1, 2, 4):
        close(pd.Series(s).autocorr(k), a ** k, 0.01)
    close(L.variance_ratio(s, 40), (1 + a) / (1 - a) * (1 - 2 * a * (1 - a ** 40) / (40 * (1 - a * a))), 0.3)


def test_unsmooth_inverts_smooth():
    rng = np.random.default_rng([3817, 97]); x = rng.standard_normal(500)
    close(np.max(np.abs(L.unsmooth(L.smooth(x, 0.6), 0.6) - x[1:])), 0.0, 1e-12)


def test_smoothed_beta_and_dimson_recovery():
    rng = np.random.default_rng([3817, 96]); m = rng.standard_normal(200000); x = 0.8 * m + 0.5 * rng.standard_normal(200000)
    s = pd.Series(L.smooth(x, 0.5)); mm = pd.Series(m)
    b, tot, _ = L.dimson(s, mm, lags=12)
    close(b[0], 0.4, 0.01); close(b[1], 0.2, 0.01); close(tot, 0.8 * (1 - 0.5 ** 13), 0.01)


def test_faustmann_rotation_maximises_land_value_and_is_shorter():
    for r in (0.03, 0.05):
        tf = L.faustmann_rotation(40.0, r, 1500.0)
        res = optimize.minimize_scalar(lambda T: -L.land_value(T, 40.0, r, 1500.0), bounds=(10, 80), method="bounded",
                                       options={"xatol": 1e-9})
        close(tf, res.x, 1e-4)
        assert tf < L.single_rotation(40.0, r)


def test_concession_value_against_numerical_sum_and_duration_limit():
    v = L.concession_value(100, 0.07, 0.025, 30)
    close(v, sum(100 * 1.025 ** (t - 1) / 1.07 ** t for t in range(1, 31)), 1e-9)
    close(L.concession_duration(0.07, 0.025, 4000), 1.07 / 0.045, 1e-6)


def test_convenience_yield_formula():
    S, r, u, y, tau = 70.0, 0.04, 0.02, 0.05, 0.5
    F = S * math.exp((r + u - y) * tau)
    close(L.net_convenience_yield(S, F, tau, r), y - u, 1e-14)


def test_real_hedge_covariance_by_simulation():
    rng = np.random.default_rng([3817, 95]); pi = 0.03 * rng.standard_normal(400000)
    R = 0.02 + 0.6 * pi + 0.1 * rng.standard_normal(400000)
    close(np.cov(R - pi, pi)[0, 1], L.real_hedge_cov(0.6, pi.var()), 2e-5)


def test_min_real_variance_weight_by_grid():
    rng = np.random.default_rng([3817, 94]); n = 100000
    pi = 0.03 * rng.standard_normal(n); x = 0.6 * pi + 0.1 * rng.standard_normal(n)
    w = L.min_real_variance_weight(np.cov(x, pi)[0, 1], x.var(ddof=1))
    grid = np.linspace(-0.2, 0.3, 501)
    close(grid[np.argmin([np.var(g * x - pi) for g in grid])], w, 1.1e-3)


def test_quoted_numbers(RAW):
    close(RAW["pfIrrSame"], 100 * 0.075, 1e-9)
    close(RAW["smCumRep"] - RAW["smCumTrue"], -RAW["smCumGap"], 1e-9)
    assert RAW["smArPre"] > RAW["smArPost"] > RAW["smArReit"]
    assert RAW["rtLeadOne"] > 2 / math.sqrt(RAW["rtN"])
    close(RAW["lvLtv"], 100 * 0.0556 / (1.25 * RAW["lvK"] / 100), 0.2)
    assert RAW["tbFFour"] < RAW["tbSFour"] < RAW["tbMai"]


def test_no_date_after_september_2026():
    for f in ("real_assets_quarterly.csv", "real_assets_monthly.csv", "eia_curves_monthly.csv"):
        d = pd.read_csv(L.ROOT / "data" / f, index_col=0, parse_dates=True)
        assert d.index.max() <= pd.Timestamp("2026-09-30")


def test_horizon_slope_of_a_lagged_price_is_one_less_one_over_k():
    rng = np.random.default_rng([3817, 91])
    pi = pd.Series(rng.normal(0.03, 0.02, 200_000))
    v = pi.shift(1)  # the asset's price change is last year's inflation
    for k in (1, 5, 10):
        b, se, n = L.horizon_beta(v, pi, k, lags=0)
        close(b, 1 - 1 / k, 0.01)


def test_intrinsic_storage_matches_brute_force_and_is_zero_for_a_falling_curve():
    rng = np.random.default_rng([3817, 92])
    for _ in range(50):
        F = rng.uniform(2, 5, 6); t = np.arange(1, 7) / 12
        v, i, j = L.intrinsic_storage(F, t, 0.05, 0.1, 0.1)
        brute = max([0.0] + [np.exp(-0.05 * t[b]) * (F[b] - 0.1) - np.exp(-0.05 * t[a]) * (F[a] + 0.1)
                             for a in range(6) for b in range(a + 1, 6)])
        close(v, brute, 1e-14)
    assert L.intrinsic_storage([4, 3, 2], [0.1, 0.2, 0.3], 0.03)[0] == 0.0


def test_monthly_payments_carried_to_year_end_cost_the_effective_annual_rate():
    """Debt service as in the leverage simulation: the loan's own IRR is (1 + i/12)^12 - 1."""
    import numpy as np
    i, n, T = 0.0625, 25, 10
    k = L.mortgage_constant(i, n); j = i / 12
    ds = k * ((1 + j) ** 12 - 1) / (12 * j)
    flows = np.r_[-1.0, ds * np.ones(T)]; flows[-1] += L.balance(i, n, T)
    assert abs(L.irr(flows) - ((1 + j) ** 12 - 1)) < 1e-10


def test_round2_slopes_and_ratios_reproduce(RAW):
    # a univariate slope is the correlation times the ratio of volatilities
    implied = RAW["rtCorrCreReit"] * RAW["rtVolCre"] / RAW["rtVolReit"]
    assert abs(RAW["rtBuniReit"] - implied) < 1e-10
    # the current-quarter coefficient inside the five-term regression is a different number
    assert abs(RAW["rtBzeroReit"] - RAW["rtBuniReit"]) > 0.01
    # the levered to unlevered ratio of IRR volatilities prints as 1.87 from the unrounded values
    assert abs(RAW["lvIlSd"] / RAW["lvIuSd"] - RAW["lvSdRatio"]) < 1e-10 and round(RAW["lvSdRatio"], 2) == 1.87
    for t in ("Bill", "Cre", "Hpi", "Reit", "Mkt", "Farm", "Timber", "Oil"):
        assert RAW[f"ihEse{t}"] > 0
