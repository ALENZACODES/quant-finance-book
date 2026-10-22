"""Library for Chapter 10: property cash flows and value, mortgage arithmetic, appraisal smoothing,
lead and lag regressions, the forestry rotation problem, concessions, the cost of carry of a stored
commodity, and inflation hedging.

Rates are annual decimals and returns are log returns per period unless a docstring says otherwise.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize

ROOT = Path(__file__).resolve().parents[2]
END = "2026-09-30"   # no observation after September 2026 is used
REIT_START = "1997-01-01"  # the Z.1 REIT revaluation has a break in the fourth quarter of 1996
SPLICE = "1996-01-01"  # the Z.1 commercial property index is appraisal based before this quarter

# ----------------------------------------------------------------------------- data

def load_quarterly():
    """Quarterly levels: Z.1 commercial real estate price index (cre), equity REIT market value (reit_mv)
    and revaluation (reit_reval), and the FHFA all-transactions house price index (hpi)."""
    q = pd.read_csv(ROOT / "data" / "real_assets_quarterly.csv", index_col=0, parse_dates=True)
    return q[q.index <= END]


def load_monthly():
    m = pd.read_csv(ROOT / "data" / "real_assets_monthly.csv", index_col=0, parse_dates=True)
    return m[m.index <= END]


def load_curves():
    e = pd.read_csv(ROOT / "data" / "eia_curves_monthly.csv", index_col=0, parse_dates=True)
    return e[e.index <= END]


def quarterly_returns():
    """Quarterly log price returns of commercial property (cre), equity REITs (reit, from 1997, the
    revaluation over the previous quarter's market value) and houses (hpi)."""
    q = load_quarterly()
    out = pd.DataFrame(index=q.index)
    out["cre"] = np.log(q["cre"]).diff()
    r = q["reit_reval"] / q["reit_mv"].shift(1)
    out["reit"] = np.log1p(r.where(q.index >= REIT_START))
    out["hpi"] = np.log(q["hpi"]).diff()
    return out.iloc[1:]


def load_market_quarterly():
    """Quarterly log total return of the US equity market (French library, market excess return plus the
    bill rate), indexed by the first day of the quarter. Not redistributed with the code."""
    f = pd.read_csv(ROOT / "data" / "ff_factors_monthly.csv")
    f.index = pd.to_datetime(f["month"].astype(str), format="%Y%m")
    lr = np.log1p((f["mkt_rf"] + f["rf"]) / 100)
    n = lr.resample("QS").count()
    return lr.resample("QS").sum()[n == 3]


# ----------------------------------------------------------------------------- property value

def gordon_value(noi1, r, g, capex=0.0):
    """Value of a property whose NOI next year is noi1 and grows at g, with a capital reserve equal to the
    share `capex` of NOI, discounted at r > g."""
    return noi1 * (1 - capex) / (r - g)


def cap_rate(r, g, capex=0.0):
    """Going-in cap rate (next year's NOI over value) implied by r, g and the capital reserve share."""
    return (r - g) / (1 - capex)


def dcf_value(noi1, g, r, T, exit_cap, capex=0.0, sale_cost=0.0):
    """Value of T years of cash flow NOI*(1-capex), growing at g, plus a sale at the end of year T at
    NOI_{T+1}/exit_cap less a share sale_cost. Annual cash flows at year ends."""
    t = np.arange(1, T + 1)
    noi = noi1 * (1 + g) ** (t - 1)
    cf = noi * (1 - capex)
    sale = noi1 * (1 + g) ** T / exit_cap * (1 - sale_cost)
    return float(np.sum(cf / (1 + r) ** t) + sale / (1 + r) ** T)


def irr(cf):
    """Internal rate of return of cash flows cf[0], cf[1], ... at yearly dates (the root above -99%)."""
    cf = np.asarray(cf, float)
    f = lambda x: np.sum(cf / (1 + x) ** np.arange(len(cf)))
    return optimize.brentq(f, -0.99, 10.0, xtol=1e-14)


def exit_cap_elasticity(r, g, T, exit_cap, capex=0.0):
    """Elasticity of the DCF value with respect to the exit cap rate: minus the share of value in the sale."""
    v = dcf_value(1.0, g, r, T, exit_cap, capex)
    sale = (1 + g) ** T / exit_cap / (1 + r) ** T
    return -sale / v

# ----------------------------------------------------------------------------- mortgages

def mortgage_constant(rate, years, m=12):
    """Annual debt service per unit of loan for a fully amortising loan with m payments a year."""
    i = rate / m
    n = years * m
    return m * i / (1 - (1 + i) ** (-n)) if abs(i) > 1e-14 else 1.0 / years


def balance(rate, years, t, m=12):
    """Balance per unit of original loan after t years of an amortisation over `years`."""
    i = rate / m
    n, k = years * m, t * m
    if abs(i) < 1e-14:
        return 1 - k / n
    return ((1 + i) ** n - (1 + i) ** k) / ((1 + i) ** n - 1)


def dscr(cap, ltv, k):
    """Debt service coverage ratio of a loan of ltv times value with mortgage constant k on a property
    whose NOI over value is cap."""
    return cap / (ltv * k)


def max_loan(noi, value, k, dscr_min, ltv_max):
    """Largest loan that meets both a minimum DSCR and a maximum LTV; returns (loan, binding test)."""
    a, b = noi / (dscr_min * k), ltv_max * value
    return (a, "DSCR") if a < b else (b, "LTV")


def levered_return(ra, rd, lam):
    """Return on equity of an asset return ra financed with a share lam of debt at rate rd."""
    return (ra - lam * rd) / (1 - lam)

# ----------------------------------------------------------------------------- smoothing

def smooth(r, alpha):
    """Appraisal smoothing r*_t = alpha r*_{t-1} + (1-alpha) r_t, started at r*_0 = r_0."""
    r = np.asarray(r, float)
    s = np.empty_like(r)
    s[0] = r[0]
    for t in range(1, len(r)):
        s[t] = alpha * s[t - 1] + (1 - alpha) * r[t]
    return s


def unsmooth(s, alpha):
    """Invert the smoothing filter: r_t = (r*_t - alpha r*_{t-1}) / (1-alpha). Loses the first point."""
    s = np.asarray(s, float)
    return (s[1:] - alpha * s[:-1]) / (1 - alpha)


def ar1(x):
    """OLS slope of x_t on x_{t-1} with an intercept."""
    x = np.asarray(pd.Series(x).dropna(), float)
    y, z = x[1:], x[:-1]
    z = z - z.mean()
    return float(np.sum(z * (y - y.mean())) / np.sum(z * z))


def variance_ratio(x, k):
    """Var of k-period sums over k times the one-period variance, from non-overlapping sums."""
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x) // k
    s = x[len(x) - n * k:].reshape(n, k).sum(axis=1)
    return float(s.var(ddof=1) / (k * x.var(ddof=1)))


def dimson(y, x, lags=3):
    """Dimson regression of y_t on x_t, x_{t-1}, ..., x_{t-lags}: returns (coefficients, sum of slopes)."""
    d = pd.concat({"y": y} | {f"x{j}": x.shift(j) for j in range(lags + 1)}, axis=1).dropna()
    X = np.column_stack([np.ones(len(d))] + [d[f"x{j}"].values for j in range(lags + 1)])
    b = np.linalg.lstsq(X, d["y"].values, rcond=None)[0]
    return b[1:], float(np.sum(b[1:])), len(d)


def ols(y, X):
    """OLS with an intercept: returns (coefficients, Newey-West standard errors with `lag` 2, residuals)."""
    import statsmodels.api as sm
    f = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(X, float))).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return f.params, f.bse, f.resid

# ----------------------------------------------------------------------------- timber and concessions

def volume(t, A=400.0, k=0.06, c=3.0):
    """Chapman-Richards stand volume, cubic metres per hectare at age t years."""
    return A * (1 - np.exp(-k * np.asarray(t, float))) ** c


def dvolume(t, A=400.0, k=0.06, c=3.0):
    t = np.asarray(t, float)
    return A * c * k * np.exp(-k * t) * (1 - np.exp(-k * t)) ** (c - 1)


def land_value(T, p, r, cost=0.0, **kw):
    """Faustmann value of bare land: an infinite chain of rotations of length T, each paying p*Q(T) at
    harvest after a planting cost `cost` at its start, at a continuously compounded rate r."""
    return (p * volume(T, **kw) * math.exp(-r * T) - cost) / (1 - math.exp(-r * T))


def faustmann_rotation(p, r, cost=0.0, lo=5.0, hi=150.0, **kw):
    """Rotation that maximises land value, by the first order condition of Proposition 10.9."""
    f = lambda T: p * dvolume(T, **kw) - r * p * volume(T, **kw) - r * land_value(T, p, r, cost, **kw)
    return optimize.brentq(f, lo, hi, xtol=1e-12)


def single_rotation(p, r, lo=5.0, hi=150.0, **kw):
    """Rotation that maximises the value of one harvest, p Q'(T) = r p Q(T) (Fisher)."""
    f = lambda T: dvolume(T, **kw) - r * volume(T, **kw)
    return optimize.brentq(f, lo, hi, xtol=1e-12)


def concession_value(cf1, r, g, n):
    """Value of n annual cash flows growing at g from cf1, discounted at r (a growing annuity)."""
    if abs(r - g) < 1e-14:
        return cf1 * n / (1 + r)
    return cf1 / (r - g) * (1 - ((1 + g) / (1 + r)) ** n)


def concession_duration(r, g, n):
    """Macaulay duration, in years, of the concession cash flows."""
    t = np.arange(1, n + 1)
    w = ((1 + g) / (1 + r)) ** (t - 1) / (1 + r)
    return float(np.sum(t * w) / np.sum(w))

# ----------------------------------------------------------------------------- storage and inflation

def net_convenience_yield(f_near, f_far, tau, r):
    """Convenience yield net of storage cost implied by two futures prices tau years apart and the
    interest rate r (continuous): r - ln(F_far/F_near)/tau."""
    return r - np.log(np.asarray(f_far) / np.asarray(f_near)) / tau


def seasonal_deviation(s, years=5):
    """Log deviation of a monthly inventory from its average in the same calendar month over the previous
    `years` years (at least three)."""
    ls = np.log(s.dropna())
    base = ls.groupby(ls.index.month).transform(lambda x: x.rolling(years, min_periods=3).mean().shift(1))
    return (ls - base).dropna()


def real_hedge_cov(b, var_pi):
    """Covariance of a real return with inflation when the nominal return has inflation beta b and an
    error uncorrelated with inflation: (b - 1) Var(pi)."""
    return (b - 1) * var_pi


def min_real_variance_weight(cov_xpi, var_x):
    """Weight in a risky asset x, with the rest in a bill whose nominal return is known in advance, that
    minimises the variance of the real return: Cov(x, pi) / Var(x)."""
    return cov_xpi / var_x


def horizon_beta(y, x, k, lags=None):
    """Slope of k period sums of y on k period sums of x, overlapping, with Newey-West standard errors
    over `lags` lags (default k): returns (slope, standard error, number of overlapping sums)."""
    import statsmodels.api as sm
    d = pd.concat([pd.Series(y), pd.Series(x)], axis=1).dropna()
    s = d.rolling(k).sum().dropna()
    f = sm.OLS(s.iloc[:, 0].to_numpy(float), sm.add_constant(s.iloc[:, 1].to_numpy(float))).fit(
        cov_type="HAC", cov_kwds={"maxlags": k if lags is None else lags})
    return float(f.params[1]), float(f.bse[1]), len(s)


def intrinsic_storage(F, t, r, c_in=0.0, c_out=0.0):
    """Intrinsic value of one cycle of storage of one unit: the largest present value, over delivery
    dates i < j, of buying for delivery at t_i and selling for delivery at t_j, less injection and
    withdrawal costs, and zero if no pair pays. Returns (value, i, j)."""
    F = np.asarray(F, float); t = np.asarray(t, float); best = (0.0, None, None)
    for i in range(len(F)):
        for j in range(i + 1, len(F)):
            v = np.exp(-r * t[j]) * (F[j] - c_out) - np.exp(-r * t[i]) * (F[i] + c_in)
            if v > best[0]:
                best = (float(v), i, j)
    return best
