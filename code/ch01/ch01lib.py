"""Chapter 1 library: return arithmetic, cash-flow based performance and return statistics.

Every function here is used by a figure script, by run_all.py (which writes the numbers quoted in
the text) and by test_ch01.py (which checks them against closed forms and against independent
implementations).
"""
from __future__ import annotations
import numpy as np
from scipy import optimize, stats

# ----------------------------------------------------------------------------- compounding
def effective_rate(r_nominal: float, m: float) -> float:
    """Effective annual rate of a nominal rate compounded m times a year (m=np.inf: continuous)."""
    if np.isinf(m):
        return float(np.expm1(r_nominal))
    return (1.0 + r_nominal / m) ** m - 1.0


def continuous_equivalent(r_effective: float) -> float:
    """Continuously compounded rate with the same one year growth as an effective rate."""
    return float(np.log1p(r_effective))


# ----------------------------------------------------------------------------- cash flows
def npv(rate, cfs, times=None):
    """Net present value of cash flows cfs at times (in periods), discount rate per period."""
    cfs = np.asarray(cfs, float)
    t = np.arange(len(cfs)) if times is None else np.asarray(times, float)
    return float(np.sum(cfs / (1.0 + rate) ** t))


def irr_all(cfs, times=None, lo=-0.99, hi=10.0, grid=20001):
    """All real IRRs in (lo, hi): scan NPV on a fine grid for sign changes, polish with Brent."""
    rs = np.linspace(lo, hi, grid)
    vals = np.array([npv(r, cfs, times) for r in rs])
    roots = []
    for a, b, fa, fb in zip(rs[:-1], rs[1:], vals[:-1], vals[1:]):
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0:
            roots.append(optimize.brentq(npv, a, b, args=(cfs, times), xtol=1e-14))
    return sorted(set(np.round(roots, 12)))


def irr(cfs, times=None):
    """The unique IRR; raises if there is not exactly one root."""
    roots = irr_all(cfs, times)
    if len(roots) != 1:
        raise ValueError(f"expected one IRR, found {len(roots)}: {roots}")
    return float(roots[0])


def mirr(cfs, finance_rate, reinvest_rate):
    """Modified IRR: negatives discounted to t=0 at the finance rate, positives compounded to
    t=N at the reinvestment rate. Same convention as numpy_financial.mirr."""
    cfs = np.asarray(cfs, float)
    n = len(cfs) - 1
    t = np.arange(n + 1)
    pv_neg = -np.sum(np.where(cfs < 0, cfs, 0) / (1 + finance_rate) ** t)
    fv_pos = np.sum(np.where(cfs > 0, cfs, 0) * (1 + reinvest_rate) ** (n - t))
    return (fv_pos / pv_neg) ** (1.0 / n) - 1.0


def twr(period_returns):
    """Time-weighted (geometrically linked) return over all periods."""
    return float(np.prod(1 + np.asarray(period_returns)) - 1)


# ----------------------------------------------------------------------------- multiples, PME
def multiples(contrib, distrib, nav_T):
    """DPI, RVPI, TVPI from positive contribution and distribution amounts."""
    paid_in = float(np.sum(contrib))
    dpi = float(np.sum(distrib)) / paid_in
    rvpi = nav_T / paid_in
    return dpi, rvpi, dpi + rvpi


def ks_pme(contrib, distrib, nav_T, index):
    """Kaplan-Schoar PME: index-adjusted distributions plus NAV over index-adjusted contributions.
    index[t] is the public index level at each date; all flows are carried forward to T."""
    index = np.asarray(index, float)
    g = index[-1] / index
    return (np.sum(np.asarray(distrib) * g) + nav_T) / np.sum(np.asarray(contrib) * g)


def ln_pme(contrib, distrib, index):
    """Long-Nickels PME: invest the same contributions in the index, take out the same
    distributions; return the terminal value of that shadow portfolio and all real IRRs of the
    shadow cash flows (contributions, distributions and the shadow terminal value)."""
    index = np.asarray(index, float)
    value = 0.0
    for t in range(len(index)):
        if t > 0:
            value *= index[t] / index[t - 1]
        value += contrib[t] - distrib[t]
    cfs = -np.asarray(contrib, float) + np.asarray(distrib, float)
    cfs[-1] += value
    return value, irr_all(cfs)


# ----------------------------------------------------------------------------- moments
def sample_moments(x):
    """Mean, sample sd (n-1), skewness and excess kurtosis (population style moment ratios)."""
    x = np.asarray(x, float)
    m = x.mean()
    d = x - m
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    return dict(n=len(x), mean=m, sd=x.std(ddof=1), skew=m3 / m2 ** 1.5,
                exkurt=m4 / m2 ** 2 - 3.0)


def jarque_bera(x):
    """JB = n/6 (S^2 + K_e^2 / 4), asymptotically chi-square with 2 df under normality."""
    s = sample_moments(x)
    jb = s["n"] / 6.0 * (s["skew"] ** 2 + s["exkurt"] ** 2 / 4.0)
    return jb, float(stats.chi2.sf(jb, 2))


def lognormal_moments(mu, sigma):
    """Mean, sd, skewness and excess kurtosis of exp(Y), Y ~ N(mu, sigma^2)."""
    w = np.exp(sigma ** 2)
    mean = np.exp(mu + sigma ** 2 / 2)
    sd = np.sqrt((w - 1) * np.exp(2 * mu + sigma ** 2))
    skew = (w + 2) * np.sqrt(w - 1)
    exk = w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 6
    return mean, sd, skew, exk


# ----------------------------------------------------------------------------- dependence
def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    c0 = np.dot(x, x)
    return np.array([np.dot(x[k:], x[:len(x) - k]) / c0 for k in range(nlags + 1)])


def durbin_watson(e):
    e = np.asarray(e, float)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def smooth(r, alpha, r0=None):
    """Exponential smoothing model of appraisal returns: R*_t = (1-a) R_t + a R*_{t-1}."""
    out = np.empty_like(np.asarray(r, float))
    prev = np.mean(r) if r0 is None else r0
    for t, x in enumerate(r):
        prev = (1 - alpha) * x + alpha * prev
        out[t] = prev
    return out


def unsmooth(rs, alpha):
    """Invert the smoothing model: R_t = (R*_t - a R*_{t-1}) / (1-a)."""
    rs = np.asarray(rs, float)
    return (rs[1:] - alpha * rs[:-1]) / (1 - alpha)


def variance_ratio_ar1(rho, q):
    """Var(sum of q returns) / (q Var(one return)) for a stationary AR(1) with coefficient rho."""
    k = np.arange(1, q)
    return 1 + 2 * np.sum((1 - k / q) * rho ** k)


def portfolio_sd_equal(sigma, rho, n):
    """sd of an equally weighted portfolio of n assets, common sd sigma and pairwise corr rho."""
    return sigma * np.sqrt(rho + (1 - rho) / n)


# ----------------------------------------------------------------------------- option helper
def bs_put(S, K, T, r, sigma):
    """Black-Scholes European put (derived in Chapter 5); used here only to price a premium."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


# ----------------------------------------------------------------------------- running fund example
def fund_example(fee=2.0):
    """Deterministic nine year fund used throughout Sections 1.3 to 1.7 ("Fund A").

    Five deals are bought at t = 0..4. Each deal's economic value grows 15% a year and the deal is
    sold at age 5. Reporting is conservative: a deal is carried at 90% of cost in its first year
    (early write-downs and expensed costs), at cost at ages 2 and 3, and at economic value from age
    4. A fee of 2 per year is paid in years 0..8: in years 0..4 it is drawn from LPs with the
    capital call, in years 5..8 it is netted from distributions. The last deal exits in year 9. Flows happen at integer dates and
    NAV is measured just after the flows of that date.
    """
    T = 9
    invest = np.array([20.0, 22.0, 20.0, 15.0, 10.0])
    g = 0.15
    t = np.arange(T + 1)
    contrib = np.zeros(T + 1)
    distrib = np.zeros(T + 1)
    nav = np.zeros(T + 1)
    for k, c in enumerate(invest):
        contrib[k] += c
        distrib[k + 5] += c * (1 + g) ** 5
    for s in range(T):
        if s <= 4:
            contrib[s] += fee
        else:
            distrib[s] -= fee
    for s in t:
        v = 0.0
        for k, c in enumerate(invest):
            age = s - k
            if age < 0 or age >= 5:
                continue
            if age == 0:
                v += 0.9 * c
            elif age <= 2:
                v += c
            else:
                v += c * (1 + g) ** age
        nav[s] = v
    return t, contrib, distrib, nav


INDEX_RETURNS = np.array([0.08, -0.12, 0.15, 0.10, 0.06, 0.09, -0.04, 0.12, 0.07])


def index_levels():
    return 100 * np.concatenate([[1.0], np.cumprod(1 + INDEX_RETURNS)])
