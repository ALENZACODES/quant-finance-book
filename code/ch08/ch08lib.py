"""Library for Chapter 8: value at risk and expected shortfall by several methods, the Cornish-Fisher
quantile, GARCH(1,1) by maximum likelihood with normal and Student t innovations, EWMA variance,
backtests of coverage (Kupiec), independence (Christoffersen) and expected shortfall
(Acerbi-Szekely), Basel traffic-light zones, peaks over threshold with the generalised Pareto law,
performance ratios, the standard error of a Sharpe ratio, drawdowns, and appraisal smoothing.

Returns are simple returns as decimals unless a docstring says otherwise. A loss is minus a return,
and VaR and ES are reported as positive losses.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy import optimize, signal, stats

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- data

def load_daily():
    """Daily US market total return (decimal), indexed by date; the bill rate as a second series."""
    import pandas as pd
    d = pd.read_csv(ROOT / "data" / "ff_daily.csv", index_col=0, parse_dates=True)
    return d["mkt"] / 100, d["rf"] / 100


def load_monthly():
    import pandas as pd
    f = pd.read_csv(ROOT / "data" / "ff_factors_monthly.csv", index_col=0) / 100
    return f


def load_usdcad(end="2026-09-30"):
    """Daily log change of the USD/CAD rate (CAD per USD), through `end`."""
    import pandas as pd
    d = pd.read_csv(ROOT / "data" / "usdcad_rates.csv", index_col=0, parse_dates=True)
    s = d["usdcad"].dropna(); s = s[s.index <= end]
    return np.log(s).diff().dropna()

# ----------------------------------------------------------------------------- VaR and ES

def var_normal(mu, sd, a):
    """VaR at confidence a of a loss whose return is N(mu, sd^2)."""
    return -(mu + sd * stats.norm.ppf(1 - a))


def es_normal(mu, sd, a):
    return -mu + sd * stats.norm.pdf(stats.norm.ppf(a)) / (1 - a)


def var_t(mu, sd, nu, a):
    """Student t scaled to standard deviation sd."""
    s = sd * math.sqrt((nu - 2) / nu)
    return -(mu + s * stats.t.ppf(1 - a, nu))


def es_t(mu, sd, nu, a):
    s = sd * math.sqrt((nu - 2) / nu)
    q = stats.t.ppf(a, nu)
    return -mu + s * stats.t.pdf(q, nu) / (1 - a) * (nu + q * q) / (nu - 1)


def var_hist(r, a):
    """Historical VaR: the empirical a-quantile of the loss, by the lower interpolation-free rule
    (the k-th largest loss with k = ceil(n(1-a)))."""
    L = np.sort(-np.asarray(r))[::-1]
    k = max(1, math.ceil(len(L) * (1 - a)))
    return L[k - 1]


def es_hist(r, a):
    """Average of the k largest losses, k = ceil(n(1-a))."""
    L = np.sort(-np.asarray(r))[::-1]
    k = max(1, math.ceil(len(L) * (1 - a)))
    return L[:k].mean()


def cf_z(z, s, k):
    """Cornish-Fisher adjusted standard normal quantile z for skewness s and excess kurtosis k."""
    return z + (z * z - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s * s / 36


def var_cf(mu, sd, s, k, a):
    return -(mu + sd * cf_z(stats.norm.ppf(1 - a), s, k))


def moments(r):
    r = np.asarray(r)
    return r.mean(), r.std(ddof=1), stats.skew(r), stats.kurtosis(r)

# ----------------------------------------------------------------------------- GARCH and EWMA

def garch_filter(e, omega, alpha, beta, s0=None):
    """Conditional variances h_t of GARCH(1,1) for residuals e; h_0 = s0 (default sample variance).
    Returns h of length len(e)+1: h[t] is the variance of e[t] given e[:t]; h[-1] forecasts the next."""
    e = np.asarray(e, float)
    s0 = e.var() if s0 is None else s0
    x = np.empty(len(e) + 1); x[0] = s0  # lfilter gives h[0] = x[0] = s0
    x[1:] = omega + alpha * e * e
    h = signal.lfilter([1.0], [1.0, -beta], x)
    return h


def _nll(theta, r, dist):
    mu, omega, alpha, beta = theta[:4]
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e12
    e = r - mu
    h = garch_filter(e, omega, alpha, beta)[:-1]
    if dist == "normal":
        return 0.5 * np.sum(np.log(2 * np.pi * h) + e * e / h)
    nu = theta[4]
    if nu <= 2.05:
        return 1e12
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return -np.sum(c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(e * e / (h * (nu - 2))))


def garch_fit(r, dist="normal"):
    """Maximum likelihood GARCH(1,1) with constant mean. r in percent works best. Returns a dict."""
    r = np.asarray(r, float)
    v = r.var()
    x0 = [r.mean(), 0.05 * v, 0.08, 0.90] + ([8.0] if dist == "t" else [])
    best = None
    for start in (x0, [r.mean(), 0.02 * v, 0.05, 0.93] + ([6.0] if dist == "t" else [])):
        res = optimize.minimize(_nll, start, args=(r, dist), method="Nelder-Mead",
                                options=dict(maxiter=6000, maxfev=12000, xatol=1e-8, fatol=1e-8))
        res = optimize.minimize(_nll, res.x, args=(r, dist), method="Nelder-Mead",
                                options=dict(maxiter=6000, maxfev=12000, xatol=1e-10, fatol=1e-10))
        if best is None or res.fun < best.fun:
            best = res
    mu, omega, alpha, beta = best.x[:4]
    out = dict(mu=mu, omega=omega, alpha=alpha, beta=beta, nll=best.fun, persistence=alpha + beta,
               lrvar=omega / (1 - alpha - beta))
    if dist == "t":
        out["nu"] = best.x[4]
    return out


def garch_forecast(h1, lrvar, persistence, k):
    """E_t of the variance k steps ahead (k = 1, 2, ...) given the one step forecast h1."""
    k = np.asarray(k)
    return lrvar + persistence ** (k - 1) * (h1 - lrvar)


def garch_kurtosis(alpha, beta):
    """Kurtosis of a GARCH(1,1) with normal innovations (requires (a+b)^2 + 2a^2 < 1)."""
    p = alpha + beta
    return 3 * (1 - p * p) / (1 - p * p - 2 * alpha * alpha)


def ewma(r, lam=0.94, s0=None):
    """RiskMetrics variance: h[t] = lam h[t-1] + (1-lam) r[t-1]^2, zero mean. Length len(r)+1."""
    r = np.asarray(r, float)
    s0 = r[:min(len(r), 60)].var() if s0 is None else s0
    x = np.empty(len(r) + 1); x[0] = s0; x[1:] = (1 - lam) * r * r
    return signal.lfilter([1.0], [1.0, -lam], x)

# ----------------------------------------------------------------------------- backtests

def kupiec(x, n, p):
    """Likelihood ratio test of unconditional coverage: x exceptions in n days at rate p."""
    def ll(q):
        return (n - x) * math.log1p(-q) + (x * math.log(q) if x > 0 else 0.0) if 0 < q < 1 else \
            (0.0 if (q == 0 and x == 0) or (q == 1 and x == n) else -math.inf)
    lr = -2 * (ll(p) - ll(x / n))
    return lr, stats.chi2.sf(lr, 1)


def christoffersen(hits):
    """Likelihood ratio test that exceptions are independent (first-order Markov alternative)."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00 = int(np.sum((a == 0) & (b == 0))); n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0))); n11 = int(np.sum((a == 1) & (b == 1)))
    def xl(k, q):
        return k * math.log(q) if k > 0 else 0.0
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)
    l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
    l0 = xl(n00 + n10, 1 - p1) + xl(n01 + n11, p1)
    lr = -2 * (l0 - l1)
    return lr, stats.chi2.sf(lr, 1), dict(n00=n00, n01=n01, n10=n10, n11=n11)


def zones(n=250, p=0.01, green=4, yellow=9):
    """Probabilities of the green (<= green), yellow and red (> yellow) zones for n days at rate p."""
    g = stats.binom.cdf(green, n, p); y = stats.binom.cdf(yellow, n, p) - g
    return g, y, 1 - g - y


def kupiec_power(n, p, q, level=0.05):
    """Exact probability that the Kupiec test at `level` rejects when the true exception rate is q."""
    x = np.arange(n + 1)
    rej = np.array([kupiec(int(k), n, p)[1] < level for k in x])
    return float(stats.binom.pmf(x, n, q)[rej].sum())


def z2_moments(law, a, nu=None, sample=None):
    """First two moments of X = (Z / ES) 1{Z > VaR} for one day's standardised loss Z under a law with
    the true VaR and ES at level a: 'normal', 't' (unit variance, nu degrees of freedom) or 'empirical'
    (the losses in sample, with VaR the k-th largest and ES the mean of the k largest,
    k = ceil(n(1-a)), as historical simulation computes them). For a continuous law m1 = 1 - a."""
    from scipy import integrate
    if law == "normal":
        q = stats.norm.ppf(a); es = stats.norm.pdf(q) / (1 - a)
        m1 = 1 - a                                       # int_q^inf z phi = phi(q) = (1-a) ES
        m2 = (q * stats.norm.pdf(q) + (1 - a)) / es ** 2  # int_q^inf z^2 phi = q phi(q) + 1 - Phi(q)
        return m1, m2
    if law == "t":
        sc = math.sqrt((nu - 2) / nu); q = stats.t.ppf(a, nu)
        es = sc * stats.t.pdf(q, nu) / (1 - a) * (nu + q * q) / (nu - 1)
        m2 = sc ** 2 * integrate.quad(lambda x: x * x * stats.t.pdf(x, nu), q, np.inf, epsabs=1e-13)[0] / es ** 2
        return 1 - a, m2
    z = np.sort(np.asarray(sample))[::-1]
    k = max(1, math.ceil(len(z) * (1 - a))); v = z[k - 1]; es = z[:k].mean()
    x = np.where(z > v, z / es, 0.0)
    return float(x.mean()), float((x ** 2).mean())


def z2_null(m1, m2, T, a, level=0.95):
    """Mean, standard deviation and upper critical value of Z2 over T independent days, by the
    central limit theorem: Z2 = mean(X) / (1 - a) - 1."""
    mean = m1 / (1 - a) - 1
    sd = math.sqrt((m2 - m1 ** 2) / T) / (1 - a)
    return mean, sd, mean + stats.norm.ppf(level) * sd


def es_z2(loss, var, es, a):
    """Acerbi-Szekely Z2 = sum_t L_t I_t / (T (1-a) ES_t) - 1; zero in expectation if ES is right."""
    loss, var, es = map(np.asarray, (loss, var, es))
    I = loss > var
    return float(np.sum(loss * I / es) / (len(loss) * (1 - a)) - 1)

# ----------------------------------------------------------------------------- extreme values

def gpd_fit(exc):
    """Maximum likelihood generalised Pareto fit to exceedances over a threshold: (xi, beta)."""
    from scipy.stats import genpareto
    xi, _, beta = genpareto.fit(exc, floc=0)
    return xi, beta


def gpd_var_es(u, xi, beta, n, nu, a):
    """POT estimates of VaR and ES at confidence a with threshold u, n observations, nu exceedances."""
    var = u + beta / xi * (((n / nu) * (1 - a)) ** (-xi) - 1)
    es = (var + beta - xi * u) / (1 - xi)
    return var, es


def hill(losses, k):
    """Hill estimator of the tail index 1/xi from the k largest positive losses."""
    L = np.sort(np.asarray(losses))[::-1]
    return 1.0 / np.mean(np.log(L[:k]) - np.log(L[k]))

# ----------------------------------------------------------------------------- performance ratios

def sharpe(x):
    x = np.asarray(x); return x.mean() / x.std(ddof=1)


def sortino(r, mar=0.0):
    r = np.asarray(r); d = np.minimum(r - mar, 0.0)
    return (r.mean() - mar) / math.sqrt(np.mean(d * d))


def omega(r, L=0.0):
    r = np.asarray(r)
    return np.mean(np.maximum(r - L, 0)) / np.mean(np.maximum(L - r, 0))


def mppm(r, rf, rho=3.0, dt=1 / 12):
    """Manipulation-proof performance measure of Goetzmann, Ingersoll, Spiegel and Welch, annualised."""
    r, rf = np.asarray(r), np.asarray(rf)
    return 1 / ((1 - rho) * dt) * math.log(np.mean(((1 + r) / (1 + rf)) ** (1 - rho)))


def mppm_law(pdf, lo, hi, rho=3.0, dt=1 / 12, floor=-0.5):
    """MPPM of a return law with density pdf on (lo, hi), by quadrature, with zero risk free rate. Any
    law with mass at or below -100% has E[(1+r)^(1-rho)] infinite and a measure of minus infinity, so the
    law is truncated at a monthly return of floor and renormalised; a law bounded below floor is exact."""
    from scipy import integrate
    a = max(lo, floor)
    f = lambda r: (1 + r) ** (1 - rho) * pdf(r)
    num = sum(integrate.quad(f, u, v, limit=400, epsabs=1e-14)[0] for u, v in _pieces(a, hi))
    den = sum(integrate.quad(pdf, u, v, limit=400, epsabs=1e-14)[0] for u, v in _pieces(a, hi))
    return 1 / ((1 - rho) * dt) * math.log(num / den)


def _pieces(a, b, cuts=(-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2)):
    pts = [a] + [c for c in cuts if a < c < b] + [b]
    return list(zip(pts[:-1], pts[1:]))


def sharpe_se(theta, T, skew=0.0, exkurt=0.0):
    """Asymptotic standard error of an estimated Sharpe ratio (per period) for iid returns."""
    return math.sqrt((1 + 0.5 * theta ** 2 - skew * theta + 0.25 * exkurt * theta ** 2) / T)


def lo_annual_factor(x, q=12):
    """Lo's factor eta(q) mapping a per-period Sharpe ratio to q periods with autocorrelation:
    SR_q = eta * SR_1, eta = q / sqrt(q + 2 sum_k (q-k) rho_k)."""
    x = np.asarray(x) - np.mean(x)
    rho = [np.sum(x[k:] * x[:-k]) / np.sum(x * x) for k in range(1, q)]
    return q / math.sqrt(q + 2 * sum((q - k) * rho[k - 1] for k in range(1, q)))

# ----------------------------------------------------------------------------- drawdowns

def drawdowns(r):
    """Wealth from simple returns, the running peak and the drawdown 1 - W/peak."""
    W = np.cumprod(1 + np.asarray(r))
    P = np.maximum.accumulate(np.concatenate([[1.0], W]))[1:]
    return W, P, 1 - W / P


def max_drawdown(r):
    return float(drawdowns(r)[2].max())

# ----------------------------------------------------------------------------- smoothing

def smooth(r, phi):
    """Appraisal-style smoothing r*_t = (1-phi) r_t + phi r*_{t-1}, started at the first return."""
    r = np.asarray(r, float)
    out = signal.lfilter([1 - phi], [1, -phi], r, zi=[phi * r[0]])[0]
    return out


def unsmooth(rs, phi):
    """Geltner's inversion r_t = (r*_t - phi r*_{t-1}) / (1 - phi); the first return is dropped."""
    rs = np.asarray(rs, float)
    return (rs[1:] - phi * rs[:-1]) / (1 - phi)


def ar1(x):
    x = np.asarray(x) - np.mean(x)
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))
