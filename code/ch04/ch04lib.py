"""Library for Chapter 4: discount curves and forward rates, one-period no-arbitrage pricing,
forwards and futures, commodity term structures and roll returns, binomial trees.

All functions are deterministic given their inputs; simulations take a numpy Generator.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import optimize, stats

# ----------------------------------------------------------------------------- discount curves


def df_simple(y, t):
    """Discount factor for a simple (money market) annual rate y over t years."""
    return 1.0 / (1.0 + y * t)


def df_periodic(y, t, m):
    """Discount factor for a rate y compounded m times a year."""
    return (1.0 + y / m) ** (-m * t)


class Curve:
    """Discount curve through knots (t_i, D_i), log-linear in between (piecewise flat forward rates).
    The knot (0, 1) is always present; beyond the last knot the last forward rate is extended."""

    def __init__(self, t, D):
        t = np.concatenate([[0.0], np.asarray(t, float)]); D = np.concatenate([[1.0], np.asarray(D, float)])
        o = np.argsort(t); self.t = t[o]; self.lnD = np.log(D[o])

    def D(self, x):
        x = np.asarray(x, float)
        t, l = self.t, self.lnD
        out = np.interp(x, t, l)
        if len(t) > 1:
            slope = (l[-1] - l[-2]) / (t[-1] - t[-2])
            out = np.where(x > t[-1], l[-1] + slope * (x - t[-1]), out)
        return np.exp(out)

    def zero(self, x):
        """Continuously compounded zero rate."""
        x = np.asarray(x, float)
        return -np.log(self.D(x)) / x

    def fwd_simple(self, t1, t2):
        """Simple forward rate for the period [t1, t2]."""
        return (self.D(t1) / self.D(t2) - 1.0) / (t2 - t1)

    def fwd_inst(self, x):
        """Instantaneous forward rate (piecewise constant), taken on the right of each knot."""
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(self.t, x, side="right") - 1, 0, len(self.t) - 2)
        return -(self.lnD[i + 1] - self.lnD[i]) / (self.t[i + 1] - self.t[i])

    def par_yield(self, T, m=2):
        """Coupon rate, compounded m times a year, of a bond maturing at T that prices at par."""
        times = np.arange(1, int(round(T * m)) + 1) / m
        return m * (1 - self.D(T)) / self.D(times).sum()


def bootstrap(bills, bonds, m=2):
    """Bootstrap a Curve from money market bills {t: simple yield} and par bonds {T: par yield, m
    coupons a year}. Each bond adds one knot at its maturity, found by a one-dimensional root search,
    with log-linear interpolation for coupon dates between knots."""
    kt = sorted(bills); kD = [df_simple(bills[t], t) for t in kt]
    for T in sorted(bonds):
        c = bonds[T]; times = np.arange(1, int(round(T * m)) + 1) / m

        def f(DT):
            cv = Curve(kt + [T], kD + [DT])
            return c / m * cv.D(times).sum() + cv.D(T) - 1.0
        DT = optimize.brentq(f, 1e-6, 1.5, xtol=1e-15)
        kt.append(T); kD.append(DT)
    return Curve(kt, kD)


def bond_price(curve, c, T, m=2):
    times = np.arange(1, int(round(T * m)) + 1) / m
    return c / m * curve.D(times).sum() + curve.D(T)


# ----------------------------------------------------------------------------- one-period markets


def state_prices(payoff, prices):
    """Solve payoff.T @ psi = prices for a complete market (payoff is states x assets, square)."""
    return np.linalg.solve(np.asarray(payoff, float).T, np.asarray(prices, float))


def arbitrage_check(payoff, prices):
    """Return (no_arbitrage, psi_or_theta). Finds psi >= t with payoff.T psi = prices, maximising t.
    If the best t is positive a strictly positive state price vector exists. Otherwise an arbitrage
    portfolio theta is returned: payoff @ theta >= 0, prices @ theta <= 0, not identically zero."""
    A = np.asarray(payoff, float); p = np.asarray(prices, float); S, N = A.shape
    # variables (psi_1..psi_S, t); maximise t
    c = np.zeros(S + 1); c[-1] = -1
    A_ub = np.hstack([-np.eye(S), np.ones((S, 1))]); b_ub = np.zeros(S)
    A_eq = np.hstack([A.T, np.zeros((N, 1))])
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=p,
                           bounds=[(0, None)] * S + [(None, 1)], method="highs")
    if res.status == 0 and res.x[-1] > 1e-10:
        return True, res.x[:S]
    # look for an arbitrage: maximise sum(payoff theta) - prices theta with payoff theta >= 0, price <= 0
    c = -(A.sum(axis=0) - p)
    res = optimize.linprog(c, A_ub=np.vstack([-A, p[None, :]]), b_ub=np.zeros(S + 1),
                           bounds=[(-1, 1)] * N, method="highs")
    return False, res.x


def price_bounds(payoff, prices, claim):
    """No-arbitrage bounds for a claim: (inf, sup) of psi @ claim over psi >= 0 with payoff.T psi =
    prices, and the super- and sub-replication costs, which are equal to them by LP duality."""
    A = np.asarray(payoff, float); p = np.asarray(prices, float); X = np.asarray(claim, float)
    S, N = A.shape
    lo = optimize.linprog(X, A_eq=A.T, b_eq=p, bounds=[(0, None)] * S, method="highs").fun
    hi = -optimize.linprog(-X, A_eq=A.T, b_eq=p, bounds=[(0, None)] * S, method="highs").fun
    sup_rep = optimize.linprog(p, A_ub=-A, b_ub=-X, bounds=[(None, None)] * N, method="highs").fun
    sub_rep = -optimize.linprog(-p, A_ub=A, b_ub=X, bounds=[(None, None)] * N, method="highs").fun
    return lo, hi, sub_rep, sup_rep


# ----------------------------------------------------------------------------- forwards and futures


def forward_price(S, r, T, q=0.0, u=0.0, y=0.0):
    """Cost of carry forward price with continuous rate r, income yield q, storage cost u and
    convenience yield y."""
    return S * math.exp((r + u - q - y) * T)


def forward_band(S_bid, S_ask, r_lend, r_borrow, T, q=0.0):
    """No-arbitrage band for a forward price when borrowing costs more than lending and the spot has
    a bid-ask spread: reverse cash and carry gives the lower end, cash and carry the upper end."""
    return S_bid * math.exp((r_lend - q) * T), S_ask * math.exp((r_borrow - q) * T)


def cip_forward(S, i_dom, i_for, tau):
    """Covered interest parity forward (domestic per foreign) with simple money market rates."""
    return S * (1 + i_dom * tau) / (1 + i_for * tau)


def vasicek_B(kappa, tau):
    return (1 - np.exp(-kappa * tau)) / kappa


def fut_fwd_logratio(kappa, sig_r, sig_S, rho, T):
    """ln(futures / forward) for an asset with volatility sig_S, correlation rho with a Vasicek short
    rate (speed kappa, volatility sig_r), no dividends:  v_I + rho sig_S sig_r int_0^T B(u,T) du,
    where v_I = sig_r^2 int_0^T B(u,T)^2 du is the variance of the integrated short rate."""
    k = kappa
    intB = (T - vasicek_B(k, T)) / k
    intB2 = (T - 2 * vasicek_B(k, T) + vasicek_B(2 * k, T)) / k ** 2
    return sig_r ** 2 * intB2 + rho * sig_S * sig_r * intB


def fut_fwd_mc(kappa, theta, r0, sig_r, sig_S, rho, T, steps, paths, rng, chunk=20000):
    """Monte Carlo futures and forward prices (S0 = 1) under the risk-neutral measure. The short rate
    moves by its exact Gaussian transition, its integral by the trapezoid rule, the asset by the
    exact lognormal step given the rate path. Returns futures, forward and the standard error of
    ln(futures / forward) from the delta method."""
    h = T / steps; e = math.exp(-kappa * h); sd = sig_r * math.sqrt((1 - e * e) / (2 * kappa))
    sS, sD, sSS, sDD, sSD, sDS2, sDSD = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sums = np.zeros(5); sq = np.zeros((5, 5)); n = 0
    left = paths
    while left > 0:
        m = min(chunk, left); left -= m
        r = np.full(m, r0); I = np.zeros(m); lnS = np.zeros(m)
        for _ in range(steps):
            z1 = rng.standard_normal(m); z2 = rng.standard_normal(m)
            rn = theta + (r - theta) * e + sd * z1
            inc = 0.5 * (r + rn) * h
            zS = rho * z1 + math.sqrt(1 - rho * rho) * z2
            lnS += inc - 0.5 * sig_S ** 2 * h + sig_S * math.sqrt(h) * zS
            I += inc; r = rn
        S = np.exp(lnS); D = np.exp(-I)
        X = np.vstack([S, D * S, D])
        sums[:3] += X.sum(axis=1); sq[:3, :3] += X @ X.T; n += m
    mean = sums[:3] / n; cov = sq[:3, :3] / n - np.outer(mean, mean)
    fut = mean[0]; fwd = mean[1] / mean[2]
    g = np.array([1 / mean[0], -1 / mean[1], 1 / mean[2]])     # gradient of ln(fut) - ln(DS) + ln(D)
    se = math.sqrt(g @ cov @ g / n)
    return fut, fwd, se


# ----------------------------------------------------------------------------- commodity term structure


def schwartz_futures(lnS, tau, kappa, alpha_star, sigma):
    """Futures price in the one-factor model of Schwartz (1997): ln S is Ornstein-Uhlenbeck with
    speed kappa, risk-neutral long-run mean alpha_star and volatility sigma."""
    e = np.exp(-kappa * tau)
    return np.exp(e * lnS + (1 - e) * alpha_star + sigma ** 2 / (4 * kappa) * (1 - e * e))


def schwartz_simulate(lnS0, kappa, alpha, sigma, h, n, rng):
    """Exact simulation of ln S under the physical measure (long-run mean alpha) on a grid of step h."""
    e = math.exp(-kappa * h); sd = sigma * math.sqrt((1 - e * e) / (2 * kappa))
    x = np.empty(n + 1); x[0] = lnS0
    z = rng.standard_normal(n)
    for k in range(n):
        x[k + 1] = alpha + (x[k] - alpha) * e + sd * z[k]
    return x


def roll_decomposition(x, h, tau_in, kappa, alpha_star, sigma):
    """Hold a futures contract with tau_in years to expiry for one step of length h, then roll. Returns
    per-step arrays: futures log excess return, spot log return and roll return, which add exactly:
    ln F_{t+h}(T)/F_t(T) = [ln S_{t+h} - ln S_t] + [b_{t+h}(T) - b_t(T)], with b = ln F - ln S."""
    b_in = np.log(schwartz_futures(x[:-1], tau_in, kappa, alpha_star, sigma)) - x[:-1]
    b_out = np.log(schwartz_futures(x[1:], tau_in - h, kappa, alpha_star, sigma)) - x[1:]
    spot = np.diff(x); roll = b_out - b_in
    return spot + roll, spot, roll


def schwartz_basis(lnS, tau, kappa, alpha_star, sigma):
    return np.log(schwartz_futures(lnS, tau, kappa, alpha_star, sigma)) - lnS


# ----------------------------------------------------------------------------- binomial trees


def bs_price(S, K, r, sigma, T, kind="call", q=0.0):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T)); d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * math.exp(-q * T) * stats.norm.cdf(-d1)


def crr_params(r, sigma, T, n, q=0.0):
    h = T / n; u = math.exp(sigma * math.sqrt(h)); d = 1 / u
    p = (math.exp((r - q) * h) - d) / (u - d)
    return u, d, p, math.exp(-r * h)


def _pp(z, n):
    """Peizer-Pratt method 2 inversion, used by the Leisen-Reimer tree (n odd)."""
    return 0.5 + math.copysign(1.0, z) * 0.5 * math.sqrt(
        1 - math.exp(-((z / (n + 1 / 3 + 0.1 / (n + 1))) ** 2) * (n + 1 / 6)))


def lr_params(S, K, r, sigma, T, n, q=0.0):
    """Leisen and Reimer (1996) tree: probabilities matched to d1 and d2 of the Black-Scholes formula."""
    assert n % 2 == 1
    h = T / n; d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T)); d2 = d1 - sigma * math.sqrt(T)
    p = _pp(d2, n); ps = _pp(d1, n); g = math.exp((r - q) * h)
    u = g * ps / p; d = (g - p * u) / (1 - p)
    return u, d, p, math.exp(-r * h)


def tree_european(S, K, n, params, kind="call"):
    """European price on a recombining tree with parameters (u, d, p, disc), as a binomial sum."""
    u, d, p, disc = params
    j = np.arange(n + 1)
    ST = S * u ** j * d ** (n - j)
    pay = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
    return disc ** n * float(np.sum(stats.binom.pmf(j, n, p) * pay))


def tree_american(S, K, n, params, kind="put", boundary=False):
    """American price by backward induction (the Snell envelope). With boundary=True also returns,
    for each step i < n, the highest stock price at which the put is exercised (nan if none)."""
    u, d, p, disc = params
    j = np.arange(n + 1)
    ST = S * u ** j * d ** (n - j)
    V = np.maximum(K - ST, 0) if kind == "put" else np.maximum(ST - K, 0)
    bnd = np.full(n, np.nan)
    for i in range(n - 1, -1, -1):
        Si = S * u ** j[:i + 1] * d ** (i - j[:i + 1])
        cont = disc * (p * V[1:i + 2] + (1 - p) * V[:i + 1])
        ex = np.maximum(K - Si, 0) if kind == "put" else np.maximum(Si - K, 0)
        V = np.maximum(cont, ex)
        if boundary:
            hit = (ex > cont + 1e-12) & (ex > 0)
            if hit.any():
                bnd[i] = Si[hit].max()
    return (float(V[0]), bnd) if boundary else float(V[0])


def tree_delta(S, K, n, params, kind="call"):
    """Hedge ratio at the root: (V_u - V_d) / (S u - S d)."""
    u, d, p, disc = params
    Vu = tree_european(S * u, K, n - 1, params, kind); Vd = tree_european(S * d, K, n - 1, params, kind)
    return (Vu - Vd) / (S * (u - d))


def perpetual_put_boundary(K, r, sigma):
    """Exercise boundary of the perpetual American put: K gamma / (1 + gamma), gamma = 2r / sigma^2."""
    g = 2 * r / sigma ** 2
    return K * g / (1 + g)


def ols_slope(x, y):
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])
