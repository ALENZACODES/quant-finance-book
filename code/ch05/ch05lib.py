"""Library for Chapter 5: the Black-Scholes formula and its Greeks, dividends, Black-76 and
Garman-Kohlhagen, Monte Carlo with antithetic and control variates, finite difference schemes,
implied volatility, jump-diffusion and stochastic volatility prices, and delta hedging.

All functions are deterministic given their inputs; simulations take a numpy Generator.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import integrate, linalg, optimize, special, stats

SQ2PI = math.sqrt(2 * math.pi)


def ncdf(x):
    return special.ndtr(x)


def npdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


# ----------------------------------------------------------------------------- closed forms

def d12(S, K, r, q, sig, T):
    v = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q) * T) / v + 0.5 * v
    return d1, d1 - v


def bs(S, K, r, sig, T, kind="call", q=0.0):
    """Black-Scholes-Merton price with continuous yield q. Works on arrays."""
    S, K, T = np.asarray(S, float), np.asarray(K, float), np.asarray(T, float)
    d1, d2 = d12(S, K, r, q, sig, T)
    Sq, Kr = S * np.exp(-q * T), K * np.exp(-r * T)
    if kind == "call":
        return Sq * ncdf(d1) - Kr * ncdf(d2)
    return Kr * ncdf(-d2) - Sq * ncdf(-d1)


def greeks(S, K, r, sig, T, kind="call", q=0.0):
    """Analytic Greeks. Theta is the derivative in calendar time t (minus the derivative in T)."""
    d1, d2 = d12(S, K, r, q, sig, T)
    eq, er, sT = np.exp(-q * T), np.exp(-r * T), np.sqrt(T)
    pdf = npdf(d1)
    gamma = eq * pdf / (S * sig * sT)
    vega = S * eq * pdf * sT
    vanna = -eq * pdf * d2 / sig
    volga = vega * d1 * d2 / sig
    if kind == "call":
        delta = eq * ncdf(d1)
        theta = -S * eq * pdf * sig / (2 * sT) + q * S * eq * ncdf(d1) - r * K * er * ncdf(d2)
        rho = K * T * er * ncdf(d2)
    else:
        delta = -eq * ncdf(-d1)
        theta = -S * eq * pdf * sig / (2 * sT) - q * S * eq * ncdf(-d1) + r * K * er * ncdf(-d2)
        rho = -K * T * er * ncdf(-d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, vanna=vanna, volga=volga)


def black76(F, K, r, sig, T, kind="call"):
    """Black (1976) price of an option on a futures or forward price F, paid at T."""
    return math.exp(-r * T) * bs(F, K, 0.0, sig, T, kind)


def gk(S, K, rd, rf, sig, T, kind="call"):
    """Garman-Kohlhagen price of a currency option: domestic units per unit of foreign currency."""
    return bs(S, K, rd, sig, T, kind, q=rf)


def escrowed(S, K, r, sig, T, divs, kind="call"):
    """Discrete cash dividends [(time, amount)] handled by subtracting their present value from S."""
    pv = sum(D * math.exp(-r * t) for t, D in divs if t < T)
    return bs(S - pv, K, r, sig, T, kind)


# ----------------------------------------------------------------------------- Monte Carlo

def mc_european(S, K, r, sig, T, n, rng, method="plain", kind="call", q=0.0):
    """Monte Carlo price. method: plain, anti (antithetic pairs), cv (control variate e^{-rT} S_T),
    anti_cv (both). n is the number of payoff evaluations. Returns (estimate, standard error)."""
    disc = math.exp(-r * T); drift = (r - q - 0.5 * sig ** 2) * T; vol = sig * math.sqrt(T)

    def pay(z):
        ST = S * np.exp(drift + vol * z)
        p = np.maximum(ST - K, 0) if kind == "call" else np.maximum(K - ST, 0)
        return disc * p, disc * ST

    if method in ("plain", "cv"):
        Y, X = pay(rng.standard_normal(n))
    else:
        z = rng.standard_normal(n // 2)
        Y1, X1 = pay(z); Y2, X2 = pay(-z)
        Y, X = 0.5 * (Y1 + Y2), 0.5 * (X1 + X2)
    if method in ("cv", "anti_cv"):
        EX = S * math.exp(-q * T)
        c = np.cov(Y, X); b = c[0, 1] / c[1, 1]
        Y = Y - b * (X - EX)
    return float(Y.mean()), float(Y.std(ddof=1) / math.sqrt(len(Y)))


def mc_delta_pathwise(S, K, r, sig, T, n, rng, q=0.0):
    """Pathwise estimator of the call delta: e^{-rT} 1{S_T > K} S_T / S."""
    ST = S * np.exp((r - q - 0.5 * sig ** 2) * T + sig * math.sqrt(T) * rng.standard_normal(n))
    Y = math.exp(-r * T) * (ST > K) * ST / S
    return float(Y.mean()), float(Y.std(ddof=1) / math.sqrt(n))


def mc_delta_lr(S, K, r, sig, T, n, rng, q=0.0):
    """Likelihood ratio estimator of the call delta: e^{-rT}(S_T-K)^+ Z / (S sigma sqrt T)."""
    z = rng.standard_normal(n)
    ST = S * np.exp((r - q - 0.5 * sig ** 2) * T + sig * math.sqrt(T) * z)
    Y = math.exp(-r * T) * np.maximum(ST - K, 0) * z / (S * sig * math.sqrt(T))
    return float(Y.mean()), float(Y.std(ddof=1) / math.sqrt(n))


# ----------------------------------------------------------------------------- finite differences

def fd_theta(S0, K, r, sig, T, M, Nt, theta=0.5, kind="call", q=0.0, width=6.0, rannacher=0,
             return_grid=False):
    """theta-scheme for the Black-Scholes equation in x = ln S on a uniform grid with M intervals and
    Nt time steps. theta = 0 explicit, 1 implicit, 1/2 Crank-Nicolson. The grid is centred on ln S0
    and spans width * sig * sqrt(T) on each side. The first `rannacher` steps are each replaced by two
    fully implicit half steps. Dirichlet values at both ends. Returns V(S0) (and the grid if asked)."""
    assert M % 2 == 0
    L = width * sig * math.sqrt(T)
    x = math.log(S0) + np.linspace(-L, L, M + 1); dx = x[1] - x[0]; S = np.exp(x)
    a, b = 0.5 * sig ** 2, r - q - 0.5 * sig ** 2
    lo, di, up = a / dx ** 2 - b / (2 * dx), -2 * a / dx ** 2 - r, a / dx ** 2 + b / (2 * dx)
    V = np.maximum(S - K, 0) if kind == "call" else np.maximum(K - S, 0)

    def bc(tau):
        if kind == "call":
            return 0.0, S[-1] * math.exp(-q * tau) - K * math.exp(-r * tau)
        return K * math.exp(-r * tau) - S[0] * math.exp(-q * tau), 0.0

    def step(V, tau0, dt, th):
        n = M - 1
        ab = np.zeros((3, n))
        ab[0, 1:] = -th * dt * up; ab[1, :] = 1 - th * dt * di; ab[2, :-1] = -th * dt * lo
        Vi = V[1:-1]
        rhs = Vi + (1 - th) * dt * (lo * V[:-2] + di * Vi + up * V[2:])
        b0, b1 = bc(tau0 + dt)
        rhs[0] += th * dt * lo * b0; rhs[-1] += th * dt * up * b1
        Vn = np.empty_like(V); Vn[0], Vn[-1] = b0, b1
        Vn[1:-1] = rhs if th == 0 else linalg.solve_banded((1, 1), ab, rhs)
        return Vn

    dt = T / Nt; tau = 0.0
    for k in range(Nt):
        if k < rannacher:
            V = step(V, tau, dt / 2, 1.0); V = step(V, tau + dt / 2, dt / 2, 1.0)
        else:
            V = step(V, tau, dt, theta)
        tau += dt
    v0 = float(V[M // 2])
    if return_grid:
        return v0, S, V, dx
    return v0


def fd_gamma(S, V, dx):
    """Gamma on the interior of a log grid: (V_xx - V_x) / S^2."""
    Vx = (V[2:] - V[:-2]) / (2 * dx); Vxx = (V[2:] - 2 * V[1:-1] + V[:-2]) / dx ** 2
    return S[1:-1], (Vxx - Vx) / S[1:-1] ** 2


def amplification(lam, w_s, omega):
    """von Neumann amplification factor A(omega) of the weighted scheme with weight w_s for u_t = a u_yy,
    lam = a dt/dy^2."""
    s = np.sin(omega / 2) ** 2
    return (1 - 4 * (1 - w_s) * lam * s) / (1 + 4 * w_s * lam * s)


# ----------------------------------------------------------------------------- implied volatility

def _norm_call(x, v):
    """Call price divided by the discounted forward, as a function of x = ln(F/K) and v = sig sqrt T."""
    return ncdf(x / v + v / 2) - math.exp(-x) * ncdf(x / v - v / 2)


def _norm_otm(x, v):
    """Price of the out-of-the-money option divided by the discounted forward: the call when
    x = ln(F/K) <= 0, the put when x > 0. Both have the same derivatives in v as the call."""
    if x <= 0:
        return _norm_call(x, v)
    return math.exp(-x) * ncdf(-(x / v - v / 2)) - ncdf(-(x / v + v / 2))


def iv_newton(price, S, K, r, T, kind="call", q=0.0, tol=1e-12, maxit=50):
    """Newton's method in v = sig sqrt(T) from the inflection point v* = sqrt(2|ln(F/K)|), applied to
    the out-of-the-money option (put-call parity converts the quote if needed).
    Returns (sigma, iterations, list of iterates in v)."""
    F = S * math.exp((r - q) * T); DF = math.exp(-r * T); x = math.log(F / K)
    otm = "call" if x <= 0 else "put"
    if kind != otm:                                             # put-call parity
        price = price - DF * (F - K) if kind == "call" else price + DF * (F - K)
    target = price / (DF * F)
    v = math.sqrt(2 * abs(x)) if x != 0 else 0.0
    path = [v]
    for it in range(1, maxit + 1):
        f = (_norm_otm(x, v) if v > 0 else 0.0) - target
        fp = float(npdf(x / v + v / 2)) if v > 0 else float(npdf(0.0))
        v_new = v - f / fp; path.append(v_new)
        if not v_new > 0:
            raise RuntimeError("Newton left the domain")
        if abs(v_new - v) < tol * max(1.0, v):
            return v_new / math.sqrt(T), it, path
        v = v_new
    raise RuntimeError("Newton did not converge")


def iv_brent(price, S, K, r, T, kind="call", q=0.0, lo=1e-4, hi=5.0, xtol=1e-14):
    """Brent's method on [lo, hi]. Returns (sigma, iterations, function calls)."""
    f = lambda s: float(bs(S, K, r, s, T, kind, q)) - price
    root, res = optimize.brentq(f, lo, hi, xtol=xtol, rtol=4 * np.finfo(float).eps, full_output=True)
    return root, res.iterations, res.function_calls


def iv_bounds(S, K, r, T, kind="call", q=0.0):
    """No-arbitrage bounds on a European price: inside them an implied volatility exists."""
    Sq, Kr = S * math.exp(-q * T), K * math.exp(-r * T)
    return (max(Sq - Kr, 0), Sq) if kind == "call" else (max(Kr - Sq, 0), Kr)


def brenner_subrahmanyam(price, S, T):
    """At-the-money-forward approximation sig = price sqrt(2 pi) / (S sqrt T)."""
    return price * SQ2PI / (S * math.sqrt(T))


# ----------------------------------------------------------------------------- smiles: jumps and stochastic volatility

def merton_jd(S, K, r, sig, T, lam, muJ, dJ, kind="call", nmax=80):
    """Merton (1976) jump-diffusion price: lognormal jumps ln(1+J) ~ N(muJ, dJ^2) at rate lam."""
    k = math.exp(muJ + 0.5 * dJ ** 2) - 1; lp = lam * (1 + k) * T
    tot = 0.0
    for n in range(nmax):
        w = math.exp(-lp + n * math.log(lp) - math.lgamma(n + 1)) if lp > 0 else float(n == 0)
        sn = math.sqrt(sig ** 2 + n * dJ ** 2 / T); rn = r - lam * k + n * math.log(1 + k) / T
        tot += w * bs(S, K, rn, sn, T, kind)
    return float(tot)


def heston_cf(u, S, r, q, T, v0, kappa, theta, xi, rho):
    """Characteristic function of ln S_T in the Heston model, in the form that avoids branch jumps."""
    iu = 1j * u
    d = np.sqrt((rho * xi * iu - kappa) ** 2 + xi ** 2 * (iu + u ** 2))
    g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)
    e = np.exp(-d * T)
    C = (kappa * theta / xi ** 2) * ((kappa - rho * xi * iu - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (v0 / xi ** 2) * (kappa - rho * xi * iu - d) * (1 - e) / (1 - g * e)
    return np.exp(iu * (math.log(S) + (r - q) * T) + C + D)


def heston(S, K, r, T, v0, kappa, theta, xi, rho, kind="call", q=0.0):
    """Heston (1993) price by the single-integral formula of Lewis (2000):
    C = DF [F - sqrt(FK)/pi int_0^inf Re(e^{iuk} psi(u - i/2)) / (u^2 + 1/4) du],
    with k = ln(F/K) and psi the characteristic function of ln(S_T/F)."""
    F = S * math.exp((r - q) * T); DF = math.exp(-r * T); k = math.log(F / K); lF = math.log(F)
    psi = lambda u: heston_cf(u, S, r, q, T, v0, kappa, theta, xi, rho) * np.exp(-1j * u * lF)
    f = lambda u: (np.exp(1j * u * k) * psi(u - 0.5j)).real / (u * u + 0.25)
    val = integrate.quad(f, 0, np.inf, limit=1000, epsabs=1e-14, epsrel=1e-12)[0]
    c = DF * (F - math.sqrt(F * K) / math.pi * val)
    return c if kind == "call" else c - DF * (F - K)


def heston_cdf(S, K, r, T, v0, kappa, theta, xi, rho, q=0.0):
    """Risk-neutral probability Q(S_T <= K) in the Heston model by the Gil-Pelaez inversion
    1/2 - (1/pi) int_0^inf Im(e^{-iu ln K} phi(u)) / u du, with phi the characteristic function of ln S_T."""
    lk = math.log(K)
    f = lambda u: (np.exp(-1j * u * lk) * heston_cf(u, S, r, q, T, v0, kappa, theta, xi, rho)).imag / u
    val = integrate.quad(f, 1e-12, np.inf, limit=1000, epsabs=1e-14, epsrel=1e-12)[0]
    return 0.5 - val / math.pi


def lewis_bs(S, K, r, sig, T):
    """The same formula with a Gaussian characteristic function: a check that equals Black-Scholes."""
    F = S * math.exp(r * T); DF = math.exp(-r * T); k = math.log(F / K)
    psi = lambda u: np.exp(-0.5 * sig ** 2 * T * (1j * u + u * u))
    f = lambda u: (np.exp(1j * u * k) * psi(u - 0.5j)).real / (u * u + 0.25)
    val = integrate.quad(f, 0, np.inf, limit=1000, epsabs=1e-14, epsrel=1e-12)[0]
    return DF * (F - math.sqrt(F * K) / math.pi * val)


def heston_mc(S, K, r, T, v0, kappa, theta, xi, rho, n, steps, rng):
    """Full truncation Euler scheme for the Heston model (log price exact given the variance path)."""
    dt = T / steps; x = np.full(n, math.log(S)); v = np.full(n, v0)
    for _ in range(steps):
        z1 = rng.standard_normal(n); z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.standard_normal(n)
        vp = np.maximum(v, 0)
        x += (r - 0.5 * vp) * dt + np.sqrt(vp * dt) * z1
        v += kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    Y = math.exp(-r * T) * np.maximum(np.exp(x) - K, 0)
    return float(Y.mean()), float(Y.std(ddof=1) / math.sqrt(n))


# ----------------------------------------------------------------------------- hedging

def hedge_sim(S0, K, r, sig_true, sig_hedge, T, steps, n, rng, mu=None, q=0.0, chunk=20000,
              keep_paths=0, sell_sig=None, cost=0.0):
    """Sell one call at the Black-Scholes price with sig_hedge, hedge with the sig_hedge delta at
    `steps` equally spaced dates, and close at T. The stock follows GBM with drift mu (default r) and
    volatility sig_true. Returns discounted P&L per path and the discrete gamma-theta sum
    sum_k e^{-r t_k} 1/2 Gamma_k S_k^2 (sig_hedge^2 dt - (dS/S)^2).
    sell_sig: volatility used for the sale price (default sig_hedge). cost: proportional cost on the
    value of every share traded, including the initial hedge (the position is settled at expiry)."""
    mu = r if mu is None else mu
    dt = T / steps; out_pnl, out_gt = [], []
    kept = None
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        S = np.full(m, float(S0))
        V0 = float(bs(S0, K, r, sig_hedge if sell_sig is None else sell_sig, T, "call", q))
        delta = float(greeks(S0, K, r, sig_hedge, T, "call", q)["delta"])
        cash = np.full(m, V0 - delta * S0 - cost * abs(delta) * S0); D = np.full(m, delta); gt = np.zeros(m)
        if keep_paths and kept is None:
            kept = np.empty((steps + 1, keep_paths)); kept[0] = S0
        for k in range(steps):
            tau = T - k * dt
            G = greeks(S, K, r, sig_hedge, tau, "call", q)["gamma"] if k > 0 else \
                float(greeks(S0, K, r, sig_hedge, T, "call", q)["gamma"])
            z = rng.standard_normal(m)
            Sn = S * np.exp((mu - q - 0.5 * sig_true ** 2) * dt + sig_true * math.sqrt(dt) * z)
            ret = Sn / S - 1
            gt += math.exp(-r * k * dt) * 0.5 * G * S ** 2 * (sig_hedge ** 2 * dt - ret ** 2)
            cash = cash * math.exp(r * dt) + D * Sn * (math.exp(q * dt) - 1)
            S = Sn
            if keep_paths and start == 0:
                kept[k + 1] = S[:keep_paths]
            if k < steps - 1:
                Dn = greeks(S, K, r, sig_hedge, T - (k + 1) * dt, "call", q)["delta"]
                cash -= (Dn - D) * S + cost * np.abs(Dn - D) * S; D = Dn
        pnl = (cash + D * S - np.maximum(S - K, 0)) * math.exp(-r * T)
        out_pnl.append(pnl); out_gt.append(gt)
    res = np.concatenate(out_pnl), np.concatenate(out_gt)
    return (res + (kept,)) if keep_paths else res


def gamma2_moment(S0, K, r, sig, t, T, mu, q=0.0):
    """E[(Gamma_t S_t^2)^2] in closed form when ln S_t ~ N(M, s^2): with v = sig^2 tau / 2,
    E = e^{-2 q tau} / (2 pi sig^2 tau) * sqrt(2 pi v) e^{2M + 2s^2} n(M + 2s^2; c, s^2 + v)."""
    tau = T - t; s2 = sig ** 2 * t; v = 0.5 * sig ** 2 * tau
    M = math.log(S0) + (mu - q - 0.5 * sig ** 2) * t
    c = math.log(K) - (r - q + 0.5 * sig ** 2) * tau
    var = s2 + v; a = M + 2 * s2
    dens = math.exp(-0.5 * (a - c) ** 2 / var) / math.sqrt(2 * math.pi * var)
    return math.exp(-2 * q * tau) / (2 * math.pi * sig ** 2 * tau) * math.sqrt(2 * math.pi * v) * \
        math.exp(2 * M + 2 * s2) * dens


def hedge_sd_theory(S0, K, r, sig, T, steps, mu=None, q=0.0):
    """Leading-order standard deviation of the discrete hedging error with the correct volatility:
    sqrt( dt/2 int_0^T e^{-2rt} E[Gamma_t^2 S_t^4] sig^4 dt ), the expectation under drift mu.
    The substitution t = T - u^2 removes the integrable singularity at expiry."""
    mu = r if mu is None else mu
    f = lambda u: 2 * u * math.exp(-2 * r * (T - u * u)) * sig ** 4 * \
        gamma2_moment(S0, K, r, sig, T - u * u, T, mu, q)
    val = integrate.quad(f, 0, math.sqrt(T), limit=400, epsabs=1e-14, epsrel=1e-11)[0]
    return math.sqrt(T / steps / 2 * val)


def ols_slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])
