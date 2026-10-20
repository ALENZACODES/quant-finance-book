"""Library for Chapter 3: random walks, Brownian motion, Ito calculus, SDEs, martingales, Girsanov and
Feynman-Kac. Pure numpy and scipy; every function is tested in code/ch03/test_*.py."""
from __future__ import annotations
import math
import numpy as np
from scipy import stats
from scipy.linalg import solve_banded

# ----------------------------------------------------------------------------- random walks
def random_walk(n, m, rng, p=0.5):
    """m paths of a simple random walk with n steps of +1 (prob p) or -1, starting at 0.
    Returns an (m, n+1) integer array."""
    steps = np.where(rng.uniform(size=(m, n)) < p, 1, -1).astype(np.int64)
    return np.concatenate([np.zeros((m, 1), np.int64), np.cumsum(steps, axis=1)], axis=1)


def srw_max_tail(n, a):
    """Exact P(max_{k<=n} S_k >= a) for the symmetric simple random walk, a >= 1 integer, from the
    reflection principle: P(S_n >= a) + P(S_n > a)."""
    k = np.arange(n + 1)                       # number of up steps; S_n = 2k - n
    pk = stats.binom.pmf(k, n, 0.5)
    s = 2 * k - n
    return float(pk[s >= a].sum() + pk[s > a].sum())


def srw_max_tail_enumerate(n, a):
    """Brute force over all 2^n paths (n <= 20), for the tests."""
    paths = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    mx = np.maximum(np.cumsum(paths, axis=1).max(axis=1), 0)
    return float(np.mean(mx >= a))


# ----------------------------------------------------------------------------- Brownian motion
def bm_paths(n, m, T, rng):
    """m Brownian paths on n equal steps of [0, T]; returns times (n+1,) and paths (m, n+1)."""
    h = T / n
    dW = rng.standard_normal((m, n)) * math.sqrt(h)
    W = np.concatenate([np.zeros((m, 1)), np.cumsum(dW, axis=1)], axis=1)
    return np.linspace(0, T, n + 1), W


def levy_bm(levels, T, rng, m=1, record=False):
    """Levy's construction: W(T) = sqrt(T) Z, then repeatedly fill in midpoints with
    W(mid) = (W(left) + W(right)) / 2 + sqrt(h / 4) Z, h = interval length.
    Returns times and paths on the dyadic grid with 2^levels steps; with record=True also the list
    of coarser approximations (piecewise linear interpolants are the Levy-Ciesielski partial sums)."""
    W = np.zeros((m, 2)); W[:, 1] = math.sqrt(T) * rng.standard_normal(m)
    hist = [W.copy()]
    for lev in range(1, levels + 1):
        h = T / 2 ** (lev - 1)
        mid = 0.5 * (W[:, :-1] + W[:, 1:]) + math.sqrt(h / 4) * rng.standard_normal((m, W.shape[1] - 1))
        new = np.empty((m, 2 * W.shape[1] - 1)); new[:, ::2] = W; new[:, 1::2] = mid
        W = new
        if record:
            hist.append(W.copy())
    t = np.linspace(0, T, W.shape[1])
    return (t, W, hist) if record else (t, W)


def quadratic_variation(W):
    return np.sum(np.diff(W, axis=-1) ** 2, axis=-1)


def total_variation(W):
    return np.sum(np.abs(np.diff(W, axis=-1)), axis=-1)


def bm_max_tail(a, T, sigma=1.0):
    """P(max_{t<=T} sigma W_t >= a) = 2 P(sigma W_T >= a), a > 0."""
    return float(2 * stats.norm.sf(a / (sigma * math.sqrt(T))))


from scipy.special import zeta as _zeta  # noqa: E402
BGK_BETA = -float(_zeta(0.5)) / math.sqrt(2 * math.pi)   # 0.5826...


def bm_max_tail_discrete(a, T, n, sigma=1.0):
    """Broadie-Glasserman-Kou continuity correction: a maximum monitored at n dates behaves like a
    continuous maximum with the barrier shifted up by beta * sigma * sqrt(T / n)."""
    return bm_max_tail(a + BGK_BETA * sigma * math.sqrt(T / n), T, sigma)


def first_passage_pdf(t, a):
    """Density of the first time standard Brownian motion hits a > 0."""
    t = np.asarray(t, float)
    return a / np.sqrt(2 * np.pi * t ** 3) * np.exp(-a ** 2 / (2 * t))


# ----------------------------------------------------------------------------- Ito integrals
def ito_sum(f_vals, W):
    """Left-point sums sum f(t_i) (W_{i+1} - W_i) along the last axis."""
    return np.sum(f_vals[..., :-1] * np.diff(W, axis=-1), axis=-1)


def ito_strat_ww(W_fine):
    """Given a path on 2n steps, return the Ito (left point) and Stratonovich (midpoint in time)
    sums for int W dW on the coarse grid of n steps."""
    Wc = W_fine[..., ::2]; Wm = W_fine[..., 1::2]
    dW = np.diff(Wc, axis=-1)
    return np.sum(Wc[..., :-1] * dW, axis=-1), np.sum(Wm * dW, axis=-1)


# ----------------------------------------------------------------------------- SDEs
def gbm_exact(S0, mu, sigma, t, W):
    return S0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * W)


def gbm_moments(S0, mu, sigma, T):
    m = S0 * math.exp(mu * T)
    v = S0 ** 2 * math.exp(2 * mu * T) * (math.exp(sigma ** 2 * T) - 1)
    med = S0 * math.exp((mu - 0.5 * sigma ** 2) * T)
    mode = S0 * math.exp((mu - 1.5 * sigma ** 2) * T)
    return m, v, med, mode


def gbm_schemes(S0, mu, sigma, T, dW):
    """Euler-Maruyama and Milstein terminal values for GBM driven by increments dW (m, n)."""
    h = T / dW.shape[1]
    e = np.full(dW.shape[0], float(S0)); mil = e.copy()
    for k in range(dW.shape[1]):
        z = dW[:, k]
        e = e * (1 + mu * h + sigma * z)
        mil = mil * (1 + mu * h + sigma * z + 0.5 * sigma ** 2 * (z ** 2 - h))
    return e, mil


def euler_gbm_mean(S0, mu, T, n):
    """Exact mean of the Euler scheme for GBM with n steps: S0 (1 + mu h)^n."""
    return S0 * (1 + mu * T / n) ** n


def euler_gbm_second(S0, mu, sigma, T, n):
    h = T / n
    return S0 ** 2 * ((1 + mu * h) ** 2 + sigma ** 2 * h) ** n


def ou_moments(x0, kappa, theta, sigma, t):
    e = np.exp(-kappa * np.asarray(t, float))
    mean = theta + (x0 - theta) * e
    var = sigma ** 2 * (1 - e ** 2) / (2 * kappa)
    return mean, var


def ou_exact(x0, kappa, theta, sigma, dt, n, m, rng):
    """Exact simulation of the Ornstein-Uhlenbeck process on n steps of length dt."""
    phi = math.exp(-kappa * dt); sd = sigma * math.sqrt((1 - phi ** 2) / (2 * kappa))
    X = np.empty((m, n + 1)); X[:, 0] = x0
    z = rng.standard_normal((m, n))
    for k in range(n):
        X[:, k + 1] = theta + phi * (X[:, k] - theta) + sd * z[:, k]
    return X


def ou_fit(x, dt):
    """Fit an OU process to equally spaced observations by OLS of x_{k+1} on x_k (the exact AR(1)
    discretisation). Returns dict with phi, c, resid sd, kappa, theta, sigma, half_life."""
    x = np.asarray(x, float)
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    beta, *_ = np.linalg.lstsq(X, x[1:], rcond=None)
    c, phi = beta
    res = x[1:] - X @ beta
    s = math.sqrt(np.sum(res ** 2) / (len(res) - 2))
    se_phi = s / math.sqrt(np.sum((x[:-1] - x[:-1].mean()) ** 2))
    out = dict(phi=phi, c=c, s=s, se_phi=se_phi)
    if 0 < phi < 1:
        kappa = -math.log(phi) / dt
        out.update(kappa=kappa, theta=c / (1 - phi), sigma=s * math.sqrt(2 * kappa / (1 - phi ** 2)),
                   half_life=math.log(2) / kappa)
    return out


# ----------------------------------------------------------------------------- martingales, Girsanov
def exit_prob_up(mu, sigma, a, b):
    """P(X hits a before -b) for X_t = mu t + sigma W_t, X_0 = 0, a, b > 0."""
    if abs(mu) < 1e-14:
        return b / (a + b)
    k = 2 * mu / sigma ** 2
    return -math.expm1(k * b) / (math.expm1(-k * a) - math.expm1(k * b))   # expm1 avoids cancellation


def exit_time_mean(mu, sigma, a, b):
    """E[tau] for the first exit of mu t + sigma W_t from (-b, a)."""
    if abs(mu) < 1e-14:
        return a * b / sigma ** 2
    p = exit_prob_up(mu, sigma, a, b)
    return (a * p - b * (1 - p)) / mu


def exit_mc(mu, sigma, a, b, m, dt, rng, max_steps=200000):
    """Monte Carlo of the exit from (-b, a) with the Brownian bridge crossing correction between
    grid dates. Returns (fraction that exit at a, mean exit time)."""
    x = np.zeros(m); alive = np.ones(m, bool); up = np.zeros(m, bool); tau = np.zeros(m)
    sd = sigma * math.sqrt(dt); s2 = sigma ** 2 * dt
    for k in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        x0 = x[idx]; x1 = x0 + mu * dt + sd * rng.standard_normal(idx.size)
        u = rng.uniform(size=idx.size)
        # probability that the bridge from x0 to x1 crossed a level between the dates
        pa = np.where(x1 < a, np.exp(-2 * (a - x0) * np.maximum(a - x1, 0) / s2), 1.0)
        pb = np.where(x1 > -b, np.exp(-2 * (x0 + b) * np.maximum(x1 + b, 0) / s2), 1.0)
        hit_a = (x1 >= a) | (u < pa)
        hit_b = ~hit_a & ((x1 <= -b) | (rng.uniform(size=idx.size) < pb))
        done = hit_a | hit_b
        x[idx] = x1
        up[idx[hit_a]] = True
        tau[idx[done]] = (k + 1) * dt - 0.5 * dt      # exit somewhere inside the last step
        alive[idx[done]] = False
    return float(up.mean()), float(tau.mean())


def girsanov_weight(theta, W_T, T):
    """Density dQ/dP on F_T that makes W_t + theta t a Q-Brownian motion: exp(-theta W_T - theta^2 T / 2)."""
    return np.exp(-theta * W_T - 0.5 * theta ** 2 * T)


# ----------------------------------------------------------------------------- Black-Scholes and Feynman-Kac
def bs_call(S, K, T, r, sigma):
    S = np.asarray(S, float)
    with np.errstate(divide="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)


def fk_theta_solve(x, T, drift, diff2, rate, payoff, bound, nt, rannacher=4):
    """Solve u_t + drift(x) u_x + 0.5 diff2(x) u_xx - rate u = 0, u(T, x) = payoff(x), backwards in time
    with Crank-Nicolson on the grid x (uniform), nt time steps, Dirichlet values bound(tau, x_edge)
    at both ends, and `rannacher` implicit half steps at the start to damp the payoff kink.
    Returns u(0, x)."""
    x = np.asarray(x, float); dx = x[1] - x[0]; n = len(x)
    a = drift(x); b = 0.5 * diff2(x)
    lo = b / dx ** 2 - a / (2 * dx)        # coefficient of u_{i-1}
    di = -2 * b / dx ** 2 - rate          # of u_i
    up = b / dx ** 2 + a / (2 * dx)        # of u_{i+1}
    u = payoff(x).astype(float)
    steps = [(T / nt / 2, 1.0)] * rannacher + [(T / nt, 0.5)] * (nt - rannacher // 2)
    tau = 0.0
    for dt, th in steps:
        tau_new = tau + dt
        # (I - th dt L) u_new = (I + (1 - th) dt L) u_old on interior points
        rhs = u[1:-1] + (1 - th) * dt * (lo[1:-1] * u[:-2] + di[1:-1] * u[1:-1] + up[1:-1] * u[2:])
        uL, uR = bound(tau_new, x[0]), bound(tau_new, x[-1])
        rhs[0] += th * dt * lo[1] * uL; rhs[-1] += th * dt * up[-2] * uR
        ab = np.zeros((3, n - 2))
        ab[0, 1:] = -th * dt * up[1:-2]
        ab[1, :] = 1 - th * dt * di[1:-1]
        ab[2, :-1] = -th * dt * lo[2:-1]
        u = np.concatenate([[uL], solve_banded((1, 1), ab, rhs), [uR]])
        tau = tau_new
    return u


def ou_second_moment(x, kappa, theta, sigma, tau):
    m, v = ou_moments(x, kappa, theta, sigma, tau)
    return m ** 2 + v


def ols_slope(x, y):
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])
