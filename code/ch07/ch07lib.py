"""Library for Chapter 7: portfolio algebra and the mean-variance frontier, long-only portfolios by
exact enumeration of supports, the CAPM and factor models, least-squares alpha and beta with ordinary
and Newey-West standard errors, the Gibbons-Ross-Shanken test, the Benjamini-Hochberg rule, betas of
smoothed returns, return attribution with multi-period linking, Black-Litterman, equal risk
contributions, covariance shrinkage and the Memmel test of two Sharpe ratios.

All functions are deterministic given their inputs; simulations take a numpy Generator.
Returns are simple monthly returns as decimals; excess returns are over the one month bill.
"""
from __future__ import annotations
import itertools, math
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]

# ----------------------------------------------------------------------------- data

def load():
    """Factors and industries as decimals, indexed by yyyymm."""
    import pandas as pd
    f = pd.read_csv(ROOT / "data" / "ff_factors_monthly.csv", index_col=0) / 100
    i = pd.read_csv(ROOT / "data" / "industry10_monthly.csv", index_col=0)
    cap = i[[f"cap_{k}" for k in INDUSTRIES]].copy(); cap.columns = INDUSTRIES
    vw = i[INDUSTRIES] / 100
    ew = i[[f"ew_{k}" for k in INDUSTRIES]] / 100; ew.columns = INDUSTRIES
    return f, vw, ew, cap


def month_name(m):
    names = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
    return f"{names[m % 100 - 1]} {m // 100}"

# ----------------------------------------------------------------------------- frontier

def abcd(mu, S):
    Si1 = np.linalg.solve(S, np.ones(len(mu))); Sim = np.linalg.solve(S, mu)
    A = Si1.sum(); B = Si1 @ mu; C = mu @ Sim
    return A, B, C, A * C - B * B


def frontier_w(mu, S, m):
    """Minimum-variance weights with mean m, fully invested, short sales allowed."""
    A, B, C, D = abcd(mu, S)
    lam = (C - B * m) / D; gam = (A * m - B) / D
    return np.linalg.solve(S, lam * np.ones(len(mu)) + gam * mu)


def frontier_var(mu, S, m):
    A, B, C, D = abcd(mu, S)
    return (A * m * m - 2 * B * m + C) / D


def gmv(S):
    w = np.linalg.solve(S, np.ones(len(S)))
    return w / w.sum()


def tangency(mx, S):
    """Maximum Sharpe ratio portfolio of excess means mx, fully invested."""
    y = np.linalg.solve(S, mx)
    return y / y.sum()


def sharpe2(mx, S):
    return float(mx @ np.linalg.solve(S, mx))


def qp_enum(S, Aeq, beq, c=None, tol=1e-11):
    """min 1/2 w'Sw - c'w  subject to Aeq w = beq and w >= 0, solved exactly by trying every support.

    For a support s the KKT system on s is linear; the solution is accepted when the weights on s are
    nonnegative and the bound multipliers off s are nonnegative. The problem is convex, so the
    accepted point is the minimum. Practical for up to about 14 assets."""
    n = len(S); c = np.zeros(n) if c is None else np.asarray(c, float)
    Aeq = np.atleast_2d(Aeq); beq = np.atleast_1d(beq); p = len(beq)
    best = None
    for k in range(1, n + 1):
        for s in itertools.combinations(range(n), k):
            s = list(s)
            M = np.zeros((k + p, k + p)); M[:k, :k] = S[np.ix_(s, s)]; M[:k, k:] = -Aeq[:, s].T; M[k:, :k] = Aeq[:, s]
            rhs = np.r_[c[s], beq]
            try:
                z = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(z)) or np.max(np.abs(M @ z - rhs)) > 1e-9:
                continue
            ws = z[:k]
            if ws.min() < -tol:
                continue
            w = np.zeros(n); w[s] = ws; lam = z[k:]
            nu = S @ w - c - Aeq.T @ lam
            off = [i for i in range(n) if i not in s]
            if off and nu[off].min() < -1e-10:
                continue
            obj = 0.5 * w @ S @ w - c @ w
            if best is None or obj < best[0] - 1e-14:
                best = (obj, w, lam, nu)
        if best is not None and k >= 1:
            pass
    if best is None:
        raise ValueError("infeasible")
    return best[1], best[2], best[3]


def long_only_frontier_w(mu, S, m):
    n = len(mu)
    w, _, _ = qp_enum(S, np.vstack([np.ones(n), mu]), np.array([1.0, m]))
    return w


def long_only_tangency(mx, S):
    """Long-only maximum Sharpe portfolio: minimise y'Sy with mx'y = 1, y >= 0, then rescale."""
    y, _, _ = qp_enum(S, mx[None, :], np.array([1.0]))
    return y / y.sum()


def long_only_gmv(S):
    n = len(S)
    w, _, _ = qp_enum(S, np.ones((1, n)), np.array([1.0]))
    return w


def sharpe2_bias(theta2, N, T):
    """E of the in-sample maximum squared Sharpe ratio, normal returns, sample covariance with T-1."""
    return (T - 1) / (T - N - 2) * (theta2 + N / T)

# ----------------------------------------------------------------------------- regressions

def ols(y, X, nw_lags=None):
    """Least squares with an intercept column already in X. Returns coef, se, nw se, t, R^2, resid."""
    y = np.asarray(y, float); X = np.asarray(X, float); T, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    s2 = e @ e / (T - k); se = np.sqrt(np.diag(XtX_inv) * s2)
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    nw = None
    if nw_lags is not None:
        u = X * e[:, None]; Sg = u.T @ u
        for L in range(1, nw_lags + 1):
            w = 1 - L / (nw_lags + 1); G = u[L:].T @ u[:-L]; Sg += w * (G + G.T)
        V = XtX_inv @ Sg @ XtX_inv * T / (T - k); nw = np.sqrt(np.diag(V))
    return dict(b=b, se=se, nw=nw, t=b / se, r2=r2, e=e, s=math.sqrt(s2))


def alpha_se_theory(sig_e, xbar, sx, T):
    """Standard error of the least-squares intercept, sx the standard deviation with divisor T."""
    return sig_e / math.sqrt(T) * math.sqrt(1 + (xbar / sx) ** 2)


def detection_months(ir_m, sr_m, t=2.0):
    """Months for the expected t statistic of alpha to reach t; information and Sharpe ratios monthly."""
    return t * t * (1 + sr_m ** 2) / ir_m ** 2


def alpha_power(ir_m, sr_m, T, crit=1.96):
    """Probability that the alpha t statistic exceeds crit, normal approximation."""
    mean_t = ir_m * math.sqrt(T) / math.sqrt(1 + sr_m ** 2)
    return float(1 - stats.norm.cdf(crit - mean_t) + stats.norm.cdf(-crit - mean_t))


def grs(R, F):
    """Gibbons-Ross-Shanken test of zero intercepts. R is T x N excess returns, F is T x K traded factors."""
    R = np.asarray(R, float); F = np.asarray(F, float).reshape(len(R), -1); T, N = R.shape; K = F.shape[1]
    X = np.c_[np.ones(T), F]; c = np.linalg.lstsq(X, R, rcond=None)[0]; a = c[0]; e = R - X @ c
    Sig = e.T @ e / T; fb = F.mean(0); Om = np.atleast_2d(np.cov(F.T, bias=True))
    q = 1 + fb @ np.linalg.solve(Om, fb)
    J = (T - N - K) / N * (a @ np.linalg.solve(Sig, a)) / q
    return float(J), float(stats.f.sf(J, N, T - N - K)), a, Sig, fb, Om


def bh(p, q):
    """Benjamini-Hochberg: reject the k smallest p-values, k the largest i with p_(i) <= q i / m."""
    p = np.asarray(p); m = len(p); o = np.argsort(p); ok = p[o] <= q * np.arange(1, m + 1) / m
    rej = np.zeros(m, bool)
    if ok.any():
        k = np.max(np.nonzero(ok)[0]); rej[o[:k + 1]] = True
    return rej


def storey_pi0(p, lam=0.5):
    return min(1.0, float(np.mean(np.asarray(p) > lam) / (1 - lam)))

# ----------------------------------------------------------------------------- smoothing

def smooth_ar1(r, phi):
    """Observed return r^o_t = (1 - phi) r_t + phi r^o_{t-1}, started at r^o_0 = r_0."""
    out = np.empty(len(r)); out[0] = r[0]
    for t in range(1, len(r)):
        out[t] = (1 - phi) * r[t] + phi * out[t - 1]
    return out


def dimson(y, x, q):
    """Sum of slopes of y_t on x_t, ..., x_{t-q} and the intercept."""
    T = len(y); Xs = [np.ones(T - q)] + [x[q - j:T - j] for j in range(q + 1)]
    r = ols(y[q:], np.column_stack(Xs))
    return float(r["b"][1:].sum()), float(r["b"][0]), r

# ----------------------------------------------------------------------------- options

def bs_put(S, K, T, r, sig):
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T)); d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

# ----------------------------------------------------------------------------- attribution

def brinson(w, b, rp, rb):
    """Allocation, selection and interaction by segment (Brinson-Fachler allocation)."""
    w, b, rp, rb = map(np.asarray, (w, b, rp, rb)); Rb = b @ rb
    return (w - b) * (rb - Rb), b * (rp - rb), (w - b) * (rp - rb)


def carino(Rp, Rb):
    """Carino factors k_t / k that make linked single-period effects add to the cumulative active return."""
    Rp, Rb = np.asarray(Rp), np.asarray(Rb)
    k_t = np.where(np.abs(Rp - Rb) > 1e-15, (np.log1p(Rp) - np.log1p(Rb)) / (Rp - Rb + 1e-300), 1 / (1 + Rp))
    P, B = np.prod(1 + Rp) - 1, np.prod(1 + Rb) - 1
    k = (math.log1p(P) - math.log1p(B)) / (P - B)
    return k_t / k, P, B

# ----------------------------------------------------------------------------- Black-Litterman

def bl_posterior(pi, S, P, q, Om, tau):
    P = np.atleast_2d(P); tS = tau * S
    M = P @ tS @ P.T + Om
    return pi + tS @ P.T @ np.linalg.solve(M, q - P @ pi)


def bl_posterior_info(pi, S, P, q, Om, tau):
    """The same posterior mean in information form."""
    P = np.atleast_2d(P); iT = np.linalg.inv(tau * S); iO = np.linalg.inv(Om)
    return np.linalg.solve(iT + P.T @ iO @ P, iT @ pi + P.T @ iO @ q)

# ----------------------------------------------------------------------------- risk budgets

def risk_contrib(w, S):
    sp = math.sqrt(w @ S @ w)
    return w * (S @ w) / sp


def erc(S, tol=1e-14, it=500):
    """Equal risk contributions by Newton's method on (S y)_i = 1 / y_i, then w = y / sum y."""
    n = len(S); y = 1 / np.sqrt(np.diag(S))
    for _ in range(it):
        g = S @ y - 1 / y; H = S + np.diag(1 / y ** 2)
        step = np.linalg.solve(H, g); a = 1.0
        while np.any(y - a * step <= 0):
            a /= 2
        y = y - a * step
        if np.max(np.abs(g)) < tol:
            break
    return y / y.sum()

# ----------------------------------------------------------------------------- estimation

def lw_identity(X):
    """Ledoit-Wolf (2004) shrinkage of the sample covariance (divisor T) towards a scaled identity."""
    X = np.asarray(X, float); T, N = X.shape; Xc = X - X.mean(0); S = Xc.T @ Xc / T
    m = np.trace(S) / N; d2 = np.sum((S - m * np.eye(N)) ** 2) / N
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in Xc) / N / T ** 2
    b2 = min(b2, d2); rho = b2 / d2
    return rho * m * np.eye(N) + (1 - rho) * S, rho


def memmel(x, y):
    """Jobson-Korkie test of equal Sharpe ratios with Memmel's correction. Returns z and two-sided p."""
    x, y = np.asarray(x), np.asarray(y); T = len(x)
    mi, mj = x.mean(), y.mean(); si, sj = x.std(ddof=1), y.std(ddof=1); sij = np.cov(x, y)[0, 1]
    th = (2 * si**2 * sj**2 - 2 * si * sj * sij + 0.5 * mi**2 * sj**2 + 0.5 * mj**2 * si**2
          - mi * mj / (2 * si * sj) * (sij**2 + si**2 * sj**2)) / T
    z = (sj * mi - si * mj) / math.sqrt(th)
    return float(z), float(2 * stats.norm.sf(abs(z)))
