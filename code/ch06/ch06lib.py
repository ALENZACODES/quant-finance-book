"""Library for Chapter 6: bond prices, duration and convexity, immunisation, the Vasicek, CIR and
Ho-Lee short-rate models, swap rates and caplets, the Merton structural model, hazard rates and
credit default swaps, the one-factor Gaussian copula and tranche losses, the screening metrics of a
commercial real estate loan, and the refinancing test of the case study.

All functions are deterministic given their inputs; simulations take a numpy Generator.
Conventions: bond yields are semi-annually compounded, Treasury bill yields are simple on an
actual/365 basis, zero rates in the short-rate models are continuously compounded, and loan rates
in the screening code are nominal annual rates compounded monthly, as in the screener.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import integrate, optimize, special, stats

# ----------------------------------------------------------------------------- bonds

def bond_cfs(c, T, freq=2, face=100.0):
    n = int(round(freq * T))
    k = np.arange(1, n + 1)
    cf = np.full(n, face * c / freq); cf[-1] += face
    return k, cf


def bond_price(y, c, T, freq=2, face=100.0):
    """Price of a bond with coupon rate c, maturity T on a coupon date, at yield y compounded freq times a year."""
    k, cf = bond_cfs(c, T, freq, face)
    return float(np.sum(cf * (1 + y / freq) ** (-k)))


def bond_derivs(y, c, T, freq=2, face=100.0):
    """Price and its first three derivatives in y."""
    k, cf = bond_cfs(c, T, freq, face)
    v = 1 + y / freq
    p0 = np.sum(cf * v ** (-k))
    p1 = -np.sum(cf * (k / freq) * v ** (-k - 1))
    p2 = np.sum(cf * k * (k + 1) / freq ** 2 * v ** (-k - 2))
    p3 = -np.sum(cf * k * (k + 1) * (k + 2) / freq ** 3 * v ** (-k - 3))
    return float(p0), float(p1), float(p2), float(p3)


def mod_duration(y, c, T, freq=2):
    p0, p1, _, _ = bond_derivs(y, c, T, freq)
    return -p1 / p0


def convexity(y, c, T, freq=2):
    p0, _, p2, _ = bond_derivs(y, c, T, freq)
    return p2 / p0


def macaulay(y, c, T, freq=2):
    return mod_duration(y, c, T, freq) * (1 + y / freq)


def par_duration(y, T, freq=2):
    """Modified duration of a par bond, closed form."""
    return (1 - (1 + y / freq) ** (-freq * T)) / y


def surplus_shift(s, z, a_times, a_amounts, l_times, l_amounts):
    """Present value of assets less liabilities after a parallel shift s of a flat continuous rate z."""
    a = sum(x * math.exp(-(z + s) * t) for t, x in zip(a_times, a_amounts))
    l = sum(x * math.exp(-(z + s) * t) for t, x in zip(l_times, l_amounts))
    return a - l


def barbell(z, T, t1, t2, L=100.0):
    """Zero-coupon amounts at t1 < T < t2 with the present value and duration of L due at T."""
    pv = L * math.exp(-z * T)
    w2 = (T - t1) / (t2 - t1); w1 = 1 - w2              # present-value weights
    return w1 * pv * math.exp(z * t1), w2 * pv * math.exp(z * t2)

# ----------------------------------------------------------------------------- short-rate models

def vas_B(kap, tau):
    return (1 - np.exp(-kap * tau)) / kap


def vas_P(r, tau, kap, rbar, sig):
    """Vasicek zero-coupon bond price P(t, t + tau) given r_t."""
    tau = np.asarray(tau, float); B = vas_B(kap, tau)
    lnA = (B - tau) * (kap ** 2 * rbar - 0.5 * sig ** 2) / kap ** 2 - sig ** 2 * B ** 2 / (4 * kap)
    return np.exp(lnA - B * r)


def cir_AB(tau, kap, rbar, sig):
    tau = np.asarray(tau, float)
    h = math.sqrt(kap ** 2 + 2 * sig ** 2)
    e = np.exp(h * tau) - 1
    den = 2 * h + (kap + h) * e
    B = 2 * e / den
    A = (2 * h * np.exp((kap + h) * tau / 2) / den) ** (2 * kap * rbar / sig ** 2)
    return A, B


def cir_P(r, tau, kap, rbar, sig):
    A, B = cir_AB(tau, kap, rbar, sig)
    return A * np.exp(-B * r)


def model_quotes(P, bill_tau=(0.25, 0.5, 1.0), bond_T=(2, 3, 5, 7, 10)):
    """Quotes in their market conventions from a discount function P(t): simple bill yields
    (actual/365) and semi-annual par yields."""
    bills = [(1 / P(t) - 1) / t for t in bill_tau]
    pars = []
    for T in bond_T:
        t = np.arange(1, 2 * T + 1) / 2
        pars.append(2 * (1 - P(T)) / np.sum(P(t)))
    return np.array(bills + pars)


def fit_short_rate(quotes, model, sig, x0):
    """Least squares fit of (r0, kappa, rbar) with sigma fixed; quotes in decimal."""
    Pf = vas_P if model == "vasicek" else cir_P

    def resid(x):
        r0, kap, rbar = x
        q = model_quotes(lambda t: Pf(r0, t, kap, rbar, sig))
        return 1e4 * (q - quotes)
    lo = [-0.05, 0.01, -0.05] if model == "vasicek" else [1e-5, 0.01, 1e-5]
    res = optimize.least_squares(resid, x0, bounds=(lo, [0.2, 5.0, 0.2]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return res.x, res.fun


def vas_moments(r0, T, kap, rbar, sig):
    """Mean and covariance of (r_T, int_0^T r_u du) under Q in the Vasicek model."""
    e1, e2 = math.exp(-kap * T), math.exp(-2 * kap * T)
    mr = rbar + (r0 - rbar) * e1
    mI = rbar * T + (r0 - rbar) * (1 - e1) / kap
    vr = sig ** 2 * (1 - e2) / (2 * kap)
    vI = sig ** 2 / kap ** 2 * (T - 2 * (1 - e1) / kap + (1 - e2) / (2 * kap))
    cRI = sig ** 2 / kap * ((1 - e1) / kap - (1 - e2) / (2 * kap))
    return np.array([mr, mI]), np.array([[vr, cRI], [cRI, vI]])


def cir_sim_bond(r0, T, kap, rbar, sig, steps, paths, rng):
    """Monte Carlo price of the T-year zero under CIR with exact (noncentral chi-square) transitions
    and the trapezoid rule for the integral. Returns (mean, standard error)."""
    dt = T / steps
    c = sig ** 2 * (1 - math.exp(-kap * dt)) / (4 * kap)
    df = 4 * kap * rbar / sig ** 2
    r = np.full(paths, r0); I = np.zeros(paths)
    for _ in range(steps):
        lam = r * math.exp(-kap * dt) / c
        rn = c * rng.noncentral_chisquare(df, lam)
        I += 0.5 * (r + rn) * dt; r = rn
    x = np.exp(-I)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(paths))


def holee_adjustment(sig, T1, T2):
    """Futures minus forward continuously compounded rate over [T1, T2] in the Ho-Lee model."""
    return 0.5 * sig ** 2 * T1 * T2

# ----------------------------------------------------------------------------- swaps and caps

def swap_rate(P, T0, Tn, freq=2):
    """Par rate of a swap starting at T0 and ending at Tn with fixed payments freq times a year."""
    n = int(round(freq * (Tn - T0)))
    t = T0 + np.arange(1, n + 1) / freq
    return float((P(T0) - P(Tn)) / (np.sum(P(t)) / freq))


def ncdf(x):
    return special.ndtr(x)


def black_caplet(F, K, sig, T, delta, P_pay):
    """Black (1976) caplet on a simple rate with forward F, fixing at T, paid at T + delta."""
    v = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * v * v) / v
    return P_pay * delta * (F * ncdf(d1) - K * ncdf(d1 - v))


def vas_zbp(r0, T, S, X, kap, rbar, sig):
    """Price at 0 of a European put, expiry T, strike X, on the zero maturing at S, in Vasicek."""
    PT, PS = vas_P(r0, T, kap, rbar, sig), vas_P(r0, S, kap, rbar, sig)
    sp = sig * math.sqrt((1 - math.exp(-2 * kap * T)) / (2 * kap)) * vas_B(kap, S - T)
    h = math.log(PS / (PT * X)) / sp + sp / 2
    return float(X * PT * ncdf(-h + sp) - PS * ncdf(-h))


def vas_caplet(r0, T, delta, K, kap, rbar, sig):
    """Caplet on the simple rate over [T, T + delta], paid at T + delta, as (1 + delta K) puts."""
    return (1 + delta * K) * vas_zbp(r0, T, T + delta, 1 / (1 + delta * K), kap, rbar, sig)


def vas_caplet_mc(r0, T, delta, K, kap, rbar, sig, paths, rng):
    """Monte Carlo of the same caplet from the exact joint law of (r_T, int_0^T r)."""
    m, C = vas_moments(r0, T, kap, rbar, sig)
    z = rng.multivariate_normal(m, C, size=paths, method="cholesky")
    rT, I = z[:, 0], z[:, 1]
    PTS = vas_P(rT, delta, kap, rbar, sig)
    L = (1 / PTS - 1) / delta
    x = np.exp(-I) * PTS * delta * np.maximum(L - K, 0)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(paths))


def black_implied(price, F, K, T, delta, P_pay):
    f = lambda s: black_caplet(F, K, s, T, delta, P_pay) - price
    return optimize.brentq(f, 1e-4, 5.0, xtol=1e-14)

# ----------------------------------------------------------------------------- Merton

def merton(V, D, r, sigV, T):
    """Equity, debt, credit spread (continuous) and risk-neutral default probability."""
    v = sigV * math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * sigV ** 2) * T) / v; d2 = d1 - v
    E = V * ncdf(d1) - D * math.exp(-r * T) * ncdf(d2)
    debt = V - E
    y = -math.log(debt / D) / T
    return dict(E=E, debt=debt, spread=y - r, pdQ=float(ncdf(-d2)), d1=d1, d2=d2)


def merton_spread_lev(L, sigV, T):
    """Spread as a function of quasi-leverage L = D e^{-rT} / V only."""
    v = sigV * np.sqrt(T)
    d1 = (-np.log(L) + 0.5 * v * v) / v; d2 = d1 - v
    return -np.log(ncdf(d2) + ncdf(-d1) / L) / T


def merton_calibrate(E, sigE, D, r, T):
    """Solve E = C(V) and sigE E = Phi(d1) sigV V for (V, sigV)."""
    def f(x):
        V, s = x
        m = merton(V, D, r, s, T)
        return [m["E"] - E, ncdf(m["d1"]) * s * V - sigE * E]
    sol = optimize.fsolve(f, [E + D * math.exp(-r * T), sigE * E / (E + D)], xtol=1e-14, full_output=True)
    return sol[0]

# ----------------------------------------------------------------------------- hazard rates and CDS

def survival(t, knots, lams):
    """Survival probability with piecewise-constant hazard lams on (knots[i-1], knots[i]], knots[-1]=inf."""
    t = np.asarray(t, float); H = np.zeros_like(t); prev = 0.0
    for k, l in zip(knots, lams):
        H += l * np.clip(np.minimum(t, k) - prev, 0, None); prev = k
    return np.exp(-H)


def cds_legs(T, knots, lams, R, P, freq=4, sub=24):
    """Risky annuity (per unit spread, with accrued premium on default) and protection leg of a CDS.
    Integrals on each premium period use `sub` midpoint steps."""
    n = int(round(freq * T)); ann = 0.0; prot = 0.0
    for i in range(1, n + 1):
        a, b = (i - 1) / freq, i / freq
        ann += (b - a) * P(b) * survival(b, knots, lams)
        u = a + (np.arange(sub) + 0.5) * (b - a) / sub
        Su = survival(u, knots, lams)
        h = np.array([lams[np.searchsorted(knots, x, side="left")] for x in u])
        dens = h * Su * (b - a) / sub
        prot += float(np.sum(P(u) * dens))
        ann += float(np.sum(P(u) * (u - a) * dens))
    return ann, (1 - R) * prot


def cds_par(T, knots, lams, R, P, freq=4):
    ann, prot = cds_legs(T, knots, lams, R, P, freq)
    return prot / ann


def bootstrap_hazard(mats, spreads, R, P):
    knots = list(mats[:-1]) + [np.inf]; lams = []
    for j, (T, s) in enumerate(zip(mats, spreads)):
        def f(l):
            lam = lams + [l] + [l] * (len(mats) - j - 1)
            return cds_par(T, knots, lam, R, P) - s
        lams.append(optimize.brentq(f, 1e-8, 2.0, xtol=1e-15))
    return knots, lams


def default_times(n, knots, lams, rng):
    """tau = H^{-1}(E) with E standard exponential, for piecewise-constant hazard."""
    E = rng.exponential(size=n); tau = np.full(n, np.inf); prev = 0.0; Hprev = 0.0
    for k, l in zip(knots, lams):
        Hk = Hprev + l * (k - prev) if np.isfinite(k) else np.inf
        m = (E > Hprev) & (E <= Hk) & ~np.isfinite(tau)
        tau[m] = prev + (E[m] - Hprev) / l
        prev, Hprev = k, Hk
    return tau

# ----------------------------------------------------------------------------- Gaussian copula



def cond_pd(p, rho, m):
    return ncdf((stats.norm.ppf(p) - math.sqrt(rho) * np.asarray(m)) / math.sqrt(1 - rho))


def pool_pmf(n, p, rho):
    """Distribution of the number of defaults in a homogeneous pool of n names; rho > 0 by adaptive
    quadrature over the common factor on [-12, 12], rho = 0 exactly binomial."""
    k = np.arange(n + 1)
    if rho == 0:
        return stats.binom.pmf(k, n, p)
    f = lambda m: stats.binom.pmf(k, n, cond_pd(p, rho, m)) * math.exp(-0.5 * m * m) / math.sqrt(2 * math.pi)
    val, _ = integrate.quad_vec(f, -12.0, 12.0, epsabs=1e-15, epsrel=1e-12, limit=400)
    return val


def lhp_cdf(x, p, rho):
    """P(default fraction <= x) in the large pool limit."""
    x = np.asarray(x, float)
    return ncdf((math.sqrt(1 - rho) * stats.norm.ppf(x) - stats.norm.ppf(p)) / math.sqrt(rho))


def tranche_loss(L, a, d):
    return np.minimum(np.maximum(L - a, 0), d - a)


def tranche_el_pool(n, p, rho, R, a, d):
    """Expected tranche loss, as a fraction of tranche size, for a pool of n names."""
    pmf = pool_pmf(n, p, rho)
    L = (1 - R) * np.arange(n + 1) / n
    return float(np.sum(pmf * tranche_loss(L, a, d)) / (d - a))


def tranche_el_lhp(p, rho, R, a, d):
    """Large pool tranche expected loss as a fraction of the tranche, integrating over the factor m.
    The integrand has kinks where the conditional loss crosses a and d; the range is split there so
    that thin regions in the far tail (where a senior tranche is hit) are not stepped over."""
    c = stats.norm.ppf(p)

    def g(m):
        return float(tranche_loss((1 - R) * cond_pd(p, rho, m), a, d)) * stats.norm.pdf(m)
    pts = [(c - math.sqrt(1 - rho) * stats.norm.ppf(x / (1 - R))) / math.sqrt(rho)
           for x in (a, d) if 0 < x < 1 - R]
    edges = [-12.0] + sorted(min(max(q, -12.0), 12.0) for q in pts) + [12.0]
    val = sum(integrate.quad(g, lo, hi, limit=400, epsabs=1e-15, epsrel=1e-12)[0]
              for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)
    return val / (d - a)


def copula_mc(n, p, rho, R, tranches, scen, rng, batch=20000):
    """Simulated tranche losses; returns means and standard errors, one per tranche."""
    c = stats.norm.ppf(p); out = []
    for s in range(0, scen, batch):
        b = min(batch, scen - s)
        M = rng.standard_normal(b)[:, None]
        X = math.sqrt(rho) * M + math.sqrt(1 - rho) * rng.standard_normal((b, n))
        L = (1 - R) * np.mean(X <= c, axis=1)
        out.append(np.stack([tranche_loss(L, a, d) / (d - a) for a, d in tranches], axis=1))
    x = np.concatenate(out)
    return x.mean(axis=0), x.std(axis=0, ddof=1) / math.sqrt(scen)

# ----------------------------------------------------------------------------- private credit

def annual_debt_service(loan, rate_pct, amort_yrs):
    """As in the screener: interest-only if amort_yrs is 0, otherwise monthly payments at rate/12."""
    if not amort_yrs:
        return loan * rate_pct / 100
    i = rate_pct / 1200; n = amort_yrs * 12
    return 12 * loan * i / (1 - (1 + i) ** (-n))


def balloon(loan, rate_pct, amort_yrs, months):
    if not amort_yrs:
        return loan
    i = rate_pct / 1200; p = annual_debt_service(loan, rate_pct, amort_yrs) / 12
    return loan * (1 + i) ** months - p * ((1 + i) ** months - 1) / i


def screener_metrics(d):
    ltv = d["loan"] / d["value"] * 100
    ds = annual_debt_service(d["loan"], d["rate"], d["amort"])
    dscr = d["noi"] / ds; dy = d["noi"] / d["loan"] * 100
    bal = balloon(d["loan"], d["rate"], d["amort"], d["termYrs"] * 12)
    exit_noi = d["noi"] * (1 + d["noiGrowth"] / 100) ** d["termYrs"]
    exit_value = exit_noi / ((d["capRate"] + d["exitCapShift"]) / 100)
    return dict(ltv=ltv, ds=ds, dscr=dscr, dy=dy, balloon=bal, exitValue=exit_value,
                exitLtv=bal / exit_value * 100)


def screener_score(d):
    m = screener_metrics(d)
    cl = lambda x: min(1.0, max(0.0, x))
    s = dict(ltv=cl((85 - m["ltv"]) / 35), dscr=cl((m["dscr"] - 1) / 0.6), dy=cl((m["dy"] - 6) / 8),
             exit=cl((90 - m["exitLtv"]) / 40), sponsor=cl((d["sponsorScore"] - 1) / 4))
    w = dict(ltv=0.25, dscr=0.25, dy=0.15, exit=0.25, sponsor=0.10)
    t = sum(w[k] * s[k] for k in w)
    total = int(math.floor(t * 100 + 0.5))          # JavaScript Math.round
    grade = "A" if total >= 75 else "B" if total >= 60 else "C" if total >= 45 else "D"
    return total, grade, m, s


def put_lognormal(K, m, s):
    """E[(K - Y)^+] for Y lognormal with E[Y] = m and log standard deviation s."""
    d1 = (math.log(m / K) + 0.5 * s * s) / s; d2 = d1 - s
    return K * ncdf(-d2) - m * ncdf(-d1)


def slice_el(lo, hi, V0, g, sigma, T, cost):
    """Expected loss of a claim on the slice (lo, hi] of collateral value at T, with recovery
    (1 - cost) V_T, where V_T is lognormal with mean V0 (1 + g)^T (growth compounded annually, as for
    NOI in the screener) and log standard deviation sigma sqrt(T). Faces lo and hi are in money at T;
    returns the expected loss in money."""
    m = (1 - cost) * V0 * (1 + g) ** T; s = sigma * math.sqrt(T)
    P = lambda K: put_lognormal(K, m, s) if K > 0 else 0.0
    return P(hi) - P(lo)


def refi_vintages(y, ltv0=0.65, cap0=0.06, spread=2.25, amort=25, g=0.02, beta=0.5, dscr_min=1.25,
                  ltv_max=0.65, term_m=60):
    """Refinancing test for a loan originated at each month-end of the yield series y (percent).
    Rates are the five year yield plus a spread, in percent, compounded monthly as in the screener.
    Returns one row per vintage: origination and maturity yields, balloon, value and NOI at maturity,
    the largest new loan the two tests allow divided by the balloon (the cover), and the binding test."""
    rows = []
    vals = np.asarray(y, float)
    for i in range(len(vals) - term_m):
        y0, y1 = vals[i], vals[i + term_m]
        r0, r1 = y0 + spread, y1 + spread
        loan, noi0 = 100 * ltv0, 100 * cap0
        B = balloon(loan, r0, amort, term_m)
        noi1 = noi0 * (1 + g) ** (term_m / 12)
        cap1 = cap0 + beta * (y1 - y0) / 100
        V1 = noi1 / cap1
        k1 = annual_debt_service(1.0, r1, amort)
        by_ltv, by_dscr = ltv_max * V1, noi1 / (dscr_min * k1)
        rows.append(dict(y0=y0, y1=y1, B=B, noi1=noi1, V1=V1, cap1=cap1, k1=k1, cover=min(by_ltv, by_dscr) / B,
                         ltv_binds=by_ltv <= by_dscr, dscr0=noi0 / annual_debt_service(loan, r0, amort),
                         dscr1=noi1 / (B * k1), ltv1=B / V1))
    return rows


def refi_pd(cover, sig):
    """Probability that a mean-one lognormal shock with log standard deviation sig falls below 1/cover."""
    return float(ncdf((math.log(1 / cover) + 0.5 * sig * sig) / sig))


def refi_el(B, a, cover, sig):
    """E[(B - a X)^+ 1{X < 1/cover}] for X mean-one lognormal with log standard deviation sig."""
    u = min(1 / cover, B / a)
    z = (math.log(u) + 0.5 * sig * sig) / sig
    return float(B * ncdf(z) - a * ncdf(z - sig))
