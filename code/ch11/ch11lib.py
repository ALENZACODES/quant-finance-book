"""Library for Chapter 11: fund waterfalls, performance measures, buyout arithmetic, valuation and discounts.

Every function is deterministic given its inputs; the random draws live in the scripts, which pass
generators in. Times are in years, rates are annual and decimal unless a name says otherwise.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------- rates and roots
def irr(times, flows, lo=-0.99, hi=10.0):
    """Annually compounded IRR of flows at the given times (investor's sign: contributions negative)."""
    t = np.asarray(times, float); c = np.asarray(flows, float)
    f = lambda r: float(np.sum(c * (1 + r) ** (-t)))
    return optimize.brentq(f, lo, hi, xtol=1e-14, maxiter=500)


def irr_cont(times, flows, lo=-5.0, hi=5.0):
    """Continuously compounded IRR: the root of sum c_j exp(-r t_j)."""
    t = np.asarray(times, float); c = np.asarray(flows, float)
    return optimize.brentq(lambda r: float(np.sum(c * np.exp(-r * t))), lo, hi, xtol=1e-14, maxiter=500)


# ---------------------------------------------------------------------------- waterfalls
def european_waterfall(times, contrib, dist, h=0.08, carry=0.20, k=1.0):
    """Whole fund (European) waterfall applied at each distribution date.

    contrib[t] are LP contributions, dist[t] gross cash available for distribution at the same dates.
    Tier 1 returns contributed capital, tier 2 pays the preferred return h, compounded annually on the
    hurdle balance, tier 3 is the GP catch-up at rate k, tier 4 splits carry/(1-carry).
    Returns a dict of arrays, one entry per date: roc, pref, catch, lp_split, gp_split, plus the
    clawback owed by the GP at the end (zero when the GP never holds more than its share).
    """
    t = np.asarray(times, float); C = np.asarray(contrib, float); X = np.asarray(dist, float)
    n = len(t); out = {k_: np.zeros(n) for k_ in ("roc", "pref", "catch", "lp_split", "gp_split")}
    bal = 0.0; unret = 0.0; lp_cum = 0.0; gp_cum = 0.0; c_cum = 0.0; prev = t[0]
    for j in range(n):
        bal *= (1 + h) ** (t[j] - prev); prev = t[j]
        bal += C[j]; unret += C[j]; c_cum += C[j]
        x = X[j]
        pay = min(x, bal); roc = min(pay, unret)
        out["roc"][j] = roc; out["pref"][j] = pay - roc
        bal -= pay; unret -= roc; lp_cum += pay; x -= pay
        if x > 0 and bal <= 1e-12:
            target = carry * (lp_cum - c_cum + gp_cum) - gp_cum      # carry still owed on profit so far
            y = max(target / (k - carry), 0.0) if k > carry else 0.0
            y = min(y, x)
            out["catch"][j] = k * y; gp_cum += k * y; lp_cum += (1 - k) * y
            out["lp_split"][j] += (1 - k) * y; x -= y
            out["lp_split"][j] += (1 - carry) * x; out["gp_split"][j] = carry * x
            lp_cum += (1 - carry) * x; gp_cum += carry * x
    profit = lp_cum + gp_cum - c_cum
    out["clawback"] = max(gp_cum - carry * max(profit, 0.0), 0.0)
    out["gp_total"] = gp_cum - out["clawback"]; out["lp_total"] = lp_cum + out["clawback"]
    return out


def deal_carry_paths(K, s_in, P, s_out, carry=0.20):
    """Cumulative carry paid by date under a whole fund rule and a deal by deal rule, no hurdle.

    Deals i cost K[i] at s_in[i] and return P[i] at s_out[i]. Whole fund: E(t) = c (realised proceeds
    less all capital called so far)^+, paid as it accrues. Deal by deal: A(t) = c * sum of positive
    realised deal profits, with a full clawback at the last exit that brings A back to the whole fund
    total. Returns the sorted event dates and the two cumulative carry paths at those dates.
    """
    K = np.asarray(K, float); P = np.asarray(P, float); si = np.asarray(s_in, float); so = np.asarray(s_out, float)
    dates = np.unique(np.concatenate([si, so]))
    E = np.zeros(len(dates)); A = np.zeros(len(dates)); paidE = 0.0
    for j, d in enumerate(dates):
        realised = so <= d + 1e-12; called = si <= d + 1e-12
        accrued = carry * max(P[realised].sum() - K[called].sum(), 0.0)
        paidE = max(paidE, accrued)                     # carry already paid is not returned in this sketch
        E[j] = paidE
        A[j] = carry * np.maximum(P[realised] - K[realised], 0).sum()
    final = carry * max(P.sum() - K.sum(), 0.0)
    A[-1] = final                                       # the clawback settles at the last exit
    return dates, E, A, final


def pv_of_path(dates, cum, r):
    """Present value at time 0 of the increments of a cumulative payment path, continuous rate r."""
    inc = np.diff(np.concatenate([[0.0], cum]))
    return float(np.sum(inc * np.exp(-r * np.asarray(dates))))


# ---------------------------------------------------------------------------- performance
def multiples(contrib, dist, nav):
    pic = float(np.sum(contrib)); d = float(np.sum(dist))
    return d / pic, nav / pic, (d + nav) / pic


def ks_pme(times, contrib, dist, nav, index, t_end):
    """Kaplan and Schoar PME: index-compounded distributions plus NAV over index-compounded contributions.
    index is a callable giving the index level at a time."""
    I_end = index(t_end)
    g = np.array([I_end / index(s) for s in times])
    return (float(np.sum(np.asarray(dist) * g)) + nav) / float(np.sum(np.asarray(contrib) * g))


def direct_alpha(times, contrib, dist, nav, index, t_end):
    """Direct alpha of Gredil, Griffiths and Stucke: the continuously compounded IRR of the fund's cash
    flows after each is carried to t_end with the index's return."""
    I_end = index(t_end)
    g = np.array([I_end / index(s) for s in times])
    flows = (np.asarray(dist) - np.asarray(contrib)) * g
    tt = np.concatenate([np.asarray(times, float), [t_end]]); ff = np.concatenate([flows, [nav]])
    return irr_cont(tt, ff)


def duration_bounds(t_contrib, t_dist):
    """Bounds of the implied duration ln(TVPI)/r for r > 0 (Proposition prop11:duration)."""
    return min(t_dist) - max(t_contrib), max(t_dist) - min(t_contrib)


def pme_beta_mean(alpha, beta, mu, sigma, T, r=0.0):
    """E[KS-PME] for one contribution of 1 and one distribution T years later of e^{alpha T} times the
    value of a position of beta in the index, rebalanced continuously and financed at the rate r:
    e^{alpha T + (1 - beta) r T - beta (beta - 1) sigma^2 T / 2} (I_T / I_0)^beta, the index a geometric
    Brownian motion with drift mu and volatility sigma. Equals exp(alpha T + (beta - 1)(mu - r - sigma^2) T)."""
    return math.exp(alpha * T + (beta - 1) * (mu - r - sigma ** 2) * T)


def levered_index_payoff(beta, x, r, sigma, T):
    """Value at T of one dollar held as a continuously rebalanced position of beta in the index financed at
    r, given the index's log return x over [0, T]."""
    return np.exp(beta * x + (1 - beta) * r * T - 0.5 * beta * (beta - 1) * sigma ** 2 * T)


# ---------------------------------------------------------------------------- buyouts
def lbo_moic(U, lam, i, tau, T):
    """Equity multiple of a buyout whose assets return U times entry value over T years, financed with
    debt of a share lam of entry value at rate i, interest deductible at tax rate tau and paid from the
    firm's cash (so the debt is in effect compounded at i(1 - tau))."""
    k = (1 + i * (1 - tau)) ** T
    return (U - lam * k) / (1 - lam)


def lbo_schedule(p):
    """Annual buyout model with a cash sweep. p is a dict of inputs (see lbo.py). Returns rows by year.
    Cash left after the sweep (or drawn on a revolver when negative) earns the after-tax debt rate, so
    net debt compounds at i(1 - tau) less free cash flow, which is the setting of the leverage
    proposition in the text."""
    rows = []; rev = p["rev0"]; debt = p["debt0"]; cash = 0.0; carry = 0.0
    cap = p.get("int_cap")                                  # deductible interest as a share of EBITDA, None for no cap
    cash_rate = p["rate"] * (1 - p["tax"])                  # cash, or a revolver, at the after-tax debt rate
    for y in range(1, p["years"] + 1):
        cash *= 1 + cash_rate
        rev_prev = rev; rev = rev * (1 + p["growth"][y - 1])
        ebitda = rev * p["margin"][y - 1]; da = p["da"] * rev; ebit = ebitda - da
        interest = p["rate"] * debt
        # with a cap, interest above cap x EBITDA is not deducted this year and is carried forward
        deduct = interest if cap is None else min(interest + carry, cap * max(ebitda, 0.0))
        carry = 0.0 if cap is None or not p.get("carry_forward", True) else interest + carry - deduct
        tax = max(ebit - deduct, 0.0) * p["tax"]
        capex = p["capex"] * rev; dnwc = p["nwc"] * (rev - rev_prev)
        fcf = ebitda - tax - capex - dnwc - interest          # cash left after paying interest
        repay = min(max(fcf, 0.0), debt) * p["sweep"]
        debt -= repay; cash += fcf - repay
        rows.append(dict(year=y, rev=rev, ebitda=ebitda, da=da, ebit=ebit, interest=interest, deduct=deduct, carry=carry, tax=tax,
                         capex=capex, dnwc=dnwc, fcf=fcf, repay=repay, debt=debt, cash=cash))
    return rows


def shapley_bridge(E0, m0, ET, mT, ND0, NDT):
    """Split the change in equity value m E - ND between EBITDA growth, the multiple and net debt, with
    the interaction of EBITDA and the multiple shared equally (the Shapley value of a product)."""
    growth = 0.5 * (m0 + mT) * (ET - E0)
    multiple = 0.5 * (E0 + ET) * (mT - m0)
    deleverage = ND0 - NDT
    return growth, multiple, deleverage


# ---------------------------------------------------------------------------- valuation
def ev_nopat_multiple(W, g, roc):
    """Enterprise value over next year's NOPAT for a firm growing at g forever with return on new
    capital roc and cost of capital W: (1 - g/roc)/(W - g)."""
    return (1 - g / roc) / (W - g)


def dcf_value(fcf, W, g_terminal, nopat_next=None, roc=None):
    """Enterprise value of explicit free cash flows fcf[0..T-1] for years 1..T plus a terminal value at
    T. If nopat_next and roc are given the terminal value is the value driver formula, otherwise the
    growing perpetuity of the last cash flow."""
    fcf = np.asarray(fcf, float); T = len(fcf)
    disc = (1 + W) ** -np.arange(1, T + 1)
    if nopat_next is not None:
        tv = nopat_next * (1 - g_terminal / roc) / (W - g_terminal)
    else:
        tv = fcf[-1] * (1 + g_terminal) / (W - g_terminal)
    return float(np.sum(fcf * disc) + tv * disc[-1]), float(tv)


def miles_ezzell_wacc(rU, rD, tau, L):
    """WACC when debt is reset each year to a share L of value (Miles and Ezzell)."""
    return rU - L * rD * tau * (1 + rU) / (1 + rD)


def apv_rebalanced(fcf, tv_fcf_next, g, rU, rD, tau, L):
    """Value by backward recursion with debt reset each year to L V_t: each year's tax shield is known
    a year ahead, so it is discounted one year at rD and before that at rU."""
    T = len(fcf)
    # terminal value under the same policy: a growing perpetuity at the Miles-Ezzell WACC
    W = miles_ezzell_wacc(rU, rD, tau, L)
    V = tv_fcf_next / (W - g)
    for t in range(T - 1, -1, -1):
        # V_{t} = (FCF_{t+1} + V_{t+1})/(1+rU) + tau rD L V_t/(1+rD)
        V = (fcf[t] + V) / (1 + rU) / (1 - tau * rD * L / (1 + rD))
    return V


def lever_cost_equity(rU, rD, DE, tau=0.0, rule="harris_pringle"):
    if rule == "harris_pringle":
        return rU + (rU - rD) * DE
    if rule == "miles_ezzell":
        return rU + (rU - rD) * DE * (1 - tau * rD / (1 + rD))
    if rule == "constant_debt":
        return rU + (rU - rD) * DE * (1 - tau)
    raise ValueError(rule)


def unlever_beta(beta_L, DE, tau, beta_D=0.0, rule="hamada"):
    if rule == "hamada":
        return (beta_L + beta_D * (1 - tau) * DE) / (1 + (1 - tau) * DE)
    if rule == "harris_pringle":
        return (beta_L + beta_D * DE) / (1 + DE)
    raise ValueError(rule)


def relever_beta(beta_U, DE, tau, beta_D=0.0, rule="hamada"):
    if rule == "hamada":
        return beta_U + (beta_U - beta_D) * (1 - tau) * DE
    if rule == "harris_pringle":
        return beta_U + (beta_U - beta_D) * DE
    raise ValueError(rule)


# ---------------------------------------------------------------------------- discounts
def combined_discount(*d):
    v = 1.0
    for x in d:
        v *= 1 - x
    return 1 - v


def bs_put(S, K, T, sigma, r=0.0, q=0.0):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T)); d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * math.exp(-q * T) * stats.norm.cdf(-d1)


def chaffe_dlom(T, sigma, r=0.0, q=0.0):
    """Price of an at the money put over the restriction period, as a share of the share price."""
    return bs_put(1.0, 1.0, T, sigma, r, q)


def finnerty_dlom(T, sigma, q=0.0):
    """Finnerty (2012): the average-strike put, the right to receive the average price over the
    restriction period in place of the price at its end, as a share of the share price. The average is
    approximated by a lognormal law with matched moments, which gives a closed form."""
    w = sigma ** 2 * T
    v2 = w + math.log(2 * (math.exp(w) - w - 1)) - 2 * math.log(math.exp(w) - 1)
    return math.exp(-q * T) * (2 * stats.norm.cdf(math.sqrt(v2) / 2) - 1)


def average_strike_mc(T, sigma, n_paths, n_steps, rng):
    """Monte Carlo of E[(A_T - S_T)^+]/S_0 for dS = sigma S dW, with A_T the time average of S over
    [0, T] by the trapezoid rule on n_steps steps."""
    dt = T / n_steps; x = np.zeros(n_paths); A = np.zeros(n_paths)
    for _ in range(n_steps):
        x1 = x - 0.5 * sigma ** 2 * dt + sigma * math.sqrt(dt) * rng.standard_normal(n_paths)
        A += 0.5 * (np.exp(x) + np.exp(x1)) * dt / T; x = x1
    y = np.maximum(A - np.exp(x), 0.0)
    return float(y.mean()), float(y.std(ddof=1) / math.sqrt(n_paths))


def longstaff_bound(T, sigma):
    """Longstaff's upper bound on the discount: E[max of a driftless geometric Brownian motion over
    [0, T]]/S_0 - 1, the value of selling at the best price of the period."""
    v = sigma ** 2 * T; s = math.sqrt(v)
    return (2 + v / 2) * stats.norm.cdf(s / 2) + s / math.sqrt(2 * math.pi) * math.exp(-v / 8) - 1


def max_gbm_mc(T, sigma, n_paths, n_steps, rng):
    """Monte Carlo of E[max_t S_t/S_0] - 1 for dS = sigma S dW, with the maximum of each step drawn
    exactly from the Brownian bridge, so there is no discretisation bias."""
    dt = T / n_steps; x = np.zeros(n_paths); m = np.zeros(n_paths)
    for _ in range(n_steps):
        x1 = x - 0.5 * sigma ** 2 * dt + sigma * math.sqrt(dt) * rng.standard_normal(n_paths)
        u = rng.random(n_paths)
        bridge_max = 0.5 * (x + x1 + np.sqrt((x1 - x) ** 2 - 2 * sigma ** 2 * dt * np.log(u)))
        m = np.maximum(m, bridge_max); x = x1
    y = np.exp(m) - 1
    return float(y.mean()), float(y.std(ddof=1) / math.sqrt(n_paths))
