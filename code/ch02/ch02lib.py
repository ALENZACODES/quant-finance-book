"""Library for Chapter 2: fund fees, distribution waterfalls, incentive fees and commitment pacing.

Every function is small, documented and tested in test_ch02.py. Time is measured in years and all
cash flows are seen from the investor side: contributions are positive amounts paid in, distributions
are positive amounts paid out.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import optimize, stats


# ----------------------------------------------------------------------------- basic tools
def npv(rate, t, cf):
    t, cf = np.asarray(t, float), np.asarray(cf, float)
    return float(np.sum(cf / (1.0 + rate) ** t))


def irr(t, cf, lo=-0.99, hi=10.0):
    """IRR of dated flows (negative = paid in) by bracketing on a grid then Brent's method."""
    t, cf = np.asarray(t, float), np.asarray(cf, float)
    grid = np.linspace(lo, hi, 4001)
    v = (cf[None, :] / (1.0 + grid[:, None]) ** t[None, :]).sum(axis=1)
    s = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if len(s) == 0:
        return float("nan")
    i = s[0]
    return float(optimize.brentq(npv, grid[i], grid[i + 1], args=(t, cf), xtol=1e-14))


def bs_call(S, K, T, r, sigma, q=0.0):
    """Black-Scholes price of a European call (derived in Chapter 5)."""
    S, K = np.asarray(S, float), np.asarray(K, float)
    if T <= 0 or sigma <= 0:
        return np.maximum(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def bs_call_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * stats.norm.pdf(d1) * np.sqrt(T)


# ----------------------------------------------------------------------------- management fees
def management_fees(commitment, years, rate_ip, rate_post, ip_years, invested_cost):
    """Annual management fees charged at the start of each year t = 0..years-1.

    During the investment period (t < ip_years) the base is committed capital; afterwards it is the
    cost of investments not yet realised, invested_cost[t] (an array indexed by year).
    """
    fees = np.zeros(years)
    for t in range(years):
        fees[t] = rate_ip * commitment if t < ip_years else rate_post * invested_cost[t]
    return fees


# ----------------------------------------------------------------------------- PE whole-fund waterfall
@dataclass
class WaterfallResult:
    lp: np.ndarray            # LP share of each distribution
    gp: np.ndarray            # GP share (carried interest) of each distribution
    tiers: np.ndarray         # columns: return of capital, preferred return, catch-up to GP,
                              # catch-up to LP, split to LP, split to GP
    hurdle_balance: np.ndarray = field(default_factory=lambda: np.zeros(0))


def european_waterfall(c_t, c, d_t, d, hurdle=0.08, carry=0.20, catchup=1.0, hard=False):
    """Whole-fund (European) waterfall with an IRR-style compounding preferred return.

    c_t, c : contribution dates and amounts (LP capital, including capital called for fees)
    d_t, d : distribution dates and amounts available for distribution (before the split)
    hurdle : annual preferred return; carry: GP share of profit; catchup: share k of each dollar paid
             to the GP in the catch-up tier (1.0 = full catch-up, k must exceed carry);
    hard   : if True there is no catch-up and the GP earns carry only on profit above the hurdle.
    The LP hurdle balance is FV(contributions) - FV(LP distributions) at the hurdle rate; it is positive
    exactly when the LP's IRR to date is below the hurdle.
    """
    c_t, c, d_t, d = map(lambda a: np.asarray(a, float), (c_t, c, d_t, d))
    if not hard and catchup <= carry:
        raise ValueError("catch-up rate must exceed the carry rate")
    lp, gp = np.zeros(len(d)), np.zeros(len(d))
    tiers = np.zeros((len(d), 6))
    bal_hist = np.zeros(len(d))
    lp_dist_t, lp_dist = [], []
    unreturned = 0.0          # contributions not yet returned
    contributed = 0.0
    cum_total, cum_gp = 0.0, 0.0
    added = np.zeros(len(c), bool)
    for j in np.argsort(d_t, kind="stable"):
        t, x = d_t[j], d[j]
        for i in range(len(c)):           # contributions dated on or before t
            if not added[i] and c_t[i] <= t + 1e-12:
                added[i] = True; unreturned += c[i]; contributed += c[i]
        mask = added
        bal = (np.sum(c[mask] * (1 + hurdle) ** (t - c_t[mask]))
               - sum(a * (1 + hurdle) ** (t - s) for s, a in zip(lp_dist_t, lp_dist)))
        # tiers 1 and 2: return of capital, then preferred return, all to the LP
        a = min(x, max(bal, 0.0))
        roc = min(a, unreturned); pref = a - roc
        unreturned -= roc
        x -= a; lp_j, gp_j = a, 0.0
        tiers[j, 0], tiers[j, 1] = roc, pref
        if not hard and x > 0:
            # tier 3: GP takes share k of each dollar until GP total = carry x (total profit distributed)
            need = (carry * (cum_total + a - contributed) - cum_gp) / (catchup - carry)
            y = min(x, max(need, 0.0))
            tiers[j, 2], tiers[j, 3] = catchup * y, (1 - catchup) * y
            gp_j += catchup * y; lp_j += (1 - catchup) * y; x -= y
        if x > 0:   # tier 4: final split
            tiers[j, 4], tiers[j, 5] = (1 - carry) * x, carry * x
            lp_j += (1 - carry) * x; gp_j += carry * x
        lp[j], gp[j] = lp_j, gp_j
        cum_total += d[j]; cum_gp += gp_j
        lp_dist_t.append(t); lp_dist.append(lp_j)
        bal_hist[j] = bal - lp_j
    return WaterfallResult(lp, gp, tiers, bal_hist)


def carry_entitlement(c_t, c, d_t, d, carry=0.20, **kw):
    """Carry the GP may keep at the end on the whole fund's flows (used for clawback).

    The sequential whole-fund waterfall can pay more than carry x total profit when capital is called
    after carry has started (for fees or follow-on investments); the end-of-fund true-up caps the
    entitlement at carry x (total distributions - total contributions)^+.
    """
    seq = float(european_waterfall(c_t, c, d_t, d, carry=carry, **kw).gp.sum())
    return min(seq, carry * max(float(np.sum(d)) - float(np.sum(c)), 0.0))


def deal_by_deal(deals, hurdle=0.08, carry=0.20, catchup=1.0, hard=False):
    """American waterfall: each deal (dict with t_in, cost, t_out, proceeds) gets its own waterfall.

    Returns per-deal GP carry and the dates at which it is paid.
    """
    out = []
    for dl in deals:
        r = european_waterfall([dl["t_in"]], [dl["cost"]], [dl["t_out"]], [dl["proceeds"]],
                               hurdle=hurdle, carry=carry, catchup=catchup, hard=hard)
        out.append(float(r.gp[0]))
    return np.array(out)


def deal_by_deal_pooled(deals, hurdle=0.08, carry=0.20, catchup=1.0, hard=False):
    """Deal by deal with netting of realised deals: at each exit the GP's entitlement is the whole-fund
    waterfall applied to all deals realised so far (written-off deals count as realised with zero
    proceeds); the carry paid at that exit is the increase in entitlement, never negative.
    Returns the carry paid at each deal's exit, in the order of `deals`.
    """
    order = sorted(range(len(deals)), key=lambda i: deals[i]["t_out"])
    paid = np.zeros(len(deals)); total = 0.0
    for k in range(len(order)):
        real = [deals[i] for i in order[: k + 1]]
        ent = carry_entitlement([x["t_in"] for x in real], [x["cost"] for x in real],
                                [x["t_out"] for x in real], [x["proceeds"] for x in real],
                                hurdle=hurdle, carry=carry, catchup=catchup, hard=hard)
        paid[order[k]] = max(0.0, ent - total); total += paid[order[k]]
    return paid


def clawback(gp_received, gp_entitled, tax_rate=0.0):
    """Clawback owed and the amount recoverable when the claim is capped at carry net of tax."""
    owed = max(0.0, gp_received - gp_entitled)
    cap = (1 - tax_rate) * gp_received
    return owed, min(owed, cap)


# ----------------------------------------------------------------------------- carry as options
def carry_payoff(V, K, H, carry=0.2, catchup=1.0, hard=False):
    """GP carry at a single terminal date as a function of fund value V.

    K: capital contributed, H: hurdle amount K(1+h)^T. With catch-up rate k > carry the payoff is
    k(V-H)^+ - (k-carry)(V-V*)^+ with V* = (kH - carry K)/(k - carry). With a hard hurdle it is
    carry (V-H)^+.
    """
    V = np.asarray(V, float)
    if hard:
        return carry * np.maximum(V - H, 0.0)
    vstar = (catchup * H - carry * K) / (catchup - carry)
    return catchup * np.maximum(V - H, 0.0) - (catchup - carry) * np.maximum(V - vstar, 0.0)


def carry_option_value(V0, K, H, T, r, sigma, carry=0.2, catchup=1.0, hard=False):
    """Risk-neutral value of carry_payoff on a lognormal V, as a portfolio of calls."""
    if hard:
        return float(carry * bs_call(V0, H, T, r, sigma))
    vstar = (catchup * H - carry * K) / (catchup - carry)
    return float(catchup * bs_call(V0, H, T, r, sigma) - (catchup - carry) * bs_call(V0, vstar, T, r, sigma))


# ----------------------------------------------------------------------------- hedge fund fees
def hedge_fund_fees(gross, nav0=100.0, mgmt=0.02, incentive=0.20, hurdle=0.0, hurdle_type="hard",
                    hwm=True, modified=None):
    """Annual hedge fund fees.

    Order each year: gross return on the opening NAV; management fee = mgmt x opening NAV; incentive fee
    on the NAV after the management fee. Without a high-water mark the base is the gain over the
    opening NAV. With a high-water mark (the highest NAV after fees at any year end) the threshold is
    HWM x (1 + hurdle); a hard hurdle charges only the excess over the threshold, a soft hurdle charges
    the whole gain over the HWM once the threshold is passed.
    modified = (reduced_rate, multiple): after a loss below the HWM the manager earns reduced_rate on
    gains above the latest post-fee NAV until NAV has recovered `multiple` times the loss measured from
    the trough; above that level the full rate applies. Returns a dict of arrays.
    """
    n = len(gross)
    nav, mark = nav0, nav0         # mark: high-water mark, or the fee mark during a modified recovery
    rec = None                     # (trough, target, old_hwm) while in a modified recovery
    cols = {k: np.zeros(n) for k in ("open", "gross_nav", "mgmt", "pre", "inc", "close", "hwm", "net")}
    for t, g in enumerate(gross):
        gnav = nav * (1 + g)
        m = mgmt * nav
        pre = gnav - m
        if rec is not None:
            low_rate, mult = modified
            trough, target, old = rec
            fee = low_rate * max(0.0, min(pre, target) - mark) + incentive * max(0.0, pre - max(target, mark))
        elif hwm:
            thr = mark * (1 + hurdle)
            fee = 0.0 if pre <= thr else incentive * ((pre - thr) if hurdle_type == "hard" else (pre - mark))
        else:
            thr = nav * (1 + hurdle)
            fee = 0.0 if pre <= thr else incentive * ((pre - thr) if hurdle_type == "hard" else (pre - nav))
        close = pre - fee
        if modified is not None:
            mult = modified[1]
            if rec is None:
                if close < mark:                           # loss below the HWM: start a recovery
                    rec = (close, close + mult * (mark - close), mark); mark = close
                else:
                    mark = close if close > mark else mark
            else:
                trough, target, old = rec
                if close < trough:                         # a new low restarts the recovery from it
                    rec = (close, close + mult * (old - close), old); mark = close
                elif pre >= target:                        # recovered: an ordinary HWM again
                    rec = None; mark = close
                else:
                    mark = max(mark, close)
        elif hwm:
            mark = max(mark, close)
        cols["open"][t], cols["gross_nav"][t], cols["mgmt"][t], cols["pre"][t] = nav, gnav, m, pre
        cols["inc"][t], cols["close"][t], cols["hwm"][t] = fee, close, mark
        cols["net"][t] = close / nav - 1
        nav = close
    return cols


def fee_annuity_pv(aum0, gross, mgmt, incentive, disc, years):
    """PV of fees when the fund earns a constant gross return above the HWM every year.

    Fees each year are aum x [mgmt + incentive (gross - mgmt)], paid at year end, and the fund grows
    at the net rate g = (1 - incentive)(gross - mgmt). Closed form of a growing annuity.
    """
    phi = mgmt + incentive * (gross - mgmt)
    g = (1 - incentive) * (gross - mgmt)
    q = (1 + g) / (1 + disc)
    if abs(q - 1) < 1e-14:
        return phi * aum0 * years / (1 + disc)
    return phi * aum0 / (1 + disc) * (1 - q ** years) / (1 - q)


def fee_annuity_pv_loop(aum0, gross, mgmt, incentive, disc, years):
    pv, a = 0.0, aum0
    for t in range(1, years + 1):
        m = mgmt * a; inc = incentive * max(0.0, a * gross - m)
        pv += (m + inc) / (1 + disc) ** t
        a = a * (1 + gross) - m - inc
    return pv


# ----------------------------------------------------------------------------- real estate promote
def tiered_waterfall(c_t, c, d_t, d, tiers, lp_share_capital=0.9):
    """Real estate style promote waterfall with IRR or equity multiple hurdles.

    tiers: list of (kind, level, promote) in order; kind 'irr' (level = annual rate) or 'mult'
    (level = equity multiple). Within tier j the sponsor first takes `promote` of each dollar, and the
    rest is split pro rata to capital (investor share lp_share_capital). The tier ends when the
    investor's own flows reach the tier's hurdle. The last tier has no ceiling (use level=None).
    Returns arrays (investor, sponsor) of each distribution, totals by tier.
    """
    c_t, c, d_t, d = map(lambda a: np.asarray(a, float), (c_t, c, d_t, d))
    s = lp_share_capital
    inv, spo = np.zeros(len(d)), np.zeros(len(d))
    by_tier = np.zeros((len(tiers), 2))
    paid_t, paid = [], []
    for j in np.argsort(d_t, kind="stable"):
        t, x = d_t[j], d[j]
        m = c_t <= t + 1e-12
        for k, (kind, level, promo) in enumerate(tiers):
            if x <= 1e-15:
                break
            inv_frac = (1 - promo) * s           # investor's cut of each dollar in this tier
            if level is None:
                y = x
            else:
                if kind == "irr":
                    bal = (np.sum(s * c[m] * (1 + level) ** (t - c_t[m]))
                           - sum(a * (1 + level) ** (t - u) for u, a in zip(paid_t, paid)) - inv[j])
                else:
                    bal = level * s * np.sum(c[m]) - sum(paid) - inv[j]
                y = min(x, max(bal, 0.0) / inv_frac)
            inv[j] += inv_frac * y; spo[j] += (1 - inv_frac) * y
            by_tier[k] += (inv_frac * y, (1 - inv_frac) * y)
            x -= y
        paid_t.append(t); paid.append(inv[j])
    return inv, spo, by_tier


# ----------------------------------------------------------------------------- commitment pacing
def takahashi_alexander(years=12, rc=(0.25, 1 / 3, 0.5), growth=0.12, bow=2.5, life=12, yld=0.0,
                        commitment=1.0):
    """Takahashi and Alexander (2002) model of one fund's contributions, distributions and NAV.

    rc: contribution rates applied to uncalled commitment in years 1, 2, 3, ... (the last value is
    repeated). Distribution rate RD_t = max(yld, (t/life)^bow); D_t = RD_t NAV_{t-1}(1+growth);
    NAV_t = NAV_{t-1}(1+growth) + C_t - D_t.
    """
    C, D, NAV = np.zeros(years + 1), np.zeros(years + 1), np.zeros(years + 1)
    uncalled = commitment
    for t in range(1, years + 1):
        r = rc[min(t - 1, len(rc) - 1)]
        C[t] = r * uncalled; uncalled -= C[t]
        rd = max(yld, min(1.0, (t / life) ** bow))
        D[t] = rd * NAV[t - 1] * (1 + growth)
        NAV[t] = NAV[t - 1] * (1 + growth) + C[t] - D[t]
    return C, D, NAV


def programme(commit_by_year, horizon, **kw):
    """Aggregate contributions, distributions, NAV and uncalled commitments of a commitment programme."""
    C, D, NAV, U = (np.zeros(horizon + 1) for _ in range(4))
    for v, k in enumerate(commit_by_year):
        if k == 0:
            continue
        c, d, n = takahashi_alexander(years=horizon - v, commitment=k, **kw)
        C[v:] += c; D[v:] += d; NAV[v:] += n
        U[v:] += k - np.cumsum(c)
    return C, D, NAV, U


# ----------------------------------------------------------------------------- simulated funds
def simulate_fund(rng, n_deals=10, invest_years=5, mult_mu=0.50, mult_sd=0.75, p_loss=0.12,
                  hold_lo=2.0, hold_hi=7.0, commitment=100.0, invested=84.0):
    """One simulated buyout fund: deal dates, costs, exits and proceeds (no fees)."""
    t_in = np.sort(rng.uniform(0, invest_years, n_deals))
    cost = np.full(n_deals, invested / n_deals)
    hold = rng.uniform(hold_lo, hold_hi, n_deals)
    mult = np.exp(mult_mu + mult_sd * rng.standard_normal(n_deals))
    mult[rng.random(n_deals) < p_loss] = 0.0
    return [dict(t_in=a, cost=b, t_out=a + h, proceeds=b * m) for a, b, h, m in zip(t_in, cost, hold, mult)]


def fund_fee_flows(deals, commitment=100.0, years=10, ip_years=5, rate_ip=0.02, rate_post=0.015, cap=False):
    """Management fee contributions at t = 0..years-1 on committed, then unrealised, cost.

    With cap=True, cumulative fee calls are limited to the commitment left after the cost of the
    deals, so paid-in capital can never exceed the commitment; the second value returned is then
    the fee schedule after the limit and `fund_fee_flows.binding` records whether it bound."""
    unreal = np.array([sum(dl["cost"] for dl in deals if dl["t_in"] <= t < dl["t_out"]) for t in range(years)])
    f = np.asarray(management_fees(commitment, years, rate_ip, rate_post, ip_years, unreal), float)
    if cap:
        room = max(commitment - sum(dl["cost"] for dl in deals), 0.0)
        cum = np.cumsum(f)
        fund_fee_flows.binding = bool(cum[-1] > room + 1e-12)
        f = np.diff(np.minimum(cum, room), prepend=0.0)
    return np.arange(years, dtype=float), f
