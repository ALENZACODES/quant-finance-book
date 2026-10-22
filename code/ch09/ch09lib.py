"""Library for Chapter 9: month-end market series built from the daily FRED, Bank of Canada and
French files, constant maturity bond returns, time series momentum with volatility scaling,
hedge fund fees with a high-water mark, the survivorship and backfill biases of a fund database,
rolling regression clones, the incentive fees of a fund of funds, and risk contributions.

Returns are simple monthly returns as decimals unless a docstring says otherwise.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
MARKETS = ["EQ", "UST", "GOC", "OIL", "EUR", "JPY", "CAD"]
NAMES = dict(EQ="US equities", UST="US 10 year bond", GOC="Canada 10 year bond", OIL="WTI crude oil",
             EUR="euro", JPY="yen", CAD="Canadian dollar")
END = "2026-08"  # the French library ends in August 2026

# ----------------------------------------------------------------------------- data

def month_end(s: pd.Series) -> pd.Series:
    """Last available daily value in each calendar month, indexed by yyyymm integers."""
    s = s.dropna()
    m = s.groupby(s.index.to_period("M")).last()
    m.index = m.index.year * 100 + m.index.month
    return m


def load_factors():
    f = pd.read_csv(ROOT / "data" / "ff_factors_monthly.csv", index_col=0) / 100
    return f


def bond_price(c, y, m):
    """Price per unit face of a bond with annual coupon rate c paid twice a year, yield y (semiannual
    compounding) and m years to maturity, treating m as a whole number of half years (no accrued interest)."""
    n = 2 * m
    v = (1 + y / 2) ** (-n)
    return np.where(np.abs(y) < 1e-12, c / 2 * n + 1, c / y * (1 - v) + v)


def bond_return(y0, y1, m=10.0):
    """Monthly return of a par bond bought at yield y0 with m years to run, priced a month later at y1."""
    y0 = np.asarray(y0, float); y1 = np.asarray(y1, float)
    return bond_price(y0, y1, m - 1 / 12) - 1 + y0 / 12


def load_markets(end=END):
    """Monthly excess returns of the seven markets of the trend program, January 1999 (February 2001 for
    the Canadian bond) to `end`, as a DataFrame indexed by yyyymm. FX returns are spot only."""
    f = load_factors()
    d = pd.read_csv(ROOT / "data" / "fred_markets_daily.csv", index_col=0, parse_dates=True)
    g = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", index_col=0, parse_dates=True)
    y = month_end(d["DGS10"]) / 100
    yc = month_end(g["BD.CDN.10YR.DQ.YLD"]) / 100
    bc = month_end(g["TB.CDN.90D.MID"]) / 100
    oil = month_end(d["DCOILWTICO"]); eur = month_end(d["DEXUSEU"])
    jpy = month_end(d["DEXJPUS"]); cad = month_end(d["DEXCAUS"])
    out = pd.DataFrame(index=f.index[(f.index >= 199901) & (f.index <= int(end.replace("-", "")))])
    out["EQ"] = f["mkt_rf"]
    out["UST"] = pd.Series(bond_return(y.shift(1), y), index=y.index) - f["rf"]
    out["GOC"] = pd.Series(bond_return(yc.shift(1), yc), index=yc.index) - bc.shift(1) / 12
    out["OIL"] = oil / oil.shift(1) - 1
    out["EUR"] = eur / eur.shift(1) - 1
    out["JPY"] = jpy.shift(1) / jpy - 1
    out["CAD"] = cad.shift(1) / cad - 1
    out = out.loc[out.index >= 199902]
    return out[MARKETS], f["rf"].reindex(out.index)


def load_crypto(end="2026-09"):
    d = pd.read_csv(ROOT / "data" / "crypto_daily.csv", index_col=0, parse_dates=True)
    d = d.loc[:end + "-30"]
    return d

# ----------------------------------------------------------------------------- trend following

def trailing_vol(r: pd.DataFrame, window=24, min_periods=12):
    """Annualised standard deviation of the last `window` monthly returns, known at the end of each month."""
    return r.rolling(window, min_periods=min_periods).std() * math.sqrt(12)


def tsmom_weights(r: pd.DataFrame, lookback=12, target=0.20, window=24):
    """Positions held during month t+1, set at the end of month t: the sign of the past `lookback` month
    return times target / (N_t * sigma_t), N_t being the number of markets with a signal and a volatility."""
    past = np.expm1(np.log1p(r).rolling(lookback, min_periods=lookback).sum())
    vol = trailing_vol(r, window)
    sig = np.sign(past)
    ok = sig.notna() & vol.notna() & (vol > 0)
    n = ok.sum(axis=1).replace(0, np.nan)
    w = (sig * target / vol).where(ok).div(n, axis=0)
    return w.fillna(0.0)


def tsmom_returns(r, w, cost=0.0005):
    """Excess return of each market sleeve in month t+1 from positions set at the end of month t,
    less `cost` per unit of notional traded. Returns (sleeve returns, program returns)."""
    wl = w.shift(1).fillna(0.0)
    turn = (w - w.shift(1).fillna(0.0)).abs().shift(1).fillna(0.0)
    sleeves = wl * r.fillna(0.0) - cost * turn
    return sleeves, sleeves.sum(axis=1)


def fees_hwm(excess, rf, mgmt=0.02, perf=0.20, months=None):
    """Net excess return of a fund with gross excess return `excess` (plus the bill rate rf), a management
    fee of `mgmt` a year charged monthly and an incentive fee of `perf` on gains above the high-water mark,
    crystallised each December. `months` holds yyyymm labels."""
    excess = np.asarray(excess, float); rf = np.asarray(rf, float)
    nav, hwm, net = 1.0, 1.0, np.empty_like(excess)
    for t in range(len(excess)):
        start = nav
        nav *= (1 + rf[t] + excess[t]) * (1 - mgmt / 12)
        if months is None or months[t] % 100 == 12 or t == len(excess) - 1:
            if nav > hwm:
                nav -= perf * (nav - hwm)
            hwm = max(hwm, nav)
        net[t] = nav / start - 1 - rf[t]
    return net


def sharpe(x, periods=12):
    x = np.asarray(x, float)
    return math.sqrt(periods) * x.mean() / x.std(ddof=1)


def max_drawdown(r):
    w = np.cumprod(1 + np.asarray(r, float))
    w = np.concatenate([[1.0], w])
    return float(np.max(1 - w / np.maximum.accumulate(w)))


def drawdown_path(r):
    w = np.concatenate([[1.0], np.cumprod(1 + np.asarray(r, float))])
    return (w / np.maximum.accumulate(w) - 1)[1:]


def compound(r):
    return float(np.prod(1 + np.asarray(r, float)) - 1)


def sign_rule_mean(rho, sd_y):
    """E[sign(X) Y] for jointly normal X, Y with mean zero, correlation rho and sd(Y) = sd_y."""
    return math.sqrt(2 / math.pi) * rho * sd_y


def combined_sharpe(theta, n, rho):
    """Sharpe ratio of an equal risk mix of n sleeves each with Sharpe theta and pairwise correlation rho."""
    return theta * math.sqrt(n / (1 + (n - 1) * rho))

# ----------------------------------------------------------------------------- database biases

def mills(a):
    return stats.norm.pdf(a) / stats.norm.sf(a)


def survivor_mean(mu, sigma, T, c):
    """E[mean | mean > c] for the mean of T iid N(mu, sigma^2) returns."""
    s = sigma / math.sqrt(T)
    return mu + s * mills((c - mu) / s)


def simulate_database(rng, n_funds=3000, months=240, mu=0.006, sigma=0.03, alpha_sd=0.002,
                      incubation=24, entry_cut=0.0, death_cut=-0.15, hazard=0.004, life=120):
    """A universe of funds with iid normal monthly returns; fund means are mu plus a fund effect with
    standard deviation alpha_sd. A fund starts at a random month, enters the database only if its
    cumulative return over the first `incubation` months exceeds `entry_cut`, then backfills that history;
    it stops reporting when its trailing twelve month return falls below `death_cut`, at random with
    monthly probability `hazard`, or after `life` months in the database. Returns a dict of arrays."""
    start = rng.integers(-incubation - 60, months - incubation, n_funds)
    a = mu + alpha_sd * rng.standard_normal(n_funds)
    R = np.full((n_funds, months + incubation + 60), np.nan)
    rec = []
    off = incubation + 60
    for i in range(n_funds):
        s = start[i] + off
        T = R.shape[1] - s
        x = a[i] + sigma * rng.standard_normal(T)
        alive_len = T
        enters = np.prod(1 + x[:incubation]) - 1 > entry_cut
        # death after incubation
        for t in range(incubation, T):
            trail = np.prod(1 + x[t - 11:t + 1]) - 1
            if trail < death_cut or rng.random() < hazard or t - incubation >= life:
                alive_len = t + 1; break
        R[i, s:s + alive_len] = x[:alive_len]
        rec.append((s, s + incubation, s + alive_len, enters, alive_len == T))
    rec = np.array(rec, dtype=object)
    return dict(R=R[:, off:], start=np.array([r[0] for r in rec]) - off, entry=np.array([r[1] for r in rec]) - off,
                stop=np.array([r[2] for r in rec]) - off, enters=np.array([bool(r[3]) for r in rec]),
                survives=np.array([bool(r[4]) for r in rec]), alpha=a)


def database_views(db, months=240):
    """Average monthly return over the window [0, months) of four samples of the same funds: every
    fund-month (the truth), the funds still alive at the end of the window, every fund in the database
    with its backfilled history, and the database without the months before entry."""
    R = db["R"]; T = months
    idx = np.arange(T)
    truth, live, bf, clean = [], [], [], []
    for i in range(R.shape[0]):
        row = R[i, :T]; m = ~np.isnan(row)
        if not m.any():
            continue
        truth.append(row[m])
        if not db["enters"][i]:
            continue
        bf.append(row[m])
        after = m & (idx >= db["entry"][i])
        if after.any():
            clean.append(row[after])
        if db["stop"][i] >= T:
            live.append(row[m])
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])
    return dict(truth=cat(truth), live=cat(live), backfilled=cat(bf), clean=cat(clean),
                n_live=len(live), n_db=len(bf), n_all=len(truth))

# ----------------------------------------------------------------------------- replication

def rolling_clone(y, X, window=24):
    """Out of sample clone: at each month t >= window, fit y on X (no intercept) over the previous
    `window` months and apply the coefficients to X in month t. Returns (clone, fitted R2 in sample)."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    out = np.full(len(y), np.nan); r2 = []
    for t in range(window, len(y)):
        Xi, yi = X[t - window:t], y[t - window:t]
        b = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        out[t] = X[t] @ b
        e = yi - Xi @ b; r2.append(1 - e @ e / ((yi - yi.mean()) @ (yi - yi.mean())))
    return out, np.array(r2)


def prediction_inflation(k, T):
    """E[out of sample squared error] / sigma^2 for least squares on T draws of k zero mean Gaussian
    regressors without intercept (needs T > k + 1)."""
    return 1 + k / (T - k - 1)

# ----------------------------------------------------------------------------- funds of funds

def call_mean(mu, s):
    """E[max(X, 0)] for X ~ N(mu, s^2)."""
    if s == 0:
        return max(mu, 0.0)
    z = mu / s
    return mu * stats.norm.cdf(z) + s * stats.norm.pdf(z)


def pool_sd(sigma, n, rho):
    return sigma * math.sqrt(rho + (1 - rho) / n)

# ----------------------------------------------------------------------------- risk contributions

def risk_share(w, a, b, c):
    """Share of the variance of (1-w) P + w B due to B, with var(P)=a, var(B)=b, cov=c (Euler split)."""
    num = w * w * b + w * (1 - w) * c
    return num / ((1 - w) ** 2 * a + 2 * w * (1 - w) * c + w * w * b)


def weight_for_share(s, a, b, c):
    """The unique weight w in (0, 1) at which B carries the share s of portfolio variance."""
    A = (1 - s) * b; Bq = (1 - 2 * s) * c; C = -s * a
    u = (-Bq + math.sqrt(Bq * Bq - 4 * A * C)) / (2 * A)
    return u / (1 + u)


def block_bootstrap_idx(rng, T, block, reps):
    """Indices for a circular block bootstrap with fixed block length."""
    nb = -(-T // block)
    starts = rng.integers(0, T, (reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
    return idx.reshape(reps, -1)[:, :T]


def plain_log_axis(ax, ticks=(0.5, 1, 2, 3, 4, 6, 8)):
    """Log scale on y with plain tick labels such as 0.5, 1, 2, rather than powers of ten."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    lo, hi = ax.get_ylim()
    ax.yaxis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

# ----------------------------------------------------------------------------- fee mechanics

def incentive_path(gross, f, hwm=True):
    """Per-unit NAV path of a fund charging an incentive fee f at the end of each period on the gain above
    the high-water mark (hwm=True) or above the previous NAV (hwm=False). gross: period returns before the
    incentive fee. Returns (fees per unit at each period end, final NAV, final high-water mark)."""
    nav, mark, fees = 1.0, 1.0, np.zeros(len(gross))
    for t, g in enumerate(gross):
        pre = nav * (1 + g)
        base = mark if hwm else nav
        fees[t] = f * max(pre - base, 0.0)
        nav = pre - fees[t]
        mark = max(mark, nav)
    return fees, nav, mark


def hurdle_fee(R, f, h, kind):
    """Incentive fee per unit of assets on a period return R with hurdle h: 'none', 'hard' (fee on the
    excess over h) or 'soft' (no fee below h, full catch-up above it, then f of the whole gain)."""
    R = np.asarray(R, float)
    if kind == "none":
        return f * np.maximum(R, 0)
    if kind == "hard":
        return f * np.maximum(R - h, 0)
    return np.minimum(np.maximum(R - h, 0), f * np.maximum(R, 0))


def hurdle_fee_mean(mu, s, f, h, kind):
    """Expected incentive fee when R ~ N(mu, s^2), from the call mean G of eq9:callmean."""
    if kind == "none":
        return f * call_mean(mu, s)
    if kind == "hard":
        return f * call_mean(mu - h, s)
    return call_mean(mu - h, s) - (1 - f) * call_mean(mu - h / (1 - f), s)


def fee_option(f, k, sigma, r, T):
    """Value per unit of assets of an incentive fee f on the gain above k times the starting value at T,
    when the assets grow at r under the pricing measure with volatility sigma (a Black-Scholes call)."""
    v = sigma * math.sqrt(T)
    d1 = (math.log(1 / k) + r * T) / v + v / 2
    return f * (stats.norm.cdf(d1) - k * math.exp(-r * T) * stats.norm.cdf(d1 - v))

# ----------------------------------------------------------------------------- liquidation at a barrier

def survival_barrier(mu, s, b, T):
    """P(min_{t<=T} X_t > -b) for X a Brownian motion with drift mu and volatility s, X_0 = 0, b > 0."""
    v = s * math.sqrt(T)
    return stats.norm.cdf((b + mu * T) / v) - math.exp(-2 * mu * b / s ** 2) * stats.norm.cdf((-b + mu * T) / v)


def survivor_mean_barrier(mu, s, b, T):
    """E[X_T | min_{t<=T} X_t > -b], from the closed form for E[X_T; survival]."""
    m, v = mu * T, s * math.sqrt(T)
    k = math.exp(-2 * mu * b / s ** 2)
    first = m * stats.norm.cdf((m + b) / v) + v * stats.norm.pdf((m + b) / v)
    second = (m - 2 * b) * stats.norm.cdf((m - b) / v) + v * stats.norm.pdf((m - b) / v)
    return (first - k * second) / survival_barrier(mu, s, b, T)


BGK = 0.5826  # -zeta(1/2)/sqrt(2 pi), the continuity correction of Broadie, Glasserman and Kou

# ----------------------------------------------------------------------------- carry

def load_cad_carry(end="2026-09"):
    """Monthly USD/CAD (CAD per USD) and one month bill yields of Canada and the US, month-end, from the
    Chapter 5 file. Returns a DataFrame indexed by yyyymm with the excess return in US dollars of holding
    Canadian bills funded at the US bill rate, split into spot, carry and cross terms."""
    d = pd.read_csv(ROOT / "data" / "usdcad_rates.csv", index_col=0, parse_dates=True).loc[:end + "-30"]
    s = month_end(d["usdcad"]); ic = month_end(d["cad_tbill_1m"]) / 100; iu = month_end(d["us_tbill_1m"]) / 100
    out = pd.DataFrame(dict(S=s, ic=ic.reindex(s.index), iu=iu.reindex(s.index)))
    spot = out["S"].shift(1) / out["S"] - 1
    carry = (out["ic"].shift(1) - out["iu"].shift(1)) / 12
    cross = out["ic"].shift(1) / 12 * spot
    out["spot"], out["carry"], out["cross"] = spot, carry, cross
    out["excess"] = (1 + out["ic"].shift(1) / 12) * out["S"].shift(1) / out["S"] - (1 + out["iu"].shift(1) / 12)
    return out.dropna()


def goc_carry(end=END):
    """Carry and roll-down of the Canada 10 year par bond each month: the excess return it would earn if
    the curve did not move, with the 9 year 11 month yield interpolated between the 7 and 10 year yields.
    Returns a DataFrame indexed by yyyymm: carry (yield less bill rate), rolldown, expected (their sum,
    known a month ahead) and realised (the same bond priced at the next month's interpolated yield)."""
    g = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", index_col=0, parse_dates=True)
    y10 = month_end(g["BD.CDN.10YR.DQ.YLD"]) / 100; y7 = month_end(g["BD.CDN.7YR.DQ.YLD"]) / 100
    b = month_end(g["TB.CDN.90D.MID"]) / 100
    df = pd.DataFrame(dict(y10=y10, y7=y7.reindex(y10.index), b=b.reindex(y10.index))).dropna()
    df = df.loc[df.index <= int(end.replace("-", ""))]
    yroll = df["y10"] - (df["y10"] - df["y7"]) / 3 / 12
    carry = (df["y10"] - df["b"]) / 12
    roll = pd.Series(bond_price(df["y10"].values, yroll.values, 10 - 1 / 12) - 1, index=df.index)
    # realised: the bond bought a month ago is priced at this month's 9 year 11 month yield
    real = pd.Series(bond_price(df["y10"].shift(1).values, yroll.values, 10 - 1 / 12) - 1, index=df.index) \
        + df["y10"].shift(1) / 12 - df["b"].shift(1) / 12
    out = pd.DataFrame(dict(carry=carry.shift(1), roll=roll.shift(1), real=real)).dropna()
    out["expected"] = out["carry"] + out["roll"]
    return out

# ----------------------------------------------------------------------------- on-chain arithmetic

def block_subsidy_sats(height, interval=210_000):
    """Block subsidy in satoshis at a block height: 50 coins halved by a right shift every `interval`
    blocks, zero after 64 halvings (the rule of GetBlockSubsidy in Bitcoin Core)."""
    h = height // interval
    return 0 if h >= 64 else (50 * 100_000_000) >> h


def total_supply_sats(interval=210_000):
    return sum(interval * ((50 * 100_000_000) >> h) for h in range(64))


def catch_up_ruin(q, z):
    """Probability that an attacker with share q of the hash rate ever overtakes an honest chain that is
    z blocks ahead (gambler's ruin)."""
    p = 1 - q
    return 1.0 if q >= p else (q / p) ** z


def catch_up_nakamoto(q, z):
    """Nakamoto's probability, which weights the ruin probability by the Poisson number of blocks the
    attacker has found while the honest chain found z."""
    p = 1 - q; lam = z * q / p; s = 1.0; pois = math.exp(-lam)
    for k in range(z + 1):
        if k > 0:
            pois *= lam / k
        s -= pois * (1 - (q / p) ** (z - k))
    return s


def fisher_test(r1, n1, r2, n2):
    """Fisher z test of equal correlations in two independent samples: statistic and two-sided p-value."""
    z = (math.atanh(r2) - math.atanh(r1)) / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return z, 2 * stats.norm.sf(abs(z))


def ols(y, X):
    """Least squares with an intercept: coefficients (intercept first), residuals and classical standard errors."""
    y = np.asarray(y, float); A = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b; s2 = e @ e / (len(y) - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))
    return b, e, se
