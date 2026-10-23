"""Section 6.8: direct lending. Exact results: the unitranche split, spreads from default and recovery,
the covenant as a barrier, the fee of a business development company at its leverage and the tender
queue of a semi-liquid fund. Historical results: the accounts of nineteen listed business development
companies from SEC EDGAR (data/bdc_quarterly.csv) and the Form PF returns of credit hedge funds.
Simulation: a Monte Carlo check of the covenant barrier.

Run on its own:  python code/ch06/direct_lending.py   (writes code/ch06/out/direct_lending.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from numbers_ch06 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT  # noqa: F401

BDC = ROOT / "data" / "bdc_quarterly.csv"
BILLS = ROOT / "data" / "usdcad_rates.csv"
OFR = ROOT / "data" / "ofr_hedge_funds_quarterly.csv"
FF = ROOT / "data" / "ff_factors_monthly.csv"
FIRST = pd.Timestamp("2021-06-30")   # quarterly XBRL coverage of BDC accounts is broad from here
MIN_FIRMS = 10

# Figures quoted from public sources; each is cited where the chapter uses it.
SOURCE = dict(
    # Cai and Haque (2024), FEDS Notes: post-default value (recovery) of direct loans, syndicated loans
    # and high-yield bonds, from KBRA DLD; mean interest coverage of private credit borrowers
    rec_direct=0.33, rec_synd=0.52, rec_hy=0.39, icr_private=2.0, icr_lev=2.7,
    # Ares Capital Corporation, Form 10-K for 2025: base fee on total assets other than cash, by
    # leverage band; quarterly hurdle and catch-up of the income based fee; asset coverage; non-accruals
    arcc_base_lo=0.015, arcc_base_hi=0.010, arcc_hurdle_q=0.0175, arcc_catchup_q=0.021875,
    arcc_inc=0.20, arcc_cover=1.89, cover_min=1.50, arcc_nonacc_cost=0.018, arcc_nonacc_fv=0.012,
    arcc_yield_cost=0.103,
    # Blackstone Private Credit Fund, Form 10-K for 2025: quarterly repurchases of up to 5% of NAV, and
    # 98% of NAV for shares held less than a year
    tender_cap=0.05, early_price=0.98,
)


# ------------------------------------------------------------------------------------------------ exact
def loss_spread(el, T):
    """Annual spread, continuously compounded, that repays an expected loss el (share of face) over T."""
    return -math.log(1 - el) / T


def unitranche_split(V0, g, sigma, T, cost, face, fo, s_u):
    """Expected losses of a unitranche of the given face and of its first-out piece (0..fo) and last-out
    piece (fo..face), by the slice formula of Proposition 6.x, and the spreads that give both pieces the
    same premium over their loss spreads while the blend of the two pays s_u."""
    el = {k: L.slice_el(a, b, V0, g, sigma, T, cost) for k, (a, b) in
          dict(u=(0, face), f=(0, fo), l=(fo, face)).items()}
    w = fo / face
    ls = dict(u=loss_spread(el["u"] / face, T), f=loss_spread(el["f"] / fo, T),
              l=loss_spread(el["l"] / (face - fo), T))
    # common premium pi with w (pi + ls_f) + (1 - w)(pi + ls_l) = s_u
    pi = s_u - (w * ls["f"] + (1 - w) * ls["l"])
    return dict(el=el, w=w, ls=ls, pi=pi, s_f=pi + ls["f"], s_l=pi + ls["l"])


def breach_prob(x0, b, nu, sigma, T):
    """P(min_{t<=T} X_t <= b) for X_t = x0 exp(nu t + sigma W_t), b < x0 (reflection principle)."""
    a = math.log(b / x0); s = sigma * math.sqrt(T)
    return L.ncdf((a - nu * T) / s) + math.exp(2 * nu * a / sigma ** 2) * L.ncdf((a + nu * T) / s)


def breach_mc(x0, b, nu, sigma, T, steps, paths, rng, shift=True, batch=50_000):
    """Monte Carlo of the same probability with monitoring at `steps` dates. A discretely monitored path
    crosses a barrier less often than a continuous one; with shift the barrier is moved up by
    exp(0.5826 sigma sqrt(dt)), the correction of Broadie, Glasserman and Kou (1997), so that the discrete
    count approximates the continuous probability."""
    dt = T / steps
    level = b * math.exp(0.5826 * sigma * math.sqrt(dt)) if shift else b
    hits = 0
    for k in range(0, paths, batch):
        z = rng.standard_normal((min(batch, paths - k), steps))
        lx = math.log(x0) + np.cumsum(nu * dt + sigma * math.sqrt(dt) * z, axis=1)
        hits += int((lx.min(axis=1) <= math.log(level)).sum())
    return hits / paths


def bdc_base_fee(de, lo=SOURCE["arcc_base_lo"], hi=SOURCE["arcc_base_hi"]):
    """Base fee as a share of net assets for a fund with debt to equity de, charging lo on assets financed
    up to 1x debt to equity and hi above (cash ignored, so total assets are 1 + de per unit of equity)."""
    return lo * (1 + min(de, 1.0)) + hi * max(de - 1.0, 0.0)


def income_fee(x, hurdle=SOURCE["arcc_hurdle_q"], catch=SOURCE["arcc_catchup_q"], f=SOURCE["arcc_inc"]):
    """Quarterly income based fee on pre-fee net investment income x (a return on net assets): nothing up
    to the hurdle, all of the excess up to the catch-up level, then f of the whole."""
    if x <= hurdle:
        return 0.0
    if x <= catch:
        return x - hurdle
    return f * x


def coverage_threshold(d, kappa=SOURCE["cover_min"]):
    """Largest share of assets a fund with debt to equity d can lose before its asset coverage, assets
    over debt, falls below kappa: (1 + d)(1 - L)/d >= kappa  <=>  L <= 1 - kappa d/(1 + d)."""
    return 1 - kappa * d / (1 + d)


def coverage_breach_prob(d, p, rho, lgd, kappa=SOURCE["cover_min"]):
    """Large pool probability, over one period, that default losses on a loan book with default
    probability p, factor correlation rho and loss given default lgd breach the asset coverage test."""
    x = coverage_threshold(d, kappa) / lgd
    if x <= 0:
        return 1.0
    if x >= 1:
        return 0.0
    return float(1 - L.lhp_cdf(x, p, rho))


def lhp_quantile(q, p, rho):
    """Default share at quantile q in the large pool limit (inverse of Proposition 6 large pool)."""
    from scipy import stats
    return float(L.ncdf((stats.norm.ppf(p) + math.sqrt(rho) * stats.norm.ppf(q)) / math.sqrt(1 - rho)))


def yield_to_life(c, u, T):
    """Continuously compounded yield of a loan bought at 1 - u that pays a coupon c continuously and
    par at T: solves 1 - u = c (1 - e^{-yT}) / y + e^{-yT}."""
    from scipy import optimize
    f = lambda y: c * (1 - math.exp(-y * T)) / y + math.exp(-y * T) - (1 - u)
    return optimize.brentq(f, 1e-9, 2.0)


def tender_exit(c, f, n):
    """Share of a holding still in the fund after n quarters for an investor who tenders all of it every
    quarter, when tenders total f of NAV each quarter, the fund buys back c, and fills pro rata."""
    fill = min(1.0, c / f)
    return (1 - fill) ** n


# ------------------------------------------------------------------------------------------- historical
def load_bdc():
    d = pd.read_csv(BDC, parse_dates=["quarter_end"])
    return d[d.quarter_end <= pd.Timestamp("2026-09-30")]


def bill_quarterly():
    b = pd.read_csv(BILLS, parse_dates=["date"]).set_index("date").us_tbill_1m.dropna()
    q = b.groupby(b.index.to_period("Q")).mean()
    q.index = q.index.to_timestamp("Q").normalize()
    return q / 100


def bdc_panel(d):
    """Per company and quarter: return on net assets (result of operations over last quarter's net
    assets), portfolio income yield on average cost, fees over income, liabilities over net assets and
    fair value over cost. Ratios that need last quarter use consecutive quarters only."""
    out = []
    for t, g in d.groupby("ticker"):
        g = g.sort_values("quarter_end").set_index("quarter_end")
        q = g.index.to_period("Q")
        consec = np.r_[False, (q[1:] - q[:-1]).map(lambda x: x.n) == 1]
        p = pd.DataFrame(index=g.index)
        p["ticker"] = t
        p["roe"] = np.where(consec, g.operations / g.net_assets.shift(1), np.nan)
        p["yield"] = np.where(consec, 4 * g.income / ((g.cost + g.cost.shift(1)) / 2), np.nan)
        p["fee_share"] = (g.mgmt_fee + g.incentive_fee) / g.income
        p["fee_na"] = np.where(consec, 4 * (g.mgmt_fee + g.incentive_fee) / g.net_assets.shift(1), np.nan)
        p["lev"] = g.liabilities / g.net_assets
        p["mark"] = g.fv / g.cost
        p["fv"] = g.fv; p["cost"] = g.cost
        out.append(p)
    p = pd.concat(out)
    return p[p.index >= FIRST]


def pooled_rho(p, col="roe"):
    """First autocorrelation of a panel, pooled across companies after removing each company's mean,
    over pairs of consecutive quarters; with its standard error 1/sqrt(pairs)."""
    xs, ys = [], []
    for _, g in p.groupby("ticker"):
        s = g[col].dropna()
        if len(s) < 4:
            continue
        s = s - s.mean()
        q = s.index.to_period("Q")
        ok = (q[1:] - q[:-1]).map(lambda x: x.n) == 1
        xs += list(s.values[:-1][ok]); ys += list(s.values[1:][ok])
    xs, ys = np.array(xs), np.array(ys)
    return float(xs @ ys / math.sqrt((xs @ xs) * (ys @ ys))), len(xs)


def ew(p, col):
    g = p.groupby(level=0)[col]
    m = g.mean()[g.count() >= MIN_FIRMS]
    return m


def desmooth(r, phi):
    """Proposition 8.x (iv): r_t = (r*_t - phi r*_{t-1}) / (1 - phi)."""
    r = np.asarray(r, float)
    return (r[1:] - phi * r[:-1]) / (1 - phi)


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se, 1 - e @ e / np.sum((y - y.mean()) ** 2)


def market_quarterly():
    f = pd.read_csv(FF)
    f.index = pd.to_datetime(f.month.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    m = (1 + (f.mkt_rf + f.rf) / 100).groupby(f.index.to_period("Q")).agg(["prod", "count"])
    m = m[m["count"] == 3]["prod"] - 1
    m.index = m.index.to_timestamp("Q").normalize()
    return m


def quarter_label(t):
    return f"{t.year}Q{(t.month - 1) // 3 + 1}"


# ------------------------------------------------------------------------------------------------ main
def exact_parts():
    S = SOURCE
    # spreads from default and recovery: a loss spread of 100 bp pays for a default rate of 1/LGD %
    for k, nm in [("rec_direct", "Direct"), ("rec_synd", "Synd"), ("rec_hy", "Hy")]:
        pct(f"dlRec{nm}", S[k], 0); pct(f"dlLgd{nm}", 1 - S[k], 0)
        put(f"dlPdPer{nm}", 1 / (1 - S[k]), "{:.2f}")
    put("dlIcrPrivate", S["icr_private"], "{:.1f}"); put("dlIcrLev", S["icr_lev"], "{:.1f}")
    # a default rate of 2% a year
    pct("dlLossDirectTwo", 0.02 * (1 - S["rec_direct"]), 2); pct("dlLossSyndTwo", 0.02 * (1 - S["rec_synd"]), 2)
    # BDC fees at the leverage of the asset coverage test
    de_max = 1 / (S["cover_min"] - 1); de_arcc = 1 / (S["arcc_cover"] - 1)
    put("dlDeMax", de_max, "{:.0f}"); put("dlDeArcc", de_arcc, "{:.2f}")
    pct("dlCoverMin", S["cover_min"], 0); pct("dlCoverArcc", S["arcc_cover"], 0)
    pct("dlBaseLo", S["arcc_base_lo"], 1); pct("dlBaseHi", S["arcc_base_hi"], 1)
    pct("dlBaseNaArcc", bdc_base_fee(de_arcc), 2); pct("dlBaseNaMax", bdc_base_fee(de_max), 2)
    pct("dlHurdleQ", S["arcc_hurdle_q"], 2); pct("dlCatchQ", S["arcc_catchup_q"], 4)
    pct("dlHurdleA", 4 * S["arcc_hurdle_q"], 0); pct("dlIncRate", S["arcc_inc"], 0)
    RAW["dlCatchCheck"] = S["arcc_hurdle_q"] / (1 - S["arcc_inc"]) - S["arcc_catchup_q"]
    pct("dlNonaccCost", S["arcc_nonacc_cost"], 1); pct("dlNonaccFv", S["arcc_nonacc_fv"], 1)
    pct("dlYieldArcc", S["arcc_yield_cost"], 1)
    # an asset yield of y on 1 + de of assets, debt at r_d: return on equity before and after fees
    y, rd = S["arcc_yield_cost"], 0.06   # the debt cost is an assumption of the example
    pct("dlExRd", rd, 0)
    gross_roe = y * (1 + de_arcc) - rd * de_arcc
    base = bdc_base_fee(de_arcc)
    pre = gross_roe - base
    inc = 4 * income_fee(pre / 4)
    pct("dlExGross", gross_roe, 1); pct("dlExBase", base, 2); pct("dlExInc", inc, 2)
    pct("dlExNet", pre - inc, 1); pct("dlExFeeShare", (base + inc) / gross_roe, 0)
    RAW["dlEx"] = dict(gross=gross_roe, base=base, inc=inc, net=pre - inc, de=de_arcc)
    # tender queue
    c = S["tender_cap"]; pct("dlCap", c, 0); pct("dlEarly", S["early_price"], 0)
    for f, nm in [(0.10, "Ten"), (0.20, "Twenty")]:
        n90 = math.log(0.1) / math.log(1 - c / f)
        pct(f"dlFill{nm}", c / f, 0); put(f"dlQuarters{nm}", n90, "{:.1f}")
        pct(f"dlLeftFour{nm}", tender_exit(c, f, 4), 0)
        RAW[f"dlN90{nm}"] = n90


def unitranche():
    # an illustration: enterprise value 100, a unitranche to 50 split 20 first out and 30 last out
    V0, g, sig, T, cost, face, fo = 100.0, 0.0, 0.35, 5, 0.20, 50.0, 20.0
    put("utV", V0, "{:.0f}"); put("utFace", face, "{:.0f}"); put("utFo", fo, "{:.0f}"); put("utLo", face - fo, "{:.0f}")
    pct("utSig", sig, 0); put("utT", T, "{:.0f}"); pct("utCost", cost, 0)
    s_u = RAW["dlSpreadLast"]
    r = unitranche_split(V0, g, sig, T, cost, face, fo, s_u)
    RAW["ut"] = dict(el={k: float(v) for k, v in r["el"].items()}, w=r["w"], ls=r["ls"], pi=r["pi"],
                     s_f=r["s_f"], s_l=r["s_l"], s_u=s_u)
    pct("utW", r["w"], 0)
    for k, nm in [("u", "U"), ("f", "F"), ("l", "L")]:
        face_k = dict(u=face, f=fo, l=face - fo)[k]
        pct(f"utEl{nm}", r["el"][k] / face_k, 1)
        put(f"utLs{nm}", 1e4 * r["ls"][k], "{:.0f}")
    put("utSu", 1e4 * s_u, "{:.0f}"); put("utPi", 1e4 * r["pi"], "{:.0f}")
    put("utSf", 1e4 * r["s_f"], "{:.0f}"); put("utSl", 1e4 * r["s_l"], "{:.0f}")
    # the whole loan's loss spread against the blend of the pieces' loss spreads
    put("utLsBlend", 1e4 * (r["w"] * r["ls"]["f"] + (1 - r["w"]) * r["ls"]["l"]), "{:.0f}")


def covenant():
    # an illustration: debt at 5 times EBITDA, a maintenance covenant at 6.5 times, a cov-lite loan that
    # defaults only when EBITDA no longer covers interest at 10% (debt at 10 times EBITDA)
    x0, D, c_cov, rate, sig, nu, T = 1.0, 5.0, 6.5, 0.10, 0.25, 0.0, 5
    b_cov, b_lite = D / c_cov, rate * D
    put("cvLev", D, "{:.0f}"); put("cvCov", c_cov, "{:.1f}"); pct("cvRate", rate, 0); pct("cvSig", sig, 0)
    put("cvT", T, "{:.0f}")
    pct("cvDrop", 1 - b_cov / x0, 0); pct("cvDropLite", 1 - b_lite / x0, 0)
    p_cov, p_lite = breach_prob(x0, b_cov, nu, sig, T), breach_prob(x0, b_lite, nu, sig, T)
    pct("cvPcov", p_cov, 0); pct("cvPlite", p_lite, 0); put("cvTimes", p_cov / p_lite, "{:.0f}")
    # recovery at the trigger if the firm is sold at a multiple m of EBITDA, after costs
    m, cost = 8.0, 0.20
    put("cvMult", m, "{:.0f}"); pct("cvCost", cost, 0)
    rec_cov = min(1.0, (1 - cost) * m * b_cov / D); rec_lite = min(1.0, (1 - cost) * m * b_lite / D)
    pct("cvRecCov", rec_cov, 0); pct("cvRecLite", rec_lite, 0)
    RAW["cv"] = dict(p_cov=p_cov, p_lite=p_lite, rec_cov=rec_cov, rec_lite=rec_lite, b_cov=b_cov, b_lite=b_lite,
                     x0=x0, nu=nu, sig=sig, T=T)
    rng = rng_for(10)
    mc = breach_mc(x0, b_cov, nu, sig, T, 60, 200_000, rng, shift=True)
    mc_raw = breach_mc(x0, b_cov, nu, sig, T, 60, 200_000, rng_for(10), shift=False)
    RAW["cv"]["mc"] = mc; RAW["cv"]["mc_raw"] = mc_raw
    pct("cvMc", mc, 1); pct("cvMcRaw", mc_raw, 1); pct("cvPcovOne", p_cov, 1)
    put("cvMcSe", 100 * math.sqrt(mc * (1 - mc) / 200_000), "{:.2f}")
    return b_cov, b_lite, x0, nu, sig, T


def historical():
    d = load_bdc(); p = bdc_panel(d)
    bills = bill_quarterly()
    put("dlFirms", p.ticker.nunique(), "{:.0f}")
    N["dlFirst"] = quarter_label(p.index.min()); N["dlLast"] = quarter_label(p.index.max())
    y = ew(p, "yield"); ymed = p.groupby(level=0)["yield"].median()[y.index]
    b = bills.reindex(y.index)
    spread = ymed - b
    RAW["dlSpreadLast"] = float(spread.iloc[-1])
    N["dlYieldFirstQ"] = quarter_label(y.index[0]); put("dlYieldN", len(y), "{:.0f}")
    pct("dlYieldFirst", ymed.iloc[0], 1); pct("dlYieldPeak", ymed.max(), 1)
    N["dlYieldPeakQ"] = quarter_label(ymed.idxmax()); pct("dlYieldLast", ymed.iloc[-1], 1)
    pct("dlBillFirst", b.iloc[0], 2); pct("dlBillPeak", b.max(), 1); pct("dlBillLast", b.iloc[-1], 1)
    put("dlSpreadLastBp", 1e4 * spread.iloc[-1], "{:.0f}"); put("dlSpreadFirstBp", 1e4 * spread.iloc[0], "{:.0f}")
    bb, se, r2 = ols(ymed.values, b.values)
    put("dlPass", bb[1]); put("dlPassSe", se[1]); put("dlPassRsq", r2)
    RAW["dlPassRaw"] = [float(bb[0]), float(bb[1])]
    # interest coverage of a floating-rate borrower across the rise in rates
    b_low = float(bills[(bills.index >= "2021-01-01") & (bills.index <= "2021-12-31")].mean())
    b_high = float(bills.max()); N["dlBillHighQ"] = quarter_label(bills.idxmax())
    s_ex = RAW["dlSpreadLast"]
    icr_high = SOURCE["icr_private"]
    icr_low = icr_high * (b_high + s_ex) / (b_low + s_ex)
    pct("icBillLow", b_low, 2); pct("icBillHigh", b_high, 1); put("icLow", icr_low, "{:.1f}")
    pct("icCostRise", (b_high + s_ex) / (b_low + s_ex) - 1, 0)
    RAW["ic"] = dict(b_low=b_low, b_high=b_high, s=s_ex, icr_low=icr_low)
    # fees
    fs = p.groupby(level=0)["fee_share"].median().dropna(); fna = p.groupby(level=0)["fee_na"].median().dropna()
    pct("dlFeeShare", float(p.fee_share.median()), 0); pct("dlFeeNa", float(p.fee_na.median()), 1)
    # leverage and marks
    last = p[p.index == p.index.max()]
    put("dlLevMed", last.lev.median(), "{:.2f}"); put("dlLevMax", last.lev.max(), "{:.2f}")
    agg = p.groupby(level=0)[["fv", "cost"]].sum(min_count=MIN_FIRMS).dropna()
    mark = agg.fv / agg.cost
    pct("dlMarkMin", mark.min(), 1); N["dlMarkMinQ"] = quarter_label(mark.idxmin())
    pct("dlMarkFirst", mark.iloc[0], 1); pct("dlMarkLast", mark.iloc[-1], 1)
    # one row per company in the last quarter
    names = {"ARCC": "Ares Capital", "BXSL": "Blackstone Secured Lending", "MAIN": "Main Street Capital",
             "OBDC": "Blue Owl Capital", "GBDC": "Golub Capital BDC", "HTGC": "Hercules Capital",
             "FSK": "FS KKR Capital", "TSLX": "Sixth Street Specialty Lending", "CSWC": "Capital Southwest",
             "OCSL": "Oaktree Specialty Lending", "GSBD": "Goldman Sachs BDC", "BBDC": "Barings BDC",
             "FDUS": "Fidus Investment", "CGBD": "Carlyle Secured Lending", "PFLT": "PennantPark Floating Rate",
             "NMFC": "New Mountain Finance", "SLRC": "SLR Investment", "TCPC": "BlackRock TCP Capital",
             "GAIN": "Gladstone Investment"}
    dd = d.set_index("quarter_end")
    rows = []
    for t in sorted(names, key=lambda t: -float(dd[dd.ticker == t].net_assets.dropna().iloc[-1])):
        g = p[p.ticker == t]; raw = dd[dd.ticker == t]
        r = dict(t=t, name=names[t], na=float(raw.net_assets.dropna().iloc[-1]) / 1e9,
                 lev=float(g.lev.dropna().iloc[-1]), mark=float(g.mark.dropna().iloc[-1]),
                 fee=float(g.fee_share.dropna().median()) if g.fee_share.notna().any() else float("nan"),
                 roe=4 * float(g.roe.mean()))
        rows.append(r)
    RAW["dlTable"] = rows
    lines = []
    for r in rows:
        fee = "--" if math.isnan(r["fee"]) else f"{100 * r['fee']:.0f}"
        lines.append(f"{r['name']} & {r['na']:.1f} & {r['lev']:.2f} & {100 * r['mark']:.1f} & {fee} & "
                     + (f"{100 * r['roe']:.1f}" if r['roe'] >= 0 else f"$-${abs(100 * r['roe']):.1f}") + r"\\")
    (ROOT / "chapters" / "ch06-bdc-table.tex").write_text(
        "% Generated by code/ch06/direct_lending.py. Do not edit by hand.\n"
        "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
        "company & net assets (\\$bn) & liab./NA & FV/cost (\\%) & fees (\\%) & return (\\%)\\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # returns on net assets
    roe = ew(p, "roe")
    pct("dlRoeMean", 4 * roe.mean(), 1); pct("dlRoeVol", 2 * roe.std(ddof=1), 1)
    pct("dlRoeMin", roe.min(), 2); N["dlRoeMinQ"] = quarter_label(roe.idxmin())
    rho, npairs = pooled_rho(p)
    put("dlRho", rho); put("dlRhoN", npairs, "{:.0f}"); put("dlRhoSe", 1 / math.sqrt(npairs))
    RAW["dlRhoRaw"] = rho
    rho_ew = float(np.corrcoef(roe.values[:-1], roe.values[1:])[0, 1]); put("dlRhoEw", rho_ew)
    put("dlRoeN", len(roe), "{:.0f}")
    # Form PF credit funds: de-smoothing a longer series
    o = pd.read_csv(OFR, index_col=0, parse_dates=True)["strategy_credit_netreturn"].dropna() / 100
    phi = float(np.corrcoef(o.values[:-1], o.values[1:])[0, 1])
    ds = desmooth(o.values, phi)
    put("dlPfPhi", phi); pct("dlPfVol", 2 * o.std(ddof=1), 1); pct("dlPfVolDs", 2 * ds.std(ddof=1), 1)
    put("dlPfRatio", math.sqrt((1 + phi) / (1 - phi))); put("dlPfN", len(o), "{:.0f}")
    pct("dlPfMin", o.min(), 1); N["dlPfMinQ"] = quarter_label(o.idxmin()); pct("dlPfMinDs", ds.min(), 1)
    RAW["dlPf"] = dict(phi=phi, vol=float(o.std(ddof=1)), vol_ds=float(ds.std(ddof=1)))
    # Dimson regression of the BDC average on the equity market, when the French file has been built
    RAW["dlHaveFF"] = float(FF.exists())
    if FF.exists():
        m = market_quarterly()
        j = pd.concat([roe, m, m.shift(1)], axis=1, keys=["r", "m0", "m1"], sort=True).dropna()
        bd, sed, r2d = ols(j.r.values, j[["m0", "m1"]].values)
        put("dlBetaZero", bd[1]); put("dlBetaOne", bd[2]); put("dlBetaSum", bd[1] + bd[2])
        put("dlBetaOneT", bd[2] / sed[2], "{:.1f}"); put("dlDimN", len(j), "{:.0f}")
        RAW["dlDimson"] = [float(x) for x in bd]
    return p, ymed, b, mark, roe


def figure(ymed, b, mark, roe):
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].plot(ymed.index, 100 * ymed, "o-", ms=2.5, lw=1.1, color=BLUE, label="median income yield on cost")
    ax[0].plot(b.index, 100 * b, "s--", ms=2, lw=0.9, color=RED, label="US one month bill")
    ax[0].set_ylabel("percent a year"); ax[0].set_title("(a) Yield and base rate"); ax[0].legend(fontsize=6.5, loc="center right")
    ax[1].plot(mark.index, 100 * (mark - 1), "o-", ms=2.5, lw=1.1, color=BLUE, label="fair value less cost (% of cost)")
    ax[1].bar(roe.index, 100 * roe, width=50, color=PALE, edgecolor=GREY, lw=0.4, label="average return on net assets, quarter")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("percent"); ax[1].set_title("(b) Marks and returns"); ax[1].legend(fontsize=6.5, loc="lower right")
    import matplotlib.dates as mdates
    for a in ax:
        a.xaxis.set_major_locator(mdates.YearLocator()); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        a.tick_params(axis="x", labelsize=7)
    fig.tight_layout(); save(fig, "fig13_direct_lending")


def covenant_figure(b_cov, b_lite, x0, nu, sig, T):
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    covs = np.linspace(5.25, 9.0, 60)
    for s, col, ls in [(0.15, GREY, ":"), (sig, BLUE, "-"), (0.35, RED, "--")]:
        ax.plot(covs, [100 * breach_prob(x0, 5.0 / c, nu, s, T) for c in covs], color=col, ls=ls,
                label=f"EBITDA volatility {int(100 * s)}%")
    ax.set_xlabel("maintenance covenant, debt over EBITDA (opening leverage 5)")
    ax.set_ylabel(f"breach within {T} years (%)"); ax.legend(fontsize=6.5)
    fig.tight_layout(); save(fig, "fig14_covenant")


def leverage_and_issue():
    S = SOURCE
    # loss share that breaks the 150% test at Ares Capital's leverage and at d = 1 and 1.5
    for d, nm in [(1 / (S["arcc_cover"] - 1), "Arcc"), (1.0, "One"), (1.5, "OneHalf")]:
        pct(f"acThr{nm}", coverage_threshold(d), 1)
    # one year: default probability 2%, factor correlation 0.2 (illustrations), recovery of direct loans
    p, rho, lgd = 0.02, 0.20, 1 - S["rec_direct"]
    pct("acP", p, 0); put("acRho", rho, "{:.1f}")
    d_arcc = 1 / (S["arcc_cover"] - 1)
    for d, nm in [(d_arcc, "Arcc"), (1.5, "OneHalf")]:
        pct(f"acBreach{nm}", coverage_breach_prob(d, p, rho, lgd), 2)
    q = lhp_quantile(0.999, p, rho)
    pct("acQ", q, 1); pct("acQloss", lgd * q, 1); pct("acQeq", (1 + d_arcc) * lgd * q, 0)
    pct("acQeqTwo", 3 * lgd * q, 0)
    RAW["ac"] = dict(p=p, rho=rho, lgd=lgd, d=d_arcc, q=q)
    # an original issue discount of 2 points on a loan paying 9% that lasts five years or is repaid in three
    c, u = 0.09, 0.02
    pct("oiC", c, 0); put("oiU", 100 * u, "{:.0f}")
    for T, nm in [(5, "Five"), (3, "Three")]:
        y = yield_to_life(c, u, T)
        put(f"oiBp{nm}", 1e4 * (y - c), "{:.0f}"); put(f"oiApprox{nm}", 1e4 * u / T, "{:.0f}")
        put(f"oiFirst{nm}", 1e4 * u * c / (1 - math.exp(-c * T)), "{:.0f}")
        RAW[f"oi{nm}"] = y


def exercises():
    """Answers to the direct lending exercises."""
    s_l = (600 - 0.4 * 350) / 0.6
    put("xdSl", s_l, "{:.0f}"); pct("xdPd", 0.02 / (1 - SOURCE["rec_direct"]), 2)
    p = breach_prob(1.0, 4 / 5.5, 0.0, 0.20, 3); pct("xdBreach", p, 1)
    put("xdA", math.log(4 / 5.5) / (0.2 * math.sqrt(3)), "{:.3f}")
    pct("xdLeft", tender_exit(0.05, 0.125, 8), 1); pct("xdFill", 0.05 / 0.125, 0)
    pct("xdThr", coverage_threshold(1.25), 1); pct("xdEq", 2.25 * 0.10, 1)
    put("xdIcr", 3.0 * (0.01 + 0.06) / (0.05 + 0.06), "{:.2f}")
    RAW["xd"] = dict(s_l=s_l, breach=p)


def main():
    exercises()
    exact_parts()
    leverage_and_issue()
    p, ymed, b, mark, roe = historical()
    unitranche()
    bc = covenant()
    figure(ymed, b, mark, roe); covenant_figure(*bc)
    return finish("direct_lending")


if __name__ == "__main__":
    main()
