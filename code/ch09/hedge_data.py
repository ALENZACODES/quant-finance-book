"""Section 9.6: the hedge fund industry in its regulatory filings. Aggregated SEC Form PF statistics of
Qualifying Hedge Funds from the Office of Financial Research (data/ofr_hedge_funds_quarterly.csv,
shipped) and the trend-following factors of Fung and Hsieh (2001) (data/fung_hsieh_ptfs.csv, rebuilt by
scripts/pull_hedge.py and not shipped). Strategy returns, fee drag and the effective fee schedule it
implies, leverage, the serial correlation of reported returns, the liquidity that investors are promised
against the liquidity of what the funds hold, and how much of the managed futures return the
lookback-straddle factors explain. Writes fig22_formpf.png and, when the factor file is present,
fig23_ptfs.png. All results here are historical: measured on reported data, with no simulation."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import numbers_ch09 as R
from numbers_ch09 import plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish

OFR = R.ROOT / "data" / "ofr_hedge_funds_quarterly.csv"
PTFS = R.ROOT / "data" / "fung_hsieh_ptfs.csv"
STRATS = ["credit", "equity", "event", "futures", "macro", "multi", "rv", "other"]
NAMES = {"credit": "Credit", "equity": "Equity", "event": "Event", "futures": "Futures", "macro": "Macro",
         "multi": "Multi", "rv": "Rv", "other": "Other"}
FACTORS = ["ptfsbd", "ptfsfx", "ptfscom", "ptfsir", "ptfsstk"]


def load_ofr():
    return pd.read_csv(OFR, index_col=0, parse_dates=True)


def load_ptfs():
    return pd.read_csv(PTFS, index_col=0, parse_dates=True)


def quarter_label(t):
    return f"{t.year}Q{(t.month - 1) // 3 + 1}"


def fee_panel(d, strats=STRATS):
    """Stack the quarterly gross and net returns (percent) of the strategies into one panel."""
    rows = []
    for s in strats:
        g, n = d[f"strategy_{s}_grossreturn"], d[f"strategy_{s}_netreturn"]
        ok = g.notna() & n.notna()
        rows.append(pd.DataFrame(dict(strategy=s, gross=g[ok], net=n[ok])))
    p = pd.concat(rows)
    p["drag"] = p.gross - p.net
    return p


def fee_fit(p):
    """Least squares fit of the quarterly drag on a constant and the positive part of the gross return:
    drag = a + b max(gross, 0). The constant estimates the quarterly management fee and b the share of a
    positive gross return taken by incentive fees. Returns a, b, their standard errors (HC1) and R^2."""
    X = np.column_stack([np.ones(len(p)), np.maximum(p.gross.values, 0.0)])
    y = p.drag.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * e ** 2) @ X @ XtXi * n / (n - k)
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return beta[0], beta[1], math.sqrt(V[0, 0]), math.sqrt(V[1, 1]), r2, n


def quarterly_ptfs(f, index):
    """Compound the monthly factor returns into the calendar quarters ending on the dates in index."""
    q = (1 + f).groupby(f.index.to_period("Q")).prod() - 1
    q = q[(1 + f).groupby(f.index.to_period("Q")).size() == 3]
    q.index = q.index.to_timestamp("Q")
    q.index = q.index.normalize()
    return q.reindex(index.normalize())


def ols(y, X):
    """OLS with a constant; coefficients, conventional t statistics and R^2."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ beta
    n, k = X1.shape
    s2 = e @ e / (n - k)
    se = np.sqrt(np.diag(np.linalg.inv(X1.T @ X1)) * s2)
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    return beta, beta / se, r2, 1 - (1 - r2) * (n - 1) / (n - k), n


def main():
    d = load_ofr()
    R.N["hdFirst"] = quarter_label(d.index[0]); R.N["hdLast"] = quarter_label(d.index[-1])
    put("hdQuarters", len(d), "{:.0f}")
    last = d.iloc[-1]
    put("hdNavLast", last.allqhf_nav_sum / 1e12, "{:.1f}"); put("hdGavLast", last.allqhf_gav_sum / 1e12, "{:.1f}")
    put("hdNavFirst", d.allqhf_nav_sum.iloc[0] / 1e12, "{:.1f}")
    put("hdCountLast", last.allqhf_count, "{:,.0f}"); put("hdCountFirst", d.allqhf_count.iloc[0], "{:,.0f}")
    put("hdLevAgg", last.allqhf_gav_sum / last.allqhf_nav_sum, "{:.1f}")
    # strategies: mean and volatility of net returns, fee drag, leverage and first autocorrelation
    rows = {}
    for s in STRATS:
        g, n = d[f"strategy_{s}_grossreturn"], d[f"strategy_{s}_netreturn"]
        ok = g.notna() & n.notna()
        nq = n[ok] / 100; gq = g[ok] / 100
        lev = d[f"strategy_{s}_leverageratio"].dropna()
        rho = float(np.corrcoef(n[ok].values[:-1], n[ok].values[1:])[0, 1])
        rows[s] = dict(n=int(ok.sum()), mean=4 * nq.mean(), vol=2 * nq.std(ddof=1), drag=4 * (gq - nq).mean(),
                       share=(gq - nq).sum() / gq.sum(), lev=lev.iloc[-1], rho=rho, sr=2 * nq.mean() / nq.std(ddof=1))
        k = NAMES[s]
        put(f"hdN{k}", rows[s]["n"], "{:.0f}")
        pct(f"hdMean{k}", rows[s]["mean"], 1); pct(f"hdVol{k}", rows[s]["vol"], 1)
        pct(f"hdDrag{k}", rows[s]["drag"], 1); pct(f"hdShare{k}", rows[s]["share"], 0)
        put(f"hdLev{k}", rows[s]["lev"], "{:.1f}"); put(f"hdRho{k}", rows[s]["rho"], "{:.2f}")
    R.RAW["hdTable"] = rows
    put("hdRhoSe", 1 / math.sqrt(rows["equity"]["n"]), "{:.2f}")
    hi = max(STRATS, key=lambda s: rows[s]["lev"]); R.N["hdLevMaxName"] = {"rv": "relative value"}.get(hi, hi)
    put("hdLevMax", rows[hi]["lev"], "{:.1f}")
    # the effective fee schedule
    p = fee_panel(d, [s for s in STRATS if s != "other"])
    a, b, sa, sb, r2, nobs = fee_fit(p)
    pct("hdFeeMgmt", 4 * a / 100, 2); pct("hdFeeMgmtSe", 4 * sa / 100, 2)
    pct("hdFeeInc", b, 1); pct("hdFeeIncSe", sb, 1); put("hdFeeRsq", r2); put("hdFeeObs", nobs, "{:.0f}")
    neg = p[p.gross < 0]
    pct("hdFeeNegDrag", 4 * neg.drag.mean() / 100, 2); put("hdFeeNegObs", len(neg), "{:.0f}")
    put("hdFeeNegQ", len(neg), "{:.0f}")
    R.RAW["hdFeeFit"] = [a, b]
    # a 2 and 20 fund on the same gross returns, with no high-water mark: drag = 0.5 + 0.2 (g - 0.5)^+
    two20 = 0.5 + 0.2 * np.maximum(p.gross.values - 0.5, 0)
    pct("hdTwoTwentyDrag", 4 * two20.mean() / 100, 1); pct("hdActualDrag", 4 * p.drag.mean() / 100, 1)
    # liquidity: what investors can redeem against what the portfolio can be sold for
    inv7, inv90 = d.allqhf_investorliquidityle7_pct, d.allqhf_investorliquidityle7_pct + d.allqhf_investorliquiditygt7le90_pct
    por7 = d.allqhf_portfolioliquidityle1_pct + d.allqhf_portfolioliquiditygt1le7_pct
    por90 = por7 + d.allqhf_portfolioliquiditygt7le90_pct
    for nm, s in (("InvSeven", inv7), ("InvNinety", inv90), ("PorSeven", por7), ("PorNinety", por90)):
        put(f"hd{nm}", s.iloc[-1], "{:.1f}"); put(f"hd{nm}Mean", s.mean(), "{:.1f}")
    put("hdPorOne", d.allqhf_portfolioliquidityle1_pct.iloc[-1], "{:.1f}")
    put("hdInvLong", d.allqhf_investorliquiditygt90_pct.iloc[-1], "{:.1f}")
    put("hdPorLong", d.allqhf_portfolioliquiditygt90_pct.iloc[-1], "{:.1f}")
    put("hdCushionMin", (por90 - inv90).min(), "{:.1f}"); R.N["hdCushionMinQ"] = quarter_label((por90 - inv90).idxmin())
    put("hdCanGate", d.allqhf_cangate_pct.iloc[-1], "{:.1f}"); put("hdCanGateFirst", d.allqhf_cangate_pct.iloc[0], "{:.1f}")
    cg = d.allqhf_currentlygates_pct
    put("hdGatedMax", cg.max(), "{:.1f}"); R.N["hdGatedMaxQ"] = quarter_label(cg.idxmax())
    put("hdGatedLast", cg.iloc[-1], "{:.1f}")
    # figure: fee drag against gross return, and the two liquidity profiles in the last quarter
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax[0].scatter(p.gross, p.drag, s=5, color=SKY, alpha=0.7, label="strategy-quarters")
    gx = np.linspace(p.gross.min(), p.gross.max(), 200)
    ax[0].plot(gx, a + b * np.maximum(gx, 0), color=BLUE, lw=1.2, label="fitted schedule")
    ax[0].plot(gx, 0.5 + 0.2 * np.maximum(gx - 0.5, 0), color=RED, lw=1, ls="--", label="2 and 20")
    ax[0].set_xlabel("gross return in the quarter (%)"); ax[0].set_ylabel("gross less net (%)")
    ax[0].set_title("(a) Fees against gross return"); ax[0].legend(fontsize=6.5)
    h = np.array([0, 1, 7, 90, 120])
    ip = np.array([0, inv7.iloc[-1], inv7.iloc[-1], inv90.iloc[-1], 100])
    pp = np.array([0, d.allqhf_portfolioliquidityle1_pct.iloc[-1], por7.iloc[-1], por90.iloc[-1], 100])
    ax[1].step(h, pp, where="post", color=BLUE, lw=1.2, label="portfolio can be sold")
    ax[1].step(h, ip, where="post", color=RED, lw=1.2, label="investors can redeem")
    ax[1].set_xscale("symlog", linthresh=1); ax[1].set_xticks([0, 1, 7, 90])
    ax[1].set_xticklabels(["0", "1", "7", "90"]); ax[1].set_xlim(0, 120)
    ax[1].set_xlabel("days"); ax[1].set_ylabel("cumulative share of net assets (%)")
    ax[1].set_title(f"(b) Liquidity, {quarter_label(d.index[-1])}"); ax[1].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(); R.save(fig, "fig22_formpf")
    # managed futures against the lookback-straddle factors, when the factor file has been built
    R.RAW["hdHavePtfs"] = float(PTFS.exists())
    if PTFS.exists():
        f = load_ptfs()
        q = quarterly_ptfs(f, d.index) * 100
        R.N["hdPtfsLast"] = f.index[-1].strftime("%B %Y")
        out = {}
        for s in STRATS:
            y = d[f"strategy_{s}_netreturn"]
            ok = y.notna() & q.notna().all(axis=1)
            beta, t, r2, ar2, n = ols(y[ok].values, q.loc[ok, FACTORS].values)
            out[s] = dict(beta=beta.tolist(), t=t.tolist(), r2=r2, ar2=ar2, n=n)
            put(f"hdRsq{NAMES[s]}", r2, "{:.2f}")
        R.RAW["hdPtfs"] = out
        # the fit for credit funds leans on one quarter, the first of 2020
        c20 = pd.Timestamp("2020-03-31")
        k20 = [i for i in d.index if i.normalize() == c20][0]
        put("hdCreditCovid", -d.loc[k20, "strategy_credit_netreturn"], "{:.1f}")
        put("hdRvCovid", -d.loc[k20, "strategy_rv_netreturn"], "{:.1f}")
        put("hdPtfsCovidMin", q.loc[k20, FACTORS].min(), "{:.0f}")
        y = d["strategy_credit_netreturn"]
        ok = y.notna() & q.notna().all(axis=1) & (d.index != k20)
        r2x = ols(y[ok].values, q.loc[ok, FACTORS].values)[2]
        put("hdRsqCreditEx", r2x, "{:.2f}"); R.RAW["hdRsqCreditExRaw"] = r2x
        fu = out["futures"]
        put("hdPtfsN", fu["n"], "{:.0f}"); put("hdPtfsAdj", fu["ar2"], "{:.2f}")
        for j, k in enumerate(["Bd", "Fx", "Com", "Ir", "Stk"]):
            put(f"hdB{k}", fu["beta"][j + 1], "{:.3f}"); put(f"hdT{k}", fu["t"][j + 1], "{:.1f}")
        put("hdAlphaQ", fu["beta"][0], "{:.2f}"); put("hdAlphaT", fu["t"][0], "{:.1f}")
        # the chapter's own trend programme against the reported managed futures return, by quarter
        prog = R.HERE / "out" / "program.csv"
        if prog.exists():
            pr = pd.read_csv(prog)
            pr.index = pd.to_datetime(pr.month.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
            pq = ((1 + pr.net + pr.rf).groupby(pr.index.to_period("Q")).prod() - 1)
            pq.index = pq.index.to_timestamp("Q").normalize()
            y = d["strategy_futures_netreturn"].copy(); y.index = y.index.normalize()
            j = pd.concat([y, pq * 100], axis=1, keys=["pf", "prog"]).dropna()
            put("hdProgCorr", j.pf.corr(j.prog)); put("hdProgN", len(j), "{:.0f}")
        fig, ax = plt.subplots(figsize=(6.4, 2.5))
        y = d["strategy_futures_netreturn"]; ok = y.notna() & q.notna().all(axis=1)
        X1 = np.column_stack([np.ones(ok.sum()), q.loc[ok, FACTORS].values])
        ax.plot(y.index[ok], y[ok], "o-", ms=2.5, lw=1, color=BLUE, label="managed futures funds, net (Form PF)")
        ax.plot(y.index[ok], X1 @ np.array(fu["beta"]), "s--", ms=2, lw=0.9, color=RED,
                label="fitted on the five trend-following factors")
        ax.axhline(0, color=GREY, lw=0.5); ax.set_ylabel("quarterly return (%)"); ax.legend(fontsize=6.8)
        fig.tight_layout(); R.save(fig, "fig23_ptfs")
    return finish("hedge_data")


if __name__ == "__main__":
    main()
