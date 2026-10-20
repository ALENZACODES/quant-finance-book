"""Section 5.11 case study: selling one-month US dollar calls against the Canadian dollar and
hedging them daily at the volatility of the previous three months.

Data: Bank of Canada USD/CAD rates (noon rate to 2016, daily average from 2017), Government of
Canada one-month bill yields, and US one-month Treasury yields (FRED series DGS1MO), all in
data/usdcad_rates.csv.
Run on its own:  python code/ch05/case_fx_hedge.py   (writes code/ch05/out/case_fx_hedge.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numbers_ch05 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT  # noqa: F401

WINDOW = 63
DATA_END = "2026-09-30"


def load():
    df = pd.read_csv(ROOT / "data" / "usdcad_rates.csv", parse_dates=["date"]).set_index("date")
    df[["cad_tbill_1m", "us_tbill_1m"]] = df[["cad_tbill_1m", "us_tbill_1m"]].ffill()
    return df.dropna(subset=["usdcad", "cad_tbill_1m", "us_tbill_1m"])


def cont(y, days=30):
    """Simple annual yield in percent to a continuously compounded rate."""
    t = days / 365
    return math.log(1 + y / 100 * t) / t


def run_month(df, i0, i1):
    """Sell a one-month at-the-money-forward USD call on date index i0, hedge daily to i1."""
    d = df.index; S = df["usdcad"].to_numpy(); n = i1 - i0
    t0 = d[i0]; tau0 = (d[i1] - t0).days / 365
    rets = np.log(S[i0 - WINDOW + 1:i0 + 1] / S[i0 - WINDOW:i0])
    span = (d[i0] - d[i0 - WINDOW]).days / 365
    sh = math.sqrt(np.sum(rets ** 2) / span)
    rd, rf = cont(df["cad_tbill_1m"].iloc[i0]), cont(df["us_tbill_1m"].iloc[i0])
    K = S[i0] * math.exp((rd - rf) * tau0)
    V0 = float(L.gk(S[i0], K, rd, rf, sh, tau0))
    g0 = L.greeks(S[i0], K, rd, sh, tau0, "call", rf)
    D = float(g0["delta"]); cash = V0 - D * S[i0]; gt = 0.0
    for k in range(i0, i1):
        dt = (d[k + 1] - d[k]).days / 365; tau = (d[i1] - d[k]).days / 365
        G = float(L.greeks(S[k], K, rd, sh, tau, "call", rf)["gamma"])
        R_ = S[k + 1] / S[k] - 1
        gt += math.exp(-rd * (d[k] - t0).days / 365) * 0.5 * G * S[k] ** 2 * (sh ** 2 * dt - R_ ** 2)
        cash = cash * math.exp(rd * dt) + D * S[k + 1] * (math.exp(rf * dt) - 1)
        if k + 1 < i1:
            Dn = float(L.greeks(S[k + 1], K, rd, sh, (d[i1] - d[k + 1]).days / 365, "call", rf)["delta"])
            cash -= (Dn - D) * S[k + 1]; D = Dn
    pnl = (cash + D * S[i1] - max(S[i1] - K, 0)) * math.exp(-rd * tau0)
    life = np.log(S[i0 + 1:i1 + 1] / S[i0:i1]); sr = math.sqrt(np.sum(life ** 2) / tau0)
    bp = 1e4 / S[i0]
    return dict(date=t0, pnl=pnl * bp, gts=gt * bp, prem=V0 * bp, sh=sh, sr=sr, vega=float(g0["vega"]) * bp,
                days=n, S0=S[i0])


def main():
    df = load(); d = df.index
    me = df.groupby([d.year, d.month]).tail(1).index
    # A month counts only once it is over. The file is cut at DATA_END, the end of September 2026, so
    # the last month in it is complete (30 September 2026 is a Canadian holiday, with no rate); a file
    # that ran part way into a later month would have its last, unfinished month dropped here.
    if d[-1] + pd.offsets.MonthEnd(0) > pd.Timestamp(DATA_END):
        me = me[:-1]
    N["fxEnd"] = d[-1].strftime("%-d %B %Y"); N["fxLastClose"] = me[-1].strftime("%-d %B %Y")
    N["fxStart"] = d[0].strftime("%B %Y")
    pos = [df.index.get_loc(x) for x in me]
    rows = [run_month(df, a, b) for a, b in zip(pos[:-1], pos[1:]) if a >= WINDOW]
    cs = pd.DataFrame(rows).set_index("date")
    cs.to_csv(ROOT / "code" / "ch05" / "out" / "case_months.csv")
    n = len(cs); put("fxN", n, "{:d}")
    N["fxFirst"] = cs.index[0].strftime("%B %Y"); N["fxLast"] = cs.index[-1].strftime("%B %Y")
    put("fxDays", df.shape[0], "{:,d}")
    put("fxYears", (me[-1] - d[0]).days / 365.25, "{:.0f}")
    put("fxWindow", WINDOW, "{:d}")
    put("fxPnlMean", cs.pnl.mean(), "{:.1f}"); put("fxPnlSE", cs.pnl.std(ddof=1) / math.sqrt(n), "{:.1f}")
    put("fxPnlMedian", cs.pnl.median(), "{:.1f}"); put("fxPnlSd", cs.pnl.std(ddof=1), "{:.1f}")
    put("fxPremMean", cs.prem.mean(), "{:.1f}")
    put("fxWin", 100 * float((cs.pnl > 0).mean()), "{:.0f}")
    put("fxShMean", 100 * cs.sh.mean(), "{:.2f}"); put("fxSrMean", 100 * cs.sr.mean(), "{:.2f}")
    put("fxShAbove", 100 * float((cs.sh > cs.sr).mean()), "{:.0f}")
    put("fxCorrGT", float(np.corrcoef(cs.pnl, cs.gts)[0, 1]), "{:.3f}")
    X = sm.add_constant(cs.vega * (cs.sh - cs.sr)); fit = sm.OLS(cs.pnl, X).fit()
    put("fxSlope", fit.params.iloc[1], "{:.2f}"); put("fxSlopeSE", fit.bse.iloc[1], "{:.2f}")
    put("fxRsq", fit.rsquared, "{:.2f}")
    RAW["fxConst"] = float(fit.params.iloc[0])
    w = cs.pnl.idxmin(); put("fxWorst", cs.pnl.min(), "{:.0f}"); N["fxWorstMonth"] = w.strftime("%B %Y")
    put("fxWorstLoss", -cs.pnl.min(), "{:.0f}")
    put("fxWorstSh", 100 * cs.sh[w], "{:.1f}"); put("fxWorstSr", 100 * cs.sr[w], "{:.1f}")
    b = cs.pnl.idxmax(); put("fxBest", cs.pnl.max(), "{:.0f}"); N["fxBestMonth"] = b.strftime("%B %Y")
    top = cs.pnl.nsmallest(5); put("fxTopFiveShare", 100 * float(-top.sum() / cs.pnl[cs.pnl < 0].abs().sum()), "{:.0f}")
    put("fxCum", cs.pnl.sum(), "{:,.0f}")
    put("fxPnlMeanExWorst", cs.pnl.drop(top.index).mean(), "{:.1f}")
    # the averaging in the post-2016 series: lag-one autocorrelation of daily returns
    r = np.log(df.usdcad).diff()
    for src, nm in [("noon", "Noon"), ("average", "Avg")]:
        x = r[df.usdcad_source == src].dropna()
        put(f"acRet{nm}", float(x.autocorr(1)), "{:.2f}")
    # split by source
    for lab, sel in [("Noon", cs.index < "2017-01-01"), ("Avg", cs.index >= "2017-01-01")]:
        put(f"fxPnlMean{lab}", cs.pnl[sel].mean(), "{:.1f}"); put(f"fxN{lab}", int(sel.sum()), "{:d}")
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax[0].plot(cs.index, 100 * cs.sh, color=BLUE, lw=1.0, label=f"hedging volatility (previous {WINDOW} days)")
    ax[0].plot(cs.index, 100 * cs.sr, color=RED, lw=0.8, alpha=0.85, label="realised over the month")
    ax[0].set_ylabel("annualised volatility (%)"); ax[0].set_title("(a) USD/CAD volatility")
    ax[0].legend(loc="upper right", fontsize=7)
    lim = [min(cs.gts.min(), cs.pnl.min()) - 5, max(cs.gts.max(), cs.pnl.max()) + 5]
    ax[1].plot(lim, lim, color=GREY, lw=0.8, ls="--")
    ax[1].scatter(cs.gts, cs.pnl, s=8, color=BLUE, alpha=0.75, lw=0)
    ax[1].set_xlabel(r"$\sum \frac{1}{2}\Gamma S^2(\sigma_h^2\Delta t-R^2)$ (bp)"); ax[1].set_ylabel("hedged P&L (bp of spot)")
    ax[1].set_title("(b) P&L against the gamma-theta sum")
    fig.tight_layout(); save(fig, "fig14_case")
    return finish("case_fx_hedge")


if __name__ == "__main__":
    main()
