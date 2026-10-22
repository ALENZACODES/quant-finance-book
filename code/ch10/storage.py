"""Section 10.6: storage and the convenience yield. The net convenience yield of WTI crude oil and of
Henry Hub natural gas implied by the first and fourth NYMEX contracts and the bill rate, against
inventories measured as a deviation from the same month of the previous five years; the share of
months in backwardation; and the seasonal slope of the gas curve. Writes fig09_working.png and
fig10_gas.png; the September to December gas spread fixed by the August futures curve against the spot
spread that followed, which writes fig12_lock.png.
Also writes fig17_curves.png, the first four crude contracts in three months against full carry."""
import numpy as np
import pandas as pd
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, PALE, ROSE, put, pct, finish

TAU = 3 / 12   # the fourth contract delivers about three months after the first


def panel():
    e = L.load_curves(); m = L.load_monthly()
    d = e.join(m[["tbill"]])
    r = np.log1p(d["tbill"] / 100)
    d["cyo"] = L.net_convenience_yield(d.cl1, d.cl4, TAU, r)
    d["cyg"] = L.net_convenience_yield(d.ng1, d.ng4, TAU, r)
    d["devo"] = L.seasonal_deviation(d.crude_stocks)
    d["devg"] = L.seasonal_deviation(d.gas_storage)
    d["slope"] = 12 * np.log(d.ng2 / d.ng1)
    return d


def fit(x, y):
    """Least squares fit of y on a + b x + c x^2 and on a + b x."""
    return np.polyfit(x, y, 2), np.polyfit(x, y, 1)


def main():
    d = panel()
    o = d[["cyo", "devo"]].dropna(); g = d[["cyg", "devg"]].dropna()
    R.N["sgOilFirst"] = o.index[0].strftime("%B %Y"); R.N["sgOilLast"] = o.index[-1].strftime("%B %Y")
    R.N["sgGasFirst"] = g.index[0].strftime("%B %Y"); R.N["sgStockLast"] = d.crude_stocks.dropna().index[-1].strftime("%B %Y")
    put("sgNOil", len(o), "{:.0f}"); put("sgNGas", len(g), "{:.0f}")
    R.N["sgGasLast"] = g.index[-1].strftime("%B %Y")
    allm = pd.period_range(g.index[0], g.index[-1], freq="M"); have = set(pd.DatetimeIndex(g.index).to_period("M"))
    miss = [p for p in allm if p not in have]
    put("sgNGasSpan", len(allm), "{:.0f}"); R.N["sgGasMissing"] = ", ".join(p.strftime("%B %Y") for p in miss)
    put("sgCorrOil", o.cyo.corr(o.devo)); put("sgCorrGas", g.cyg.corr(g.devg))
    q2, b1 = fit(o.devo, o.cyo); put("sgSlopeOil", b1[0], "{:.2f}"); put("sgQuadOil", q2[0], "{:.1f}")
    q2g, b1g = fit(g.devg, g.cyg); put("sgSlopeGas", b1g[0], "{:.2f}")
    # quintiles of the inventory deviation
    qq = pd.qcut(o.devo, 5, labels=False)
    means = o.cyo.groupby(qq).mean()
    pct("sgQLow", means.iloc[0], 1); pct("sgQHigh", means.iloc[-1], 1)
    dv = o.devo.groupby(qq).mean(); pct("sgDevLow", dv.iloc[0], 1); pct("sgDevHigh", dv.iloc[-1], 1)
    # backwardation share over the whole curve file
    c = d[["cl1", "cl4"]].dropna(); c = c[c.index >= "1990-01-01"]
    R.N["sgBackFirst"] = c.index[0].strftime("%B %Y"); R.N["sgCurveLast"] = c.index[-1].strftime("%B %Y")
    pct("sgBackShare", np.mean(c.cl4 < c.cl1), 0)
    pct("sgCyMean", d.cyo.dropna()[d.cyo.dropna().index >= "1990-01-01"].mean(), 1)
    # April 2020
    a = d.loc["2020-04-01"]
    put("sgAprOne", a.cl1, "{:.2f}"); put("sgAprFour", a.cl4, "{:.2f}"); pct("sgAprCy", a.cyo, 0); pct("sgAprCyAbs", -a.cyo, 0)
    pct("sgAprDev", a.devo, 1)
    # the cost of carry bound: the full carry spread at a storage cost of u, compared with the largest contango
    pct("sgMaxContango", (np.log(d.cl4 / d.cl1) / TAU).max(), 0)
    # gas seasonality: annualised slope from the first to the second contract by calendar month
    s = d.slope.dropna(); sm = s.groupby(s.index.month).mean()
    pct("sgSlopeSep", sm[9], 0); pct("sgSlopeJan", sm[1], 0); pct("sgSlopeDec", sm[12], 0)
    R.N["sgGasCurveFirst"] = s.index[0].strftime("%B %Y")
    # figure: the Working curve
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8))
    for a_, x, y, q, t in ((ax[0], o.devo, o.cyo, q2, "WTI crude oil"), (ax[1], g.devg, g.cyg, q2g, "Henry Hub gas")):
        a_.scatter(100 * x, 100 * y, s=5, color=SKY, alpha=0.7)
        xx = np.linspace(x.min(), x.max(), 100); a_.plot(100 * xx, 100 * np.polyval(q, xx), color=RED, lw=1.2)
        a_.axhline(0, color="#333333", lw=0.6); a_.axvline(0, color="#333333", lw=0.6)
        a_.set_xlabel("inventory against 5 year norm, %"); a_.set_title(t)
    ax[0].set_ylabel("net convenience yield, % a year")
    ax[0].set_ylim(-150, 60); ax[1].set_ylim(-150, 120)
    fig.tight_layout(); R.save(fig, "fig09_working"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.bar(range(1, 13), 100 * sm.values, color=[RED if v > 0 else BLUE for v in sm.values], width=0.7)
    ax.axhline(0, color="#333333", lw=0.6); ax.set_xticks(range(1, 13)); ax.set_xticklabels(mon)
    ax.set_ylabel("slope, % a year"); ax.set_xlabel("month of the average"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig10_gas"); plt.close(fig)
    lock(d)
    curves_figure(d)
    return finish("storage")


def curves_figure(d):
    """The first four crude contracts relative to the first, in April 2020, in the month of steepest
    backwardation since 1990 and in the month whose slope is the median, against full carry at the bill
    rate with no storage cost, F_k / F_1 = exp(r (k - 1) / 12)."""
    c = d[["cl1", "cl2", "cl3", "cl4", "tbill"]].dropna(); c = c[c.index >= "1990-01-01"]
    slope = np.log(c.cl4 / c.cl1)
    back = slope.idxmin(); med = (slope - slope.median()).abs().idxmin()
    R.N["cvBackWhen"] = back.strftime("%B %Y"); R.N["cvLastWhen"] = c.index[-1].strftime("%B %Y"); R.N["cvMedWhen"] = med.strftime("%B %Y")
    pct("cvBackSpread", c.loc[back, "cl4"] / c.loc[back, "cl1"] - 1, 1)
    pct("cvBackSpreadAbs", 1 - c.loc[back, "cl4"] / c.loc[back, "cl1"], 1)
    pct("cvMedSpread", c.loc[med, "cl4"] / c.loc[med, "cl1"] - 1, 1)
    pct("cvAprSpread", c.loc["2020-04-01", "cl4"] / c.loc["2020-04-01", "cl1"] - 1, 0)
    pct("cvAprCarry", np.exp(3 * np.log1p(c.loc["2020-04-01", "tbill"] / 100) / 12) - 1, 2)
    k = np.arange(1, 5)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for when, col, mk in ((pd.Timestamp("2020-04-01"), RED, "o"), (back, BLUE, "s"), (med, GREY, "^")):
        row = c.loc[when]; f = row[["cl1", "cl2", "cl3", "cl4"]].to_numpy(float)
        r = np.log1p(row.tbill / 100)
        ax.plot(k, 100 * f / f[0], color=col, marker=mk, ms=4, label=when.strftime("%B %Y"))
        ax.plot(k, 100 * np.exp(r * (k - 1) / 12), color=col, ls=":", lw=0.9)
    ax.set_xticks(k); ax.set_xlabel("contract"); ax.set_ylabel("price, first contract = 100")
    ax.legend(loc="upper left", title="dotted: full carry at the bill", title_fontsize=7)
    fig.tight_layout(); R.save(fig, "fig17_curves"); plt.close(fig)


def lock(d, month=8):
    """Proposition 10.12: the spread a storage owner can fix in August, buying the first contract (about
    September) and selling the fourth (about December), against the change in the monthly average spot
    price from September to December of the same year."""
    sp = np.log(L.load_monthly().gas.dropna())
    rows = []
    for t in d.index[d.index.month == month]:
        t1, t4 = t + pd.DateOffset(months=1), t + pd.DateOffset(months=4)
        if pd.notna(d.at[t, "ng4"]) and pd.notna(d.at[t, "ng1"]) and t1 in sp.index and t4 in sp.index:
            rows.append((t.year, np.log(d.at[t, "ng4"] / d.at[t, "ng1"]), sp[t4] - sp[t1]))
    k = pd.DataFrame(rows, columns=["year", "lock", "real"]).set_index("year")
    put("sgLockN", len(k), "{:.0f}"); R.N["sgLockFirst"] = str(k.index[0]); R.N["sgLockLast"] = str(k.index[-1])
    pct("sgLockMean", k.lock.mean(), 1); pct("sgLockSd", k.lock.std(ddof=1), 1); pct("sgLockMin", k.lock.min(), 1)
    pct("sgRealMean", k.real.mean(), 1); pct("sgRealSd", k.real.std(ddof=1), 1)
    pct("sgRealPos", (k.real > 0).mean(), 0); pct("sgLockPos", (k.lock > 0).mean(), 0)
    gap = k.lock - k.real
    pct("sgGapMean", gap.mean(), 1); pct("sgGapSe", gap.std(ddof=1) / np.sqrt(len(gap)), 1)
    put("sgGapT", gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap))), "{:.1f}")
    # the July to January spot spread, the round trip of a whole storage season
    jj = [(y, sp[pd.Timestamp(y + 1, 1, 1)] - sp[pd.Timestamp(y, 7, 1)]) for y in range(sp.index[0].year, sp.index[-1].year)
          if pd.Timestamp(y + 1, 1, 1) in sp.index and pd.Timestamp(y, 7, 1) in sp.index]
    jj = pd.Series(dict(jj))
    pct("sgJanMean", jj.mean(), 1); pct("sgJanSd", jj.std(ddof=1), 1); pct("sgJanPos", (jj > 0).mean(), 0)
    put("sgJanN", len(jj), "{:.0f}"); R.N["sgJanFirst"] = str(jj.index[0]); R.N["sgJanLast"] = str(jj.index[-1] + 1)
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    x = np.arange(len(k))
    ax.bar(x - 0.2, 100 * k.lock, width=0.4, color=BLUE, label="fixed by the August curve")
    ax.bar(x + 0.2, 100 * k.real, width=0.4, color=ROSE, label="spot, September to December")
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xticks(x[::3]); ax.set_xticklabels([str(y) for y in k.index[::3]])
    ax.set_ylabel("log spread, %"); ax.legend(fontsize=7, frameon=False, loc="lower left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig12_lock"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
