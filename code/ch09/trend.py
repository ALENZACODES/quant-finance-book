"""Sections 9.2 and 9.3: a time series momentum program on seven markets, January 2001 to August 2026.
Sharpe ratios gross and net of a 2 and 20 fee, correlation with equities, the crisis record, the
quarterly smile, the choice of lookback and the gain from diversification across markets.
Writes fig01_trend.png, fig02_smile.png, fig04_lookback.png, fig15_positions.png and
fig16_sleevecorr.png, and out/program.csv."""
import math
import numpy as np
import pandas as pd
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish

START = 200101


def program(r, lookback=12, cost=0.0005):
    w = L.tsmom_weights(r, lookback)
    sl, p = L.tsmom_returns(r, w, cost)
    keep = r.index >= START
    return w[keep], sl[keep], p[keep]


def label(yyyymm):
    y, m = divmod(int(yyyymm), 100)
    return ["January", "February", "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December"][m - 1] + f" {y}"


def main():
    r, rf = L.load_markets()
    w, sl, p = program(r)
    months = p.index.values; rfp = rf.loc[p.index].values
    eq = r.loc[p.index, "EQ"].values
    net = L.fees_hwm(p.values, rfp, months=months)
    pd.DataFrame(dict(month=months, gross=p.values, net=net, eq=eq, ust=r.loc[p.index, "UST"].values,
                      rf=rfp)).to_csv(R.HERE / "out" / "program.csv", index=False)
    put("tfFirst", 0, "{}"); R.N["tfFirst"] = label(months[0]); R.N["tfLast"] = label(months[-1])
    put("tfMonths", len(p), "{:.0f}"); put("tfMarkets", r.shape[1], "{:.0f}")
    gocfirst = w.index[w["GOC"] != 0][0]; R.N["tfGocFirst"] = label(gocfirst)   # month of the first signal
    R.N["tfGocFirstPos"] = label(sl.index[sl["GOC"] != 0][0])                   # month of the first position
    for k in L.MARKETS:
        pct(f"tfVol{k.title()}", r[k].std() * math.sqrt(12), 1)
    pct("tfMean", 12 * p.mean(), 1); pct("tfVolP", math.sqrt(12) * p.std(), 1)
    put("tfSR", L.sharpe(p)); put("tfSRnet", L.sharpe(net))
    pct("tfMeanNet", 12 * net.mean(), 1); pct("tfFeeDrag", 12 * (p.mean() - net.mean()), 1)
    # the split of the fee: the same fund with no incentive fee gives the management fee's share; the
    # incentive fee is charged on the whole gain above the high-water mark, bill rate included
    mg = L.fees_hwm(p.values, rfp, perf=0.0, months=months)
    pct("tfFeeExact", 12 * (p.mean() - net.mean()), 2); pct("tfFeeMgmt", 12 * (p.mean() - mg.mean()), 2)
    pct("tfFeeInc", 12 * (mg.mean() - net.mean()), 2); pct("tfLeftMgmt", 12 * mg.mean(), 2)
    gain_mg = 12 * (mg.mean() + rfp.mean())                 # gain after the management fee, bills included
    pct("tfGainMgmt", gain_mg, 1); pct("tfIncShareGain", 12 * (mg.mean() - net.mean()) / gain_mg, 0)
    pct("tfIncShareLeft", (mg.mean() - net.mean()) / mg.mean(), 0)
    pct("tfRfMean", 12 * rfp.mean(), 2); pct("tfVolNet", math.sqrt(12) * net.std(ddof=1), 1)
    put("tfSReq", L.sharpe(eq))
    put("tfCorrEq", np.corrcoef(p, eq)[0, 1]); put("tfBetaEq", np.polyfit(eq, p, 1)[0])
    put("tfCorrUst", np.corrcoef(p, r.loc[p.index, "UST"])[0, 1])
    pct("tfMdd", L.max_drawdown(p), 0); pct("tfMddEq", L.max_drawdown(eq), 0)
    gross_lev = w.abs().sum(axis=1)
    put("tfLevMed", gross_lev.median(), "{:.1f}"); put("tfLevMax", gross_lev.max(), "{:.1f}")
    turn = (w.diff().abs().sum(axis=1)).iloc[1:]
    put("tfTurn", 12 * turn.mean(), "{:.1f}")
    # crisis windows
    def window(a, b, x):
        m = (months >= a) & (months <= b); return L.compound(x[m])
    pct("tfGfcP", window(200711, 200902, p.values + rfp), 0); pct("tfGfcEq", window(200711, 200902, eq + rfp), 0)
    pct("tfTwentyTwoP", window(202201, 202212, p.values + rfp), 0)
    pct("tfTwentyTwoEq", window(202201, 202212, eq + rfp), 0)
    pct("tfTwentyTwoUst", window(202201, 202212, r.loc[p.index, "UST"].values + rfp), 0)
    pct("tfCovidP", window(202002, 202003, p.values + rfp), 1); pct("tfCovidEq", window(202002, 202003, eq + rfp), 1)
    for k in ("tfGfcEq", "tfTwentyTwoEq", "tfTwentyTwoUst"):
        put(k + "Abs", -R.RAW[k], "{:.0f}")
    put("tfCovidEqAbs", -R.RAW["tfCovidEq"], "{:.1f}")
    # annual returns: count of years
    yrs = months // 100
    ann = pd.Series(p.values).groupby(yrs).apply(lambda x: L.compound(x))
    full = pd.Series(yrs).value_counts().sort_index(); ann = ann[full[full == 12].index]
    put("tfYears", len(ann), "{:.0f}"); put("tfYearsUp", (ann > 0).sum(), "{:.0f}")
    pct("tfBestYear", ann.max(), 0); R.N["tfBestYearLab"] = str(int(ann.idxmax()))
    pct("tfWorstYear", ann.min(), 0); R.N["tfWorstYearLab"] = str(int(ann.idxmin()))
    # worst ten equity months
    order = np.argsort(eq)[:10]
    pct("tfWorstTenEq", eq[order].mean(), 1); pct("tfWorstTenP", p.values[order].mean(), 1)
    put("tfWorstTenUp", (p.values[order] > 0).sum(), "{:.0f}")
    # quarterly smile
    q = (months // 100) * 10 + ((months % 100) - 1) // 3
    dfq = pd.DataFrame(dict(q=q, p=p.values, e=eq)).groupby("q").agg(lambda x: L.compound(x))
    counts = pd.Series(q).value_counts(); dfq = dfq.loc[counts[counts == 3].index.sort_values()]
    x = dfq["e"].values; y = dfq["p"].values
    X = np.column_stack([np.ones_like(x), x, x * x])
    b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b; s2 = e @ e / (len(y) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    put("slQuarters", len(y), "{:.0f}"); put("slLin", b[1]); put("slQuad", b[2], "{:.2f}")
    put("slQuadT", b[2] / se[2], "{:.1f}"); put("slLinT", b[1] / se[1], "{:.1f}")
    pct("slIcpt", b[0], 2)
    lo = x < np.quantile(x, 0.2); hi = x > np.quantile(x, 0.8)
    pct("slLowP", y[lo].mean(), 1); pct("slMidP", y[~lo & ~hi].mean(), 1); pct("slHighP", y[hi].mean(), 1)
    # per market and lookbacks
    srm = {k: L.sharpe(sl[k][sl[k] != 0]) for k in L.MARKETS}
    for k, v in srm.items():
        put(f"tfSR{k.title()}", v)
    put("tfSRavg", np.mean(list(srm.values())))
    C = sl[(sl != 0).all(axis=1)].corr().values
    rho = C[np.triu_indices(7, 1)].mean(); put("tfRhoAvg", rho)
    put("tfSRpred", L.combined_sharpe(np.mean(list(srm.values())), 7, rho))
    looks = [1, 3, 6, 9, 12, 18, 24]; srl = []
    for k in looks:
        srl.append(L.sharpe(program(r, k)[2]))
    for k, v in zip(looks, srl):
        put(f"lbSR{R.word(k) if k <= 10 else {12:'Twelve',18:'Eighteen',24:'TwentyFour'}[k]}", v)
    put("lbSE", 1 / math.sqrt(len(p) / 12))
    # figures ----------------------------------------------------------------
    t = np.arange(len(p)) / 12 + 2001
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    ax[0].plot(t, np.cumprod(1 + p.values), color=BLUE, label="trend, before fees")
    ax[0].plot(t, np.cumprod(1 + net), color=SKY, label="trend, net of 2 and 20")
    ax[0].plot(t, np.cumprod(1 + eq), color=GREY, lw=0.9, label="US equities")
    ax[0].set_yscale("log"); L.plain_log_axis(ax[0]); ax[0].set_ylabel("growth of 1, excess of bills"); ax[0].legend(loc="upper left")
    ax[1].plot(t, 100 * L.drawdown_path(eq), color=GREY, lw=0.9, label="US equities")
    ax[1].plot(t, 100 * L.drawdown_path(p.values), color=BLUE, label="trend, before fees")
    ax[1].set_ylabel("drawdown, %"); ax[1].set_ylim(-75, 2)
    ax[1].legend(loc="lower right", frameon=True, framealpha=1, edgecolor="none").set_zorder(5)
    fig.tight_layout(); R.save(fig, "fig01_trend"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.scatter(100 * x, 100 * y, s=9, color=BLUE)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(100 * xx, 100 * (b[0] + b[1] * xx + b[2] * xx * xx), color=RED)
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlabel("US equity excess return in the quarter, %"); ax.set_ylabel("trend programme, %")
    fig.tight_layout(); R.save(fig, "fig02_smile"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ax[0].bar(range(len(looks)), srl, color=BLUE); ax[0].set_xticks(range(len(looks)))
    ax[0].set_xticklabels([str(k) for k in looks]); ax[0].set_xlabel("lookback, months")
    ax[0].set_ylabel("Sharpe ratio, before fees"); ax[0].grid(axis="x", visible=False)
    ax[1].barh(range(7), [srm[k] for k in L.MARKETS], color=[RED if srm[k] < 0 else BLUE for k in L.MARKETS])
    ax[1].set_yticks(range(7)); ax[1].set_yticklabels([L.NAMES[k] for k in L.MARKETS])
    ax[1].invert_yaxis(); ax[1].set_xlabel("Sharpe ratio of the sleeve"); ax[1].grid(axis="y", visible=False)
    fig.tight_layout(); R.save(fig, "fig04_lookback"); plt.close(fig)
    positions_figure(r, months)
    sleeve_corr_figure(sl, C)
    return finish("trend")


CRISES = ((200711, 200902), (202002, 202003), (202201, 202212))


def positions_figure(r, months):
    """Positions held in each month (set at the end of the month before), in units of capital."""
    from matplotlib.colors import LinearSegmentedColormap
    held = L.tsmom_weights(r).shift(1).fillna(0.0).loc[months]   # set at the end of t, held in t + 1
    eq = held["EQ"]
    put("psEqFebTwenty", eq.loc[202002], "{:.2f}"); put("psEqMarTwenty", eq.loc[202003], "{:.2f}")
    put("psEqAprTwenty", eq.loc[202004], "{:.2f}")
    g = (months >= 200711) & (months <= 200902)
    put("psGfcShortEq", (eq.values[g] < 0).sum(), "{:.0f}"); put("psGfcMonths", g.sum(), "{:.0f}")
    R.N["psGfcFirstShort"] = label(months[g][eq.values[g] < 0][0])
    t22 = (months >= 202201) & (months <= 202212)
    put("psTwoTwoShortUst", (held["UST"].values[t22] < 0).sum(), "{:.0f}")
    put("psTwoTwoShortGoc", (held["GOC"].values[t22] < 0).sum(), "{:.0f}")
    lim = float(np.abs(held.values).max()); put("psMax", lim, "{:.2f}")
    t0 = 2001
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    cmap = LinearSegmentedColormap.from_list("bwr_book", [BLUE, "#FFFFFF", RED])
    ax.imshow(held.T.values, aspect="auto", cmap=cmap.reversed(), vmin=-lim, vmax=lim, interpolation="nearest",
              extent=(t0, t0 + len(months) / 12, len(L.MARKETS) - 0.5, -0.5))
    for a, b in CRISES:
        xa = t0 + np.searchsorted(months, a) / 12; xb = t0 + (np.searchsorted(months, b) + 1) / 12
        ax.add_patch(plt.Rectangle((xa, -0.5), xb - xa, len(L.MARKETS), fill=False, ec="#222222", lw=0.8))
    ax.set_yticks(range(len(L.MARKETS))); ax.set_yticklabels([L.NAMES[k] for k in L.MARKETS], fontsize=7)
    ax.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap.reversed(), norm=plt.Normalize(-lim, lim))
    cb = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.04); cb.set_label("position, units of capital", fontsize=7)
    fig.tight_layout(); R.save(fig, "fig15_positions"); plt.close(fig)


def sleeve_corr_figure(sl, C):
    """Correlations of the seven sleeves over the months in which all seven held a position."""
    from matplotlib.colors import LinearSegmentedColormap
    both = (sl != 0).all(axis=1); put("scMonths", both.sum(), "{:.0f}")
    iu = np.triu_indices(7, 1); vals = C[iu]
    i, j = iu[0][np.argmax(vals)], iu[1][np.argmax(vals)]
    R.N["scMaxPair"] = f"{L.NAMES[L.MARKETS[i]]} and {L.NAMES[L.MARKETS[j]]}"; put("scMax", vals.max())
    i, j = iu[0][np.argmin(vals)], iu[1][np.argmin(vals)]
    R.N["scMinPair"] = f"{L.NAMES[L.MARKETS[i]]} and {L.NAMES[L.MARKETS[j]]}"; put("scMin", vals.min())
    put("scBonds", C[1, 2]); put("scSdPairs", vals.std(ddof=1))
    put("scNeg", (vals < 0).sum(), "{:.0f}"); put("scPairs", len(vals), "{:.0f}")
    lim = 1.0
    short = dict(EQ="US eq.", UST="US 10y", GOC="Can. 10y", OIL="WTI", EUR="euro", JPY="yen", CAD="CAD")
    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    cmap = LinearSegmentedColormap.from_list("rwb_book", [RED, "#FFFFFF", BLUE])   # blue positive, as Fig 9.4
    M = C.copy(); np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, cmap=cmap, vmin=-0.7, vmax=0.7)
    for a in range(7):
        for b in range(7):
            if a != b:
                v = f"{C[a, b]:.2f}"; v = "0.00" if v == "-0.00" else v.replace("-", "\u2212")
                ax.text(b, a, v, ha="center", va="center", fontsize=6,
                        color="white" if abs(C[a, b]) > 0.45 else "#222222")
    ax.set_xticks(range(7)); ax.set_xticklabels([short[k] for k in L.MARKETS], rotation=90, fontsize=7)
    ax.set_yticks(range(7)); ax.set_yticklabels([short[k] for k in L.MARKETS], fontsize=7)
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(); R.save(fig, "fig16_sleevecorr"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
