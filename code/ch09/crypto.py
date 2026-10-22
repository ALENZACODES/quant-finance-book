"""Section 9.7: bitcoin and ether, month-end Coinbase prices from December 2014 (ether from May 2016)
to September 2026. Volatility, drawdowns, the standard error of the mean, correlation with US equities
before and after 2020, and the weight at which bitcoin carries a given share of the risk of a 60/40
portfolio, over the whole period and each half. Writes fig09_crypto.png, fig19_riskshare.png and
out/crypto_monthly.csv."""
import math
import numpy as np
import pandas as pd
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, put, pct, finish
from scipy import stats


def main():
    d = L.load_crypto()
    btc = L.month_end(d["CBBTCUSD"]); eth = L.month_end(d["CBETHUSD"])
    rb = (btc / btc.shift(1) - 1).dropna(); re_ = (eth / eth.shift(1) - 1).dropna()
    r, rf = L.load_markets()
    f = L.load_factors()
    pd.DataFrame(dict(btc=rb, eth=re_.reindex(rb.index))).to_csv(R.HERE / "out" / "crypto_monthly.csv", index_label="month")
    from trend import label
    R.N["btFirst"] = label(rb.index[0]); R.N["btLast"] = label(rb.index[-1]); R.N["ceFirst"] = label(re_.index[0])
    put("btMonths", len(rb), "{:.0f}"); put("ceMonths", len(re_), "{:.0f}")
    first = d["CBBTCUSD"].dropna(); dec = first[first.index.to_period("M") == first.index[0].to_period("M")]
    R.N["btSeriesStart"] = f"{first.index[0].day} {first.index[0]:%B %Y}"; R.N["btStartDay"] = f"{dec.index[-1].day} {dec.index[-1]:%B %Y}"
    nxt = first.index[first.index > dec.index[-1]][0]; R.N["btNextDay"] = f"{nxt.day} {nxt:%B %Y}"
    R.N["btFirstPrice"] = label(btc.index[0])
    put("btPriceStart", btc.iloc[0], "{:,.0f}"); put("btPriceEnd", btc.iloc[-1], "{:,.0f}")
    for tag, x, px in (("bt", rb, d["CBBTCUSD"].dropna()), ("ce", re_, d["CBETHUSD"].dropna())):
        pct(f"{tag}Vol", math.sqrt(12) * x.std(), 0)
        pct(f"{tag}Geo", (1 + x).prod() ** (12 / len(x)) - 1, 0)
        pct(f"{tag}Arith", 12 * x.mean(), 0)
        pct(f"{tag}MeanSe", 12 * x.std() / math.sqrt(len(x)), 0)
        pct(f"{tag}Worst", x.min(), 0); pct(f"{tag}Best", x.max(), 0)
        put(f"{tag}Skew", stats.skew(x)); put(f"{tag}Kurt", stats.kurtosis(x), "{:.1f}")
        dd = 1 - px / px.cummax()
        pct(f"{tag}MddDaily", dd.max(), 0)
        # longest time under water, in months, from the daily series
        under = (px < px.cummax()).astype(int)
        runs = under.groupby((under == 0).cumsum()).sum()
        put(f"{tag}UnderYears", runs.max() / 365.25, "{:.1f}")
        put(f"{tag}SR", L.sharpe(x - f["rf"].reindex(x.index).fillna(f["rf"].iloc[-1])))
    put("btYearsSig", (1.96 * rb.std() / rb.mean()) ** 2 / 12, "{:.0f}")
    # correlation with US equities (equities end in August 2026)
    e = f["mkt_rf"]; j = rb.index[rb.index <= e.index[-1]]
    early = j[j < 202001]; late = j[j >= 202001]
    put("btCorrEarly", np.corrcoef(rb[early], e[early])[0, 1]); put("btCorrLate", np.corrcoef(rb[late], e[late])[0, 1])
    put("btBetaLate", np.polyfit(e[late], rb[late], 1)[0], "{:.1f}")
    put("btCorrEth", np.corrcoef(rb.reindex(re_.index), re_)[0, 1])
    # risk share in a 60/40 portfolio, January 2015 to August 2026
    m = j
    p6040 = 0.6 * e[m] + 0.4 * r.loc[m, "UST"]
    a, b = p6040.var(), (rb[m] - f["rf"].reindex(m)).var(); c = np.cov(p6040, rb[m] - f["rf"].reindex(m))[0, 1]
    pct("btVolSixty", math.sqrt(12 * a), 1)
    for s in (0.05, 0.10, 0.25, 0.50):
        tag = {0.05: "Five", 0.10: "Ten", 0.25: "Quarter", 0.50: "Half"}[s]
        pct(f"btW{tag}", L.weight_for_share(s, a, b, c), 1)
    for w in (0.01, 0.02, 0.05):
        tag = {0.01: "One", 0.02: "Two", 0.05: "Five"}[w]
        pct(f"btShare{tag}", L.risk_share(w, a, b, c), 0)
    put("btCovSign", c / math.sqrt(a * b))
    share_figure(e, r, rb, f, j)
    # figure: drawdowns and rolling correlation
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    for px, col, lab in ((d["CBBTCUSD"].dropna(), BLUE, "bitcoin"), (d["CBETHUSD"].dropna(), RED, "ether")):
        t = px.index.year + (px.index.dayofyear - 1) / 365.25
        ax[0].plot(t, 100 * (px / px.cummax() - 1), color=col, lw=0.8, label=lab)
    ax[0].set_ylabel("drawdown from peak, %")
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)   # above the axes
    ax[0].set_xticks([2016, 2018, 2020, 2022, 2024, 2026]); ax[0].set_xticklabels(["2016", "2018", "2020", "2022", "2024", "2026"])
    rc = pd.Series(rb[j].values).rolling(24).corr(pd.Series(e[j].values))
    tj = np.array([(k // 100) + ((k % 100) - 0.5) / 12 for k in j])
    ax[1].plot(tj, rc, color=BLUE); ax[1].axhline(0, color=GREY, lw=0.6)
    ax[1].set_ylabel("24 month correlation with US equities")
    ax[1].set_xticks([2018, 2020, 2022, 2024, 2026]); ax[1].set_xticklabels(["2018", "2020", "2022", "2024", "2026"])
    fig.tight_layout(); R.save(fig, "fig09_crypto"); plt.close(fig)
    return finish("crypto")


def share_figure(e, r, rb, f, j):
    """Bitcoin's share of the variance of a 60/40 portfolio against its weight, with the variances and
    covariance estimated over the whole period and over each half on its own."""
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ww = np.linspace(0, 0.15, 151)
    periods = (("full", j, BLUE, "-", "January 2015 to August 2026"), ("Early", j[j < 202001], GREY, "--", "2015 to 2019"),
               ("Late", j[j >= 202001], RED, "-.", "2020 to August 2026"))
    for tag, m, col, ls, lab in periods:
        p6040 = 0.6 * e[m] + 0.4 * r.loc[m, "UST"]; xb = rb[m] - f["rf"].reindex(m)
        a, b = p6040.var(), xb.var(); c = np.cov(p6040, xb)[0, 1]
        if tag != "full":
            put(f"rsCorr{tag}", c / math.sqrt(a * b))
            pct(f"rsVolBtc{tag}", math.sqrt(12 * b), 0); pct(f"rsVolSixty{tag}", math.sqrt(12 * a), 1)
            for sh, st in ((0.10, "Ten"), (0.25, "Quarter"), (0.50, "Half")):
                pct(f"rsW{st}{tag}", L.weight_for_share(sh, a, b, c), 1)
        ax.plot(100 * ww, 100 * np.array([L.risk_share(w, a, b, c) for w in ww]), color=col, ls=ls, label=lab)
    lines = [l.get_ydata() for l in ax.get_lines()]                   # full, early, late
    above = np.nonzero(lines[1][1:] <= lines[2][1:])[0]
    put("rsCross", 100 * ww[1:][above.max() + 1] if len(above) else 0.0, "{:.1f}")   # late below early beyond this
    for h in (10, 25, 50):
        ax.axhline(h, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel("weight of bitcoin, %"); ax.set_ylabel("share of portfolio variance, %")
    ax.set_ylim(0, 80); ax.legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig19_riskshare"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
