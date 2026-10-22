"""Section 10.4: REITs and private property. Correlations of REIT and private property returns with the
equity market, the lead of REITs over the property index, Dimson betas, the volatility of REIT assets
after removing leverage, and the agreement of the two at longer horizons. Writes fig06_leadlag.png."""
import numpy as np
import pandas as pd
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, PALE, put, pct, finish

LEVERAGE = (0.3, 0.4, 0.5)


def main():
    r = L.quarterly_returns()
    mk = L.load_market_quarterly()
    d = pd.concat({"cre": r["cre"], "reit": r["reit"], "mkt": mk}, axis=1, sort=True)
    d = d[(d.index >= L.REIT_START)].dropna()
    R.N["rtFirst"] = f"{d.index[0].year}Q{d.index[0].quarter}"; R.N["rtLast"] = f"{d.index[-1].year}Q{d.index[-1].quarter}"
    put("rtN", len(d), "{:.0f}")
    put("rtCorrReitMkt", d.reit.corr(d.mkt)); put("rtCorrCreMkt", d.cre.corr(d.mkt)); put("rtCorrCreReit", d.cre.corr(d.reit))
    pct("rtVolMkt", 2 * d.mkt.std(ddof=1), 1)
    # lead of REITs: correlation of property this quarter with REITs k quarters earlier
    cc = {k: d.cre.corr(r["reit"].shift(k).reindex(d.index)) for k in range(-4, 9)}
    put("rtLeadOne", cc[1]); put("rtLeadTwo", cc[2]); put("rtLeadFour", cc[4]); put("rtLagOne", cc[-1])
    # regression of property on current and lagged REIT returns
    b, tot, n = L.dimson(d.cre, r["reit"].reindex(d.index), lags=4)
    put("rtSumReit", tot); put("rtBzeroReit", b[0])
    # slope on the current quarter alone, for comparison with b[0] inside the five-term regression
    put("rtBuniReit", np.polyfit(d.reit, d.cre, 1)[0])
    # Dimson betas on the market
    b0 = np.polyfit(d.mkt, d.cre, 1)[0]
    b, tot, _ = L.dimson(d.cre, d.mkt, lags=4)
    put("rtBetaCre", b0); put("rtBetaCreDim", tot)
    bR = np.polyfit(d.mkt, d.reit, 1)[0]; _, totR, _ = L.dimson(d.reit, d.mkt, lags=4)
    put("rtBetaReit", bR); put("rtBetaReitDim", totR)
    # annual horizon: calendar year sums of the quarterly log returns
    y = d.groupby(d.index.year).agg(["sum", "count"])
    full = y[("cre", "count")] == 4
    ya = pd.DataFrame({c: y[(c, "sum")][full] for c in ("cre", "reit", "mkt")})
    R.N["rtYFirst"] = str(ya.index[0]); R.N["rtYLast"] = str(ya.index[-1]); put("rtNY", len(ya), "{:.0f}")
    put("rtCorrY", ya.cre.corr(ya.reit)); put("rtCorrYLag", ya.cre.corr(ya.reit.shift(1)))
    pct("rtVolCreY", ya.cre.std(ddof=1), 1); pct("rtVolReitY", ya.reit.std(ddof=1), 1)
    # removing leverage
    s = 2 * d.reit.std(ddof=1)
    for lam, tag in zip(LEVERAGE, ("Three", "Four", "Five")):
        pct(f"rtUnlev{tag}", (1 - lam) * s, 1)
    pct("rtVolReit", s, 1); pct("rtVolCre", 2 * d.cre.std(ddof=1), 1)
    pct("rtUvolCre", 2 * L.unsmooth(d.cre.values, L.ar1(d.cre)).std(ddof=1), 1)
    # equity correlation of unlevered REIT and private property: leverage does not change correlation
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    ks = np.array(list(cc)); ax[0].bar(ks, [cc[k] for k in ks], color=[RED if k > 0 else SKY for k in ks], width=0.7)
    n = len(d); ax[0].axhspan(-2 / np.sqrt(n), 2 / np.sqrt(n), color="#EEEEEE", zorder=0)
    ax[0].axhline(0, color="#333333", lw=0.6)
    ax[0].set_xticks(range(-4, 9, 2)); ax[0].set_xlabel("quarters by which REITs lead"); ax[0].set_ylabel("correlation"); ax[0].grid(axis="x", visible=False)
    ax[1].scatter(100 * ya.reit, 100 * ya.cre, s=10, color=BLUE)
    z = np.polyfit(ya.reit, ya.cre, 1); xx = np.linspace(ya.reit.min(), ya.reit.max(), 10)
    ax[1].plot(100 * xx, 100 * np.polyval(z, xx), color=RED, lw=1)
    ax[1].set_xlabel("REIT price return in the year, %"); ax[1].set_ylabel("property price return, %")
    fig.tight_layout(); R.save(fig, "fig06_leadlag"); plt.close(fig)
    return finish("reits")


if __name__ == "__main__":
    print(main()["N"])
