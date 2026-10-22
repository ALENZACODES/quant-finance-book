"""Section 8.8: what appraisal smoothing does to measured risk. Monthly US market returns smoothed
with phi = 0.6, the measured volatility, Sharpe ratio, VaR and drawdown before and after, the
correction for serial correlation, and Geltner's unsmoothing. Writes fig08_smoothing.png."""
import math
import numpy as np
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish

PHI = 0.6


def stats_of(r, rf):
    ex = r - rf
    return dict(vol=math.sqrt(12) * r.std(ddof=1), exvol=math.sqrt(12) * ex.std(ddof=1), exmean=12 * ex.mean(), sr=math.sqrt(12) * L.sharpe(ex),
                srlo=L.lo_annual_factor(ex, 12) * L.sharpe(ex), var=L.var_hist(r, 0.99),
                mdd=L.max_drawdown(r), ac=L.ar1(r))


def main():
    f = L.load_monthly()
    rf = f["rf"].values; r = f["mkt_rf"].values + rf
    rs = L.smooth(r, PHI)
    put("smPhi", PHI, "{:.1f}")
    put("smVolRatioTheory", math.sqrt((1 - PHI) / (1 + PHI)), "{:.2f}")
    a = stats_of(r, rf); b = stats_of(rs, rf)
    phat = L.ar1(rs); ru = L.unsmooth(rs, phat); c = stats_of(ru, rf[1:])
    put("smPhiHat", phat, "{:.3f}")
    for tg, d in (("T", a), ("S", b), ("U", c)):
        pct(f"smVol{tg}", d["vol"], 1); pct(f"smExVol{tg}", d["exvol"], 2); pct(f"smExMean{tg}", d["exmean"], 2); put(f"smSR{tg}", d["sr"]); put(f"smSRlo{tg}", d["srlo"])
        pct(f"smVar{tg}", d["var"], 1); pct(f"smMdd{tg}", d["mdd"], 0); put(f"smAc{tg}", d["ac"])
    put("smRatio", b["vol"] / a["vol"], "{:.2f}")
    # beta of smoothed on the market, contemporaneous and summed over lags (Dimson, 3 lags)
    X = np.column_stack([np.ones(len(r) - 3)] + [r[3 - j:len(r) - j] for j in range(4)])
    coef = np.linalg.lstsq(X, rs[3:], rcond=None)[0]
    put("smBetaZero", np.polyfit(r, rs, 1)[0]); put("smBetaSum", coef[1:].sum())
    # calendar year volatility (compounded annual returns)
    yrs = f.index.values // 100
    def annual(x):
        return np.array([np.prod(1 + x[yrs == y]) - 1 for y in np.unique(yrs) if np.sum(yrs == y) == 12])
    pct("smAnnVolT", annual(r).std(ddof=1), 1); pct("smAnnVolS", annual(rs).std(ddof=1), 1)
    put("smCorrU", np.corrcoef(ru, r[1:])[0, 1], "{:.4f}")
    # figure: twelve months around 2008 and wealth paths
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8))
    m = (f.index.values >= 200701) & (f.index.values <= 201012)
    t = np.arange(m.sum())
    ax[0].plot(t, 100 * r[m], color=GREY, lw=0.9, label="market")
    ax[0].plot(t, 100 * rs[m], color=RED, label="smoothed")
    ax[0].set_xticks([0, 12, 24, 36]); ax[0].set_xticklabels(["2007", "2008", "2009", "2010"])
    ax[0].set_ylabel("monthly return, %"); ax[0].legend(loc="lower left")
    labs = ["volatility", "99% VaR", "max drawdown"]
    vals = [[a["vol"], a["var"], a["mdd"]], [b["vol"], b["var"], b["mdd"]], [c["vol"], c["var"], c["mdd"]]]
    xs = np.arange(3); w = 0.27
    for j, (nm, col) in enumerate((("market", GREY), ("smoothed", RED), ("unsmoothed", BLUE))):
        ax[1].bar(xs + (j - 1) * w, 100 * np.array(vals[j]), w, color=col, label=nm)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labs); ax[1].set_ylabel("%"); ax[1].legend(loc="upper left")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig08_smoothing"); plt.close(fig)
    return finish("smoothing")


if __name__ == "__main__":
    print(main()["N"])
