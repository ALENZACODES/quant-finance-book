"""Alpha, beta and the Sharpe ratio of a smoothed return series, and the Dimson sum of lagged betas.
The true returns are the excess returns of the Other industry; the reported series is an AR(1)
smoothing of them. Figures fig08_smoothing and fig16_appraisal. No random numbers."""
import math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

QMAX = 12


def main():
    f, vw, ew, cap = L.load()
    m = f.mkt_rf.values; r = vw["Other"].values - f.rf.values; T = len(r)
    true = L.ols(r, np.c_[np.ones(T), m])
    R.put("smoBeta", true["b"][1]); R.put("smoAlpha", 1200 * true["b"][0])
    R.put("smoSR", math.sqrt(12) * r.mean() / r.std(ddof=1)); R.put("smoVol", 100 * math.sqrt(12) * r.std(ddof=1), "{:.1f}")
    # reconcile phi = 0.5 with the factor sqrt((1 - phi) / (1 + phi)), which assumes uncorrelated returns
    x = r - r.mean(); acv = lambda h: float((x[h:] * x[:T - h]).sum() / T)
    R.put("smoOwnAcf", acv(1) / acv(0)); ph = 0.5
    R.put("smoVolFormula", 100 * math.sqrt(12) * r.std(ddof=1) * math.sqrt((1 - ph) / (1 + ph)), "{:.1f}")
    R.put("smoSRFormula", math.sqrt(12) * r.mean() / r.std(ddof=1) * math.sqrt((1 + ph) / (1 - ph)))
    v = sum((1 - ph) ** 2 * ph ** (j + k) * acv(abs(j - k)) for j in range(60) for k in range(60))
    R.put("smoVolExact", 100 * math.sqrt(12 * v), "{:.1f}")
    R.put("smoMeanKept", 100 * L.smooth_ar1(r, ph).mean() / r.mean(), "{:.1f}")
    curves = {}
    for phi, nm in ((0.5, "Half"), (0.8, "Eight")):
        ro = L.smooth_ar1(r, phi)
        c0 = L.ols(ro, np.c_[np.ones(T), m])
        R.put(f"smoBeta{nm}", c0["b"][1]); R.put(f"smoAlpha{nm}", 1200 * c0["b"][0])
        R.put(f"smoAlphaT{nm}", c0["t"][0]); R.put(f"smoSR{nm}", math.sqrt(12) * ro.mean() / ro.std(ddof=1))
        R.put(f"smoVol{nm}", 100 * math.sqrt(12) * ro.std(ddof=1), "{:.1f}")
        R.put(f"smoAcf{nm}", np.corrcoef(ro[1:], ro[:-1])[0, 1])
        bs = []; al = []
        for q in range(QMAX + 1):
            b, a, _ = L.dimson(ro, m, q); bs.append(b); al.append(a)
        curves[phi] = np.array(bs)
        R.put(f"smoDimson{nm}", bs[QMAX]); R.put(f"smoDimsonAlpha{nm}", 1200 * al[QMAX])
        R.put(f"smoDimsonSix{nm}", bs[6])
        R.put(f"smoSpur{nm}", 1200 * al[QMAX] - R.RAW["smoAlpha"])     # alpha left after the lags
        R.RAW[f"smoDimsonAll{nm}"] = bs
        # unsmoothing recovers the series exactly when phi is known
        back = (ro[1:] - phi * ro[:-1]) / (1 - phi)
        R.RAW[f"smoUnsmoothErr{nm}"] = float(np.max(np.abs(back - r[1:])))
    R.put("smoQ", QMAX, "{:.0f}"); R.put("smoMktAcf", np.corrcoef(m[1:], m[:-1])[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    q = np.arange(QMAX + 1); b = true["b"][1]
    for phi, col in ((0.5, BLUE), (0.8, RED)):
        ax.plot(q, curves[phi], "o", color=col, ms=4, label=f"estimated, $\\phi={phi}$")
        ax.plot(q, b * (1 - phi ** (q + 1)), color=col, lw=0.9, ls="--")
    ax.axhline(b, color=GREY, lw=1.0, label="beta of the true returns")
    ax.set_xlabel("number of lagged market returns $q$"); ax.set_ylabel("sum of slopes")
    ax.set_ylim(0, 1.4); ax.legend(loc="lower right"); R.save(fig, "fig08_smoothing")
    crisis_figure(f, r)
    return R.finish("smoothing_beta")


def crisis_figure(f, r):
    """Wealth from the excess returns of Other, October 2007 to December 2009, true and smoothed with
    phi = 0.5 and 0.8 (the smoothing runs over the whole sample, as in the regressions)."""
    idx = f.index.values; win = (idx >= 200710) & (idx <= 200912)
    series = [("true", r, GREY, "True"), ("$\\phi=0.5$", L.smooth_ar1(r, 0.5), BLUE, "Half"),
              ("$\\phi=0.8$", L.smooth_ar1(r, 0.8), RED, "Eight")]
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    t = np.arange(win.sum() + 1)
    for lab, x, col, tag in series:
        W = np.concatenate([[1.0], np.cumprod(1 + x[win])])
        dd = 1 - W / np.maximum.accumulate(W)
        R.put(f"smoDd{tag}", 100 * dd.max(), "{:.0f}")
        ax.plot(t, 100 * W, color=col, lw=1.2, label=f"{lab}, worst drawdown {100 * dd.max():.0f}%")
    R.N["smoDdFirst"] = L.month_name(200710); R.N["smoDdLast"] = L.month_name(200912)
    ax.set_xticks([0, 15, 27]); ax.set_xticklabels(["end of Sep 2007", "Dec 2008", "Dec 2009"])
    ax.set_ylabel("wealth, start = 100"); ax.legend(loc="lower left")
    R.save(fig, "fig16_appraisal")


if __name__ == "__main__":
    main()
