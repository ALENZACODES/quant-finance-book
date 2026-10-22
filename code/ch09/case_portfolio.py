"""Section 9.8, case study: what a trend fund (net of a 2 and 20 fee) and a small bitcoin sleeve did to a
60/40 portfolio of US equities and the US 10 year bond, with block bootstrap intervals for the change
in the Sharpe ratio, the log wealth gap the trend fund made, and a simulation of choosing the best of
twenty tokens with no expected return (random stream 21). Writes fig10_case.png, fig20_wealthgap.png
and fig21_selection.png."""
import math
import numpy as np
import pandas as pd
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, PALE, put, pct, finish

BLOCK, REPS = 12, 5000


def stats_of(x):
    return dict(m=12 * x.mean(), v=math.sqrt(12) * x.std(ddof=1), sr=L.sharpe(x), mdd=L.max_drawdown(x),
                worst=x.min())


def boot_diff(rng, a, b):
    idx = L.block_bootstrap_idx(rng, len(a), BLOCK, REPS)
    A, B = a[idx], b[idx]
    sa = math.sqrt(12) * A.mean(1) / A.std(1, ddof=1); sb = math.sqrt(12) * B.mean(1) / B.std(1, ddof=1)
    d = sb - sa
    return np.quantile(d, [0.025, 0.975]), np.mean(d <= 0)


def main():
    pr = pd.read_csv(R.HERE / "out" / "program.csv")
    months = pr["month"].values
    A = 0.6 * pr["eq"].values + 0.4 * pr["ust"].values           # excess returns
    B = 0.9 * A + 0.1 * pr["net"].values
    G = 0.9 * A + 0.1 * pr["gross"].values
    rng = R.rng_for(6)
    for tag, x in (("A", A), ("B", B), ("G", G)):
        s = stats_of(x)
        pct(f"cs{tag}Mean", s["m"], 1); pct(f"cs{tag}Vol", s["v"], 1); put(f"cs{tag}SR", s["sr"])
        pct(f"cs{tag}Mdd", s["mdd"], 0); pct(f"cs{tag}Worst", s["worst"], 1)
    rf = pr["rf"].values
    def window(a, b, x):
        m = (months >= a) & (months <= b); return L.compound(x[m] + rf[m])
    pct("csAGfc", window(200711, 200902, A), 0); pct("csBGfc", window(200711, 200902, B), 0)
    pct("csATwo", window(202201, 202212, A), 0); pct("csBTwo", window(202201, 202212, B), 0)
    for k in ("csAGfc", "csBGfc", "csATwo", "csBTwo"):
        put(k + "Abs", -R.RAW[k], "{:.0f}")
    put("csCorrAT", np.corrcoef(A, pr["net"].values)[0, 1])
    ci, p = boot_diff(rng, A, B)
    put("csDiff", L.sharpe(B) - L.sharpe(A), "{:.3f}"); put("csDiffTab", L.sharpe(B) - L.sharpe(A), "{:.2f}")
    put("csDiffLo", ci[0], "{:.2f}"); put("csDiffHi", ci[1], "{:.2f}"); pct("csDiffP", p, 0)
    ci, p = boot_diff(rng, A, G); put("csDiffG", L.sharpe(G) - L.sharpe(A), "{:.2f}")
    put("csDiffGLo", ci[0], "{:.2f}"); put("csDiffGHi", ci[1], "{:.2f}")
    put("csBlock", BLOCK, "{:.0f}"); put("csReps", REPS, "{:,.0f}")
    # bitcoin, January 2015 to August 2026
    cr = pd.read_csv(R.HERE / "out" / "crypto_monthly.csv", index_col=0)
    k = (months >= 201501); mk = months[k]
    xb = cr["btc"].reindex(mk).values - rf[k]
    A2 = A[k]
    for w, tag in ((0.02, "Two"), (0.05, "Five")):
        C = (1 - w) * A2 + w * xb
        s = stats_of(C)
        pct(f"cc{tag}Mean", s["m"], 1); pct(f"cc{tag}Vol", s["v"], 1); put(f"cc{tag}SR", s["sr"])
        pct(f"cc{tag}Mdd", s["mdd"], 0); pct(f"cc{tag}Worst", s["worst"], 1)
        ci, p = boot_diff(rng, A2, C); put(f"cc{tag}Diff", L.sharpe(C) - L.sharpe(A2), "{:.2f}")
        put(f"cc{tag}Lo", ci[0], "{:.2f}"); put(f"cc{tag}Hi", ci[1], "{:.2f}")
    s = stats_of(A2)
    pct("ccAMean", s["m"], 1); pct("ccAVol", s["v"], 1); put("ccASR", s["sr"]); pct("ccAMdd", s["mdd"], 0); pct("ccAWorst", s["worst"], 1)
    # in 2022 bitcoin
    m22 = (mk >= 202201) & (mk <= 202212)
    pct("ccBtcTwo", L.compound(xb[m22] + rf[k][m22]), 0)
    pct("ccTwoTwo", L.compound((0.98 * A2 + 0.02 * xb)[m22] + rf[k][m22]), 0)
    pct("ccATwo", L.compound(A2[m22] + rf[k][m22]), 0)
    for k_ in ("ccBtcTwo", "ccTwoTwo", "ccATwo"):
        put(k_ + "Abs", -R.RAW[k_], "{:.0f}")
    wealth_gap_figure(months, A, B, G, rf)
    selection_figure(A2, xb)
    # figure
    t = np.arange(len(A)) / 12 + 2001
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    ax[0].plot(t, np.cumprod(1 + A + rf), color=GREY, label="60/40")
    ax[0].plot(t, np.cumprod(1 + B + rf), color=BLUE, label="with 10% trend fund")
    ax[0].set_yscale("log"); L.plain_log_axis(ax[0]); ax[0].set_ylabel("growth of 1"); ax[0].legend(loc="upper left")
    ax[1].plot(t, 100 * L.drawdown_path(A), color=GREY, label="60/40")
    ax[1].plot(t, 100 * L.drawdown_path(B), color=BLUE, label="with 10% trend fund")
    ax[1].set_ylabel("drawdown, %")
    ax[1].legend(loc="lower left", bbox_to_anchor=(0.36, 0.0), frameon=True, framealpha=1, edgecolor="none").set_zorder(5)
    fig.tight_layout(); R.save(fig, "fig10_case"); plt.close(fig)
    return finish("case_portfolio")


def wealth_gap_figure(months, A, B, G, rf):
    """Log wealth of 60/40 with 10% in the trend fund less that of 60/40, net and before fees, in percent."""
    lw = lambda x: np.cumsum(np.log1p(x + rf))
    dn, dg = 100 * (lw(B) - lw(A)), 100 * (lw(G) - lw(A))
    put("cwEnd", dn[-1], "{:.1f}"); put("cwEndGross", dg[-1], "{:.1f}"); put("cwFeeWedge", dg[-1] - dn[-1], "{:.1f}")
    put("cwEndAbs", -dn[-1], "{:.1f}"); put("cwEndGrossAbs", -dg[-1], "{:.1f}")
    g0, g1 = np.searchsorted(months, 200710), np.searchsorted(months, 200902)      # end Oct 2007 to end Feb 2009
    put("cwGfc", dn[g1] - dn[g0], "{:.1f}")
    k = int(np.argmax(dn)); put("cwPeak", dn[k], "{:.1f}"); R.N["cwPeakWhen"] = _label(months[k])
    put("cwFromPeak", dn[k] - dn[-1], "{:.1f}")
    put("cwFromPeakUp", math.ceil(dn[k] - dn[-1]), "{:.0f}")       # "almost" this many points
    t = np.arange(len(A)) / 12 + 2001 + 1 / 12
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.plot(t, dg, color=SKY, label="before fees"); ax.plot(t, dn, color=BLUE, label="net of 2 and 20")
    ax.axhline(0, color=GREY, lw=0.6); ax.set_ylabel("log wealth difference, %")
    ax.legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig20_wealthgap"); plt.close(fig)


def _label(yyyymm):
    from trend import label
    return label(yyyymm)


TOKENS, SELREPS = 20, 2000


def selection_figure(A2, xb):
    """Twenty tokens with bitcoin's volatility and correlation with 60/40 but no expected excess return,
    drawn conditionally on the actual 60/40 months; the best one ex post is added at 2%. Repeated."""
    rng = R.rng_for(21)
    sd, rho = xb.std(ddof=1), np.corrcoef(A2, xb)[0, 1]
    za = (A2 - A2.mean()) / A2.std(ddof=1)
    base = L.sharpe(A2); best, one = np.empty(SELREPS), np.empty(SELREPS)
    for i in range(SELREPS):
        Z = rng.standard_normal((TOKENS, len(A2)))
        tok = sd * (rho * za + math.sqrt(1 - rho * rho) * Z)                   # mean zero by construction
        d = np.array([L.sharpe(0.98 * A2 + 0.02 * x) for x in tok]) - base
        best[i], one[i] = d.max(), d[0]
    actual = R.RAW["ccTwoDiff"]
    put("sxTokens", TOKENS, "{:.0f}"); put("sxReps", SELREPS, "{:,.0f}")
    put("sxMedian", np.median(best)); put("sxQ", np.quantile(best, 0.95))
    pct("sxAbove", np.mean(best >= actual), 0); put("sxCount", int(np.sum(best >= actual)), "{:.0f}")
    put("sxActualFine", actual, "{:.3f}")
    put("sxMedianOne", np.median(one), "{:.3f}")
    R.RAW["sxActual"] = actual
    # how many candidates would make a gain this large the typical best one: the tail of a single token
    hits, n1 = 0, 400_000
    for _ in range(n1 // 20_000):
        tok = sd * (rho * za + math.sqrt(1 - rho * rho) * rng.standard_normal((20_000, len(A2))))
        C = 0.98 * A2 + 0.02 * tok
        hits += int(np.sum(math.sqrt(12) * C.mean(1) / C.std(1, ddof=1) - base >= actual))
    q1 = hits / n1; put("sxSingle", n1, "{:,.0f}"); R.RAW["sxTail"] = q1
    put("sxNeeded", round(math.log(0.5) / math.log(1 - q1), -2), "{:,.0f}")   # Monte Carlo error about 9%
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.hist(best, bins=40, color=PALE, edgecolor=BLUE, lw=0.4)
    ax.axvline(actual, color=RED, lw=1.4, label="bitcoin, actual")
    ax.annotate(f"{actual:.2f}", (actual, 0.92), xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", color=RED, fontsize=9, fontweight="bold")
    ax.set_xlabel("change in Sharpe ratio, best of 20 tokens at 2%"); ax.set_ylabel("simulations")
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig21_selection"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
