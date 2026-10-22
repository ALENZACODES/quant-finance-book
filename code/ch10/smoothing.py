"""Section 10.3: appraisal smoothing. Autocorrelation and volatility of the Z.1 commercial property
index before and after 1996, when it changes from appraisal-based series to a repeat-sales index, of the
FHFA house price index and of equity REITs; unsmoothing of the appraisal-era index, and the identity of
Proposition 10.7(a) for its cumulative return. Writes fig04_levels.png and
fig05_acf.png.
Also writes fig15_smoothsim.png, simulated true and reported returns (random stream 2)."""
import math
import numpy as np
import pandas as pd
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish

def acf(x, k):
    x = pd.Series(x).dropna()
    return [x.autocorr(j) for j in range(1, k + 1)]


def main():
    r = L.quarterly_returns()
    pre = r["cre"][r.index < L.SPLICE].dropna()
    post = r["cre"][r.index > L.SPLICE].dropna()       # the splice quarter itself is dropped
    reit = r["reit"].dropna(); hpi = r["hpi"].dropna()
    R.N["smPreFirst"] = f"{pre.index[0].year}Q{pre.index[0].quarter}"; R.N["smPreLast"] = f"{pre.index[-1].year}Q{pre.index[-1].quarter}"
    R.N["smPostFirst"] = f"{post.index[0].year}Q{post.index[0].quarter}"; R.N["smLast"] = f"{post.index[-1].year}Q{post.index[-1].quarter}"
    R.N["smReitFirst"] = f"{reit.index[0].year}Q{reit.index[0].quarter}"; R.N["smHpiFirst"] = f"{hpi.index[0].year}Q{hpi.index[0].quarter}"
    q0 = L.load_quarterly(); brk = q0["reit_reval"] / q0["reit_mv"].shift(1)
    pct("smReitBreak", brk.loc["1996-10-01"], 0); put("smNHpi", len(hpi), "{:.0f}")
    put("smNPre", len(pre), "{:.0f}"); put("smNPost", len(post), "{:.0f}"); put("smNReit", len(reit), "{:.0f}")
    out = {}
    for tag, x in (("Pre", pre), ("Post", post), ("Reit", reit), ("Hpi", hpi)):
        a = L.ar1(x); out[tag] = a
        put(f"smAr{tag}", a, "{:.2f}"); put(f"smAr{tag}X", a, "{:.3f}"); pct(f"smVol{tag}", 2 * x.std(ddof=1), 1)
        pct(f"smMean{tag}", 4 * x.mean(), 1)
        put(f"smAcfFour{tag}", x.autocorr(4), "{:.2f}")
    # unsmoothing with the estimated alpha
    for tag, x in (("Pre", pre), ("Post", post), ("Hpi", hpi)):
        u = L.unsmooth(x.values, out[tag])
        pct(f"smUvol{tag}", 2 * u.std(ddof=1), 1); put(f"smUar{tag}", L.ar1(u), "{:.2f}")
        put(f"smTheory{tag}", np.sqrt((1 + out[tag]) / (1 - out[tag])), "{:.2f}")
        put(f"smRatio{tag}", u.std(ddof=1) / x.std(ddof=1), "{:.2f}")
    # 2008 to 2009: peak to trough of the CRE index and of REITs
    q = L.load_quarterly()
    c = q["cre"]["2006":"2012"]; pct("smCreFall", c.min() / c["2006":"2008"].max() - 1, 0); pct("smCreFallAbs", 1 - c.min() / c["2006":"2008"].max(), 0)
    R.N["smCrePeak"] = f"{c['2006':'2008'].idxmax().year}Q{c['2006':'2008'].idxmax().quarter}"
    R.N["smCreTrough"] = f"{c.idxmin().year}Q{c.idxmin().quarter}"
    lr = reit["2006":"2012"].cumsum()
    lev = np.exp(lr - lr["2006":"2007"].max())
    pct("smReitFall", lev.min() - 1, 0); pct("smReitFallAbs", 1 - lev.min(), 0)
    pk = lr["2006":"2007"].idxmax(); R.N["smReitPeak"] = f"{pk.year}Q{pk.quarter}"
    R.N["smReitTrough"] = f"{lev.idxmin().year}Q{lev.idxmin().quarter}"
    # Proposition 10.7(a): the cumulative return of the appraisal era, reported and unsmoothed
    a_ = out["Pre"]; u = L.unsmooth(pre.values, a_)
    rep, true = pre.values[1:].sum(), u.sum()
    pct("smCumRep", rep, 1); pct("smCumTrue", true, 1)
    pct("smCumGap", a_ / (1 - a_) * (pre.values[-1] - pre.values[0]), 1)
    assert abs(rep - true + a_ / (1 - a_) * (pre.values[-1] - pre.values[0])) < 1e-12
    put("smCorrShrink", np.sqrt(1 - a_ ** 2), "{:.2f}"); put("smCorrHalf", 0.5 * np.sqrt(1 - a_ ** 2), "{:.2f}")
    # figures
    lv = pd.DataFrame({"cre": q["cre"], "hpi": q["hpi"]}).loc["1997":]
    lv = np.log(lv / lv.iloc[0])
    rl = reit.cumsum(); rl = rl - rl.iloc[0]
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.plot(lv.index, 100 * np.expm1(lv["cre"]) + 100, color=BLUE, label="commercial property (Z.1)")
    ax.plot(rl.index, 100 * np.exp(rl), color=RED, label="equity REITs, price")
    ax.plot(lv.index, 100 * np.expm1(lv["hpi"]) + 100, color=GREY, label="houses (FHFA)")
    ax.set_yscale("log"); ax.set_ylabel(f"index, {R.N['smReitFirst']} = 100"); ax.set_yticks([50, 100, 150, 200, 300, 400])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}")); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout(); R.save(fig, "fig04_levels"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    k = np.arange(1, 9); w = 0.2
    for i, (lab, x, col) in enumerate((("property, appraisal era", pre, BLUE), ("property, repeat sales", post, SKY),
                                       ("houses", hpi, GREY), ("equity REITs", reit, RED))):
        ax.bar(k + (i - 1.5) * w, acf(x, 8), width=w, color=col, label=lab)
    ax.axhline(0, color="#333333", lw=0.6)
    n = len(post); ax.axhspan(-2 / np.sqrt(n), 2 / np.sqrt(n), color="#EEEEEE", zorder=0)
    ax.set_xlabel("lag, quarters"); ax.set_ylabel("autocorrelation"); ax.set_xticks(k); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7.8)
    fig.tight_layout(); R.save(fig, "fig05_acf"); plt.close(fig)
    simulated_figure(out["Pre"], post)
    return finish("smoothing")


QSIM = 40


def simulated_figure(alpha, post):
    """Simulated quarterly true log returns, normal with the mean and volatility of the repeat-sales
    era, smoothed at the appraisal-era alpha: cumulative true and reported returns over 40 quarters, and
    the model's volatility factor sqrt((1 - alpha) / (1 + alpha)). Random stream 2."""
    rng = R.rng_for(2)
    r = post.mean() + post.std(ddof=1) * rng.standard_normal(QSIM + 1)
    r[0] = post.mean()                                  # the filter starts from the mean report
    s = L.smooth(r, alpha)
    ct, cr = np.cumsum(r[1:]), np.cumsum(s[1:])
    gap = ct - cr                                       # Proposition 10.7(a): an end-point term
    R.RAW["ssIdErr"] = float(np.max(np.abs(gap - alpha / (1 - alpha) * (s[1:] - s[0]))))
    pct("ssGapMax", np.abs(gap).max(), 1); pct("ssGapEnd", gap[-1], 1); put("ssQ", QSIM, "{:.0f}")
    pct("ssVolTrue", 2 * r[1:].std(ddof=1), 1); pct("ssVolRep", 2 * s[1:].std(ddof=1), 1)
    fac = math.sqrt((1 - alpha) / (1 + alpha))          # volatility factor of the smoothing model
    put("ssVolFactor", fac, "{:.2f}"); pct("ssVolModel", 2 * r[1:].std(ddof=1) * fac, 1)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    q = np.arange(1, QSIM + 1)
    ax.plot(q, 100 * ct, color=GREY, label="true"); ax.plot(q, 100 * cr, color=BLUE, label="reported")
    ax.set_xlabel("quarter"); ax.set_ylabel("cumulative log return, %"); ax.legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig15_smoothsim"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
