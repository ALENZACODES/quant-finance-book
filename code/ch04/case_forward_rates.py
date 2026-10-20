"""Section 4.9: case study. Do Government of Canada forward rates forecast the one year bill rate?
A regression in the style of Fama and Bliss (1987) on month-end data with Newey-West errors.

Run on its own:  python code/ch04/case_forward_rates.py   (writes code/ch04/out/case_forward_rates.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from numbers_ch04 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

COLS = ["TB.CDN.180D.MID", "TB.CDN.1Y.MID", "BD.CDN.2YR.DQ.YLD", "BD.CDN.3YR.DQ.YLD"]


def month_end_forwards():
    df = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", parse_dates=["date"]).set_index("date")
    df = df[COLS].apply(pd.to_numeric, errors="coerce").dropna()
    me = df.groupby(df.index.to_period("M")).tail(1).copy()
    if df.index.max() < (df.index.max().to_period("M").to_timestamp(how="end") - pd.Timedelta(days=5)):
        me = me.iloc[:-1]
    f12, f23 = [], []
    for _, r in me.iterrows():
        c = L.bootstrap({0.5: r[COLS[0]] / 100, 1.0: r[COLS[1]] / 100}, {2.0: r[COLS[2]] / 100, 3.0: r[COLS[3]] / 100})
        f12.append(float(c.fwd_simple(1.0, 2.0))); f23.append(float(c.fwd_simple(2.0, 3.0)))
    me["y1"] = me[COLS[1]] / 100; me["f12"] = f12; me["f23"] = f23
    me.index = me.index.to_period("M")
    return me


def fb_regression(me, fwd, h):
    y = me["y1"].shift(-h) - me["y1"]; x = me[fwd] - me["y1"]
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    X = sm.add_constant(d["x"].values)
    res = sm.OLS(d["y"].values, X).fit(cov_type="HAC", cov_kwds=dict(maxlags=h, use_correction=False))
    return d, res


def main():
    me = month_end_forwards()
    put("caseMonths", len(me), "{:.0f}")
    N["fbStart"] = me.index[0].strftime("%B %Y"); N["fbEnd"] = me.index[-1].strftime("%B %Y")
    fv = me["f12"].dropna().index   # forward observed at these month ends, plotted twelve months later
    N["fbFwdStart"] = (fv[0] + 12).strftime("%B %Y"); N["fbFwdEnd"] = (fv[-1] + 12).strftime("%B %Y")
    out = {}
    for fwd, h, nm in [("f12", 12, "One"), ("f23", 24, "Two")]:
        d, res = fb_regression(me, fwd, h)
        b, a = res.params[1], res.params[0]; se_b = res.bse[1]
        put(f"fb{nm}Slope", b, "{:.2f}"); put(f"fb{nm}SE", se_b, "{:.2f}")
        put(f"fb{nm}TZero", b / se_b, "{:.2f}"); put(f"fb{nm}TOne", (b - 1) / se_b, "{:.2f}")
        put(f"fb{nm}RSq", res.rsquared, "{:.2f}"); put(f"fb{nm}N", len(d), "{:.0f}")
        put(f"fb{nm}Lo", b - 1.96 * se_b, "{:.2f}"); put(f"fb{nm}Hi", b + 1.96 * se_b, "{:.2f}")
        N[f"fb{nm}LastDate"] = me.index[-1].strftime("%B %Y")
        pct(f"fb{nm}Const", a, 2)
        RAW[f"fb{nm}Int"] = float(a)
        out[nm] = (d, res)
    # forecast errors of the one year forward rate, a year later
    err = (me["f12"] - me["y1"].shift(-12)).dropna()
    m = float(err.mean())
    res = sm.OLS(err.values, np.ones(len(err))).fit(cov_type="HAC", cov_kwds=dict(maxlags=12, use_correction=False))
    put("fwdErrMean", 1e4 * m, "{:.0f}"); put("fwdErrSE", 1e4 * res.bse[0], "{:.0f}")
    put("fwdErrT", m / res.bse[0], "{:.1f}")
    pct("fwdAbove", float((err > 0).mean()), 0)
    put("fwdErrMAE", 1e4 * float(err.abs().mean()), "{:.0f}")
    # the naive forecast (no change) against the forward
    naive = (me["y1"] - me["y1"].shift(-12)).dropna()
    put("naiveMAE", 1e4 * float(naive.abs().mean()), "{:.0f}")
    # spread statistics
    spr = (me["f12"] - me["y1"]); put("sprMean", 1e4 * spr.mean(), "{:.0f}"); put("sprSD", 1e4 * spr.std(), "{:.0f}")
    pct("sprPos", float((spr > 0).mean()), 0)
    # figure
    d1, r1 = out["One"]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw=dict(width_ratios=[1.55, 1]))
    target = (me.index + 12).to_timestamp()
    ax[0].plot(me.index.to_timestamp(), 100 * me["y1"], color=BLUE, lw=1.1, label="one year bill rate")
    ax[0].plot(target, 100 * me["f12"], color=RED, lw=1.0, label="forward rate, plotted at the date it targets")
    ax[0].set_ylabel("percent"); ax[0].set_title("(a) forward rate and the rate it forecasts")
    ax[0].legend(fontsize=7.0, loc="upper center", ncol=1); ax[0].set_ylim(0, 7.4)
    ax[1].scatter(1e2 * d1["x"], 1e2 * d1["y"], s=4, color=SKY, alpha=0.7, lw=0)
    xx = np.linspace(d1["x"].min(), d1["x"].max(), 2)
    ax[1].plot(1e2 * xx, 1e2 * (r1.params[0] + r1.params[1] * xx), color=RED, lw=1.2, label=f"fit, slope {r1.params[1]:.2f}")
    ax[1].plot(1e2 * xx, 1e2 * xx, color=GREY, lw=0.9, ls="--", label="slope 1")
    ax[1].axhline(0, color="#999999", lw=0.5); ax[1].axvline(0, color="#999999", lw=0.5)
    ax[1].set_xlabel("forward less spot (points)"); ax[1].set_ylabel("change over next year (points)")
    ax[1].set_title("(b) the regression"); ax[1].legend(fontsize=7.0, loc="lower right")
    fig.tight_layout(); save(fig, "fig10_case")
    return finish("case_forward_rates")


if __name__ == "__main__":
    main()
