"""Section 12.5: forecasting the monthly US equity premium out of sample, January 1965 to December 2025,
with the fourteen predictors of Goyal and Welch: the historical mean, each predictor alone, all of them
by least squares, the mean of the single-predictor forecasts, ridge, the lasso, a random forest and
gradient boosting. Campbell and Thompson's R^2, the Clark and West test and the certainty equivalent
gain of a mean-variance investor. Also writes fig19_dp.png, the log dividend-price ratio beside a
simulated autoregression (random stream 51), and fig21_ct.png, the analytic gain of eq. 12.7.

Needs data/goyal_welch_monthly.csv, built by scripts/pull_ch12.py (not redistributed).
Run on its own:  python code/ch12/equity_premium.py   (writes code/ch12/out/equity_premium.json)
"""
from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

NAMES = ["dp", "dy", "ep", "de", "svar", "bm", "ntis", "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl"]
LABEL = dict(dp="dividend-price ratio", dy="dividend yield", ep="earnings-price ratio", de="dividend payout ratio",
             svar="stock variance", bm="book to market", ntis="net equity expansion", tbl="Treasury bill rate",
             lty="long-term yield", ltr="long-term return", tms="term spread", dfy="default yield spread",
             dfr="default return spread", infl="inflation")
OOS0, GAMMA, WLO, WHI, VARWIN, VALID = 196501, 3.0, 0.0, 1.5, 60, 120
RIDGE_A = np.logspace(-2, 2.5, 19)
LASSO_F = np.logspace(-2.5, 0, 16)
FOREST_B, FOREST_MTRY, LEAF, BOOST_M, BOOST_NU = 100, 5, 40, 150, 0.02


def load():
    d = pd.read_csv(ROOT / "data" / "goyal_welch_monthly.csv")
    x = pd.DataFrame({"month": d.month})
    x["dp"] = np.log(d.d12) - np.log(d["index"]); x["dy"] = np.log(d.d12) - np.log(d["index"].shift(1))
    x["ep"] = np.log(d.e12) - np.log(d["index"]); x["de"] = np.log(d.d12) - np.log(d.e12)
    for c in ["svar", "bm", "ntis", "tbl", "lty", "ltr"]:
        x[c] = d[c]
    x["tms"] = d.lty - d.tbl; x["dfy"] = d.baa - d.aaa; x["dfr"] = d.corpr - d.ltr
    x["infl"] = d.infl.shift(1)                        # inflation is published with a lag of a month
    x["r"] = np.log1p(d.ret) - np.log1p(d.rf)          # log equity premium
    x["rx"] = d.ret - d.rf; x["rf"] = d.rf
    return x.dropna().reset_index(drop=True)


def tune_ridge(X, y):
    n = len(y); tr, va = slice(0, n - VALID), slice(n - VALID, n)
    Z, m, s = L.standardise(X[tr]); Zv = (X[va] - m) / s
    errs = [np.mean((y[va] - (y[tr].mean() + Zv @ L.ridge(Z, y[tr] - y[tr].mean(), a * len(Z)))) ** 2) for a in RIDGE_A]
    return RIDGE_A[int(np.argmin(errs))]


def tune_lasso(X, y):
    n = len(y); tr, va = slice(0, n - VALID), slice(n - VALID, n)
    Z, m, s = L.standardise(X[tr]); Zv = (X[va] - m) / s; yc = y[tr] - y[tr].mean()
    lm = L.lasso_lam_max(Z, yc); errs, b = [], None
    for f in LASSO_F[::-1]:
        b = L.lasso(Z, yc, f * lm, b); errs.append(np.mean((y[va] - (y[tr].mean() + Zv @ b)) ** 2))
    return LASSO_F[::-1][int(np.argmin(errs))]


def forecasts(x):
    X, r, mon = x[NAMES].values, x.r.values, x.month.values
    oos = np.nonzero(mon >= OOS0)[0]
    F = {k: np.empty(len(oos)) for k in ["hm", *NAMES, "ols", "comb", "ridge", "lasso", "forest", "boost", "olsct"]}
    rng = rng_for(5); a_r = f_l = None; forest = boost = None; bl = None; nnz = []
    for k, j in enumerate(oos):
        Xi, yi = X[:j - 1], r[1:j]                       # pairs (x_i, r_{i+1}) with i + 1 <= j - 1
        xj = X[j - 1]
        F["hm"][k] = yi.mean()
        for c, nm in enumerate(NAMES):
            b = L.ols_fit(Xi[:, c], yi); F[nm][k] = b[0] + b[1] * xj[c]
        # payout = dp - ep and term spread = lty - tbl hold exactly, so in every window the design has
        # rank 1 + 12 and lstsq returns the minimum-norm solution
        rank = np.linalg.matrix_rank(np.column_stack([np.ones(len(Xi)), Xi]))
        assert rank == 13, (k, rank)
        if k == 0: RAW["eqNfirst"] = len(yi)
        if k == len(oos) - 1: RAW["eqNlast"] = len(yi)
        b = L.ols_fit(Xi, yi); F["ols"][k] = b[0] + b[1:] @ xj; F["olsct"][k] = max(F["ols"][k], 0.0)
        F["comb"][k] = np.mean([F[nm][k] for nm in NAMES])
        january = mon[j] % 100 == 1
        if january or a_r is None:
            a_r = tune_ridge(Xi, yi); f_l = tune_lasso(Xi, yi)
            forest = L.forest(Xi, yi, FOREST_B, rng, FOREST_MTRY, max_depth=8, min_leaf=LEAF)
            f0, trees, _ = L.boost(Xi, yi, BOOST_M, BOOST_NU, 2, LEAF); boost = (f0, trees)
        Z, m, s = L.standardise(Xi); zj = (xj - m) / s; yc = yi - yi.mean()
        F["ridge"][k] = yi.mean() + zj @ L.ridge(Z, yc, a_r * len(Z))
        bl = L.lasso(Z, yc, f_l * L.lasso_lam_max(Z, yc), bl if bl is not None else None); nnz.append(int((bl != 0).sum()))
        F["lasso"][k] = yi.mean() + zj @ bl
        if k == 0:                                       # the first forecast, for the Octave check
            RAW["eqRidgeAFirst"], RAW["eqLassoFFirst"] = float(a_r), float(f_l)
            RAW["eqRidgeFirst"], RAW["eqLassoFirst"] = float(F["ridge"][k]), float(F["lasso"][k])
        F["forest"][k] = L.forest_predict(forest, xj[None, :])[0]
        F["boost"][k] = L.boost_predict(boost[0], boost[1], xj[None, :], BOOST_NU)[0]
    return oos, F, np.array(nnz)


def dp_figure(x):
    """The log dividend-price ratio from the data, beside one simulated AR(1) path with coefficient 0.98
    (the predictor of cv_time) on random stream 51, scaled to the data's mean and standard deviation."""
    dp = x.dp.values; t = pd.to_datetime(pd.Series(x.month).astype(str), format="%Y%m")
    a1 = float(np.corrcoef(dp[:-1], dp[1:])[0, 1])
    put("dpRho", a1, "{:.4f}"); put("dpFrom", x.month.iloc[0] // 100, "{:.0f}"); put("dpTo", x.month.iloc[-1] // 100, "{:.0f}")
    rng = rng_for(51); e = rng.standard_normal(len(dp) + 200); z = np.zeros_like(e)
    for k in range(1, len(e)):
        z[k] = 0.98 * z[k - 1] + e[k]
    z = z[200:]; z = dp.mean() + dp.std() * (z - z.mean()) / z.std()
    put("dpSimRho", np.corrcoef(z[:-1], z[1:])[0, 1], "{:.3f}")
    put("dpHalf", math.log(0.5) / math.log(a1) / 12, "{:.0f}")          # half-life in years, data
    put("dpSimHalf", math.log(0.5) / math.log(0.98) / 12, "{:.0f}")     # and for the coefficient 0.98
    pct("dpMoveMed", np.median(np.abs(np.diff(dp))) / dp.std(), 0)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True)
    ax[0].plot(t, dp, color=BLUE, lw=0.8); ax[0].set_title(f"data, first autocorrelation {a1:.3f}", fontsize=8)
    ax[1].plot(np.arange(len(z)) / 12, z, color=RED, lw=0.8)
    ax[1].set_title("simulated AR(1), coefficient 0.98", fontsize=8)
    ax[0].set_ylabel("log dividend-price ratio"); ax[1].set_xlabel("years")
    save(fig, "fig19_dp")


def ct_figure(S):
    """Campbell and Thompson's proportional gain against R^2 for four monthly Sharpe ratios."""
    R2 = np.linspace(0, 0.02, 201)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for s, col, ls in ((0.05, RED, "-."), (S, BLUE, "-"), (0.12, SKY, "--"), (0.15, GREY, ":")):
        ax.plot(100 * R2, 100 * L.ct_gain(s, R2), color=col, ls=ls, label=rf"$SR_m={s:.3f}$")
    ax.plot([0.5], [100 * L.ct_gain(S, 0.005)], "o", color=BLUE, ms=4)   # the sample's Sharpe ratio only
    put("ctLowHalf", L.ct_gain(0.05, 0.005) * 100, "{:.0f}")
    ax.set_xlabel(r"monthly $R^2$ of the predictor (%)"); ax.set_ylabel("rise in mean excess return (%)")
    ax.set_ylim(0, 400); ax.legend(loc="upper left")
    save(fig, "fig21_ct")


def main():
    x = load()
    dp_figure(x)
    oos, F, nnz = forecasts(x)
    r, rx, rf, mon = x.r.values[oos], x.rx.values[oos], x.rf.values[oos], x.month.values[oos]
    put("eqFirst", x.month.iloc[0] // 100, "{:.0f}"); put("eqOosN", len(oos), "{:.0f}")
    put("eqOosFrom", mon[0] // 100, "{:.0f}"); put("eqOosTo", mon[-1] // 100, "{:.0f}"); put("eqPred", len(NAMES), "{:.0f}")
    put("eqGamma", GAMMA, "{:.0f}"); put("eqWhi", WHI, "{:.1f}"); put("eqVarWin", VARWIN, "{:.0f}"); put("eqValid", VALID, "{:.0f}")
    put("eqForestB", FOREST_B, "{:.0f}"); put("eqForestMtry", FOREST_MTRY, "{:.0f}"); put("eqLeaf", LEAF, "{:.0f}")
    put("eqBoostM", BOOST_M, "{:.0f}"); put("eqBoostNu", BOOST_NU, "{:.2f}")
    put("eqLassoNz", nnz.mean(), "{:.1f}")
    put("eqWinFirst", RAW["eqNfirst"], "{:,.0f}"); put("eqWinLast", RAW["eqNlast"], "{:,.0f}")
    # variance for the investor: the previous VARWIN months of the log premium
    allr = x.r.values
    var = np.array([allr[j - VARWIN:j].var(ddof=1) for j in oos])
    w_hm = L.mv_weights(F["hm"], var, GAMMA, WLO, WHI); ce_hm = L.ce_annual(rx, w_hm, rf, GAMMA)
    pct("eqCeHm", ce_hm, 2)
    stats, cw_p = {}, {}
    tag = dict(hm="Hm", ols="Ols", comb="Comb", ridge="Ridge", lasso="Lasso", forest="Forest", boost="Boost", olsct="Olsct")
    tag.update({nm: nm.capitalize() for nm in NAMES})
    early, late = mon < 200001, mon >= 200001
    for k, f in F.items():
        w = L.mv_weights(f, var, GAMMA, WLO, WHI); port = w * rx
        sr = port.mean() / port.std(ddof=1)
        s = dict(r2=L.r2_os(r, f, F["hm"]) if k != "hm" else 0.0, ce=L.ce_annual(rx, w, rf, GAMMA) - ce_hm, sr=sr,
                 skew=float(pd.Series(port).skew()), exkurt=float(pd.Series(port).kurt()),
                 r2a=L.r2_os(r[early], f[early], F["hm"][early]), r2b=L.r2_os(r[late], f[late], F["hm"][late]))
        if k != "hm":
            s["cw"], s["p"] = L.clark_west(r, f, F["hm"])
            pct(f"eqRtwo{tag[k]}", s["r2"], 2); put(f"eqCw{tag[k]}", s["cw"], "{:.2f}"); put(f"eqP{tag[k]}", s["p"], "{:.3f}")
            pct(f"eqCe{tag[k]}", s["ce"], 2)
            pct(f"eqRtwoA{tag[k]}", s["r2a"], 2); pct(f"eqRtwoB{tag[k]}", s["r2b"], 2)
        put(f"eqSr{tag[k]}", np.sqrt(12) * sr, "{:.2f}")
        stats[k] = s
    multi = ["ols", "olsct", "comb", "ridge", "lasso", "forest", "boost"]
    put("eqUniPos", sum(stats[nm]["r2"] > 0 for nm in NAMES), "{:.0f}")
    put("eqUniSig", sum(stats[nm]["p"] < 0.05 for nm in NAMES), "{:.0f}")
    best_uni = max(NAMES, key=lambda nm: stats[nm]["r2"]); N["eqBestUni"] = LABEL[best_uni]
    pct("eqBestUniRtwo", stats[best_uni]["r2"], 2)
    worst = min(NAMES, key=lambda nm: stats[nm]["r2"]); N["eqWorstUni"] = LABEL[worst]; pct("eqWorstUniRtwo", stats[worst]["r2"], 1)
    # Campbell and Thompson's gain, at the in-sample Sharpe ratio of the premium before 1965
    pre = x.r.values[: oos[0]]; S = pre.mean() / pre.std(ddof=1)
    put("eqS", S, "{:.3f}"); put("eqSAnn", S * np.sqrt(12), "{:.2f}")
    pct("eqCtHalf", L.ct_gain(S, 0.005), 0)
    pct("eqCtComb", L.ct_gain(S, max(stats["comb"]["r2"], 0)), 0)
    ct_figure(S)
    # where least squares on all fourteen lost ground: change in the cumulative squared error difference by decade
    gain = (r - F["hm"]) ** 2 - (r - F["ols"]) ** 2
    dec = pd.Series(gain).groupby((mon // 100) // 10 * 10).sum()
    up = [int(d) for d, v in dec.items() if v > 0]
    assert len(up) == 1, dec
    N["eqOlsGainDecade"] = f"{up[0]}s"
    yr = pd.Series(gain).groupby(mon // 100).sum().sort_values()
    put("eqOlsWorstYear", yr.index[0], "{:.0f}"); put("eqOlsSecondYear", yr.index[1], "{:.0f}")
    # save for overfit.py and the R check (kept out of the public copy)
    out = pd.DataFrame({"month": mon, "r": r, "rx": rx, "rf": rf, "var": var, **{f"f_{k}": v for k, v in F.items()}})
    out.to_csv(HERE / "out" / "eq_forecasts.csv", index=False)
    (HERE / "out" / "eq_stats.json").write_text(json.dumps({k: {a: float(b) for a, b in v.items()} for k, v in stats.items()}, indent=1))
    # figure 9: cumulative squared error of the historical mean less that of the model
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    t = pd.to_datetime(pd.Series(mon).astype(str), format="%Y%m")
    for k, col, ls, lab in [("dp", GREY, ":", "dividend-price ratio"), ("ols", ROSE, "--", "all fourteen, least squares"),
                            ("comb", BLUE, "-", "mean of the fourteen forecasts"), ("ridge", SKY, "-", "ridge"),
                            ("forest", RED, "-.", "random forest")]:
        ax.plot(t, np.cumsum((r - F["hm"]) ** 2 - (r - F[k]) ** 2), color=col, ls=ls, lw=1.1, label=lab)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_ylabel("cumulative SSE, historical mean\nless model"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.2)
    save(fig, "fig09_cum_sse")
    # figure 10: R^2_OS of every model
    keys = NAMES + multi
    lab = [LABEL[k] for k in NAMES] + ["all, least squares", "all, floored at zero", "mean of forecasts", "ridge", "lasso",
                                       "random forest", "boosting"]
    vals = np.array([100 * stats[k]["r2"] for k in keys])
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ypos = np.arange(len(keys))[::-1]
    cols = [BLUE if stats[k]["p"] < 0.05 else (SKY if v > 0 else ROSE) for k, v in zip(keys, vals)]
    ax.barh(ypos, np.clip(vals, -4, None), color=cols, height=0.7)
    for yp, v in zip(ypos, vals):
        if v < -4:
            ax.text(-3.95, yp, f"{v:.1f}", va="center", ha="left", fontsize=7, color="white")
    ax.set_yticks(ypos); ax.set_yticklabels(lab, fontsize=7.8); ax.axvline(0, color="#333333", lw=0.6)
    ax.axhline(len(multi) - 0.5, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel(r"out-of-sample $R^2$ against the historical mean (%)"); ax.set_xlim(-4, max(2.0, vals.max() + 0.5))
    ax.grid(axis="y", visible=False)
    save(fig, "fig10_r2os")
    return finish("equity_premium")


if __name__ == "__main__":
    import time; t = time.time(); main(); print(f"{time.time() - t:.1f}s")
