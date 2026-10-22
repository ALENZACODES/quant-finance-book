"""Section 12.7: what a search does to a backtest. The expected maximum of many zero-skill Sharpe
ratios, a search over trading rules on simulated prices with no predictability, the probability of
backtest overfitting by combinatorially symmetric cross-validation, the deflated Sharpe ratio of the
best equity premium model, and Holm's correction of the fourteen single-predictor tests. Also writes
fig20_dsr.png, the deflated Sharpe ratio against the assumed number of trials.

Reads code/ch12/out/eq_forecasts.csv and eq_stats.json, written by equity_premium.py.
Run on its own:  python code/ch12/overfit.py   (writes code/ch12/out/overfit.json)
"""
from __future__ import annotations
import json, math
import numpy as np
import pandas as pd
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, HERE  # noqa: F401

NS = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
TB, VOLB, LOOK, REPS, S_FIG, S_REP = 240, 0.045, 50, 200, 16, 10
from equity_premium import LABEL  # noqa: E402
EQ_NAMES = ["dp", "dy", "ep", "de", "svar", "bm", "ntis", "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl"]


def rules(r):
    """Returns of 4 x LOOK rules on a series of monthly returns r: long or flat, and long or short, when
    the price is above its moving average over 2 to LOOK + 1 months, and the two contrarian versions."""
    p = np.concatenate([[0.0], np.cumsum(r)])            # log price
    out = []
    for Lk in range(2, LOOK + 2):
        ma = pd.Series(p).rolling(Lk).mean().values
        sig = (p - ma)[:-1]                               # known at the end of month t, applied to month t + 1
        up = np.where(np.isnan(sig), 0.0, (sig > 0).astype(float))
        for pos in (up, 2 * up - 1, 1 - up, 1 - 2 * up):
            out.append(pos[LOOK + 1:] * r[LOOK + 1:])
    return np.array(out).T                                # rows: months after the warm-up


def sr(x):
    """Per-period Sharpe ratios of the columns; a rule that never trades in the window scores zero."""
    sd = x.std(0, ddof=1)
    return np.divide(x.mean(0), sd, out=np.zeros(x.shape[1]), where=sd > 0)


def main():
    rng = rng_for(7)
    # expected maximum of N standard normals
    ex = np.array([L.expected_max_normal(n) for n in NS]); ap = np.array([L.expected_max_approx(n) for n in NS])
    sim = np.array([rng.standard_normal((20000, n)).max(1).mean() for n in NS])
    for n, nm in [(10, "Ten"), (100, "Hundred"), (1000, "Thousand")]:
        i = NS.index(n)
        put(f"ovMax{nm}", ex[i], "{:.3f}"); put(f"ovApprox{nm}", ap[i], "{:.3f}"); put(f"ovSim{nm}", sim[i], "{:.3f}")
        put(f"ovBound{nm}", math.sqrt(2 * math.log(n)), "{:.3f}")
    put("ovSimReps", 20000, "{:,.0f}")
    put("ovYears", 10, "{:.0f}"); put("ovMaxSrHundred", ex[NS.index(100)] / math.sqrt(10), "{:.2f}")
    RAW["ovApproxErr"] = float(np.max(np.abs(ap - ex)[2:]))
    # the search on noise
    nrule = 4 * LOOK
    put("ovRules", nrule, "{:.0f}"); put("ovLook", LOOK, "{:.0f}"); put("ovT", TB, "{:.0f}"); put("ovReps", REPS, "{:.0f}")
    pct("ovVol", VOLB, 1); put("ovHalf", (TB - LOOK - 1) // 2, "{:.0f}")
    best_is, best_oos, pbos, med_corr = [], [], [], []
    rr = rng_for(71)
    for k in range(REPS):
        R = rules(VOLB * rr.standard_normal(TB))
        h = len(R) // 2
        si, so = sr(R[:h]), sr(R[h:]); b = int(np.argmax(si))
        best_is.append(si[b]); best_oos.append(so[b])
        if k < 50:
            pbos.append(L.cscv_pbo(R, S_REP)[0])
        if k == 0:
            C = np.corrcoef(R.T); med_corr = float(np.median(np.abs(C[np.triu_indices(nrule, 1)])))
    a12 = math.sqrt(12)
    put("ovBestIs", a12 * np.mean(best_is), "{:.2f}"); put("ovBestOos", a12 * np.mean(best_oos), "{:.2f}")
    put("ovBestOosSe", a12 * np.std(best_oos, ddof=1) / math.sqrt(REPS), "{:.2f}")
    put("ovPbo", np.mean(pbos), "{:.2f}"); put("ovPboRuns", len(pbos), "{:.0f}"); put("ovSRep", S_REP, "{:.0f}")
    put("ovCorr", med_corr, "{:.2f}")
    # the effective number of independent trials implied by the best in-sample Sharpe ratio
    h = (TB - LOOK - 1) // 2
    z = np.mean(best_is) * math.sqrt(h)
    neff = 10 ** np.interp(z, ex, np.log10(NS)) if z < ex[-1] else NS[-1]
    put("ovNeff", neff, "{:.0f}")
    # figure 14 data: one run with S_FIG blocks
    R = rules(VOLB * rng_for(72).standard_normal(TB))
    pbo, logits, pairs = L.cscv_pbo(R, S_FIG)
    put("ovPboFig", pbo, "{:.2f}"); put("ovSFig", S_FIG, "{:.0f}"); put("ovSplits", len(logits), "{:,.0f}")
    put("ovSlope", np.polyfit(pairs[:, 0], pairs[:, 1], 1)[0], "{:.2f}")
    # the deflated Sharpe ratio of the best equity premium model, on its return over the historical mean's portfolio
    f = pd.read_csv(HERE / "out" / "eq_forecasts.csv")
    models = [c[2:] for c in f.columns if c.startswith("f_") and c != "f_hm"]
    w = lambda c: L.mv_weights(f[f"f_{c}"].values, f["var"].values, 3.0, 0.0, 1.5)
    act = {m: (w(m) - w("hm")) * f.rx.values for m in models}
    srs = {m: a.mean() / a.std(ddof=1) for m, a in act.items()}
    best = max(srs, key=srs.get); a = act[best]; T = len(a)
    sk = float(pd.Series(a).skew()); ku = float(pd.Series(a).kurt() + 3)
    put("ovEqModels", len(models), "{:.0f}"); N["ovEqBest"] = LABEL.get(best, best)
    put("ovEqSr", srs[best], "{:.4f}"); put("ovEqSrAnn", a12 * srs[best], "{:.2f}"); put("ovEqT", T, "{:.0f}")
    put("ovEqSkew", sk, "{:.2f}"); put("ovEqKurt", ku, "{:.1f}")
    put("ovEqPsr", L.psr(srs[best], T, sk, ku), "{:.3f}")
    vs = float(np.var(list(srs.values()), ddof=1))
    d, bench = L.dsr(srs[best], T, len(models), vs, sk, ku)
    put("ovEqDsr", d, "{:.3f}"); put("ovEqBench", bench, "{:.4f}"); put("ovEqSrSd", math.sqrt(vs), "{:.4f}")
    # least squares and its floored version hold the same portfolio: negative forecasts give a zero weight either way
    distinct = {tuple(np.round(act[m], 12)) for m in models}; put("ovEqDistinct", len(distinct), "{:.0f}")
    assert np.allclose(act["ols"], act["olsct"])
    # counting the two identical series once: N = 20 and the spread over the distinct series
    vs20 = float(np.var([v for m, v in srs.items() if m != "olsct"], ddof=1))
    put("ovEqDsrDistinct", L.dsr(srs[best], T, len(distinct), vs20, sk, ku)[0], "{:.3f}")
    put("ovEqSe", L.sharpe_se(srs[best], T, sk, ku), "{:.4f}")
    dsr_figure(srs[best], T, vs, sk, ku, len(models))
    # Holm, Bonferroni and Benjamini-Hochberg on the fourteen Clark-West p-values
    st = json.loads((HERE / "out" / "eq_stats.json").read_text())
    p = np.array([st[m]["p"] for m in EQ_NAMES])
    for alpha, nm in [(0.10, "Ten"), (0.05, "Five")]:
        put(f"ovRaw{nm}", int((p <= alpha).sum()), "{:.0f}"); put(f"ovHolm{nm}", int(L.holm(p, alpha).sum()), "{:.0f}")
        put(f"ovBonf{nm}", int((p <= alpha / len(p)).sum()), "{:.0f}"); put(f"ovBh{nm}", int(L.bh(p, alpha).sum()), "{:.0f}")
    put("ovPmin", p.min(), "{:.3f}"); put("ovHolmFirst", 0.10 / len(p), "{:.4f}")
    # figure 13: expected maximum against the number of trials
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.semilogx(NS, ex, color=BLUE, marker="o", ms=3, label="exact, by quadrature")
    ax.semilogx(NS, ap, color=RED, ls="--", label="approximation of Bailey and L\u00f3pez de Prado")
    ax.semilogx(NS, sim, ls="", marker="x", ms=4, color=GREY, label="simulation")
    ax.semilogx(NS, np.sqrt(2 * np.log(NS)), color=SKY, ls=":", label=r"upper bound $\sqrt{2\ln N}$")
    ax.set_xlabel("number of independent trials $N$"); ax.set_ylabel("expected maximum (standard errors)")
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "fig13_expected_max")
    # figure 14: in-sample against out-of-sample Sharpe ratio of the chosen rule, and the logits
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ax[0].scatter(a12 * pairs[:, 0], a12 * pairs[:, 1], s=2, color=SKY, alpha=0.3, lw=0)
    ax[0].axhline(0, color="#333333", lw=0.6)
    ax[0].set_xlabel("in-sample Sharpe ratio of the chosen rule"); ax[0].set_ylabel("its out-of-sample Sharpe ratio")
    ax[1].hist(logits, bins=40, color=PALE, edgecolor=BLUE, lw=0.4)
    ax[1].axvline(0, color=RED, lw=1.0)
    ax[1].set_xlabel("logit of the out-of-sample rank"); ax[1].set_ylabel("splits")
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig14_pbo")
    return finish("overfit")


def dsr_figure(s, T, vs, sk, ku, n_models):
    """The deflated Sharpe ratio of the best model against the assumed number of independent trials."""
    Ns = np.arange(2, 101); d = np.array([L.dsr(s, T, n, vs, sk, ku)[0] for n in Ns])
    below = Ns[d < 0.95]
    put("dsCross", below[0], "{:.0f}"); put("dsTwo", d[0], "{:.3f}"); put("dsHundred", d[-1], "{:.3f}")
    put("dsAtModels", d[Ns == n_models][0], "{:.3f}")
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.plot(Ns, d, color=BLUE)
    ax.axhline(0.95, color=RED, lw=0.8, ls="--"); ax.axvline(n_models, color=GREY, lw=0.8, ls=":")
    ax.text(98, 0.955, "0.95", fontsize=7.5, color=RED, ha="right", va="bottom")
    ax.plot([n_models], [d[Ns == n_models][0]], "o", color=BLUE, ms=4)
    ax.annotate(f"{n_models} models, {d[Ns == n_models][0]:.3f}", (n_models, d[Ns == n_models][0]), xytext=(6, 6),
                textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("assumed number of independent trials $N$"); ax.set_ylabel("deflated Sharpe ratio")
    ax.set_ylim(0, 1.02)
    save(fig, "fig20_dsr")


if __name__ == "__main__":
    import time; t = time.time(); main(); print(f"{time.time() - t:.1f}s")
