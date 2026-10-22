"""Section 9.2: what a trend rule earns in theory. The sign rule under joint normality checked market by
market, the straddle identity for a linear trend rule on simulated paths, and a permutation test of
the trend program: shuffling the months destroys any serial dependence, so by the martingale argument the
shuffled programs earn nothing on average, and a descriptive picture of next month's return by decile of
the past year's. Writes fig03_straddle.png and fig17_deciles.png."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, PALE, put, pct, finish
from trend import START

NPATH, NDAY, SD = 4000, 252, 0.01
NPERM = 400


def main():
    r, rf = L.load_markets()
    # sign rule: X = past 12 month return, Y = next month return
    pred, real, rhos, drift, timing = [], [], [], [], []
    for k in L.MARKETS:
        x = r[k]; past = np.expm1(np.log1p(x).rolling(12).sum()).shift(1)
        ok = past.notna() & x.notna() & (x.index >= START)
        X, Y = past[ok].values, x[ok].values
        rho = np.corrcoef(X, Y)[0, 1]; rhos.append(rho)
        pred.append(L.sign_rule_mean(rho, Y.std())); real.append(np.mean(np.sign(X) * Y))
        # sample identity: mean(sY) = mean(s) mean(Y) + cov(s, Y), with cov dividing by n
        sg = np.sign(X); drift.append(sg.mean() * Y.mean() / Y.std()); timing.append(np.mean((sg - sg.mean()) * (Y - Y.mean())) / Y.std())
        if k in ("EQ", "JPY", "OIL"):
            put(f"srRho{k.title()}", rho, "{:.3f}")
    put("srRhoAvg", np.mean(rhos), "{:.3f}")
    put("srDriftAvg", math.sqrt(12) * np.mean(drift)); put("srTimingAvg", math.sqrt(12) * np.mean(timing))
    k = L.MARKETS.index("EQ"); put("srDriftEq", math.sqrt(12) * drift[k]); put("srTimingEq", math.sqrt(12) * timing[k])
    # ratio of realised to predicted annual Sharpe of a unit-vol sign rule, averaged
    put("srPredSRavg", math.sqrt(12) * math.sqrt(2 / math.pi) * np.mean(rhos))
    put("srRealSRavg", np.mean([math.sqrt(12) * m / r[k][r.index >= START].std() for m, k in zip(real, L.MARKETS)]))
    decile_figure(r)
    # straddle identity on simulated random walks
    rng = R.rng_for(1)
    x = SD * rng.standard_normal((NPATH, NDAY))
    S = np.cumsum(x, axis=1); Sprev = np.concatenate([np.zeros((NPATH, 1)), S[:, :-1]], axis=1)
    pnl = np.sum(Sprev * x, axis=1)
    ident = 0.5 * (S[:, -1] ** 2 - np.sum(x * x, axis=1))
    R.sci("stMaxErr", np.max(np.abs(pnl - ident)) / np.max(np.abs(pnl)), 0)
    pct("stLossShare", np.mean(pnl < 0), 1); pct("stLossTheory", math.erf(1 / math.sqrt(2)), 1)
    put("stMeanPnl", pnl.mean() / (NDAY * SD * SD), "{:.3f}")
    put("stNpath", NPATH, "{:,.0f}"); put("stNday", NDAY, "{:.0f}")
    put("stSkew", float(((pnl - pnl.mean()) ** 3).mean() / pnl.std() ** 3), "{:.1f}")
    # the exact loss probability over NDAY days: with mean-zero normal returns the rule loses exactly
    # when |t| < 1 for the t statistic of the mean, so it is P(|t_{n-1}| < 1); and the standard error
    # of the share in the paths above
    from scipy import stats
    pct("stLossFinite", 1 - 2 * stats.t.sf(1, NDAY - 1), 2)
    R.RAW["stLossTEquiv"] = float(np.mean((S[:, -1] ** 2 < np.sum(x * x, axis=1))
                                          == (np.abs(x.mean(1) / (x.std(1, ddof=1) / math.sqrt(NDAY))) < 1)))
    q = np.mean(pnl < 0); se = 100 * math.sqrt(q * (1 - q) / NPATH); put("stLossSe", se, "{:.2f}")
    put("stLossZ", (R.RAW["stLossFinite"] - 100 * q) / se, "{:.1f}")   # standard errors, from unrounded values
    # permutation test of the program
    from trend import program
    actual = L.sharpe(program(r)[2])
    rng2 = R.rng_for(2); sh = []; sh0 = []
    live = np.flatnonzero(r.notna().all(axis=1).values)   # months with all seven returns
    put("pmLive", len(live), "{:.0f}"); put("pmHeld", len(r) - len(live), "{:.0f}")
    from trend import label as month_label
    held = np.setdiff1d(np.arange(len(r)), live)
    assert (np.diff(held) == 1).all()                    # the held months are one block at the start
    R.N["pmHeldFirst"] = month_label(r.index[held[0]]); R.N["pmHeldLast"] = month_label(r.index[held[-1]])
    for _ in range(NPERM):
        perm = np.arange(len(r)); perm[live] = rng2.permutation(live)   # earlier months stay put
        rp = r.iloc[perm].copy(); rp.index = r.index
        sh.append(L.sharpe(program(rp)[2])); sh0.append(L.sharpe(program(rp, cost=0.0)[2]))
    sh = np.array(sh)
    put("pmN", NPERM, "{:.0f}"); put("pmMean", sh.mean()); put("pmSd", sh.std(ddof=1))
    put("pmCount", np.sum(sh >= actual), "{:.0f}"); put("pmMeanGross", np.mean(sh0))
    put("pmQ", np.quantile(sh, 0.95)); put("pmP", (1 + np.sum(sh >= actual)) / (NPERM + 1), "{:.3f}")
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    k = slice(0, 1500)
    ax[0].scatter(100 * S[k, -1], 1e4 * pnl[k], s=3, color=SKY, alpha=0.6)
    zz = np.linspace(S[:, -1].min(), S[:, -1].max(), 200)
    ax[0].plot(100 * zz, 1e4 * 0.5 * (zz * zz - NDAY * SD * SD), color=RED)
    ax[0].set_xlabel("total move over the year, %"); ax[0].set_ylabel(r"profit, units of $10^{-4}$")
    ax[1].hist(sh, bins=30, color=PALE, edgecolor=BLUE, lw=0.5)
    ax[1].axvline(actual, color=RED, lw=1.4); ax[1].set_xlabel("Sharpe ratio of the programme, before fees")
    ax[1].set_ylabel("shuffled samples"); ax[1].grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig03_straddle"); plt.close(fig)
    return finish("theory")


def decile_figure(r):
    """Next month's return against deciles of the past year's return, pooled over the seven markets, each
    series divided by its own standard deviation. Descriptive: the months overlap in the signal and the
    markets are correlated, so the decile means are not independent."""
    xs, ys = [], []
    for k in L.MARKETS:
        x = r[k]; past = np.expm1(np.log1p(x).rolling(12).sum()).shift(1)
        ok = past.notna() & x.notna() & (x.index >= START)
        X, Y = past[ok].values, x[ok].values
        xs.append(X / X.std()); ys.append(Y / Y.std())
    X, Y = np.concatenate(xs), np.concatenate(ys)
    q = np.quantile(X, np.linspace(0, 1, 11)); d = np.clip(np.searchsorted(q, X, side="right") - 1, 0, 9)
    mx = np.array([X[d == i].mean() for i in range(10)]); my = np.array([Y[d == i].mean() for i in range(10)])
    b = np.polyfit(X, Y, 1); rho = np.corrcoef(X, Y)[0, 1]
    put("dqObs", len(X), "{:,.0f}"); put("dqRho", rho, "{:.3f}")
    put("dqBottom", my[0]); put("dqTop", my[9]); put("dqMid", Y[(d > 0) & (d < 9)].mean())
    g = np.sign(X) * Y; tails = (d == 0) | (d == 9)
    pct("dqTailShare", g[tails].sum() / g.sum(), 0); put("dqTailShareAbs", -R.RAW["dqTailShare"], "{:.0f}")
    R.RAW["dqMeans"] = list(map(float, my))
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ax.bar(mx, my, width=0.18, color=[RED if v < 0 else BLUE for v in my])
    xx = np.linspace(mx[0], mx[-1], 50); ax.plot(xx, b[1] + b[0] * xx, color=GREY, ls="--", lw=1.0,
                                                 label="least-squares line")
    ax.axhline(0, color=GREY, lw=0.6); ax.legend(loc="upper left")
    ax.set_xlabel("past 12 month return, standard deviations (decile mean)")
    ax.set_ylabel("next month's return, s.d.")
    fig.tight_layout(); R.save(fig, "fig17_deciles"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
