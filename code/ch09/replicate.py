"""Section 9.5: replicating a trend manager. The target is a hypothetical manager who blends six and
twelve month trend rules on the same seven markets and adds returns of its own; clones are rolling
24 month regressions on the market returns themselves and on three trend factors. Checks
the prediction error formula by simulation. Writes fig07_clone.png and fig18_clonecoef.png."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish
from trend import program

WIN = 24
IDIO = 0.03  # a year, the manager's own returns


def oos_r2(y, yhat):
    e = y - yhat
    return 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))


def main():
    r, rf = L.load_markets()
    rng = R.rng_for(4)
    p6, p12 = program(r, 6)[2], program(r, 12)[2]
    f1, f3 = program(r, 1)[2], program(r, 3)[2]
    target = (0.5 * p6 + 0.5 * p12).values
    target = target + IDIO / math.sqrt(12) * rng.standard_normal(len(target))
    X1 = r.loc[p12.index].fillna(0.0).values
    X2 = np.column_stack([f1.values, f3.values, p12.values])
    c1, r21 = L.rolling_clone(target, X1, WIN)
    c2, r22 = L.rolling_clone(target, X2, WIN)
    ok = ~np.isnan(c1)
    y = target[ok]; a = c1[ok]; b = c2[ok]
    put("rpWin", WIN, "{:.0f}"); pct("rpIdio", IDIO, 0); put("rpMonths", ok.sum(), "{:.0f}")
    put("rpSRt", L.sharpe(y)); put("rpSRa", L.sharpe(a)); put("rpSRb", L.sharpe(b))
    put("rpCorrA", np.corrcoef(y, a)[0, 1]); put("rpCorrB", np.corrcoef(y, b)[0, 1])
    pct("rpTeA", math.sqrt(12) * (y - a).std(), 1); pct("rpTeB", math.sqrt(12) * (y - b).std(), 1)
    put("rpInA", r21.mean()); put("rpInB", r22.mean())
    put("rpOutA", oos_r2(y, a)); put("rpOutB", oos_r2(y, b))
    pct("rpMeanGapA", 12 * (y.mean() - a.mean()), 1); pct("rpMeanGapB", 12 * (y.mean() - b.mean()), 1)
    # full sample linear fit on markets: how much of the target is linear in the markets
    Xf = np.column_stack([np.ones(len(target)), X1])
    e = target - Xf @ np.linalg.lstsq(Xf, target, rcond=None)[0]
    put("rpFullA", 1 - e.var() / target.var())
    # the prediction error formula by simulation
    k, T, n = 7, WIN, 20000
    errs = np.empty(n)
    for i in range(n):
        Xs = rng.standard_normal((T + 1, k)); beta = np.ones(k) * 0.3
        ys = Xs @ beta + rng.standard_normal(T + 1)
        bh = np.linalg.lstsq(Xs[:T], ys[:T], rcond=None)[0]
        errs[i] = (ys[T] - Xs[T] @ bh) ** 2
    put("pvK", k, "{:.0f}"); put("pvT", T, "{:.0f}"); put("pvSims", n, "{:,.0f}")
    put("pvTheory", L.prediction_inflation(k, T), "{:.3f}"); put("pvSim", errs.mean(), "{:.3f}")
    put("pvSimSe", errs.std(ddof=1) / math.sqrt(n), "{:.3f}")
    put("pvInSample", (T - k) / T, "{:.3f}")
    # figure
    t = np.arange(len(target))[ok] / 12 + 2001
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.plot(t, np.cumprod(1 + y), color=BLUE, label="manager")
    ax.plot(t, np.cumprod(1 + b), color=RED, label="clone on trend factors")
    ax.plot(t, np.cumprod(1 + a), color=GREY, lw=0.9, label="clone on market returns")
    ax.set_yscale("log"); L.plain_log_axis(ax); ax.set_ylabel("growth of 1, excess of bills"); ax.legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig07_clone"); plt.close(fig)
    coef_figure(target, X1, X2, ok)
    return finish("replicate")


def coefs(y, X, window=WIN):
    """Coefficients of each refit of L.rolling_clone: the regression on the `window` months before t."""
    B = np.full((len(y), X.shape[1]), np.nan)
    for t in range(window, len(y)):
        B[t] = np.linalg.lstsq(X[t - window:t], y[t - window:t], rcond=None)[0]
    return B


def coef_figure(target, X1, X2, ok):
    B1, B2 = coefs(target, X1)[ok], coefs(target, X2)[ok]
    # check against the clone's own forecasts
    R.RAW["cfFitErr"] = float(np.max(np.abs(np.sum(B1 * X1[ok], axis=1) - L.rolling_clone(target, X1, WIN)[0][ok])))
    ch1 = np.abs(np.diff(B1, axis=0)).mean(); ch2 = np.abs(np.diff(B2, axis=0)).mean()
    put("cfChangeA", ch1); put("cfChangeB", ch2); put("cfRatio", ch1 / ch2, "{:.1f}")
    flips = (np.diff(np.sign(B1), axis=0) != 0).sum(axis=0)
    put("cfFlipsMed", np.median(flips), "{:.0f}"); put("cfMonths", ok.sum(), "{:.0f}")
    put("cfRangeA", B1.max() - B1.min(), "{:.1f}")
    put("cfTwelveMin", B2[:, 2].min()); put("cfTwelveMax", B2[:, 2].max())
    t = np.arange(len(target))[ok] / 12 + 2001
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
    sty = [(BLUE, "-"), (RED, "-"), (GREY, "-"), (SKY, "-"), (ROSE, "-"), (BLUE, "--"), (RED, "--")]
    for k, nm in enumerate(L.MARKETS):
        ax[0].plot(t, B1[:, k], color=sty[k][0], ls=sty[k][1], lw=0.7, label=L.NAMES[nm])
    for k, (nm, col) in enumerate(zip(["one month", "three months", "twelve months"], [GREY, SKY, BLUE])):
        ax[1].plot(t, B2[:, k], color=col, lw=0.9, label=f"trend factor, {nm}")
    for a in ax:
        a.axhline(0, color=GREY, lw=0.5)
    ax[0].set_ylabel("coefficient"); ax[0].set_title("clone on market returns", fontsize=8)
    ax[1].set_title("clone on trend factors", fontsize=8)
    ax[0].set_ylim(-4, 3.5)                           # the 2020 spike near -3.5 clears the frame
    for a, fs, nc in ((ax[0], 5.5, 2), (ax[1], 6, 1)):
        a.legend(loc="lower left", fontsize=fs, ncol=nc, frameon=True, framealpha=1, edgecolor="none").set_zorder(5)
    fig.tight_layout(); R.save(fig, "fig18_clonecoef"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
