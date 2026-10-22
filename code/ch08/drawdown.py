"""Section 8.7: drawdowns of the US market since 1963, the stationary law of a drawdown for a
random walk with drift, and the spread of the worst drawdown over ten years.
Writes fig07_drawdown.png and fig15_ten_year.png."""
import math
import numpy as np
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish, rng_for


def episodes(dd, idx, top=3):
    """The deepest drawdown episodes: (peak date, trough date, recovery date or None, depth)."""
    out = []; i = 0; n = len(dd)
    while i < n:
        if dd[i] > 0:
            j = i
            while j < n and dd[j] > 0:
                j += 1
            t = i + int(np.argmax(dd[i:j]))
            out.append((idx[i - 1] if i > 0 else idx[0], idx[t], idx[j] if j < n else None, dd[t], t - i + 1, j - i))
            i = j
        else:
            i += 1
    return sorted(out, key=lambda e: -e[3])[:top]


def main():
    r, rf = L.load_daily()
    x = r.values
    W, P, dd = L.drawdowns(x)
    mdd = dd.max(); pct("ddMax", mdd, 0)
    ep = episodes(dd, r.index)
    fmt = "%-d %B %Y"
    pk, tr, rc, d, ndown, nall = ep[0]
    R.N["ddPeak"] = f"{pk:{fmt}}"; R.N["ddTrough"] = f"{tr:{fmt}}"; R.N["ddRecover"] = f"{rc:{fmt}}"
    put("ddDownYears", ndown / 252, "{:.1f}"); put("ddAllYears", nall / 252, "{:.1f}")
    pct("ddSecond", ep[1][3], 0); R.N["ddSecondPeak"] = f"{ep[1][0]:%B %Y}"; R.N["ddSecondTrough"] = f"{ep[1][1]:%B %Y}"
    pct("ddThird", ep[2][3], 0); R.N["ddThirdPeak"] = f"{ep[2][0]:%B %Y}"; R.N["ddThirdTrough"] = f"{ep[2][1]:%B %Y}"
    pct("ddShareUnder", np.mean(dd > 0), 0); pct("ddShareTen", np.mean(dd > 0.10), 0)
    pct("ddShareTwenty", np.mean(dd > 0.20), 0)
    years = len(x) / 252
    cagr = W[-1] ** (1 / years) - 1
    pct("ddCagr", cagr, 1); put("ddCalmar", cagr / mdd)
    # stationary law for log drawdown y = ln(peak / W): exponential with mean s^2 / (2 m)
    lx = np.log1p(x); m, s2 = lx.mean(), lx.var(ddof=1)
    y = -np.log1p(-dd)
    meanlaw = s2 / (2 * m)
    put("ddLawMean", 100 * meanlaw, "{:.1f}"); put("ddEmpMean", 100 * y.mean(), "{:.1f}")
    pct("ddLawTwenty", math.exp(-(-math.log(0.8)) / meanlaw), 1)
    pct("ddLogMu", 252 * m, 2); pct("ddLogVol", math.sqrt(252 * s2), 2)
    # the band of log drawdowns, 40% to 55%, where the data lie above the law, as simple drawdowns
    pct("ddSimpleLo", 1 - math.exp(-0.40), 0); pct("ddSimpleHi", 1 - math.exp(-0.55), 0)
    # simulated worst drawdown over ten years of daily returns with the market's log mean and volatility
    rng = rng_for(7); M = 4000; n10 = 2520
    sims = rng.normal(m, math.sqrt(s2), size=(M, n10))
    cum = np.cumsum(sims, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((M, 1)), cum], axis=1), axis=1)[:, 1:]
    mdds = 1 - np.exp(-(peak - cum).max(axis=1))
    pct("ddSimMedian", np.median(mdds), 0); pct("ddSimNinety", np.quantile(mdds, 0.9), 0)
    pct("ddSimThirty", np.mean(mdds > 0.30), 0); pct("ddSimFifty", np.mean(mdds > 0.5), 0)
    put("ddSimM", M, "{:,.0f}")
    # empirical ten-year windows of the actual series (non-overlapping start each year)
    w10 = [L.max_drawdown(x[i:i + n10]) for i in range(0, len(x) - n10 + 1, 252)]
    pct("ddWinMedian", np.median(w10), 0); put("ddWinN", len(w10), "{:.0f}")
    pct("ddWinMin", np.min(w10), 0); pct("ddWinMax", np.max(w10), 0)
    # figure: underwater chart and the tail of the log drawdown vs the exponential law
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax[0].fill_between(r.index, -100 * dd, 0, color=BLUE, lw=0)
    ax[0].set_ylabel("drawdown, %"); ax[0].set_ylim(-60, 0)
    yy = np.linspace(0, 0.8, 200)
    emp = [np.mean(y > v) for v in yy]
    ax[1].semilogy(100 * yy, emp, color=BLUE, label="data")
    ax[1].semilogy(100 * yy, np.exp(-yy / meanlaw), color=RED, label="exponential law")
    ax[1].set_xlabel("log drawdown, %"); ax[1].set_ylabel("share of days deeper")
    ax[1].set_ylim(1e-3, 1.05); ax[1].legend(loc="lower left")
    fig.tight_layout(); R.save(fig, "fig07_drawdown"); plt.close(fig)
    ten_year_figure(mdds, np.array(w10))
    return finish("drawdown")


def ten_year_figure(mdds, w10):
    """Histogram of the maximum drawdown over ten years of simulated normal log returns, with the
    maximum drawdowns of the actual ten year windows (one start a year, overlapping) marked below it."""
    q90 = np.quantile(mdds, 0.9)
    put("ddWinAboveNinety", int(np.sum(w10 > q90)), "{:.0f}")
    pct("ddSimMax", mdds.max(), 0)
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    ax.hist(100 * mdds, bins=np.arange(5, 81, 2.5), color="#C9CED8", edgecolor="white", lw=0.4,
            density=True, label="simulated random walk")
    ax.plot(100 * w10, np.full(len(w10), -0.002), "|", ms=9, color=RED, mew=1.0, label="actual windows")
    ax.axvline(100 * np.median(mdds), color=BLUE, lw=0.9, label="simulated median")
    ax.axvline(100 * np.median(w10), color=RED, lw=0.9, ls="--", label="actual median")
    ax.axvline(100 * q90, color=BLUE, lw=0.9, ls=":", label="simulated 90th percentile")
    ax.set_xlabel("maximum drawdown over ten years, %"); ax.set_ylabel("density")
    ax.set_xlim(5, 80); ax.set_ylim(-0.005, None); ax.legend(loc="upper right")
    fig.tight_layout(); R.save(fig, "fig15_ten_year"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
