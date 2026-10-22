"""Section 8.4: peaks over threshold. A generalised Pareto law fitted to daily market losses above
their 95th percentile, VaR and ES far in the tail, the Hill estimator, and a check of the method on
simulated Student t losses with a known tail. Writes fig04_evt.png and fig13_threshold.png."""
import math
import numpy as np
from scipy import stats
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, sci, finish, rng_for


def pot(loss, qu=0.95):
    u = float(np.quantile(loss, qu))
    exc = loss[loss > u] - u
    xi, beta = L.gpd_fit(exc)
    return u, exc, xi, beta


def main():
    r, _ = L.load_daily()
    loss = -r.values
    n = len(loss)
    u, exc, xi, beta = pot(loss)
    nu = len(exc)
    pct("evU", u, 2); put("evNu", nu, "{:,.0f}"); put("evXi", xi, "{:.4f}"); pct("evBeta", beta, 3)
    put("evAlphaTail", 1 / xi, "{:.2f}")
    # standard error of xi from the observed information (GPD, xi > -1/2): var = (1+xi)^2 / nu
    put("evXiSe", (1 + xi) / math.sqrt(nu), "{:.4f}")
    se = (1 + xi) / math.sqrt(nu)                      # tail index at xi plus and minus two standard errors
    put("evAlphaLo", 1 / (xi + 2 * se), "{:.1f}"); put("evAlphaHi", 1 / (xi - 2 * se), "{:.1f}")
    for a, tg in ((0.99, "N"), (0.999, "T"), (0.9999, "F")):
        v, e = L.gpd_var_es(u, xi, beta, n, nu, a)
        pct(f"evVar{tg}", v, 2); pct(f"evEs{tg}", e, 2)
        pct(f"evHsVar{tg}", L.var_hist(r.values, a), 2)
        pct(f"evNorVar{tg}", L.var_normal(r.values.mean(), r.values.std(ddof=1), a), 2)
    # how often a loss as large as October 1987 occurs under the fitted tail
    big = loss.max()
    pexc = nu / n * (1 + xi * (big - u) / beta) ** (-1 / xi)
    put("evYears", 1 / (pexc * 252), "{:,.0f}")
    z = (big - loss.mean()) / loss.std(ddof=1)
    sci("evNormalYears", 1 / (stats.norm.sf(z) * 252), 0)
    # Hill estimator over k
    ks = np.arange(50, 1601, 10)
    H = np.array([L.hill(loss[loss > 0], int(k)) for k in ks])
    put("evHillTwoHundred", L.hill(loss[loss > 0], 200), "{:.1f}"); put("evHillEight", L.hill(loss[loss > 0], 800), "{:.1f}")
    # check on simulated t losses with 4 degrees of freedom: true xi = 1/4
    rng = rng_for(4); M = 400
    xs, vv = [], []
    s = math.sqrt(2 / 4)
    true = s * stats.t.ppf(0.999, 4)
    for _ in range(M):
        Ls = s * rng.standard_t(4, n)
        us, es_, xis, bs = pot(Ls)
        xs.append(xis); vv.append(L.gpd_var_es(us, xis, bs, n, len(es_), 0.999)[0])
    xs, vv = np.array(xs), np.array(vv)
    put("evSimXi", xs.mean(), "{:.3f}"); put("evSimXiSd", xs.std(), "{:.3f}"); put("evSimM", M, "{:.0f}")
    pct("evSimBias", vv.mean() / true - 1, 1); pct("evSimSd", vv.std() / true, 1)
    hs = []
    for _ in range(M):
        Ls = s * rng.standard_t(4, n); hs.append(np.quantile(Ls, 0.999))
    pct("evSimHsSd", np.std(hs) / true, 1)
    # figure: left, empirical tail vs GPD and normal on log scale; right, Hill plot
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8))
    Ls = np.sort(loss)[::-1]
    emp = np.arange(1, n + 1) / n
    m = Ls > u
    ax[0].loglog(100 * Ls[m], emp[m], ".", ms=2.2, color=GREY, label="data")
    xx = np.linspace(u, 0.2, 400)
    ax[0].loglog(100 * xx, nu / n * (1 + xi * (xx - u) / beta) ** (-1 / xi), color=RED, label="generalised Pareto")
    mu_, sd_ = r.values.mean(), r.values.std(ddof=1)
    ax[0].loglog(100 * xx, stats.norm.sf((xx + mu_) / sd_), color=BLUE, label="normal")
    ax[0].set_ylim(3e-5, 0.06); ax[0].set_xlabel("daily loss, % (log scale)"); ax[0].set_ylabel("probability of a larger loss")
    ax[0].legend(loc="upper right", framealpha=1)
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    tk = [1.5, 2, 3, 5, 10, 20]
    ax[0].xaxis.set_major_locator(FixedLocator(tk)); ax[0].xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in tk]))
    ax[0].xaxis.set_minor_locator(NullLocator())
    ax[1].plot(ks, H, color=BLUE); ax[1].axhline(1 / xi, color=RED, lw=0.8, ls="--")
    ax[1].set_xlabel("number of largest losses used, $k$"); ax[1].set_ylabel("Hill estimate of $1/\\xi$")
    fig.tight_layout(); R.save(fig, "fig04_evt"); plt.close(fig)
    threshold_figure(loss, u, xi, beta)
    return finish("evt")


def threshold_figure(loss, u0, xi0, beta0):
    """Left: the generalised Pareto shape fitted above each threshold from the 90th to the 99th
    percentile, with plus and minus two standard errors at each threshold taken alone (pointwise,
    not a band for the whole curve). Right: the mean excess over a threshold, which is linear in the
    threshold for a generalised Pareto tail, with the line implied by the fit at the 95th percentile."""
    qs = np.round(np.arange(0.90, 0.99001, 0.0025), 4)
    fits = []
    for q in qs:
        uq, exc, xq, bq = pot(loss, q)
        fits.append((uq, len(exc), xq, (1 + xq) / math.sqrt(len(exc))))
    fits = np.array(fits)
    i, j = int(np.argmin(fits[:, 2])), int(np.argmax(fits[:, 2]))
    put("evXiMin", fits[i, 2], "{:.2f}")
    put("evXiMax", fits[j, 2], "{:.2f}"); pct("evXiMaxQ", qs[j], 0)
    put("evXiSeTop", fits[-1, 3], "{:.2f}"); put("evNuTop", fits[-1, 1], "{:.0f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ax[0].fill_between(100 * qs, fits[:, 2] - 2 * fits[:, 3], fits[:, 2] + 2 * fits[:, 3], color="#D9DEE8", lw=0)
    ax[0].plot(100 * qs, fits[:, 2], color=BLUE)
    ax[0].axvline(95, color="#999999", lw=0.8, ls=":")
    ax[0].set_xlabel("threshold, percentile of daily losses"); ax[0].set_ylabel("fitted shape $\\xi$")
    ax[0].set_xlim(90, 99)
    us = np.quantile(loss, np.linspace(0.80, 0.995, 200))
    me = np.array([np.mean(loss[loss > v] - v) for v in us])
    ax[1].plot(100 * us, 100 * me, color=BLUE, label="data")
    ax[1].plot(100 * us, 100 * (beta0 + xi0 * (us - u0)) / (1 - xi0), color=RED, lw=0.9, ls="--",
               label="fit at the 95th percentile")
    ax[1].axvline(100 * u0, color="#999999", lw=0.8, ls=":")
    ax[1].set_xlabel("threshold, daily loss in %"); ax[1].set_ylabel("mean excess, %")
    ax[1].legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig13_threshold"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
