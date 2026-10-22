"""Section 8.3: GARCH(1,1) with normal and Student t innovations fitted to daily US market returns,
the EWMA alternative, and forecasts of volatility over the next year from four dates.
Writes fig02_garch.png, fig03_forecast.png, fig12_clusters.png and out/garch_params.json."""
import json, math
import numpy as np
from scipy import stats
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish

DATES = ["1987-10-20", "2008-11-20", "2020-03-16", "2017-07-14"]


def main():
    r, _ = L.load_daily()
    x = 100 * r.values  # percent
    fits = {d: L.garch_fit(x, d) for d in ("normal", "t")}
    (R.HERE / "out" / "garch_params.json").write_text(json.dumps({k: {a: float(b) for a, b in v.items()}
                                                                   for k, v in fits.items()}, indent=1))
    for tag, d in (("N", "normal"), ("T", "t")):
        g = fits[d]
        put(f"gaOmega{tag}", g["omega"], "{:.4f}"); put(f"gaAlpha{tag}", g["alpha"], "{:.3f}")
        put(f"gaBeta{tag}", g["beta"], "{:.3f}"); put(f"gaPers{tag}", g["persistence"], "{:.4f}")
        put(f"gaLrVol{tag}", math.sqrt(252 * g["lrvar"]), "{:.1f}")
        put(f"gaHalf{tag}", math.log(0.5) / math.log(g["persistence"]), "{:.0f}")
        put(f"gaLogLik{tag}", -g["nll"], "{:,.1f}")
    put("gaNu", fits["t"]["nu"], "{:.1f}"); put("gaNuFine", fits["t"]["nu"], "{:.3f}")
    put("gaLR", 2 * (fits["normal"]["nll"] - fits["t"]["nll"]), "{:,.0f}")
    gn = fits["normal"]
    put("gaFourth", gn["alpha"] ** 2 * 2 + (gn["alpha"] + gn["beta"]) ** 2, "{:.3f}")
    # standardised residuals: kurtosis left after GARCH
    hn = L.garch_filter(x - gn["mu"], gn["omega"], gn["alpha"], gn["beta"])
    zr = (x - gn["mu"]) / np.sqrt(hn[:-1])
    put("gaResKurt", stats.kurtosis(zr), "{:.1f}"); put("gaResSkew", stats.skew(zr))
    put("gaResAcSq", L.ar1(zr * zr), "{:.2f}")
    pct("gaResVarHs", -np.quantile(zr, 0.01) / 100, 2)  # 99% quantile in sd units, stored as number
    R.RAW["gaResQ"] = float(-np.quantile(zr, 0.01)); R.N["gaResQ"] = f"{-np.quantile(zr, 0.01):.2f}"
    del R.RAW["gaResVarHs"], R.N["gaResVarHs"]
    # EWMA
    he = L.ewma(x, 0.94)
    vol_g = np.sqrt(252 * hn[:-1]); vol_e = np.sqrt(252 * he[:-1])
    put("ewCorr", np.corrcoef(vol_g, vol_e)[0, 1], "{:.3f}")
    put("ewLLdiff", 0.5 * np.sum(np.log(he[1:-1]) + x[1:] ** 2 / he[1:-1]) -
        0.5 * np.sum(np.log(hn[1:-1]) + (x[1:] - gn["mu"]) ** 2 / hn[1:-1]), "{:,.0f}")
    put("volMax", vol_g.max(), "{:.0f}"); R.N["volMaxDate"] = f"{r.index[int(np.argmax(vol_g))]:%-d %B %Y}"
    put("volMin", vol_g.min(), "{:.1f}")
    put("volMedian", np.median(vol_g), "{:.1f}")
    # forecasts from four dates: term structure of annualised average volatility over k days
    k = np.arange(1, 253)
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    cols = [RED, BLUE, ROSE, GREY]
    idx = {d: int(np.searchsorted(r.index.values, np.datetime64(d))) for d in DATES}
    tags = ["Ka", "Kb", "Kc", "Kd"]
    for (d, i), c, tg in zip(idx.items(), cols, tags):
        h1 = hn[i]  # forecast for date d made at the close of the previous day
        path = L.garch_forecast(h1, gn["lrvar"], gn["persistence"], k)
        avg = np.sqrt(252 * np.cumsum(path) / k)
        lab = f"{r.index[i]:%-d %b %Y}"
        ax.plot(k, avg, color=c, label=lab)
        put(f"fc{tg}Now", math.sqrt(252 * h1), "{:.1f}"); put(f"fc{tg}Year", avg[-1], "{:.0f}")
        R.N[f"fc{tg}Date"] = f"{r.index[i]:%-d %B %Y}"
        # realised volatility over the next 252 days
        put(f"fc{tg}Real", math.sqrt(252) * np.std(x[i:i + 252], ddof=0), "{:.0f}")
    ax.axhline(math.sqrt(252 * gn["lrvar"]), color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("horizon, trading days"); ax.set_ylabel("forecast volatility, % a year")
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 0.2)); ax.set_xlim(0, 252); ax.set_ylim(0, 100)
    fig.tight_layout(); R.save(fig, "fig03_forecast"); plt.close(fig)
    # figure: conditional volatility and absolute returns
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    t = r.index
    ax.plot(t, math.sqrt(252) * np.abs(x - gn["mu"]), color="#C9CED8", lw=0.4, label="$\\sqrt{252}\\,|r_t-\\mu|$")
    ax.plot(t, vol_g, color=BLUE, lw=0.8, label="GARCH(1,1) volatility")
    ax.set_ylim(0, 120); ax.set_ylabel("% a year"); ax.legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig02_garch"); plt.close(fig)
    cluster_figure(x, zr, fits["t"]["nu"])
    return finish("garch")


def acf(y, K):
    y = np.asarray(y) - np.mean(y)
    return np.array([np.sum(y[j:] * y[:-j]) / np.sum(y * y) for j in range(1, K + 1)])


def cluster_figure(x, zr, nu):
    """Left: autocorrelations of returns, squared and absolute returns at lags 1 to 100 with the
    band +-2/sqrt(n). Right: normal GARCH residuals against standard normal quantiles and against
    the quantiles of a Student t law rescaled to unit variance, the law the residuals are meant to have."""
    n = len(x); K = 100; band = 2 / math.sqrt(n)
    acs = {nm: acf(y, K) for nm, y in (("r", x), ("sq", x * x), ("abs", np.abs(x)))}
    put("acLags", K, "{:.0f}")
    put("acSqAbove", int(np.sum(acs["sq"] > band)), "{:.0f}"); put("acAbsAbove", int(np.sum(acs["abs"] > band)), "{:.0f}")
    put("acRetOut", int(np.sum(np.abs(acs["r"]) > band)), "{:.0f}")
    put("acSqHundred", acs["sq"][-1], "{:.3f}"); put("acAbsHundred", acs["abs"][-1], "{:.2f}")
    sc = math.sqrt((nu - 2) / nu)                          # t(nu) times sc has variance one
    put("gaResQThou", np.quantile(zr, 0.001)); put("gaTQThou", sc * stats.t.ppf(0.001, nu))
    put("gaTQ", sc * stats.t.ppf(0.01, nu)); put("gaTQRaw", stats.t.ppf(0.01, nu))
    put("gaNQThou", stats.norm.ppf(0.001))
    R.RAW["gaResQTop"] = float(np.quantile(zr, 0.999)); R.RAW["gaTQTop"] = sc * stats.t.ppf(0.999, nu)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    lag = np.arange(1, K + 1)
    ax[0].axhspan(-band, band, color="#E8E3E3", lw=0)
    ax[0].plot(lag, acs["abs"], color=RED, lw=1, label="absolute returns")
    ax[0].plot(lag, acs["sq"], color=BLUE, lw=1, label="squared returns")
    ax[0].plot(lag, acs["r"], color=GREY, lw=0.8, label="returns")
    ax[0].set_xlabel("lag, trading days"); ax[0].set_ylabel("autocorrelation"); ax[0].set_xlim(0, K)
    ax[0].set_ylim(-0.06, 0.42); ax[0].legend(loc="upper right")
    q = np.sort(zr); pp = (np.arange(1, n + 1) - 0.5) / n
    ax[1].plot(stats.norm.ppf(pp), q, ".", ms=1.5, color=GREY, label="against normal")
    ax[1].plot(sc * stats.t.ppf(pp, nu), q, ".", ms=1.5, color=BLUE, label=f"against unit variance $t$, {nu:.1f} df")
    lim = [-7, 6]; ax[1].plot(lim, lim, color=RED, lw=1)
    ax[1].set_xlim(lim); ax[1].set_ylim(np.floor(q[0]) - 0.5, 7)
    ax[1].set_xlabel("reference quantile"); ax[1].set_ylabel("residual quantile")
    ax[1].legend(loc="lower right", markerscale=5)
    fig.tight_layout(); R.save(fig, "fig12_clusters"); plt.close(fig)


if __name__ == "__main__":
    out = main(); print(out["N"])
