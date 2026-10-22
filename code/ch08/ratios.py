"""Section 8.6: performance ratios. The Sharpe ratio of the US market with its standard error, the
effect of serial correlation on annualising, the usual ratios for one industry against the market,
and three return laws with the same mean and volatility that the Sharpe ratio cannot tell apart.
Writes fig06_ratios.png and fig14_sharpe_se.png."""
import math
import numpy as np, pandas as pd
from scipy import stats
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish, rng_for


def main():
    f = L.load_monthly()
    x = f["mkt_rf"].values; rf = f["rf"].values; T = len(x)
    th = L.sharpe(x)
    put("prT", T, "{:.0f}")
    import datetime as _dt
    ym = lambda v: _dt.date(int(str(v)[:4]), int(str(v)[4:6]), 1).strftime("%B %Y")
    R.N["prFirst"] = ym(f.index[0]); R.N["prLast"] = ym(f.index[-1])
    put("prSR", math.sqrt(12) * th); put("prSRm", th, "{:.3f}")
    s, k = stats.skew(x), stats.kurtosis(x)
    put("prSkew", s); put("prKurt", k, "{:.1f}")
    se0 = L.sharpe_se(th, T); se1 = L.sharpe_se(th, T, s, k)
    put("prSeIid", math.sqrt(12) * se0, "{:.3f}"); put("prSeNn", math.sqrt(12) * se1, "{:.3f}")
    put("prSeNaive", math.sqrt(12) / math.sqrt(T), "{:.3f}")
    put("prEta", L.lo_annual_factor(x, 12), "{:.2f}")
    put("prSRlo", L.lo_annual_factor(x, 12) * th)
    # ten years of data: how wide is the interval?
    put("prSeTen", math.sqrt(12) * L.sharpe_se(th, 120, s, k), "{:.2f}")
    # an industry against the market: HiTec
    ind = pd.read_csv(L.ROOT / "data" / "industry10_monthly.csv", index_col=0)
    p = ind["HiTec"].values / 100 - rf
    b, a = np.polyfit(x, p, 1)
    e = p - a - b * x
    put("prHtSR", math.sqrt(12) * L.sharpe(p)); pct("prHtVol", math.sqrt(12) * p.std(ddof=1), 1)
    pct("prMktVol", math.sqrt(12) * x.std(ddof=1), 1)
    put("prHtBeta", b); pct("prHtTreynor", 12 * p.mean() / b, 2); pct("prMktTreynor", 12 * x.mean(), 2)
    pct("prHtAlpha", 12 * a, 2); pct("prHtTE", math.sqrt(12) * e.std(ddof=2), 1)
    put("prHtIR", math.sqrt(12) * a / e.std(ddof=2))
    m2 = 12 * p.mean() * x.std(ddof=1) / p.std(ddof=1)
    pct("prHtMsq", m2, 2); pct("prMktMean", 12 * x.mean(), 2)
    put("prHtSortino", math.sqrt(12) * L.sortino(p)); put("prMktSortino", math.sqrt(12) * L.sortino(x))
    put("prHtOmega", L.omega(p)); put("prMktOmega", L.omega(x))
    # three laws with the same mean and volatility (monthly), drawn in a large sample
    rng = rng_for(6); M = 1_000_000
    mu, sd = 0.005, 0.04
    E = rng.exponential(1.0, M); Z = rng.standard_normal(M)
    laws = {"normal": mu + sd * Z, "left": mu - sd * (E - 1), "right": mu + sd * (E - 1)}
    put("thMu", 12 * 100 * mu, "{:.0f}"); put("thVol", math.sqrt(12) * 100 * sd, "{:.1f}")
    put("thSR", math.sqrt(12) * mu / sd)
    rfm = np.zeros(M)
    res = {}
    for tg, y in laws.items():
        c = tg.capitalize()
        res[tg] = dict(sr=math.sqrt(12) * L.sharpe(y), so=math.sqrt(12) * L.sortino(y), om=L.omega(y),
                       mp=L.mppm(y, rfm, 3.0), var=L.var_hist(y, 0.99), es=L.es_hist(y, 0.99))
        put(f"th{c}SR", res[tg]["sr"]); put(f"th{c}So", res[tg]["so"]); put(f"th{c}Om", res[tg]["om"])
        pct(f"th{c}MpSample", res[tg]["mp"], 2); pct(f"th{c}Var", res[tg]["var"], 1); pct(f"th{c}Es", res[tg]["es"], 1)
    # the MPPM of each law by quadrature: the normal and left skewed laws put mass below -100%, where
    # (1+r)^(-2) is not integrable, so they are truncated at a monthly return of -50%; the right skewed law
    # is bounded below at mu - sd and is exact
    exact = {"normal": L.mppm_law(lambda r: stats.norm.pdf(r, mu, sd), -np.inf, np.inf),
             "left": L.mppm_law(lambda r: stats.expon.pdf((mu + sd - r) / sd) / sd, -np.inf, mu + sd),
             "right": L.mppm_law(lambda r: stats.expon.pdf((r - mu + sd) / sd) / sd, mu - sd, np.inf)}
    for tg, v in exact.items():
        pct(f"th{tg.capitalize()}Mp", v, 2); res[tg]["mp"] = v
    put("thFloor", 50, "{:.0f}")
    pl = math.exp(-(mu + sd + 1) / sd)                    # P(r < -100%) for the left skewed law
    e = math.floor(math.log10(pl)); R.RAW["thLeftTail"] = pl
    R.N["thLeftTail"] = f"{pl / 10 ** e:.1f}\\times10^{{{e}}}"
    # figure: densities and ratios relative to the normal
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8))
    grid = np.linspace(-0.15, 0.2, 400)
    ax[0].plot(100 * grid, stats.norm.pdf(grid, mu, sd) / 100, color=GREY, label="normal")
    gl = grid[grid <= mu + sd]; ax[0].plot(100 * gl, stats.expon.pdf((mu + sd - gl) / sd) / sd / 100, color=RED, label="left skewed")
    gr = grid[grid >= mu - sd]; ax[0].plot(100 * gr, stats.expon.pdf((gr - mu + sd) / sd) / sd / 100, color=BLUE, label="right skewed")
    ax[0].set_xlabel("monthly return, %"); ax[0].set_ylabel("density"); ax[0].legend(loc="center right")
    ax[0].set_xlim(-15, 20)
    names = ["Sharpe", "Sortino", "Omega", "MPPM"]; keys = ["sr", "so", "om", "mp"]
    w = 0.38; xs = np.arange(4)
    for j, (tg, c) in enumerate((("left", RED), ("right", BLUE))):
        rel = [res[tg][kk] / res["normal"][kk] for kk in keys]
        ax[1].bar(xs + (j - 0.5) * w, rel, w, color=c, label=f"{tg} skewed")
    ax[1].axhline(1, color=GREY, lw=0.8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(names); ax[1].set_ylabel("relative to the normal law")
    ax[1].legend(loc="upper left", ncol=2); ax[1].set_ylim(0, 1.9); ax[1].grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig06_ratios"); plt.close(fig)
    sharpe_se_figure(th, s, k, T)
    return finish("ratios")


def sharpe_se_figure(th, s, k, T):
    """Years of independent monthly returns needed before a two standard error interval for the
    Sharpe ratio excludes zero, against the true annualised ratio, with the market's skewness and
    excess kurtosis and with a more negatively skewed, fatter tailed law for comparison."""
    def years(th_m, sk, ku):
        c = 1 + 0.5 * th_m ** 2 - sk * th_m + 0.25 * ku * th_m ** 2
        return c / (3 * th_m * th_m)                       # solves sqrt(c / (12 Y)) = th_m / 2
    put("prYearsTwo", years(th, s, k), "{:.0f}")
    R.RAW["prYearsCheck"] = L.sharpe_se(th, 12 * years(th, s, k), s, k) / (th / 2)  # one, for the test
    SK, KU = -1.0, 6.0
    put("prAltSkew", SK, "{:.0f}"); put("prAltKurt", KU, "{:.0f}")
    put("prYearsOne", years(1 / math.sqrt(12), s, k), "{:.1f}")
    put("prYearsOneAlt", years(1 / math.sqrt(12), SK, KU), "{:.1f}")
    sr = np.linspace(0.2, 1.5, 300); tm = sr / math.sqrt(12)
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    ax.plot(sr, years(tm, s, k), color=BLUE, label="market skewness and kurtosis")
    ax.plot(sr, years(tm, SK, KU), color=RED, ls="--", label=f"skewness {SK:.0f}, excess kurtosis {KU:.0f}")
    ax.plot([math.sqrt(12) * th], [years(th, s, k)], "o", color=BLUE, ms=4)
    ax.set_yscale("log"); ax.set_yticks([1, 2, 5, 10, 20, 50, 100]); ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("true Sharpe ratio, annualised"); ax.set_ylabel("years of monthly data")
    ax.set_xlim(0.2, 1.5); ax.legend(loc="upper right")
    fig.tight_layout(); R.save(fig, "fig14_sharpe_se"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
