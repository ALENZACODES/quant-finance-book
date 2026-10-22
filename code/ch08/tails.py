"""Section 8.1 and 8.2: the shape of daily US market returns, VaR and ES by four methods, the
failure of subadditivity for VaR, and the square root of time.
Writes fig01_tails.png, fig10_loans.png, fig11_cornish.png and the numbers registered below."""
import math
import numpy as np
from scipy import stats
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, put, pct, sci, finish


def main():
    r, rf = L.load_daily()
    x = r.values
    n = len(x)
    put("tdN", n, "{:,.0f}")
    R.N["tdFirst"] = "1 July 1963"; R.N["tdLast"] = "31 August 2026"
    mu, sd, s, k = L.moments(x)
    pct("tdMeanAnn", 252 * mu, 1); pct("tdVolAnn", math.sqrt(252) * sd, 1)
    pct("tdSd", sd, 2); put("tdSkew", s); put("tdKurt", k, "{:.1f}")
    i = int(np.argmin(x)); pct("tdWorst", -x[i], 1); R.N["tdWorstDate"] = f"{r.index[i]:%-d %B %Y}"
    put("tdWorstSd", -(x[i] - mu) / sd, "{:.0f}")
    j = int(np.argmax(x)); pct("tdBest", x[j], 1); R.N["tdBestDate"] = f"{r.index[j]:%-d %B %Y}"
    z = (x - mu) / sd
    beyond = int(np.sum(np.abs(z) > 5)); put("tdBeyondFive", beyond, "{:.0f}")
    put("tdBeyondFiveNormal", n * 2 * stats.norm.sf(5), "{:.3f}")
    beyond4 = int(np.sum(np.abs(z) > 4)); put("tdBeyondFour", beyond4, "{:.0f}")
    put("tdBeyondFourNormal", n * 2 * stats.norm.sf(4), "{:.1f}")
    # Student t fitted by maximum likelihood (location and scale free)
    nu, loc, scale = stats.t.fit(x)
    put("tdNu", nu, "{:.1f}")
    # VaR and ES over the whole sample, four methods
    rows = []
    for a, tag in ((0.99, "N"), (0.975, "S")):
        vals = dict(
            nor=(L.var_normal(mu, sd, a), L.es_normal(mu, sd, a)),
            tt=(-(loc + scale * stats.t.ppf(1 - a, nu)),
                -loc + scale * stats.t.pdf(stats.t.ppf(a, nu), nu) / (1 - a) * (nu + stats.t.ppf(a, nu) ** 2) / (nu - 1)),
            cf=(L.var_cf(mu, sd, s, k, a), np.nan),
            hs=(L.var_hist(x, a), L.es_hist(x, a)))
        for m, (v, e) in vals.items():
            pct(f"vr{m.capitalize()}{tag}", v, 2)
            if not np.isnan(e):
                pct(f"es{m.capitalize()}{tag}", e, 2)
        rows.append(vals)
    # ES of the empirical law: the k-th loss takes the part of a weight inside the tail
    Ls = np.sort(-x)[::-1]
    for a, tag in ((0.99, "N"), (0.975, "S")):
        m = n * (1 - a); kk = math.ceil(m - 1e-9)
        exact = (Ls[:kk - 1].sum() + (m - (kk - 1)) * Ls[kk - 1]) / m
        R.RAW[f"esExact{tag}"] = float(exact)
        put(f"esGap{tag}", 100 * (exact - L.es_hist(x, a)), "{:.4f}")
        put(f"esTail{tag}", m, "{:.3f}"); put(f"esK{tag}", kk, "{:.0f}")
    # ES 97.5 vs VaR 99 under normal: nearly equal (the Basel switch)
    put("esRatioNormal", L.es_normal(0, 1, 0.975) / L.var_normal(0, 1, 0.99), "{:.3f}")
    put("esRatioHist", L.es_hist(x, 0.975) / L.var_hist(x, 0.99), "{:.2f}")
    # Cornish-Fisher at 99.9 and its failure when kurtosis is large
    put("cfZnine", L.cf_z(stats.norm.ppf(0.001), s, k), "{:.1f}")
    put("cfLossNine", -L.cf_z(stats.norm.ppf(0.001), s, k), "{:.1f}")   # as a loss quantile
    pct("vrHsThree", L.var_hist(x, 0.999), 2); pct("vrNorThree", L.var_normal(mu, sd, 0.999), 2)
    # subadditivity counterexample: two independent loans, default probability 0.9%, loss 100
    p = 0.009
    put("saP", 100 * p, "{:.1f}"); put("saBoth", 100 * (1 - (1 - p) ** 2), "{:.2f}")
    # ES of one loan at 99%: tail of 1% contains the 0.9% default and 0.1% of zero losses
    put("saEsOne", 100 * p / 0.01, "{:.0f}")
    pb = (1 - p) ** 2; p1 = 2 * p * (1 - p); p2 = p * p
    # ES of the pair at 99%: worst 1% of the loss distribution
    tail = 0.01; es2 = (min(p2, tail) * 200 + max(0, tail - p2) * 100) / tail
    put("saEsTwo", es2, "{:.1f}")
    # square root of time: 10-day VaR from 1-day vs overlapping 10-day historical
    lx = np.log1p(x)
    ten = np.exp(np.convolve(lx, np.ones(10), "valid")) - 1
    pct("srOne", L.var_hist(x, 0.99), 2); pct("srScaled", math.sqrt(10) * L.var_hist(x, 0.99), 2)
    pct("srTen", L.var_hist(ten, 0.99), 2)
    pct("srOneEs", L.es_hist(x, 0.99), 2); pct("srScaledEs", math.sqrt(10) * L.es_hist(x, 0.99), 2)
    pct("srTenEs", L.es_hist(ten, 0.99), 2)
    # autocorrelation of returns and of squared returns
    put("acRet", L.ar1(x)); put("acSq", L.ar1(x * x)); put("acAbs", L.ar1(np.abs(x)))
    # figure: left, log density; right, QQ against normal
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.9))
    zz = np.linspace(-10, 10, 801)
    hist, edges = np.histogram(z, bins=np.linspace(-10, 10, 101), density=True)
    c = 0.5 * (edges[1:] + edges[:-1]); ok = hist > 0
    ax[0].semilogy(c[ok], hist[ok], "o", ms=2.4, color=GREY, label="data")
    ax[0].semilogy(zz, stats.norm.pdf(zz), color=BLUE, label="normal")
    ts = scale / sd
    ax[0].semilogy(zz, stats.t.pdf((zz - (loc - mu) / sd) / ts, nu) / ts, color=RED, label=f"t, {nu:.1f} df")
    ax[0].set_ylim(1e-6, 1); ax[0].set_xlim(-10, 8)
    ax[0].set_xlabel("daily return, standard deviations"); ax[0].set_ylabel("density (log scale)")
    ax[0].legend(loc="upper left")
    q = np.sort(z); pp = (np.arange(1, n + 1) - 0.5) / n
    ax[1].plot(stats.norm.ppf(pp), q, ".", ms=1.5, color=BLUE)
    lim = [-5, 5]; ax[1].plot(lim, lim, color=RED, lw=1)
    ax[1].set_xlabel("normal quantile"); ax[1].set_ylabel("sample quantile")
    fig.tight_layout(); R.save(fig, "fig01_tails"); plt.close(fig)
    loans_figure()
    cornish_figure(x, mu, sd, s, k)
    return finish("tails")


def tail_measures(values, probs, a):
    """VaR and ES at confidence a of a discrete loss law (values ascending, probs summing to one).
    VaR is the smallest value v with P(L > v) <= 1 - a; ES averages the worst 1 - a of probability,
    taking part of the mass at the boundary value when the tail cuts through it."""
    values, probs = np.asarray(values, float), np.asarray(probs, float)
    above = 1 - np.cumsum(probs)                           # P(L > values[i])
    var = values[int(np.argmax(above <= 1 - a + 1e-15))]
    left, es = 1 - a, 0.0
    for v, q in zip(values[::-1], probs[::-1]):           # worst outcomes first
        take = min(q, left); es += take * v; left -= take
        if left <= 0:
            break
    return var, es / (1 - a)


def loans_figure():
    """VaR and ES at 99% of one loan (doubled) and of two independent loans, against the default
    probability p. VaR fails subadditivity on the band where the pair's VaR exceeds the sum."""
    a = 0.99
    ps = np.linspace(0, 0.02, 2001)
    one = np.array([tail_measures([0, 100], [1 - q, q], a) for q in ps])
    two = np.array([tail_measures([0, 100, 200], [(1 - q) ** 2, 2 * q * (1 - q), q * q], a) for q in ps])
    lo = 1 - math.sqrt(a)                                  # pair's VaR jumps to 100 above this
    pct("saBandLo", lo, 4); pct("saBandHi", 1 - a, 0)
    bad = ps[two[:, 0] > 2 * one[:, 0] + 1e-9]
    R.RAW["saBadMin"] = float(bad.min()); R.RAW["saBadMax"] = float(bad.max())
    R.RAW["saEsWorst"] = float(np.max(two[:, 1] - 2 * one[:, 1]))  # ES never exceeds the sum
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    for j, (lab, c) in enumerate((("99% VaR", 0), ("99% ES", 1))):
        ax[j].axvspan(100 * lo, 100 * (1 - a), color="#E8E3E3", lw=0)
        ax[j].plot(100 * ps, 2 * one[:, c], color=GREY, label="sum of the two loans")
        ax[j].plot(100 * ps, two[:, c], color=RED, label="the pair")
        ax[j].axvline(0.9, color="#999999", lw=0.8, ls=":")
        ax[j].set_xlabel("default probability of each loan, %"); ax[j].set_title(lab)
        ax[j].set_xlim(0, 2)
    ax[0].set_ylabel("loss"); ax[0].set_ylim(-5, 215); ax[0].legend(loc="lower right")
    fig.tight_layout(); R.save(fig, "fig10_loans"); plt.close(fig)


def cornish_figure(x, mu, sd, s, k):
    """Loss quantile in standard deviations against the confidence level: normal, historical and
    Cornish-Fisher with the sample skewness and excess kurtosis."""
    a = np.linspace(0.5, 0.999, 999)
    z = stats.norm.ppf(1 - a)
    cf = -L.cf_z(z, s, k)
    hs = np.array([(L.var_hist(x, q) + mu) / sd for q in a])
    falls = a[1:][np.diff(cf) < 0]                         # where the expansion is not monotone
    pct("cfDownHi", falls.max(), 0)                        # the grid starts at 50%
    pct("cfNegHi", a[cf < 0].max(), 0)                     # below this level the "loss" is a gain
    put("cfHsThree", (L.var_hist(x, 0.999) + mu) / sd, "{:.1f}")
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(100 * a, -z, color=GREY, label="normal")
    ax.plot(100 * a, hs, color=BLUE, label="historical")
    ax.plot(100 * a, cf, color=RED, label="Cornish-Fisher")
    ax.set_xscale("function", functions=(lambda v: -np.log10(1 - np.clip(v, 0, 99.95) / 100),
                                         lambda w: 100 * (1 - 10 ** (-w))))
    ax.set_xticks([50, 90, 99, 99.9]); ax.set_xticklabels(["50", "90", "99", "99.9"])
    ax.set_xlim(50, 99.9); ax.set_xlabel("confidence level, %")
    ax.set_ylabel("loss, standard deviations"); ax.legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig11_cornish"); plt.close(fig)


if __name__ == "__main__":
    out = main(); print({k: out["N"][k] for k in list(out["N"])})
