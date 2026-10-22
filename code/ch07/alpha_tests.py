"""Alpha and beta by least squares for the ten industries under the CAPM and a six-factor model, the
standard error of alpha and the time needed to detect it, the Gibbons-Ross-Shanken test, rolling
betas, and a simulated population of managers screened by t statistics with and without the
Benjamini-Hochberg rule. Figures fig06_detect and fig07_rolling; table tab7:alphas. Random stream 3."""
import math
import numpy as np
from scipy import stats
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

STREAM = 3
NW = 6


def table(f, X):
    T = len(X); m = f.mkt_rf.values; F = f[L.FACTORS].values; rows = []
    for i, nm in enumerate(L.INDUSTRIES):
        c = L.ols(X[:, i], np.c_[np.ones(T), m], NW); s = L.ols(X[:, i], np.c_[np.ones(T), F], NW)
        rows.append(dict(name=nm, a1=1200 * c["b"][0], t1=c["b"][0] / c["nw"][0], b1=c["b"][1], t1ols=c["t"][0],
                         a6=1200 * s["b"][0], t6=s["b"][0] / s["nw"][0], r26=s["r2"], t6ols=s["t"][0], r21=c["r2"]))
    R.RAW["alphaTable"] = rows
    lines = []
    for r in rows:
        fm = lambda v, d: (f"{v:.{d}f}" if v >= 0 else f"\\ensuremath{{-}}{-v:.{d}f}")
        lines.append(f"{r['name']} & {fm(r['a1'], 2)} & {fm(r['t1'], 2)} & {r['b1']:.2f} & {r['r21']:.2f} & {fm(r['a6'], 2)} & "
                     f"{fm(r['t6'], 2)} & {r['r26']:.2f} \\\\")
    R.N["alTableRows"] = "\n".join(lines)
    hi = next(r for r in rows if r["name"] == "HiTec"); ot = next(r for r in rows if r["name"] == "Other")
    hl = next(r for r in rows if r["name"] == "Hlth")
    R.put("alHiTecA", hi["a6"]); R.put("alHiTecT", hi["t6"]); R.put("alHiTecCapm", hi["a1"])
    R.put("alOtherA", ot["a6"]); R.put("alOtherT", ot["t6"]); R.put("alHlthA", hl["a1"]); R.put("alHlthT", hl["t1"])
    R.put("alNsigCapm", sum(abs(r["t1"]) > 1.96 for r in rows), "{:.0f}")
    R.put("alNsigSix", sum(abs(r["t6"]) > 1.96 for r in rows), "{:.0f}")
    R.N["alNsigCapmWord"] = R.word(sum(abs(r["t1"]) > 1.96 for r in rows)).lower()
    R.N["alNsigSixWord"] = R.word(sum(abs(r["t6"]) > 1.96 for r in rows)).lower()
    R.put("alRsqMinCapm", min(r["r21"] for r in rows)); R.put("alRsqMaxCapm", max(r["r21"] for r in rows))
    R.put("alRsqMinSix", min(r["r26"] for r in rows)); R.put("alRsqMaxSix", max(r["r26"] for r in rows))
    # the closed form for the standard error of alpha, checked on Hlth
    i = L.INDUSTRIES.index("Hlth"); c = L.ols(X[:, i], np.c_[np.ones(T), m])
    R.RAW["alSeOls"] = float(c["se"][0]); R.RAW["alSeTheory"] = L.alpha_se_theory(c["s"], m.mean(), m.std(), T)
    R.put("alHlthSe", 1200 * c["se"][0]); R.put("alHlthResid", 100 * math.sqrt(12) * c["s"], "{:.1f}")
    R.put("alHlthIR", (c["b"][0] / c["s"]) * math.sqrt(12), "{:.2f}")
    # GRS
    for nm, Fm in (("Capm", m), ("Five", f[L.FACTORS[:5]].values), ("Six", F)):
        J, p, *_ = L.grs(X, Fm)
        R.put(f"grs{nm}", J, "{:.2f}")
        if p < 0.001:
            R.sci(f"grsP{nm}", p, 1)
        else:
            R.put(f"grsP{nm}", p, "{:.2f}")
    # Sharpe ratio form of the GRS quadratic form
    J, p, a, Sig, fb, Om = L.grs(X, F)
    both = np.c_[F, X]; mb = both.mean(0); Sb = np.cov(both.T, bias=True)
    R.RAW["grsIdentityLhs"] = float(a @ np.linalg.solve(Sig, a))
    R.RAW["grsIdentityRhs"] = float(mb @ np.linalg.solve(Sb, mb) - fb @ np.linalg.solve(Om, fb))
    R.put("grsSRfac", math.sqrt(12 * fb @ np.linalg.solve(Om, fb)), "{:.2f}")
    R.put("grsSRall", math.sqrt(12 * mb @ np.linalg.solve(Sb, mb)), "{:.2f}")
    R.put("grsSRcapm", math.sqrt(12) * m.mean() / m.std(), "{:.2f}")
    bothc = np.c_[m, X]; mc = bothc.mean(0); Sc = np.cov(bothc.T, bias=True)
    R.put("grsSRcapmAll", math.sqrt(12 * mc @ np.linalg.solve(Sc, mc)), "{:.2f}")


def rolling(f, X):
    m = f.mkt_rf.values; W = 60; T = len(X); out = {}
    for nm in ("HiTec", "Utils", "Enrgy"):
        i = L.INDUSTRIES.index(nm); b = np.full(T, np.nan); se = np.full(T, np.nan)
        for t in range(W, T + 1):
            r = L.ols(X[t - W:t, i], np.c_[np.ones(W), m[t - W:t]]); b[t - 1] = r["b"][1]; se[t - 1] = r["se"][1]
        out[nm] = (b, se)
        R.put(f"rbMin{nm}", np.nanmin(b)); R.put(f"rbMax{nm}", np.nanmax(b)); R.put(f"rbLast{nm}", b[-1])
        R.put(f"rbSe{nm}", np.nanmedian(se))
        R.N[f"rbMinWhen{nm}"] = L.month_name(int(f.index[int(np.nanargmin(b))]))
        R.N[f"rbMaxWhen{nm}"] = L.month_name(int(f.index[int(np.nanargmax(b))]))
    # share of 60-month betas of HiTec outside the band of the full-sample beta +- 2 se
    i = L.INDUSTRIES.index("HiTec"); full = L.ols(X[:, i], np.c_[np.ones(T), m])["b"][1]
    b, se = out["HiTec"]; ok = ~np.isnan(b)
    R.put("rbOutside", 100 * np.mean(np.abs(b[ok] - full) > 2 * se[ok]), "{:.0f}")
    yrs = f.index.values // 100 + (f.index.values % 100 - 0.5) / 12
    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    for nm, col in (("HiTec", BLUE), ("Enrgy", RED), ("Utils", GREY)):
        b, se = out[nm]
        ax.plot(yrs, b, color=col, lw=1.1, label=nm); ax.fill_between(yrs, b - 2 * se, b + 2 * se, color=col, alpha=0.12, lw=0)
    ax.axhline(1, color="#333333", lw=0.6); ax.set_ylabel("beta, 60 month window"); ax.set_xlabel("end of window")
    ax.legend(loc="lower left", ncol=3); R.save(fig, "fig07_rolling")
    R.put("rbW", W, "{:.0f}")


def managers(f, rng):
    """1,000 managers over ten years, nine in ten with no alpha, the rest with an information ratio of 0.5."""
    m_all = f.mkt_rf.values; Nm, T = 1000, 120; share_skill = 0.1; ir = 0.5; te = 0.08
    a_m = ir * te / 12; s_e = te / math.sqrt(12)
    skilled = np.zeros(Nm, bool); skilled[: int(share_skill * Nm)] = True
    t_stats = np.empty(Nm)
    mu_m, sd_m = m_all.mean(), m_all.std(ddof=1)
    for j in range(Nm):
        mk = mu_m + sd_m * rng.standard_normal(T); beta = 0.6 + 0.8 * rng.random()
        y = (a_m if skilled[j] else 0.0) + beta * mk + s_e * rng.standard_normal(T)
        r = L.ols(y, np.c_[np.ones(T), mk]); t_stats[j] = r["t"][0]
    p = 2 * stats.t.sf(np.abs(t_stats), T - 2)
    naive = np.abs(t_stats) > stats.t.ppf(0.975, T - 2)
    pos = naive & (t_stats > 0)
    bhr = L.bh(p, 0.10)
    R.put("mgN", Nm, "{:,.0f}"); R.put("mgT", T, "{:.0f}"); R.put("mgSkill", int(share_skill * Nm), "{:.0f}")
    R.put("mgNaive", naive.sum(), "{:.0f}"); R.put("mgNaiveFalse", (naive & ~skilled).sum(), "{:.0f}")
    R.put("mgNaivePos", pos.sum(), "{:.0f}"); R.put("mgNaivePosFalse", (pos & ~skilled).sum(), "{:.0f}")
    R.put("mgBh", bhr.sum(), "{:.0f}"); R.put("mgBhFalse", (bhr & ~skilled).sum(), "{:.0f}")
    R.put("mgBhTrue", (bhr & skilled).sum(), "{:.0f}"); R.put("mgNaiveTrue", (naive & skilled).sum(), "{:.0f}")
    R.put("mgPiZero", L.storey_pi0(p), "{:.2f}")
    R.RAW["mgP"] = list(map(float, p)); R.RAW["mgSkilled"] = list(map(bool, skilled))
    R.put("mgPowerTheory", 100 * L.alpha_power(ir / math.sqrt(12), mu_m / sd_m, T), "{:.0f}")
    # the same population with twenty year records
    T2 = 240; t2 = np.empty(Nm)
    for j in range(Nm):
        mk = mu_m + sd_m * rng.standard_normal(T2); beta = 0.6 + 0.8 * rng.random()
        y = (a_m if skilled[j] else 0.0) + beta * mk + s_e * rng.standard_normal(T2)
        t2[j] = L.ols(y, np.c_[np.ones(T2), mk])["t"][0]
    p2 = 2 * stats.t.sf(np.abs(t2), T2 - 2); b2 = L.bh(p2, 0.10)
    R.put("mgLongBh", b2.sum(), "{:.0f}"); R.put("mgLongBhFalse", (b2 & ~skilled).sum(), "{:.0f}")
    n2 = np.abs(t2) > stats.t.ppf(0.975, T2 - 2)
    R.put("mgLongNaive", n2.sum(), "{:.0f}"); R.put("mgLongNaiveFalse", (n2 & ~skilled).sum(), "{:.0f}")
    R.put("mgLongPower", 100 * L.alpha_power(ir / math.sqrt(12), mu_m / sd_m, T2), "{:.0f}")
    R.RAW["mgLongP"] = list(map(float, p2))
    np.savetxt(R.HERE / "out" / "managers_p.csv", np.column_stack([p, p2, skilled]), delimiter=",",
               header="p120,p240,skilled", comments="", fmt="%.15g")
    return t_stats, skilled, mu_m / sd_m


def detection(f, t_stats, skilled, sr_m):
    R.put("dtSRm", sr_m * math.sqrt(12), "{:.2f}")
    for ir, nm in ((0.25, "Quarter"), (0.5, "Half"), (1.0, "One")):
        R.put(f"dtYrs{nm}", L.detection_months(ir / math.sqrt(12), sr_m) / 12, "{:.0f}")
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.9))
    yrs = np.linspace(1, 40, 200)
    for ir, col in ((0.25, GREY), (0.5, RED), (1.0, BLUE)):
        axs[0].plot(yrs, [100 * L.alpha_power(ir / math.sqrt(12), sr_m, 12 * y) for y in yrs], color=col,
                    label=f"IR {ir:g}")
    axs[0].set_xlabel("years of monthly data"); axs[0].set_ylabel("probability that |t| > 1.96, %")
    axs[0].legend(loc="center right"); axs[0].set_ylim(0, 100)
    bins = np.linspace(-4, 6, 41)
    axs[1].hist([t_stats[~skilled], t_stats[skilled]], bins=bins, stacked=True, color=[SKY, RED],
                label=["no alpha", "IR 0.5"])
    axs[1].axvline(1.98, color="#333333", lw=0.7, ls="--")
    axs[1].set_xlabel("t statistic of alpha"); axs[1].set_ylabel("managers"); axs[1].legend(loc="upper right")
    R.save(fig, "fig06_detect")


def main():
    rng = R.rng_for(STREAM)
    f, vw, ew, cap = L.load()
    X = vw.sub(f.rf, axis=0).values
    R.put("alNW", NW, "{:.0f}")
    table(f, X)
    rolling(f, X)
    t_stats, skilled, sr_m = managers(f, rng)
    detection(f, t_stats, skilled, sr_m)
    return R.finish("alpha_tests")


if __name__ == "__main__":
    main()
