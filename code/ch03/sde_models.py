"""Section 3.7: geometric Brownian motion, Euler and Milstein schemes, the Ornstein-Uhlenbeck process
and a case study fitting an OU model to the five year Government of Canada yield.

Run on its own:  python code/ch03/sde_models.py   (writes code/ch03/out/sde_models.json)
"""
from __future__ import annotations
import math, runpy, io, contextlib
import numpy as np
import pandas as pd
from scipy import stats
SAMPLE_END = "2026-09-30"   # no October 2026 month-end exists on 19 October 2026
from numbers_ch03 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401


def gbm(rng):
    S0, T = 100.0, 30.0
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    for j, (mu, sig) in enumerate([(0.08, 0.25), (0.08, 0.50)]):
        n = 360; t, W = L.bm_paths(n, 40, T, rng)
        S = L.gbm_exact(S0, mu, sig, t, W)
        for p in S:
            ax[j].plot(t, p, color=SKY, lw=0.4, alpha=0.6)
        m = S0 * np.exp(mu * t); med = S0 * np.exp((mu - 0.5 * sig ** 2) * t)
        q = lambda z: S0 * np.exp((mu - 0.5 * sig ** 2) * t + z * sig * np.sqrt(t))
        ax[j].fill_between(t, q(-1.645), q(1.645), color=PALE, alpha=0.6, lw=0, label="5% to 95% band")
        ax[j].plot(t, m, color=RED, lw=1.3, label="mean")
        ax[j].plot(t, med, color=BLUE, lw=1.3, label="median")
        ax[j].set_yscale("log"); ax[j].set_xlabel("years")
        ax[j].set_title(f"({'ab'[j]}) $\\mu={mu:.2f}$, $\\sigma={sig:.2f}$")
    ax[0].set_ylabel("$S_t$ (log scale)"); ax[0].legend(fontsize=7.2, loc="upper left")
    fig.tight_layout(); save(fig, "fig07_gbm")
    for sig, nm in [(0.25, "Low"), (0.50, "High")]:
        mean, var, med, mode = L.gbm_moments(S0, 0.08, sig, T)
        put("gbmMean" + nm, mean, "{:,.0f}"); put("gbmMedian" + nm, med, "{:,.1f}")
        put("gbmProbLoss" + nm, 100 * stats.norm.cdf(-(0.08 - 0.5 * sig ** 2) * math.sqrt(T) / sig), "{:.1f}")
        put("gbmGrowth" + nm, 100 * (0.08 - 0.5 * sig ** 2), "{:.3f}")
    put("gbmMedianLowTen", L.gbm_moments(S0, 0.08, 0.25, 10)[2], "{:.1f}")
    put("gbmMeanLowTen", L.gbm_moments(S0, 0.08, 0.25, 10)[0], "{:.1f}")


def schemes(rng):
    S0, mu, sig, T = 100.0, 0.08, 0.40, 1.0
    nf, M = 2 ** 11, 4000
    ks = np.arange(2, 10); errs_e = np.zeros(len(ks)); errs_m = np.zeros(len(ks))
    for _ in range(4):
        dWf = rng.standard_normal((M // 4, nf)) * math.sqrt(T / nf)
        ST = S0 * np.exp((mu - 0.5 * sig ** 2) * T + sig * dWf.sum(axis=1))
        for i, k in enumerate(ks):
            n = 2 ** k
            dW = dWf.reshape(M // 4, n, nf // n).sum(axis=2)
            e, m = L.gbm_schemes(S0, mu, sig, T, dW)
            errs_e[i] += np.sum(np.abs(e - ST)); errs_m[i] += np.sum(np.abs(m - ST))
    errs_e /= M; errs_m /= M
    h = T / 2.0 ** ks
    put("strongEuler", L.ols_slope(h, errs_e), "{:.2f}"); put("strongMilstein", L.ols_slope(h, errs_m), "{:.2f}")
    put("strongEulerSixteen", errs_e[2], "{:.3f}"); put("strongMilSixteen", errs_m[2], "{:.3f}")
    put("strongPaths", M, "{:,d}")
    RAW["strongErrE"] = errs_e.tolist(); RAW["strongErrM"] = errs_m.tolist()
    ns = 2 ** np.arange(1, 11)
    w1 = np.abs([L.euler_gbm_mean(S0, mu, T, int(n)) - S0 * math.exp(mu * T) for n in ns])
    w2 = np.abs([L.euler_gbm_second(S0, mu, sig, T, int(n)) - S0 ** 2 * math.exp((2 * mu + sig ** 2) * T) for n in ns])
    put("weakSlopeMean", L.ols_slope(T / ns, w1), "{:.3f}"); put("weakSlopeSecond", L.ols_slope(T / ns, w2), "{:.3f}")
    put("weakLead", -0.5 * S0 * math.exp(mu * T) * mu ** 2 * T / 12, "{:.4f}")
    put("weakMeanTwelve", L.euler_gbm_mean(S0, mu, T, 12), "{:.4f}"); put("exactMeanOne", S0 * math.exp(mu * T), "{:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].loglog(h, errs_e, "o-", color=RED, ms=3, lw=1, label="Euler")
    ax[0].loglog(h, errs_m, "s-", color=BLUE, ms=3, lw=1, label="Milstein")
    ax[0].loglog(h, errs_e[-1] * (h / h[-1]) ** 0.5, color=GREY, ls=":", lw=0.9, label="slope 1/2")
    ax[0].loglog(h, errs_m[-1] * (h / h[-1]), color=GREY, ls="--", lw=0.9, label="slope 1")
    ax[0].set_xlabel("step $h$"); ax[0].set_ylabel(r"$\mathbb{E}|S_T-\hat S_T|$"); ax[0].set_title("(a) Strong error")
    ax[0].legend(fontsize=7.0)
    ax[1].loglog(T / ns, w1, "o-", color=RED, ms=3, lw=1, label=r"first moment")
    ax[1].loglog(T / ns, w2 / 100, "s-", color=BLUE, ms=3, lw=1, label=r"second moment / 100")
    ax[1].set_xlabel("step $h$"); ax[1].set_ylabel("error of the Euler moment"); ax[1].set_title("(b) Weak error, exact")
    ax[1].legend(fontsize=7.0)
    fig.tight_layout(); save(fig, "fig08_schemes")


def ou(rng):
    x0, kappa, theta, sigma = 0.10, 1.2, 0.03, 0.02
    dt, n = 1 / 252, 4 * 252
    X = L.ou_exact(x0, kappa, theta, sigma, dt, n, 30, rng)
    t = np.arange(n + 1) * dt
    m, v = L.ou_moments(x0, kappa, theta, sigma, t)
    sd_inf = sigma / math.sqrt(2 * kappa); hl = math.log(2) / kappa
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    for p in X:
        ax.plot(t, 100 * p, color=SKY, lw=0.4, alpha=0.7)
    ax.plot(t, 100 * m, color=RED, lw=1.3, label="mean")
    ax.plot(t, 100 * (m + 1.96 * np.sqrt(v)), color=BLUE, lw=0.9, ls="--", label="95% band")
    ax.plot(t, 100 * (m - 1.96 * np.sqrt(v)), color=BLUE, lw=0.9, ls="--")
    ax.axhline(100 * theta, color=GREY, lw=0.8, ls=":")
    ax.axvline(hl, color=GREY, lw=0.8, ls=":")
    ax.text(hl + 0.05, -0.6, f"half-life {hl:.2f} years", fontsize=8, color=GREY)
    ax.set_xlabel("years"); ax.set_ylabel("rate (%)"); ax.legend(fontsize=7.5)
    ax.set_title(r"Ornstein-Uhlenbeck paths: $\kappa=1.2$, $\theta=3\%$, $\sigma=2\%$, $x_0=10\%$")
    fig.tight_layout(); save(fig, "fig09_ou")
    put("ouHalfLife", hl, "{:.3f}"); put("ouSdInf", 100 * sd_inf, "{:.2f}")
    pct("ouProbNeg", stats.norm.cdf(-theta / sd_inf), 2)
    put("ouMeanOne", 100 * L.ou_moments(x0, kappa, theta, sigma, 1.0)[0], "{:.2f}")
    # the exact discretisation is an AR(1): check its coefficient on the simulated paths
    fits = [L.ou_fit(p, dt) for p in L.ou_exact(theta, kappa, theta, sigma, 1 / 12, 600, 400, rng)]
    phi_hat = np.mean([f["phi"] for f in fits])
    put("ouKappaSd", np.std([-math.log(f["phi"]) * 12 for f in fits if f["phi"] > 0], ddof=1), "{:.2f}")
    put("ouKendallKappa", 12 * (1 + 3 * math.exp(-kappa / 12)) / 600 / math.exp(-kappa / 12), "{:.2f}")
    put("ouPhiMonthly", math.exp(-kappa / 12), "{:.4f}"); put("ouPhiHatMean", phi_hat, "{:.4f}")
    put("ouKendall", -(1 + 3 * math.exp(-kappa / 12)) / 600, "{:.4f}")
    put("ouPhiBias", phi_hat - math.exp(-kappa / 12), "{:.4f}")
    amp = 1 / (kappa / 12); put("ouAmp", amp, "{:.0f}")     # relative error in kappa per relative error in phi
    rel = abs(phi_hat - math.exp(-kappa / 12)) / math.exp(-kappa / 12)
    put("ouPhiBiasRel", 100 * rel, "{:.1f}"); put("ouKappaBiasRel", 100 * rel * amp, "{:.0f}")
    # the listing in the text
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(HERE / "snippets" / "ou_fit.py"), run_name="__main__")
    out = buf.getvalue().strip(); RAW["snippetOut"] = out
    kh = float(out.split("=")[1].split(",")[0]); put("snipKappa", kh, "{:.3f}"); put("snipHalf", math.log(2) / kh, "{:.3f}")
    kb = 12 * (1 + 3 * math.exp(-kappa / 12)) / 600 / math.exp(-kappa / 12)
    put("snipExcess", kh - kappa, "{:.3f}"); put("snipExcessRest", kh - kappa - kb, "{:.2f}")


def yield_case(rng):
    df = pd.read_csv(ROOT / "data" / "daily_policy_bond.csv", parse_dates=["date"]).dropna(subset=["BD.CDN.5YR.DQ.YLD"])
    y = df.set_index("date")["BD.CDN.5YR.DQ.YLD"].resample("ME").last().dropna()
    y = y[y.index <= SAMPLE_END]                  # last complete month-end in the data
    dt = 1 / 12
    f = L.ou_fit(y.values, dt)
    put("caseN", len(y), "{:d}"); N["caseStart"] = y.index[0].strftime("%B %Y"); N["caseEnd"] = y.index[-1].strftime("%B %Y")
    put("casePhi", f["phi"], "{:.4f}"); put("caseKappa", f["kappa"], "{:.3f}"); put("caseTheta", f["theta"], "{:.2f}")
    put("caseSigma", f["sigma"], "{:.3f}"); put("caseHalf", f["half_life"], "{:.2f}")
    put("caseLast", y.values[-1], "{:.2f}")
    # Kendall's first-order correction (mean estimated): E[phi_hat] - phi ~ -(1 + 3 phi) / n
    nk = len(y) - 1; phi_k = f["phi"] + (1 + 3 * f["phi"]) / nk
    put("caseAmp", 1 / (-math.log(f["phi"])), "{:.0f}"); put("caseNK", nk, "{:d}"); put("casePhiKendallAlt", f["phi"] + (1 + 3 * f["phi"]) / len(y), "{:.4f}")
    put("casePhiKendall", phi_k, "{:.4f}"); put("caseKappaKendall", -math.log(phi_k) / dt, "{:.3f}")
    sd_inf = f["sigma"] / math.sqrt(2 * f["kappa"]); put("caseSdInf", sd_inf, "{:.2f}"); put("caseBand", 1.96 * sd_inf, "{:.2f}")
    # parametric bootstrap of the estimator: simulate the fitted model, refit
    B = 2000
    sims = L.ou_exact(y.values[0], f["kappa"], f["theta"], f["sigma"], dt, len(y) - 1, B, rng)
    kh = np.array([L.ou_fit(s, dt).get("kappa", np.nan) for s in sims])
    ok = np.isfinite(kh); kh = kh[ok]
    put("caseBootValid", 100 * ok.mean(), "{:.1f}")
    put("caseKappaBootMean", kh.mean(), "{:.3f}"); put("caseKappaBias", kh.mean() - f["kappa"], "{:.3f}")
    put("caseBiasRatio", kh.mean() / f["kappa"], "{:.2f}")
    # can the data reject a random walk (phi = 1)? augmented Dickey-Fuller test with a constant
    from statsmodels.tsa.stattools import adfuller
    adf = adfuller(y.values, maxlag=0, regression="c", autolag=None, result_object=False)
    put("caseADF", adf[0], "{:.2f}"); put("caseADFp", adf[1], "{:.2f}"); put("caseADFcrit", adf[4]["5%"], "{:.2f}")
    RAW["caseDFt"] = float((f["phi"] - 1) / f["se_phi"])
    lo, hi = np.percentile(kh, [5, 95])
    put("caseHalfLo", math.log(2) / hi, "{:.2f}"); put("caseHalfHi", math.log(2) / lo, "{:.2f}")
    put("caseBoot", B, "{:,d}")
    # robustness: the same fit and test on month-ends from January 2001 (the longer Valet series of Section 3.4)
    g = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", parse_dates=["date"]).set_index("date")["BD.CDN.5YR.DQ.YLD"]
    yl = g.dropna().resample("ME").last().dropna(); yl = yl[yl.index <= SAMPLE_END]
    fl = L.ou_fit(yl.values, dt)
    adl = adfuller(yl.values, maxlag=0, regression="c", autolag=None, result_object=False)
    put("caseLongN", len(yl), "{:d}"); N["caseLongStart"] = yl.index[0].strftime("%B %Y")
    put("caseLongKappa", fl["kappa"], "{:.3f}"); put("caseLongHalf", fl["half_life"], "{:.2f}")
    put("caseLongTheta", fl["theta"], "{:.2f}")
    put("caseLongADF", adl[0], "{:.2f}"); put("caseLongADFp", adl[1], "{:.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax[0].plot(y.index, y.values, color=BLUE, lw=1, label="month-end yield")
    ax[0].axhline(f["theta"], color=RED, lw=0.9, ls="--", label=r"fitted $\hat\theta$")
    ax[0].fill_between(y.index, f["theta"] - 1.96 * sd_inf, f["theta"] + 1.96 * sd_inf, color=PALE, alpha=0.45, lw=0,
                       label="stationary 95% band")
    ax[0].set_ylabel("yield (%)"); ax[0].set_title("(a) Five year Government of Canada yield")
    ax[0].set_ylim(-0.2, 6.0); ax[0].legend(fontsize=6.6, loc="upper left", ncol=2)
    ax[1].hist(kh, bins=50, color=PALE, ec=SKY, lw=0.3)
    ax[1].axvline(f["kappa"], color=RED, lw=1, label=r"value used to simulate")
    ax[1].axvline(kh.mean(), color=BLUE, lw=1, ls="--", label="mean estimate")
    ax[1].set_xlabel(r"estimated $\kappa$"); ax[1].set_title(r"(b) Bias of $\hat\kappa$"); ax[1].legend(fontsize=6.8)
    fig.tight_layout(); save(fig, "fig10_yield_ou")


def main():
    rng = rng_for(4)
    gbm(rng); schemes(rng); ou(rng); yield_case(rng)
    return finish("sde_models")


if __name__ == "__main__":
    main()
