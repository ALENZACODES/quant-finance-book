"""Section 5.7: Monte Carlo pricing, antithetic variates, control variates and delta estimators.

Run on its own:  python code/ch05/monte_carlo.py   (writes code/ch05/out/monte_carlo.json)
"""
from __future__ import annotations
import math
import numpy as np
from scipy import integrate
from numbers_ch05 import HERE, N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
METHODS = [("plain", "plain"), ("anti", "antithetic"), ("cv", "control variate"), ("anti_cv", "both")]
NS = [250, 1000, 4000, 16000, 64000]
REPS = 200


def theory():
    """Exact variances of the discounted payoff Y, of the control X = e^{-rT} S_T, and of the
    antithetic pair average, from lognormal moments and one-dimensional quadrature."""
    c = float(L.bs(S0, K, R, SIG, T)); d1, d2 = L.d12(S0, K, R, 0.0, SIG, T); v = SIG * math.sqrt(T)
    g = math.exp(-2 * R * T)
    EY2 = g * (S0 ** 2 * math.exp((2 * R + SIG ** 2) * T) * L.ncdf(d1 + v) - 2 * K * S0 * math.exp(R * T) * L.ncdf(d1)
               + K ** 2 * L.ncdf(d2))
    EYX = g * (S0 ** 2 * math.exp((2 * R + SIG ** 2) * T) * L.ncdf(d1 + v) - K * S0 * math.exp(R * T) * L.ncdf(d1))
    varY = EY2 - c ** 2; varX = S0 ** 2 * (math.exp(SIG ** 2 * T) - 1); cov = EYX - c * S0
    rho = cov / math.sqrt(varY * varX)
    m = (R - 0.5 * SIG ** 2) * T
    f = lambda z: math.exp(-R * T) * max(S0 * math.exp(m + v * z) - K, 0)
    zmax = (m + math.log(S0 / K)) / v
    Eanti = integrate.quad(lambda z: f(z) * f(-z) * float(L.npdf(z)), -abs(zmax), abs(zmax), epsabs=1e-14)[0] \
        if zmax > 0 else 0.0
    covA = Eanti - c ** 2
    varPair = 0.5 * (varY + covA)
    # both methods: the pair average (the even part of the payoff in Z) against the even part of the control
    xe = lambda z: math.exp(-R * T) * S0 * math.exp(m) * math.cosh(v * z)
    ye = lambda z: 0.5 * (f(z) + f(-z))
    E = lambda h: integrate.quad(lambda z: h(z) * float(L.npdf(z)), -12, 12, epsabs=1e-13, limit=400,
                                 points=[-abs(zmax), abs(zmax)])[0]
    EXe = S0; varXe = E(lambda z: xe(z) ** 2) - EXe ** 2
    covE = E(lambda z: ye(z) * xe(z)) - c * EXe
    rhoE = covE / math.sqrt(varPair * varXe)
    return dict(c=c, varY=varY, varX=varX, rho=rho, corrA=covA / varY, varPair=varPair,
                vrAnti=varY / (2 * varPair), vrCV=1 / (1 - rho ** 2), rhoE=rhoE,
                vrBoth=varY / (2 * varPair * (1 - rhoE ** 2)))


def main():
    th = theory()
    put("mcExact", th["c"], "{:.4f}"); put("mcSdY", math.sqrt(th["varY"]), "{:.3f}")
    put("mcRho", th["rho"], "{:.4f}"); put("mcCorrAnti", th["corrA"], "{:.4f}")
    put("vrAntiTheory", th["vrAnti"], "{:.2f}"); put("vrCVTheory", th["vrCV"], "{:.2f}")
    put("vrBothTheory", th["vrBoth"], "{:.2f}"); put("mcRhoEven", th["rhoE"], "{:.3f}")
    RAW.update(mcVarY=th["varY"], mcVarPair=th["varPair"])
    put("mcSEMillionTheory", math.sqrt(th["varY"] / 1e6), "{:.4f}")
    # one large run of each method: estimates, standard errors, variance per payoff evaluation
    big = 1_000_000; rng = rng_for(1)
    est = {}
    for key, _ in METHODS:
        m, se = L.mc_european(S0, K, R, SIG, T, big, rng, key)
        est[key] = (m, se)
    names = dict(plain="Plain", anti="Anti", cv="CV", anti_cv="Both")
    for key, (m, se) in est.items():
        put(f"mcEst{names[key]}", m, "{:.4f}"); put(f"mcSE{names[key]}", se, "{:.4f}")
        RAW[f"mcZ{names[key]}"] = (m - th["c"]) / se
    for key in ("anti", "cv", "anti_cv"):
        put(f"vrMC{names[key]}", (est["plain"][1] / est[key][1]) ** 2, "{:.2f}")
    put("vrMCProduct", RAW["vrMCAnti"] * RAW["vrMCCV"], "{:.1f}")
    put("mcPairs", big // 2, "{:,.0f}")
    put("mcBig", big, "{:,.0f}")
    # replications: root mean square error against n, and coverage of 95% intervals
    rng = rng_for(2); rmse = {k: [] for k, _ in METHODS}; cover = {k: 0 for k, _ in METHODS}
    for n in NS:
        for key, _ in METHODS:
            errs = np.empty(REPS)
            for i in range(REPS):
                m, se = L.mc_european(S0, K, R, SIG, T, n, rng, key)
                errs[i] = m - th["c"]
                if n == 4000:
                    cover[key] += abs(m - th["c"]) <= 1.96 * se
            rmse[key].append(math.sqrt(np.mean(errs ** 2)))
    for key, _ in METHODS:
        put(f"mcSlope{names[key]}", L.ols_slope(NS, rmse[key]), "{:.2f}")
        put(f"mcCover{names[key]}", 100 * cover[key] / REPS, "{:.1f}")
    put("mcReps", REPS, "{:d}")
    RAW["mcRmse"] = {k: v for k, v in rmse.items()}
    # delta estimators
    put("deltaPaths", 100_000, "{:,d}")
    dP, seP = L.mc_delta_pathwise(S0, K, R, SIG, T, 100_000, rng_for(3))
    dL, seL = L.mc_delta_lr(S0, K, R, SIG, T, 100_000, rng_for(3))
    put("deltaPW", dP, "{:.4f}"); put("deltaPWSE", seP, "{:.4f}")
    put("deltaLR", dL, "{:.4f}"); put("deltaLRSE", seL, "{:.4f}")
    put("deltaSERatio", seL / seP, "{:.2f}"); put("deltaVarRatio", (seL / seP) ** 2, "{:.1f}")
    RAW["deltaExact"] = float(L.greeks(S0, K, R, SIG, T)["delta"])
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    rng = rng_for(4); z = rng.standard_normal(64000)
    ST = S0 * np.exp((R - 0.5 * SIG ** 2) * T + SIG * math.sqrt(T) * z)
    Y = math.exp(-R * T) * np.maximum(ST - K, 0); n_ = np.arange(1, len(Y) + 1)
    run = np.cumsum(Y) / n_; sd = np.sqrt(np.maximum(np.cumsum(Y ** 2) / n_ - run ** 2, 0) / n_)
    sel = n_ >= 100
    ax[0].fill_between(n_[sel], (run - 2 * sd)[sel], (run + 2 * sd)[sel], color=PALE)
    ax[0].plot(n_[sel], run[sel], color=BLUE, lw=1.0, label="running estimate")
    ax[0].axhline(th["c"], color=RED, ls="--", lw=1.0, label="Black-Scholes")
    ax[0].set_xscale("log"); ax[0].set_xlabel("paths $n$"); ax[0].set_ylabel("call price")
    ax[0].set_title("(a) one run, plain"); ax[0].legend(loc="lower right"); ax[0].set_ylim(9.0, 12.0)
    for (key, lab), col, mk in zip(METHODS, [BLUE, SKY, ROSE, RED], ["o", "s", "^", "D"]):
        ax[1].plot(NS, rmse[key], color=col, marker=mk, ms=3.5, label=lab)
    ax[1].plot(NS, rmse["plain"][0] * (np.array(NS) / NS[0]) ** -0.5, color=GREY, ls=":", lw=1, label=r"$n^{-1/2}$")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("payoff evaluations $n$")
    ax[1].set_ylabel("root mean square error"); ax[1].set_title(f"(b) {REPS} replications")
    ax[1].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig05_mc")
    snippet()
    return finish("monte_carlo")


def snippet():
    """Run the listing printed in the chapter and keep what it prints."""
    import contextlib, io, re, runpy
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g = runpy.run_path(str(HERE / "snippets" / "bs_mc.py"), run_name="__main__")
    a, b, c = map(float, re.findall(r"[0-9]+\.[0-9]+", buf.getvalue()))
    put("snipFormula", a, "{:.4f}"); put("snipMC", b, "{:.4f}"); put("snipSE", c, "{:.4f}")
    # the z-score from the unrounded estimate, standard error and formula value, not from the printed figures
    exact = float(g["bs_call"](g["S"], g["K"], g["r"], g["sigma"], g["T"]))
    put("snipZ", abs(float(g["m"]) - exact) / float(g["se"]), "{:.2f}")
    # how often a miss at least this large, in either direction, happens by chance
    from scipy.stats import norm
    p = 2 * norm.sf(RAW["snipZ"]); put("snipProb", 100 * p, "{:.1f}"); put("snipOdds", 1 / p, "{:.0f}")
    src = (HERE / "snippets" / "bs_mc.py").read_text()
    put("snipSeed", int(re.search(r"default_rng\((\d+)\)", src).group(1)), "{:d}")
    put("snipPairs", int(re.search(r"standard_normal\(([\d_]+)\)", src).group(1).replace("_", "")), "{:,d}")


if __name__ == "__main__":
    main()
