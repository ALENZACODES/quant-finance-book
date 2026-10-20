"""Section 5.9: implied volatility by Newton and Brent, smiles from jumps and stochastic volatility,
and the risk-neutral density implied by a smile.

Run on its own:  python code/ch05/implied_vol.py   (writes code/ch05/out/implied_vol.json)
"""
from __future__ import annotations
import math, warnings
import numpy as np
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numbers_ch05 import N, RAW, put, sci, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401
from closed_form import HOUSE_CMAP, _surface

warnings.filterwarnings("ignore", message=".*roundoff error.*")
S0, R = 100.0, 0.05
MERTON = dict(sig=0.15, lam=0.5, muJ=-0.10, dJ=0.15)
HESTON = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
TS = [0.1, 0.5, 2.0]


def newton_example():
    K, T, sig = 120.0, 0.5, 0.30
    price = float(L.bs(S0, K, R, sig, T)); put("ivExPrice", price, "{:.4f}")
    s, it, path = L.iv_newton(price, S0, K, R, T)
    put("ivExIter", it, "{:d}"); put("ivExStart", path[0] / math.sqrt(T), "{:.4f}")
    for i, v in enumerate(path[1:4]):
        put("ivExStep" + "ABC"[i], v / math.sqrt(T), "{:.6f}")
    RAW["ivExErr"] = abs(s - sig)
    # error of the fourth iterate; the fifth step is the one that meets the stopping rule
    sci("ivExErrFour", abs(path[4] / math.sqrt(T) - sig), 0)
    sb, itb, fc = L.iv_brent(price, S0, K, R, T); put("ivExBrentIter", itb, "{:d}"); put("ivExBrentCalls", fc, "{:d}")
    RAW["ivExBrentErr"] = abs(sb - sig)
    return K, T, sig, price


def grid_test():
    worst, its, bits, nonmono, n = 0.0, [], [], 0, 0
    for T in (0.05, 0.5, 2.0):
        F = S0 * math.exp(R * T)
        for sig in (0.05, 0.2, 0.5, 1.0):
            for m in np.linspace(-1.0, 1.0, 41):
                K = F * math.exp(m * sig * math.sqrt(T))          # strikes within one standard deviation band scaled
                for kind in ("call", "put"):
                    p = float(L.bs(S0, K, R, sig, T, kind))
                    s, it, path = L.iv_newton(p, S0, K, R, T, kind)
                    sb, itb, _ = L.iv_brent(p, S0, K, R, T, kind, hi=6.0)
                    worst = max(worst, abs(s - sig), abs(sb - sig)); its.append(it); bits.append(itb); n += 1
                    d = np.diff(path[:-1])
                    if len(d) and not (np.all(d >= -1e-15) or np.all(d <= 1e-15)):
                        nonmono += 1
    put("ivGridN", n, "{:,d}"); sci("ivGridWorst", worst, 0)
    put("ivNewtonMax", max(its), "{:d}"); put("ivNewtonMed", float(np.median(its)), "{:.0f}")
    put("ivBrentMax", max(bits), "{:d}"); put("ivBrentMed", float(np.median(bits)), "{:.0f}")
    RAW["ivNonMonotone"] = nonmono
    # Brenner-Subrahmanyam at the money forward
    T = 1.0; F = S0 * math.exp(R * T); p = float(L.bs(S0, F, R, 0.2, T))
    put("bsApprox", 100 * L.brenner_subrahmanyam(p * math.exp(R * T) * S0 / F, S0, T), "{:.3f}")
    put("bsApproxStrike", F, "{:.2f}")
    # the same rule applied to the base call, struck at spot 100 and so in the money forward
    p0 = float(L.bs(S0, 100.0, R, 0.2, T))
    put("bsApproxSpot", 100 * L.brenner_subrahmanyam(p0 * math.exp(R * T) * S0 / F, S0, T), "{:.1f}")


def fig_newton(K, T, sig, price):
    xi_ = -0.5; vstar = math.sqrt(2 * abs(xi_))
    v = np.linspace(0.01, 2.2, 500); c = np.array([L._norm_call(xi_, vi) for vi in v])
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.1))
    ax[0].axvspan(0, vstar, color=PALE, alpha=0.6, lw=0)
    ax[0].text(0.08, 0.5, "convex", color=GREY, fontsize=8); ax[0].text(1.55, 0.06, "concave", color=GREY, fontsize=8)
    ax[0].plot(v, c, color=BLUE, lw=1.6)
    ax[0].text(vstar + 0.03, 0.52, r"$\nu^*$", color=GREY)
    for target, col in [(0.35, RED), (1.9, ROSE)]:
        tgt = L._norm_call(xi_, target); ax[0].axhline(tgt, color=col, lw=0.6, ls="--")
        vk = vstar
        for _ in range(3):
            fk = L._norm_call(xi_, vk); fp = float(L.npdf(xi_ / vk + vk / 2)); vn = vk - (fk - tgt) / fp
            ax[0].plot([vk, vn], [fk, tgt], color=col, lw=1.0)
            ax[0].plot([vn, vn], [tgt, L._norm_call(xi_, vn)], color=col, lw=0.7, ls=":")
            ax[0].plot([vk], [fk], "o", color=col, ms=3)
            vk = vn
    ax[0].set_xlim(0, 2.2); ax[0].set_ylim(0, 0.6)
    ax[0].set_xlabel(r"total volatility $\nu=\sigma\sqrt{T}$"); ax[0].set_ylabel("normalised call price")
    ax[0].set_title(r"(a) Newton steps from $\nu^*$, $\ln(F/K)=-0.5$")
    # error against function evaluations
    calls = []
    f = lambda s: (calls.append(s), float(L.bs(S0, K, R, s, T)) - price)[1]
    optimize.brentq(f, 1e-4, 5.0, xtol=1e-14, rtol=4 * np.finfo(float).eps)
    _, _, path = L.iv_newton(price, S0, K, R, T)
    en = [abs(p / math.sqrt(T) - sig) for p in path]; eb = [abs(s - sig) for s in calls]
    ax[1].semilogy(2 * np.arange(len(en)), np.maximum(en, 1e-16), color=BLUE, marker="o", ms=3.5,
                   label="Newton (price and vega)")
    ax[1].semilogy(np.arange(1, len(eb) + 1), np.maximum(eb, 1e-16), color=RED, marker="s", ms=3.5, label="Brent")
    ax[1].set_xlabel("evaluations"); ax[1].set_ylabel(r"$|\sigma_k-\sigma|$"); ax[1].set_title("(b) convergence")
    ax[1].set_ylim(1e-16, 10)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.5, frameon=False)  # below the axes
    fig.tight_layout(); save(fig, "fig08_newton")


def smiles():
    ms = np.linspace(-0.45, 0.35, 33)
    sm = {"merton": {}, "heston": {}}
    for T in TS:
        F = S0 * math.exp(R * T)
        for mdl in sm:
            ivs = []
            for m in ms:
                K = F * math.exp(m); kind = "put" if m < 0 else "call"
                if mdl == "merton":
                    p = L.merton_jd(S0, K, R, MERTON["sig"], T, MERTON["lam"], MERTON["muJ"], MERTON["dJ"], kind)
                else:
                    p = L.heston(S0, K, R, T, kind=kind, **HESTON)
                ivs.append(L.iv_newton(p, S0, K, R, T, kind)[0])
            sm[mdl][T] = np.array(ivs)
    i0 = int(np.argmin(abs(ms))); im, ip = int(np.argmin(abs(ms + 0.1))), int(np.argmin(abs(ms - 0.1)))
    for mdl, nm in [("merton", "Mer"), ("heston", "Hes")]:
        for T, tn in zip(TS, ["Short", "Mid", "Long"]):
            put(f"atm{nm}{tn}", 100 * sm[mdl][T][i0], "{:.2f}")
            put(f"skew{nm}{tn}", 100 * (sm[mdl][T][im] - sm[mdl][T][ip]), "{:.2f}")
    # Merton long-run variance check: sig^2 + lam (muJ^2 + dJ^2) is the variance rate of log returns
    put("merTotalVol", 100 * math.sqrt(MERTON["sig"] ** 2 + MERTON["lam"] * (MERTON["muJ"] ** 2 + MERTON["dJ"] ** 2)), "{:.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for k, (mdl, title) in enumerate([("merton", "(a) jump diffusion"), ("heston", "(b) stochastic volatility")]):
        for T, col in zip(TS, [RED, SKY, BLUE]):
            ax[k].plot(ms, 100 * sm[mdl][T], color=col, label=f"$T={T:g}$")
        ax[k].set_xlabel(r"log moneyness $\ln(K/F)$"); ax[k].set_title(title)
    ax[0].set_ylabel("implied volatility (%)"); ax[0].legend(loc="upper right")
    fig.tight_layout(); save(fig, "fig09_smiles")
    return sm, ms


def heston_surface():
    ms = np.linspace(-0.4, 0.3, 29); Ts = np.linspace(0.1, 2.0, 20)
    Z = np.empty((len(Ts), len(ms)))
    for i, T in enumerate(Ts):
        F = S0 * math.exp(R * T)
        for j, m in enumerate(ms):
            K = F * math.exp(m); kind = "put" if m < 0 else "call"
            Z[i, j] = 100 * L.iv_newton(L.heston(S0, K, R, T, kind=kind, **HESTON), S0, K, R, T, kind)[0]
    X, Y = np.meshgrid(ms, Ts)
    fig = plt.figure(figsize=(5.0, 3.4)); ax = fig.add_subplot(111, projection="3d")
    _surface(ax, X, Y, Z, "", "implied vol. (%)", stride=1)
    ax.set_xlabel(r"$\ln(K/F)$", labelpad=-4); ax.set_ylabel("$T$", labelpad=-4); ax.view_init(elev=24, azim=-55)
    fig.subplots_adjust(left=0.0, right=0.93, bottom=0.03, top=0.98); save(fig, "fig10_heston_surface")
    put("hesSurfMin", Z.min(), "{:.1f}"); put("hesSurfMax", Z.max(), "{:.1f}")


def density_and_checks():
    T = 0.5; F = S0 * math.exp(R * T)
    Ks = np.linspace(20, 260, 2401); h = Ks[1] - Ks[0]
    C = np.array([L.heston(S0, K, R, T, **HESTON) for K in Ks])
    # the second difference of call prices is the density times the discount factor e^{-rT}
    # (Breeden-Litzenberger), so the density itself is e^{rT} times the second difference
    d2 = (C[2:] - 2 * C[1:-1] + C[:-2]) / h ** 2; Kd = Ks[1:-1]
    put("densIntRaw", float(np.sum(d2) * h), "{:.4f}"); put("densDisc", math.exp(-R * T), "{:.4f}")
    assert abs(RAW["densIntRaw"] - RAW["densDisc"]) < 1e-4
    dens = math.exp(R * T) * d2
    put("densInt", float(np.sum(dens) * h), "{:.4f}"); put("densMean", float(np.sum(Kd * dens) * h), "{:.3f}")
    put("densFwd", F, "{:.3f}")
    # the text says the mean equals the forward to within densTol; check that it does
    RAW["densGapAbs"] = abs(RAW["densMean"] - F); put("densTol", 1e-4, "{:.4f}"); assert RAW["densGapAbs"] < RAW["densTol"]
    RAW["densMinNeg"] = float(dens.min()); RAW["callIncr"] = int(np.sum(np.diff(C) > 1e-9)); RAW["callConvexViol"] = int(np.sum(np.diff(C, 2) < -1e-9))
    atm = L.iv_newton(float(L.heston(S0, F, R, T, **HESTON)), S0, F, R, T)[0]
    put("densAtmVol", 100 * atm, "{:.2f}")
    from scipy import stats
    ln = stats.lognorm(s=atm * math.sqrt(T), scale=F * math.exp(-0.5 * atm ** 2 * T))
    lt = 0.8 * F    # tail probabilities from the characteristic function at the exact cut points. (Summing the gridded
    # density over nodes below 0.8F, as an earlier draft did, misplaces the cut by part of a grid cell
    # and moved the second decimal.) The grid sums are kept in RAW for the test that compares them.
    put("tailHes", 100 * L.heston_cdf(S0, lt, R, T, **HESTON), "{:.2f}"); put("tailLogn", 100 * ln.cdf(lt), "{:.2f}")
    put("rtailHes", 100 * (1 - L.heston_cdf(S0, 1.2 * F, R, T, **HESTON)), "{:.2f}"); put("rtailLogn", 100 * ln.sf(1.2 * F), "{:.2f}")
    RAW["tailHesGrid"] = 100 * float(np.sum(dens[Kd < lt]) * h); RAW["rtailHesGrid"] = 100 * float(np.sum(dens[Kd > 1.2 * F]) * h)
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    sel = (Kd > 50) & (Kd < 160)
    ax.plot(Kd[sel], dens[sel], color=BLUE, label="implied by the Heston smile")
    ax.plot(Kd[sel], ln.pdf(Kd[sel]), color=RED, ls="--", label=f"lognormal, volatility {100 * atm:.1f}%")
    ax.set_xlabel(r"$S_T$"); ax.set_ylabel("risk-neutral density"); ax.legend(loc="upper right")
    fig.tight_layout(); save(fig, "fig11_density")
    # Heston checks: Monte Carlo, and the limit of a small volatility of variance
    HES_PATHS, HES_STEPS = 200_000, 200
    m, se = L.heston_mc(S0, F, R, T, n=HES_PATHS, steps=HES_STEPS, rng=rng_for(1), **HESTON)
    put("hesMC", m, "{:.4f}"); put("hesMCSE", se, "{:.4f}"); put("hesFourier", float(L.heston(S0, F, R, T, **HESTON)), "{:.4f}")
    put("hesMCZ", abs(m - RAW["hesFourier"]) / se, "{:.1f}")
    for xi, nm in [(0.01, "A"), (0.001, "B")]:
        RAW["hesSmallXiGap" + nm] = float(L.heston(S0, F, R, T, **dict(HESTON, xi=xi)) - L.bs(S0, F, R, 0.2, T))
    RAW["lewisBSGap"] = float(L.lewis_bs(S0, 120, R, 0.2, 0.1) - L.bs(S0, 120, R, 0.2, 0.1))
    put("hesMCPaths", HES_PATHS, "{:,d}"); put("hesMCSteps", HES_STEPS, "{:d}")


def main():
    K, T, sig, price = newton_example(); grid_test(); fig_newton(K, T, sig, price)
    smiles(); heston_surface(); density_and_checks()
    return finish("implied_vol")


if __name__ == "__main__":
    main()
