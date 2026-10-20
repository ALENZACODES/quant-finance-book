"""Section 4.6: commodity term structures in the one-factor model of Schwartz (1997), contango and
backwardation, and the exact split of a rolled futures position into spot, roll and collateral returns.

Run on its own:  python code/ch04/commodity_roll.py   (writes code/ch04/out/commodity_roll.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch04 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

KAPPA, SIGMA = 1.2, 0.35
ALPHA_STAR = math.log(70.0)          # risk-neutral long-run level of ln S
PREMIUM = 0.05                       # alpha - alpha_star: physical long-run level is higher
ALPHA = ALPHA_STAR + PREMIUM
H, TAU_IN, YEARS, RATE = 1 / 12, 2 / 12, 40, 0.03


def expected_roll(h=H, tau=TAU_IN):
    """Stationary expected roll return per step (also the expected futures excess return)."""
    k = KAPPA
    a = math.exp(-k * (tau - h)) - math.exp(-k * tau)
    b = math.exp(-2 * k * (tau - h)) - math.exp(-2 * k * tau)
    return a * (ALPHA - ALPHA_STAR) - SIGMA ** 2 / (4 * k) * b


def term_structures():
    tau = np.linspace(0, 3, 301)
    Finf = math.exp(ALPHA_STAR + SIGMA ** 2 / (4 * KAPPA))
    put("schFinf", Finf, "{:.2f}"); put("schSStar", Finf, "{:.2f}")
    # the curve is flat at tau=0 slope zero exactly when ln S = alpha* + sigma^2/(2 kappa)
    put("schSFlat", math.exp(ALPHA_STAR + SIGMA ** 2 / (2 * KAPPA)), "{:.2f}")
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for S, col in zip([50, 60, 70, 85, 100], [BLUE, SKY, GREY, ROSE, RED]):
        ax.plot(tau, L.schwartz_futures(math.log(S), tau, KAPPA, ALPHA_STAR, SIGMA), color=col, lw=1.3,
                label=f"spot {S}")
    ax.axhline(Finf, color="#999999", lw=0.7, ls="--")
    ax.annotate(f"long-maturity limit {Finf:.2f}", (2.4, Finf), xytext=(1.75, 58), fontsize=7.6, color=GREY,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    ax.set_xlabel("time to expiry $\\tau$ (years)"); ax.set_ylabel("futures price $G(\\tau)$")
    ax.set_title("Futures curves in the one-factor model ($\\kappa=1.2$, $\\sigma=0.35$)")
    ax.legend(loc="upper right", fontsize=7.6, ncol=1); ax.set_xlim(0, 3)
    fig.tight_layout(); save(fig, "fig05_term")
    for S, nm in [(50, "Low"), (100, "High")]:
        put(f"schOne{nm}", float(L.schwartz_futures(math.log(S), 1.0, KAPPA, ALPHA_STAR, SIGMA)), "{:.2f}")


def wealth_log_return(ex, h=H, r=RATE):
    """Exact log growth of a fully collateralised position: wealth V holds V in bills and V/G contracts,
    so each step V' = V (e^{rh} + G'/G - 1). ex are the log futures-price changes ln(G'/G)."""
    return np.sum(np.log(np.exp(r * h) + np.exp(ex) - 1), axis=-1)


def rolling(rng):
    n = int(YEARS / H); ann = 1 / H
    # many paths from the stationary law: the cross-section of 40-year average returns
    m = 10000; e = math.exp(-KAPPA * H); sd = SIGMA * math.sqrt((1 - e * e) / (2 * KAPPA))
    X = np.empty((m, n + 1)); X[:, 0] = ALPHA + SIGMA / math.sqrt(2 * KAPPA) * rng.standard_normal(m)
    for k in range(n):
        X[:, k + 1] = ALPHA + (X[:, k] - ALPHA) * e + sd * rng.standard_normal(m)
    b_in = L.schwartz_basis(X[:, :-1], TAU_IN, KAPPA, ALPHA_STAR, SIGMA)
    b_out = L.schwartz_basis(X[:, 1:], TAU_IN - H, KAPPA, ALPHA_STAR, SIGMA)
    rolls = b_out - b_in; spots = np.diff(X, axis=1); exs = spots + rolls
    rp = rolls.mean(axis=1) * ann; spp = spots.mean(axis=1) * ann
    pct("rollMCMean", rp.mean(), 2); pct("rollMCSE", rp.std(ddof=1) / math.sqrt(m), 2)
    pct("rollMCLo", np.quantile(rp, 0.05), 1); pct("rollMCHi", np.quantile(rp, 0.95), 1)
    pct("rollMCSpotLo", np.quantile(spp, 0.05), 1); pct("rollMCSpotHi", np.quantile(spp, 0.95), 1)
    pct("rollMCContango", float(np.mean(b_in > 0)), 1)
    put("rollPaths", m, "{:,.0f}"); pct("rollTheory", expected_roll() * ann, 2)
    k = KAPPA; a = math.exp(-k * (TAU_IN - H)) - math.exp(-k * TAU_IN); b = math.exp(-2 * k * (TAU_IN - H)) - math.exp(-2 * k * TAU_IN)
    pct("rollPremTerm", a * (ALPHA - ALPHA_STAR) * ann, 2); pct("rollConvTerm", -SIGMA ** 2 / (4 * k) * b * ann, 2)
    RAW["rollMCz"] = float((rp.mean() - expected_roll() * ann) / (rp.std(ddof=1) / math.sqrt(m)))
    # exact collateralised wealth growth against the sum of log changes plus the bill rate
    wl = wealth_log_return(exs) / YEARS; approx = exs.sum(axis=1) / YEARS + RATE
    gap = wl - approx
    put("rollGapMedBp", 1e4 * np.median(np.abs(gap)), "{:.1f}"); put("rollGapMaxBp", 1e4 * np.abs(gap).max(), "{:.1f}")
    pct("rollWealthMean", wl.mean(), 2); pct("rollApproxMean", approx.mean(), 2)
    pct("rollRate", RATE, 0); put("rollYears", YEARS, "{:.0f}")
    # the path in the figure is the median one by its average roll, not a tail path
    i = int(np.argsort(rp)[m // 2])
    x = X[i]
    ex, spot, roll = L.roll_decomposition(x, H, TAU_IN, KAPPA, ALPHA_STAR, SIGMA)
    RAW["rollIdentityErr"] = float(np.abs(ex - spot - roll).max())
    RAW["rollPathIndex"] = i
    basis = L.schwartz_basis(x[:-1], TAU_IN, KAPPA, ALPHA_STAR, SIGMA)
    pct("rollContango", float(np.mean(basis > 0)), 1)
    pct("rollSpotAnn", spot.mean() * ann, 2); pct("rollSpotAnnAbs", abs(spot.mean() * ann), 2)
    pct("rollSpotFortyFall", 1 - math.exp(spot.mean() * ann * 40), 0); pct("rollRollAnn", roll.mean() * ann, 2)
    pct("rollExAnn", ex.mean() * ann, 2)
    pct("rollWealthAnn", wealth_log_return(ex) / YEARS, 2); pct("rollTotalAnn", ex.mean() * ann + RATE, 2)
    se = roll.std(ddof=1) / math.sqrt(len(roll)) * ann
    pct("rollRollSE", se, 2)
    RAW["rollBasisCorr"] = float(np.corrcoef(basis, roll)[0, 1]); put("rollBasisCorr", RAW["rollBasisCorr"], "{:.2f}")
    # how far out was the tallest path? (the earlier draft showed a tail path)
    pct("rollMCSd", rp.std(ddof=1), 2)
    t = np.arange(1, n + 1) * H
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 3.1), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax[0].plot(t, 100 * np.cumsum(spot), color=GREY, lw=1.0, label="spot")
    ax[0].plot(t, 100 * np.cumsum(roll), color=RED, lw=1.2, label="roll")
    ax[0].plot(t, 100 * np.cumsum(ex), color=BLUE, lw=1.3, label="sum of log futures changes")
    ax[0].set_xlabel("years"); ax[0].set_ylabel("cumulative log change (%)")
    ax[0].set_title("(a) the median path of 10,000"); ax[0].legend(fontsize=7.0, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)  # below the axes, clear of the lines
    ax[1].scatter(100 * basis, 100 * roll, s=2, color=SKY, alpha=0.5, lw=0)
    bb = np.linspace(basis.min(), basis.max(), 2)
    co = np.polyfit(basis, roll, 1); ax[1].plot(100 * bb, 100 * np.polyval(co, bb), color=RED, lw=1.1)
    ax[1].axhline(0, color="#999999", lw=0.6); ax[1].axvline(0, color="#999999", lw=0.6)
    ax[1].set_xlabel("basis $\\ln(G/S)$ at entry (%)"); ax[1].set_ylabel("roll (%)")
    ax[1].set_title("(b) roll against basis")
    fig.tight_layout(); save(fig, "fig06_roll")


def main():
    term_structures(); rolling(rng_for(4))
    return finish("commodity_roll")


if __name__ == "__main__":
    main()
