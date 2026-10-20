"""Section 3.10: Feynman-Kac. The same expectation computed three ways: a Crank-Nicolson solution of
the PDE, Monte Carlo on the SDE, and a closed form.

Run on its own:  python code/ch03/feynman_kac.py   (writes code/ch03/out/feynman_kac.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch03 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

K, T, R, SIG = 100.0, 1.0, 0.03, 0.20
KA, TH, SG, TOU = 1.5, 0.2, 0.5, 2.0


def call_pde(nx, nt, lo=math.log(1.0), hi=math.log(1000.0)):
    x = np.linspace(lo, hi, nx + 1)
    u = L.fk_theta_solve(x, T, lambda z: (R - 0.5 * SIG ** 2) + 0 * z, lambda z: SIG ** 2 + 0 * z, R,
                         lambda z: np.maximum(np.exp(z) - K, 0),
                         lambda tau, xe: max(math.exp(xe) - K * math.exp(-R * tau), 0.0), nt)
    return x, u


def ou_pde(nx, nt):
    x = np.linspace(-3.0, 3.0, nx + 1)
    u = L.fk_theta_solve(x, TOU, lambda z: KA * (TH - z), lambda z: SG ** 2 + 0 * z, 0.0, lambda z: z ** 2,
                         lambda tau, xe: L.ou_second_moment(xe, KA, TH, SG, tau), nt)
    return x, u


def at(x, u, x0):
    # cubic interpolation through the four nearest nodes
    i = int(np.clip(np.searchsorted(x, x0) - 2, 0, len(x) - 4))
    return float(np.polyval(np.polyfit(x[i:i + 4], u[i:i + 4], 3), x0))


def main():
    rng = rng_for(6)
    x, u = call_pde(1600, 800)
    bs = float(L.bs_call(100.0, K, T, R, SIG))
    put("fkPDE", at(x, u, math.log(100.0)), "{:.4f}"); put("fkBS", bs, "{:.4f}")
    M = 1_000_000
    z = rng.standard_normal(M // 2); z = np.concatenate([z, -z])
    pay = math.exp(-R * T) * np.maximum(100 * np.exp((R - 0.5 * SIG ** 2) * T + SIG * math.sqrt(T) * z) - K, 0)
    pm = pay.reshape(2, -1).mean(axis=0)            # antithetic pairs
    put("fkMC", pay.mean(), "{:.4f}"); put("fkMCse", pm.std() / math.sqrt(M // 2), "{:.4f}"); put("fkPaths", M, "{:,d}")
    xo, uo = ou_pde(1200, 600)
    put("fkOUPDE", at(xo, uo, 1.0), "{:.5f}"); put("fkOUExact", L.ou_second_moment(1.0, KA, TH, SG, TOU), "{:.5f}")
    X = L.ou_exact(1.0, KA, TH, SG, TOU / 50, 50, 200000, rng)[:, -1]
    put("fkOUMC", np.mean(X ** 2), "{:.4f}"); put("fkOUMCse", np.std(X ** 2) / math.sqrt(len(X)), "{:.4f}")
    # grid convergence
    nxs = np.array([100, 200, 400, 800, 1600])
    ec = np.array([abs(at(*call_pde(n, n // 2), math.log(100.0)) - bs) for n in nxs])
    eo = np.array([abs(at(*ou_pde(n, n // 2), 1.0) - L.ou_second_moment(1.0, KA, TH, SG, TOU)) for n in nxs])
    put("fkSlopeCall", L.ols_slope(nxs, ec), "{:.2f}"); put("fkSlopeOU", L.ols_slope(nxs, eo), "{:.2f}")
    put("fkErrCallBig", ec[-1], "{:.5f}"); RAW["fkErrCall"] = ec.tolist(); RAW["fkErrOU"] = eo.tolist()
    # figure
    S = np.linspace(60, 160, 101)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].plot(S, [at(x, u, math.log(s)) for s in S], color=BLUE, lw=1.6, label="Crank-Nicolson PDE")
    ax[0].plot(S, L.bs_call(S, K, T, R, SIG), color=RED, lw=0.9, ls="--", label="closed form")
    Sm = np.array([70, 85, 100, 115, 130, 145])
    mcv = [math.exp(-R * T) * np.mean(np.maximum(s * np.exp((R - 0.5 * SIG ** 2) * T + SIG * z[:200000]) - K, 0)) for s in Sm]
    ax[0].plot(Sm, mcv, "o", color=GREY, ms=3.5, label="Monte Carlo")
    ax[0].plot(S, np.maximum(S - K, 0), color=GREY, lw=0.7, ls=":")
    ax[0].set_xlabel("$S_0$"); ax[0].set_ylabel(r"$e^{-rT}\,\mathbb{E}[(S_T-K)^+]$")
    ax[0].set_title("(a) One expectation, three methods"); ax[0].legend(fontsize=7.0, loc="upper left")
    ax[1].loglog(nxs, ec, "o-", color=BLUE, ms=3, lw=1, label="call, $S_0=100$")
    ax[1].loglog(nxs, eo, "s-", color=RED, ms=3, lw=1, label=r"OU, $\mathbb{E}[X_T^2]$")
    ax[1].loglog(nxs, ec[0] * (nxs / nxs[0]) ** -2.0, color=GREY, ls="--", lw=0.8, label="slope $-2$")
    ax[1].set_xlabel("space steps (time steps = half)"); ax[1].set_ylabel("absolute error")
    ax[1].set_title("(b) Grid convergence"); ax[1].legend(fontsize=7.0, loc="lower left")
    fig.tight_layout(); save(fig, "fig13_feynman_kac")
    return finish("feynman_kac")


if __name__ == "__main__":
    main()
