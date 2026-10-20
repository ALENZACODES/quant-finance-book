"""Sections 5.2 to 5.5: the Black-Scholes price, put-call parity, the Greeks and their surfaces.

Run on its own:  python code/ch05/closed_form.py   (writes code/ch05/out/closed_form.json)
"""
from __future__ import annotations
import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm, colors
from numbers_ch05 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
HOUSE_CMAP = colors.LinearSegmentedColormap.from_list("house", [BLUE, "#7A8FB8", "#F2F2F2", "#C98080", RED])


def base_case():
    c = float(L.bs(S0, K, R, SIG, T)); p = float(L.bs(S0, K, R, SIG, T, "put"))
    put("callBase", c, "{:.4f}"); put("putBase", p, "{:.4f}")
    d1, d2 = L.d12(S0, K, R, 0.0, SIG, T); put("dOne", d1, "{:.4f}"); put("dTwo", d2, "{:.4f}")
    put("NdOne", L.ncdf(d1), "{:.4f}"); put("NdTwo", L.ncdf(d2), "{:.4f}")
    RAW["parityGap"] = c - p - (S0 - K * math.exp(-R * T))
    put("parityRHS", S0 - K * math.exp(-R * T), "{:.4f}")
    gc, gp = L.greeks(S0, K, R, SIG, T), L.greeks(S0, K, R, SIG, T, "put")
    put("deltaC", gc["delta"], "{:.4f}"); put("deltaP", gp["delta"], "{:.4f}")
    put("gammaB", gc["gamma"], "{:.5f}"); put("vegaB", gc["vega"], "{:.4f}")
    put("vegaPoint", gc["vega"] / 100, "{:.4f}")
    put("thetaC", gc["theta"], "{:.4f}"); put("thetaP", gp["theta"], "{:.4f}")
    put("thetaCDay", gc["theta"] / 365, "{:.4f}"); put("thetaPDay", gp["theta"] / 365, "{:.4f}")
    put("rhoC", gc["rho"], "{:.4f}"); put("rhoP", gp["rho"], "{:.4f}")
    put("vannaB", gc["vanna"], "{:.4f}"); put("volgaB", gc["volga"], "{:.4f}")
    # finite difference checks of every Greek, kept for the tests
    h = 1e-3
    f = lambda **kw: float(L.bs(kw.get("S", S0), K, kw.get("r", R), kw.get("s", SIG), kw.get("T", T)))
    fd = dict(delta=(f(S=S0 + h) - f(S=S0 - h)) / (2 * h),
              gamma=(f(S=S0 + h) - 2 * f() + f(S=S0 - h)) / h ** 2,
              vega=(f(s=SIG + h) - f(s=SIG - h)) / (2 * h),
              theta=-(f(T=T + h) - f(T=T - h)) / (2 * h),
              rho=(f(r=R + h) - f(r=R - h)) / (2 * h))
    for k, v in fd.items():
        RAW[f"fdErr{k.capitalize()}"] = abs(v - float(gc[k]))
    # the identity S phi(d1) = K e^{-rT} phi(d2) and the Black-Scholes equation at the base point
    RAW["phiIdentity"] = float(S0 * L.npdf(d1) - K * math.exp(-R * T) * L.npdf(d2))
    RAW["pdeResidual"] = float(gc["theta"] + 0.5 * SIG ** 2 * S0 ** 2 * gc["gamma"] + R * S0 * gc["delta"] - R * c)
    put("thetaGammaPart", -0.5 * SIG ** 2 * S0 ** 2 * gc["gamma"], "{:.4f}")
    put("thetaCarryPart", R * c - R * S0 * gc["delta"], "{:.4f}")
    # where gamma peaks: S* = K exp(-(r + 3 sig^2 / 2) T) maximises gamma in S
    Sg = K * math.exp(-(R + 1.5 * SIG ** 2) * T); put("gammaPeakS", Sg, "{:.2f}")
    grid = np.linspace(80, 120, 400001); g = L.greeks(grid, K, R, SIG, T)["gamma"]
    RAW["gammaPeakNumeric"] = float(grid[np.argmax(g)])
    # a deep in and out of the money pair, for the bounds
    put("callDeep", float(L.bs(150, K, R, SIG, T)), "{:.4f}"); put("callDeepLower", 150 - K * math.exp(-R * T), "{:.4f}")


def fig_price():
    S = np.linspace(60, 140, 401)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for tau, col, ls in [(1.0, BLUE, "-"), (0.5, SKY, "-"), (0.1, ROSE, "-")]:
        ax[0].plot(S, L.bs(S, K, R, SIG, tau), color=col, ls=ls, label=f"$T-t={tau:g}$")
        ax[1].plot(S, L.greeks(S, K, R, SIG, tau)["delta"], color=col, ls=ls)
    ax[0].plot(S, np.maximum(S - K, 0), color=GREY, lw=1, ls="--", label="payoff")
    ax[1].plot(S, (S > K).astype(float), color=GREY, lw=1, ls="--")
    ax[0].set_xlabel("stock price $S$"); ax[0].set_ylabel("call value"); ax[0].set_title("(a) price")
    ax[1].set_xlabel("stock price $S$"); ax[1].set_ylabel(r"$\Delta$"); ax[1].set_title("(b) delta")
    ax[0].legend(loc="upper left")
    fig.tight_layout(); save(fig, "fig01_price")


def _surface(ax, S, tau, Z, title, zlabel, stride=4):
    ax.plot_surface(S, tau, Z, cmap=HOUSE_CMAP, linewidth=0.15, edgecolor="#555555", rstride=stride, cstride=stride,
                    antialiased=True, alpha=0.95)
    ax.set_xlabel("$S$", labelpad=-4); ax.set_ylabel("$T-t$", labelpad=-4); ax.set_title(title, pad=0)
    ax.set_zlabel(zlabel, labelpad=-2)
    ax.tick_params(labelsize=7, pad=-2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor("white"); a.pane.set_edgecolor("#DDDDDD"); a._axinfo["grid"]["color"] = "#EEEEEE"
    ax.view_init(elev=24, azim=-58)


def fig_surfaces():
    S, tau = np.meshgrid(np.linspace(60, 140, 81), np.linspace(0.02, 1.0, 50))
    g = L.greeks(S, K, R, SIG, tau)
    for name, pairs in [("fig02_delta_gamma", [("delta", r"(a) delta $\Delta$", r"$\Delta$"),
                                                ("gamma", r"(b) gamma $\Gamma$", r"$\Gamma$")]),
                        ("fig03_vega_theta", [("vega", r"(a) vega $\mathcal{V}$", r"$\mathcal{V}$"),
                                              ("theta", r"(b) theta $\Theta$", r"$\Theta$")])]:
        fig = plt.figure(figsize=(6.6, 3.0))
        for i, (k, title, zl) in enumerate(pairs):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            _surface(ax, S, tau, g[k], title, zl)
        fig.subplots_adjust(left=0.0, right=0.96, wspace=0.08, bottom=0.04, top=0.92)
        save(fig, name)


def main():
    base_case(); fig_price(); fig_surfaces()
    return finish("closed_form")


if __name__ == "__main__":
    main()
