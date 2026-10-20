"""Section 5.8: explicit, implicit and Crank-Nicolson schemes, stability and convergence.

Run on its own:  python code/ch05/finite_diff.py   (writes code/ch05/out/finite_diff.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch05 import N, RAW, put, sci, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0
WIDTH = 6.0
MS = [50, 100, 200, 400, 800, 1600]


def nt_for_lambda(M, lam, T_=T):
    dx = 2 * WIDTH * SIG * math.sqrt(T_) / M
    return int(math.ceil(T_ * 0.5 * SIG ** 2 / (lam * dx ** 2)))


def convergence():
    ex = float(L.bs(S0, K, R, SIG, T)); errs = {k: [] for k in ("expl", "impl", "cn", "cnr")}
    for M in MS:
        errs["expl"].append(abs(L.fd_theta(S0, K, R, SIG, T, M, nt_for_lambda(M, 0.4), 0.0, width=WIDTH) - ex))
        errs["impl"].append(abs(L.fd_theta(S0, K, R, SIG, T, M, M // 2, 1.0, width=WIDTH) - ex))
        errs["cn"].append(abs(L.fd_theta(S0, K, R, SIG, T, M, M // 2, 0.5, width=WIDTH) - ex))
        errs["cnr"].append(abs(L.fd_theta(S0, K, R, SIG, T, M, M // 2, 0.5, width=WIDTH, rannacher=2) - ex))
    i = MS.index(400)
    for k, nm in [("expl", "Expl"), ("impl", "Impl"), ("cn", "CN"), ("cnr", "CNR")]:
        (sci(f"fdErr{nm}", errs[k][i], 1) if errs[k][i] < 1e-3 else put(f"fdErr{nm}", errs[k][i], "{:.4f}"))
        put(f"fdOrder{nm}", -L.ols_slope(MS[2:], errs[k][2:]), "{:.2f}")
        RAW[f"fdErrs{nm}"] = errs[k]
    put("fdStepsExpl", nt_for_lambda(400, 0.4), "{:,d}"); put("fdStepsCN", 200, "{:d}")
    put("fdCNValue", L.fd_theta(S0, K, R, SIG, T, 400, 200, 0.5, width=WIDTH), "{:.4f}")
    # the Crank-Nicolson error at M = 400 is spatial: eight times as many steps barely change it
    sci("fdErrCNFine", abs(L.fd_theta(S0, K, R, SIG, T, 400, 1600, 0.5, width=WIDTH) - ex), 1)
    put("fdStepsCNFine", 1600, "{:,d}")
    put("fdStepRatio", nt_for_lambda(400, 0.4) / 200, "{:.1f}")
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for k, lab, col, mk in [("expl", r"explicit, $\lambda=0.4$", GREY, "o"), ("impl", "implicit", SKY, "s"),
                            ("cn", "Crank-Nicolson", BLUE, "^"), ("cnr", "Crank-Nicolson, Rannacher start", RED, "D")]:
        ax.plot(MS, errs[k], color=col, marker=mk, ms=3.5, label=lab, ls="-" if k != "cnr" else "--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("space intervals $M$ (time steps $M/2$ for the implicit schemes)")
    ax.set_ylabel(r"error at $S_0$"); ax.legend(loc="lower left")
    fig.tight_layout(); save(fig, "fig06_fd_convergence")
    return ex


def stability(ex):
    M = 200
    for lam, nm in [(0.45, "A"), (0.50, "B"), (0.52, "C")]:
        # one rounding rule for all three: round the step count up, so the Courant number is at most lam
        dx = 2 * WIDTH * SIG / M; Nt = math.ceil(T * 0.5 * SIG ** 2 / (lam * dx ** 2) - 1e-9)
        RAW[f"explLamAct{nm}"] = T / Nt * 0.5 * SIG ** 2 / dx ** 2
        v = L.fd_theta(S0, K, R, SIG, T, M, Nt, 0.0, width=WIDTH)
        RAW[f"explLam{nm}"] = abs(v - ex)
        put(f"explSteps{nm}", Nt, "{:d}")
    put("explAmpC", abs(1 - 4 * RAW["explLamActC"]), "{:.2f}"); put("explCourC", RAW["explLamActC"], "{:.4f}")
    sci("explErrA", RAW["explLamA"]); sci("explErrC", RAW["explLamC"]); put("explErrB", RAW["explLamB"], "{:.4f}")
    # Crank-Nicolson with a large time step near expiry: gamma oscillates around the strike
    T2, M2, Nt2 = 0.25, 800, 10
    lam2 = T2 / Nt2 * 0.5 * SIG ** 2 / (2 * WIDTH * SIG * math.sqrt(T2) / M2) ** 2
    put("cnLambda", lam2, "{:.0f}")
    out = {}
    for nm, th, rn in [("cn", 0.5, 0), ("cnr", 0.5, 2), ("impl", 1.0, 0)]:
        _, S, V, dx = L.fd_theta(S0, K, R, SIG, T2, M2, Nt2, th, width=WIDTH, rannacher=rn, return_grid=True)
        Sg, G = L.fd_gamma(S, V, dx); out[nm] = (Sg, G)
    Sg = out["cn"][0]; Gex = L.greeks(Sg, K, R, SIG, T2)["gamma"]
    win = (Sg > 80) & (Sg < 125)
    for nm, lab in [("cn", "CN"), ("cnr", "CNR"), ("impl", "Impl")]:
        put(f"gammaErr{lab}", float(np.max(np.abs(out[nm][1] - Gex)[win])), "{:.4f}")
        RAW[f"gammaNeg{lab}"] = float(np.min(out[nm][1][win]))
    put("gammaPeakExact", float(Gex[win].max()), "{:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    w = np.linspace(0, math.pi, 400)
    for lam, th, lab, col, ls in [(0.4, 0.0, r"explicit, $\lambda=0.4$", GREY, "-"), (0.6, 0.0, r"explicit, $\lambda=0.6$", ROSE, "-"),
                                  (5.0, 1.0, r"implicit, $\lambda=5$", SKY, "-"), (5.0, 0.5, r"Crank-Nicolson, $\lambda=5$", BLUE, "-")]:
        ax[0].plot(w, L.amplification(lam, th, w), color=col, ls=ls, label=lab)
    ax[0].axhline(-1, color=RED, lw=0.8, ls="--"); ax[0].axhline(1, color=RED, lw=0.8, ls="--")
    ax[0].set_xlabel(r"frequency $\omega$"); ax[0].set_ylabel(r"amplification $A(\omega)$")
    ax[0].set_title("(a) von Neumann factors"); ax[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=6.8); ax[0].set_ylim(-1.6, 1.1)
    sel = (Sg > 80) & (Sg < 125)   # the window in which the text measures the gamma error
    ax[1].plot(Sg[sel], Gex[sel], color=GREY, lw=2.2, alpha=0.5, label="exact")
    ax[1].plot(Sg[sel], out["cn"][1][sel], color=BLUE, lw=0.9, label="Crank-Nicolson")
    ax[1].plot(Sg[sel], out["cnr"][1][sel], color=RED, lw=1.0, ls="--", label="with Rannacher start")
    ax[1].set_ylim(-0.04, 0.10)
    imax = int(np.argmax(out["cn"][1]))
    ax[1].annotate(f"Crank-Nicolson reaches {out['cn'][1][imax]:.1f}\nand {RAW['gammaNegCN']:.1f} at the strike",
                   xy=(Sg[imax] + 0.6, 0.085), xytext=(103, 0.075), fontsize=7, color=BLUE,
                   arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))
    ax[1].set_xlabel("stock price $S$"); ax[1].set_ylabel(r"$\Gamma$")
    ax[1].set_title(f"(b) gamma, {Nt2} steps, $T={T2:g}$"); ax[1].legend(loc="lower right", fontsize=7)
    fig.tight_layout(); save(fig, "fig07_fd_stability")
    put("cnSteps", Nt2, "{:d}"); put("cnSpace", M2, "{:d}")


def main():
    ex = convergence(); stability(ex)
    return finish("finite_diff")


if __name__ == "__main__":
    main()
