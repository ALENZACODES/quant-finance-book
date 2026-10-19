"""Sections 1.2 and 1.8: compounding conventions, log and simple returns, notional principal.

Run on its own:  python code/ch01/rates_returns.py   (writes code/ch01/out/rates_returns.json)
run_all.py runs every script of the chapter, in a fixed order with one random generator, and writes
chapters/ch01-numbers.tex.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from numbers_ch01 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, SEED  # noqa: F401


# =============================================================== 1.2 compounding and log returns
def compounding():
    r = 0.08
    for m, nm in [(1, "A"), (2, "B"), (4, "C"), (12, "D"), (365, "E"), (np.inf, "F")]:
        pct("ear" + nm, L.effective_rate(r, m), 4)
    pct("ccEquivEight", L.continuous_equivalent(0.08), 4)
    pct("upLog", np.log(1.5), 2); pct("downLog", np.log(100 / 150), 2)
    ud = np.array([0.5, -1 / 3]); m = ud.mean()
    pct("udMean", m, 2); pct("udHalfVar", 0.5 * np.var(ud), 2); pct("udApprox", m - 0.5 * np.var(ud), 2)
    pct("udApproxSample", m - 0.5 * np.var(ud, ddof=1), 2)

    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    m = np.logspace(0, 3, 200)
    ax[0].semilogx(m, 100 * ((1 + r / m) ** m - 1), color=BLUE, label=r"$(1+r/m)^m-1$")
    ax[0].axhline(100 * np.expm1(r), color=RED, ls="--", lw=1, label=r"$e^{r}-1$")
    for mm in [1, 2, 4, 12, 365]:
        ax[0].plot(mm, 100 * ((1 + r / mm) ** mm - 1), "o", ms=3.5, color=BLUE)
    ax[0].set_xlabel("compounding periods per year, $m$"); ax[0].set_ylabel("effective annual rate (%)")
    ax[0].set_title("(a) Nominal 8% under different compounding"); ax[0].legend(loc="lower right")
    R = np.linspace(-0.6, 1.0, 400)
    ax[1].plot(100 * R, 100 * np.log1p(R), color=BLUE, label=r"log return $\ln(1+R)$")
    ax[1].plot(100 * R, 100 * R, color=GREY, ls=":", lw=1, label="simple return $R$")
    ax[1].set_xlabel("simple return $R$ (%)"); ax[1].set_ylabel("return (%)")
    ax[1].set_title("(b) Log return lies below simple return"); ax[1].legend(loc="upper left")
    fig.tight_layout(); save(fig, "fig01_compounding")


# =============================================================== 1.3 notional principal
def notional():
    notl, carry = 10_000_000, 0.004
    gain = notl * 0.02
    for coll, nm in [(notl, "Full"), (0.25 * notl, "Part")]:
        pnl = gain + coll * carry
        put("pnl" + nm, pnl, "{:,.0f}")
        pct("simple" + nm, pnl / coll, 2)
        pct("log" + nm, np.log1p(pnl / coll), 3)
    # a loss case for the partially collateralised position
    pnl = -notl * 0.02 + 0.25 * notl * carry
    pct("simplePartLoss", pnl / (0.25 * notl), 2)
    pct("logPartLoss", np.log1p(pnl / (0.25 * notl)), 3)


def main(rng=None):
    compounding()
    notional()
    return finish('rates_returns')


if __name__ == "__main__":
    main()
