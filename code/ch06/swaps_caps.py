"""Section 6.3: swap rates from the fitted curve, Vasicek caplets in closed form and by exact
simulation, and the Black volatilities they imply.

Run on its own:  python code/ch06/swaps_caps.py   (writes code/ch06/out/swaps_caps.json)
"""
from __future__ import annotations
import json, math
import numpy as np
from numbers_ch06 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401
import short_rates


def params():
    if "fitNowvasicek" not in RAW:
        short_rates.main()
    return tuple(RAW["fitNowvasicek"])


def swaps(p):
    r0, kap, rb, sig = p
    P = lambda t: L.vas_P(r0, t, kap, rb, sig)
    s5 = L.swap_rate(P, 0, 5); par5 = L.model_quotes(P, bill_tau=(), bond_T=(5,))[0]
    pct("swFive", s5, 3); RAW["swParGap"] = s5 - par5
    pct("swTen", L.swap_rate(P, 0, 10), 3); pct("swFiveFive", L.swap_rate(P, 5, 10), 3)
    ann = sum(P(k / 2) for k in range(11, 21)) / 2
    put("swAnnFiveFive", ann, "{:.4f}")


def caps(p):
    r0, kap, rb, sig = p
    P = lambda t: float(L.vas_P(r0, t, kap, rb, sig))
    dl = 0.25
    fwd = lambda T: (P(T) / P(T + dl) - 1) / dl
    for T, nm in [(1, "One"), (2, "Two"), (5, "Five")]:
        F = fwd(T); c = L.vas_caplet(r0, T, dl, F, kap, rb, sig)
        pct(f"cpF{nm}", F); put(f"cpAtm{nm}", 1e4 * c, "{:.2f}")      # basis points of notional
        pct(f"cpVol{nm}", L.black_implied(c, F, F, T, dl, P(T + dl)), 1)
    T = 2; F = fwd(T); paths = 400000
    m, se = L.vas_caplet_mc(r0, T, dl, F, kap, rb, sig, paths, rng_for(4))
    exact = L.vas_caplet(r0, T, dl, F, kap, rb, sig)
    put("cpMc", 1e4 * m, "{:.3f}"); put("cpMcExact", 1e4 * exact, "{:.3f}"); put("cpMcSE", 1e4 * se, "{:.4f}"); put("cpMcZ", (m - exact) / se, "{:.1f}")
    put("cpMcZabs", abs(m - exact) / se, "{:.1f}"); N["cpMcDir"] = "above" if m > exact else "below"
    put("cpMcPaths", paths, "{:,.0f}")
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    Ks = np.linspace(0.015, 0.06, 46)
    for T, col, ls in [(1, BLUE, "-"), (3, RED, "--"), (5, GREY, ":")]:
        F = fwd(T)
        v = [L.black_implied(L.vas_caplet(r0, T, dl, K, kap, rb, sig), F, K, T, dl, P(T + dl)) for K in Ks]
        ax.plot(100 * Ks, 100 * np.array(v), color=col, ls=ls, label=f"{T} year expiry")
        ax.axvline(100 * F, color=col, lw=0.5, ls=ls)
        if T == 3:
            put("cpSkewLo", 100 * L.black_implied(L.vas_caplet(r0, T, dl, 0.02, kap, rb, sig), F, 0.02, T, dl, P(T + dl)), "{:.1f}")
            put("cpSkewHi", 100 * L.black_implied(L.vas_caplet(r0, T, dl, 0.05, kap, rb, sig), F, 0.05, T, dl, P(T + dl)), "{:.1f}")
    ax.set_xlabel("strike (%, simple, quarterly)"); ax.set_ylabel("Black volatility (%)")
    ax.legend(loc="upper right")
    save(fig, "fig05_caplet_vols")


def main():
    p = params(); swaps(p); caps(p)
    return finish("swaps_caps")


if __name__ == "__main__":
    main()
