"""Section 6.5: hazard rates bootstrapped from a CDS curve, the credit triangle, and simulation checks
of the default time and of the protection leg.

Run on its own:  python code/ch06/hazard_cds.py   (writes code/ch06/out/hazard_cds.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch06 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401
import swaps_caps

MATS = [1, 3, 5, 7, 10]
QUOTES = [0.0040, 0.0055, 0.0070, 0.0080, 0.0090]
REC = 0.40


def curve():
    r0, kap, rb, sig = swaps_caps.params()
    return lambda t: L.vas_P(r0, t, kap, rb, sig)


def main():
    P = curve()
    knots, lams = L.bootstrap_hazard(MATS, QUOTES, REC, P)
    names = ["One", "Three", "Five", "Seven", "Ten"]
    for nm, q, l in zip(names, QUOTES, lams):
        put(f"hzQ{nm}", 1e4 * q, "{:.0f}"); pct(f"hzL{nm}", l, 3); pct(f"hzTri{nm}", q / (1 - REC), 3)
    RAW["hzReprice"] = max(abs(L.cds_par(T, knots, lams, REC, P) - q) for T, q in zip(MATS, QUOTES))
    S5 = float(L.survival(5.0, knots, lams)); pct("hzSFive", S5, 2); pct("hzPdFive", 1 - S5, 2)
    pct("hzAvgFive", -math.log(S5) / 5, 3); pct("hzRec", REC, 0)
    # flat curve: quarterly premium against the continuous credit triangle
    s_flat = L.cds_par(5, [np.inf], [0.01], REC, P)
    put("hzFlatPar", 1e4 * s_flat, "{:.2f}"); put("hzFlatTri", 1e4 * 0.01 * (1 - REC), "{:.2f}")
    # simulation of the default time and the protection leg
    n = 1_000_000; tau = L.default_times(n, knots, lams, rng_for(5))
    hit = (tau <= 5).astype(float)
    put("hzMcPd", 100 * hit.mean(), "{:.3f}"); put("hzPdFiveX", 100 * (1 - S5), "{:.3f}"); put("hzMcPdSE", 100 * hit.std(ddof=1) / math.sqrt(n), "{:.3f}")
    put("hzMcPdZ", (hit.mean() - (1 - S5)) / (hit.std(ddof=1) / math.sqrt(n)), "{:.1f}")
    pv = np.where(tau <= 5, (1 - REC) * P(np.minimum(tau, 5)), 0.0)
    _, prot = L.cds_legs(5, knots, lams, REC, P)
    put("hzMcProt", 1e4 * pv.mean(), "{:.2f}"); put("hzMcProtSE", 1e4 * pv.std(ddof=1) / math.sqrt(n), "{:.2f}")
    put("hzProt", 1e4 * prot, "{:.2f}"); put("hzMcN", n, "{:,.0f}")
    put("hzMcProtZ", (pv.mean() - prot) / (pv.std(ddof=1) / math.sqrt(n)), "{:.1f}")
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    t = np.linspace(0, 10, 1001)
    h = np.array([lams[np.searchsorted(knots, x, side="left")] if x > 0 else lams[0] for x in t])
    ax.plot(t, 100 * h, color=BLUE, drawstyle="steps-post", label="bootstrapped hazard rate")
    ax.scatter(MATS, [100 * q / (1 - REC) for q in QUOTES], color=RED, s=14, zorder=5,
               label=r"$s_T/(1-R)$ at each maturity")
    ax.set_xlabel("years"); ax.set_ylabel("hazard rate (% a year)"); ax.set_ylim(0, 2.6)
    ax2 = ax.twinx(); ax2.plot(t, 100 * L.survival(t, knots, lams), color=GREY, ls="--", label="survival (right)")
    ax2.set_ylabel("survival probability (%)"); ax2.set_ylim(84, 100); ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center")
    save(fig, "fig07_hazard")
    return finish("hazard_cds")


if __name__ == "__main__":
    main()
