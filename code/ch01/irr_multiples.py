"""Sections 1.3 to 1.7: IRR and its problems, MIRR, TWR and MWR, multiples, PME and the J-curve.

Run on its own:  python code/ch01/irr_multiples.py   (writes code/ch01/out/irr_multiples.json)
run_all.py runs every script of the chapter, in a fixed order with one random generator, and writes
chapters/ch01-numbers.tex.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from numbers_ch01 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, SEED  # noqa: F401


# =============================================================== 1.4 to 1.6 IRR family
def irr_family():
    t, c, d, nav = L.fund_example()
    net = d - c
    for s in range(len(t)):
        put(f"fc{chr(65+s)}", c[s], "{:.2f}"); put(f"fd{chr(65+s)}", d[s], "{:.2f}")
        put(f"fn{chr(65+s)}", nav[s], "{:.2f}")
    put("fundPaidIn", c.sum()); put("fundDistrib", d.sum())
    cfs = net.copy()
    irr_final = L.irr(cfs)
    pct("fundIRR", irr_final, 2)
    # the same fund with no management fee: every deal earns 15%, and so does the fund
    t0, c0, d0, _ = L.fund_example(fee=0.0)
    pct("fundIRRNoFee", L.irr(d0 - c0), 2)
    put("feeDrag", RAW["fundIRRNoFee"] - RAW["fundIRR"], "{:.2f}")
    # interim (since inception) IRR at the end of year 3, 5 and 7
    interim = {}
    for s in range(1, 10):
        f = net[: s + 1].copy(); f[-1] += nav[s]
        interim[s] = L.irr(f)
    for s, nm in [(3, "Three"), (5, "Five"), (7, "Seven")]:
        pct("interim" + nm, interim[s], 2)
    RAW["interim"] = interim

    # NPV profile figure with the multiple and no-root examples
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.5))
    rr = np.linspace(-0.05, 0.40, 400)
    ax[0].plot(100 * rr, [L.npv(x, cfs) for x in rr], color=BLUE)
    ax[0].axhline(0, color="#333333", lw=0.6); ax[0].plot(100 * irr_final, 0, "o", color=RED, ms=4)
    ax[0].set_title("(a) Fund A: one root"); ax[0].set_xlabel("discount rate (%)"); ax[0].set_ylabel("NPV")
    multi = [-100, 230, -132]
    rr2 = np.linspace(0.0, 0.32, 400)
    ax[1].plot(100 * rr2, [L.npv(x, multi) for x in rr2], color=BLUE)
    ax[1].axhline(0, color="#333333", lw=0.6)
    for z in L.irr_all(multi):
        ax[1].plot(100 * z, 0, "o", color=RED, ms=4)
    ax[1].set_title("(b) $-100, +230, -132$"); ax[1].set_xlabel("discount rate (%)")
    none = [100, -300, 250]
    rr3 = np.linspace(-0.3, 1.0, 400)
    ax[2].plot(100 * rr3, [L.npv(x, none) for x in rr3], color=BLUE)
    ax[2].axhline(0, color="#333333", lw=0.6); ax[2].set_ylim(-5, 60)
    ax[2].set_title("(c) $+100, -300, +250$"); ax[2].set_xlabel("discount rate (%)")
    fig.tight_layout(); save(fig, "fig02_npv_profiles")
    r_multi = L.irr_all(multi); pct("multiLo", r_multi[0], 2); pct("multiHi", r_multi[1], 2)
    RAW["multiCount"] = len(r_multi); RAW["noneCount"] = len(L.irr_all(none))
    put("noneMinNPV", min(L.npv(x, none) for x in np.linspace(-0.5, 5, 20001)), "{:.2f}")

    # timing and scale: projects A (long) and B (short)
    A, B = [-100, 0, 0, 200], [-100, 130]
    pct("irrLong", L.irr(A)); pct("irrShort", L.irr(B))
    cross = np.sqrt(200 / 130) - 1; pct("crossRate", cross)
    pct("npvLongTen", L.npv(0.10, A) / 100, 2); put("npvLongTenAbs", L.npv(0.10, A))
    put("npvShortTenAbs", L.npv(0.10, B))
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    rr = np.linspace(0.0, 0.36, 400)
    ax.plot(100 * rr, [L.npv(x, A) for x in rr], color=BLUE, label="Long project: $-100$ now, $+200$ in year 3")
    ax.plot(100 * rr, [L.npv(x, B) for x in rr], color=RED, label="Short project: $-100$ now, $+130$ in year 1")
    ax.axhline(0, color="#333333", lw=0.6)
    ax.axvline(100 * cross, color=GREY, ls=":", lw=1)
    ax.annotate(f"crossover {100*cross:.2f}%", (100 * cross, L.npv(cross, B)), xytext=(100 * cross + 2.5, 42),
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7), fontsize=8.5, color=GREY)
    ax.set_xlabel("discount rate (%)"); ax.set_ylabel("NPV"); ax.legend(loc="upper right")
    fig.tight_layout(); save(fig, "fig03_crossover")

    # aggregation
    X, Y = [-100, 150, 0], [-100, 0, 121]
    comb = np.add(X, Y)
    pct("irrX", L.irr(X)); pct("irrY", L.irr(Y)); pct("irrXY", L.irr(comb))

    # MIRR
    m = L.mirr(multi, 0.08, 0.12); pct("mirrMulti", m)
    put("mirrPVneg", 100 + 132 / 1.08 ** 2); put("mirrFVpos", 230 * 1.12)
    mf = L.mirr(cfs, 0.08, 0.08); pct("mirrFundEight", mf)
    mf2 = L.mirr(cfs, 0.08, 0.12); pct("mirrFundTwelve", mf2)
    pct("mirrFundTwelveTwelve", L.mirr(cfs, 0.12, 0.12))

    # TWR vs DWR
    tw = L.twr([0.20, -0.10]); pct("twrTotal", tw); pct("twrPer", np.sqrt(1 + tw) - 1)
    dw = L.irr([-100, -200, 288]); pct("dwrPer", dw)
    return t, c, d, nav, interim


# =============================================================== 1.7 multiples and PME; 1.8 J-curve
def multiples_pme(t, c, d, nav, interim):
    dpi, rvpi, tvpi = L.multiples(c[:6], d[:6], nav[5])
    put("dpiFive", dpi); put("rvpiFive", rvpi); put("tvpiFive", tvpi)
    dpi, rvpi, tvpi = L.multiples(c, d, nav[-1])
    put("dpiEnd", dpi); put("rvpiEnd", rvpi); put("tvpiEnd", tvpi)
    I = L.index_levels()
    for s in range(len(I)):
        put(f"ix{chr(65+s)}", I[s], "{:.2f}")
    for s in range(len(L.INDEX_RETURNS)):
        pct(f"ir{chr(66+s)}", L.INDEX_RETURNS[s], 0)
    ks = L.ks_pme(c, d, nav[-1], I); put("ksPME", ks, "{:.3f}")
    val, ln_roots = L.ln_pme(c, d, I); put("lnValue", val); RAW["lnRoots"] = [float(x) for x in ln_roots]
    idx_irr = (I[-1] / I[0]) ** (1 / 9) - 1; pct("indexCAGR", idx_irr)
    # shadow portfolio path for the figure
    shadow = np.zeros(len(I)); v = 0.0
    for s in range(len(I)):
        if s > 0:
            v *= I[s] / I[s - 1]
        v += c[s] - d[s]; shadow[s] = v
    RAW["shadowMin"] = float(shadow.min())

    # J-curve figure
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    cum = np.cumsum(d - c)
    ax[0].bar(t, d, color=BLUE, width=0.6, label="distributions")
    ax[0].bar(t, -c, color=RED, width=0.6, label="contributions")
    ax[0].plot(t, cum, color=GREY, marker="o", ms=3, label="cumulative net cash flow")
    ax[0].axhline(0, color="#333333", lw=0.6)
    ax[0].set_xlabel("year"); ax[0].set_ylabel("cash flow to LPs"); ax[0].set_title("(a) Cash flows of Fund A")
    ax[0].legend(loc="upper left", fontsize=7.5)
    yrs = sorted(interim)
    ax[1].plot(yrs, [100 * interim[s] for s in yrs], color=BLUE, marker="o", ms=3)
    ax[1].axhline(0, color="#333333", lw=0.6)
    ax[1].set_xlabel("year of measurement"); ax[1].set_ylabel("since-inception IRR (%)")
    ax[1].set_title("(b) Interim IRR traces the J-curve")
    fig.tight_layout(); save(fig, "fig04_jcurve")
    put("cumTrough", cum.min()); put("cumTroughYear", int(np.argmin(cum)), "{:d}")
    pos = int(np.argmax(cum > 0)); put("breakEvenYear", pos, "{:d}")
    pct("interimOne", interim[1]); pct("interimTwo", interim[2])

    # PME figure: fund value (NAV + cumulative distributions) vs shadow
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(t, nav, color=BLUE, marker="o", ms=3, label="Fund A NAV")
    ax.plot(t, shadow, color=RED, marker="s", ms=3, label="shadow index portfolio (Long-Nickels)")
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xlabel("year"); ax.set_ylabel("value"); ax.legend(loc="lower left")
    fig.tight_layout(); save(fig, "fig05_pme")


def main(rng=None):
    t, c, d, nav, interim = irr_family()
    multiples_pme(t, c, d, nav, interim)
    return finish('irr_multiples')


if __name__ == "__main__":
    main()
