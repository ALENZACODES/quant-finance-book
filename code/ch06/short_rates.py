"""Section 6.2: the Vasicek and CIR models fitted to the Government of Canada curve on two dates, the
probability of negative rates, a simulation check of the CIR bond price, and the Ho-Lee convexity
adjustment.

Run on its own:  python code/ch06/short_rates.py   (writes code/ch06/out/short_rates.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats
from numbers_ch06 import N, RAW, sci, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

COLS = ["TB.CDN.90D.MID", "TB.CDN.180D.MID", "TB.CDN.1Y.MID", "BD.CDN.2YR.DQ.YLD", "BD.CDN.3YR.DQ.YLD",
        "BD.CDN.5YR.DQ.YLD", "BD.CDN.7YR.DQ.YLD", "BD.CDN.10YR.DQ.YLD"]
MATS = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
DATES = {"Now": "2026-09-29", "Inv": "2023-07-31"}


def load():
    return pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", parse_dates=["date"])


def vol(d):
    """Volatility of monthly changes in the 90 day bill yield over the whole file, and its mean."""
    m = d.set_index("date")["TB.CDN.90D.MID"].resample("ME").last().dropna() / 100
    return float(np.std(np.diff(m.values), ddof=1) * math.sqrt(12)), float(m.mean()), len(m)


def fits(d):
    out = {}
    sigV, mbar, nm_ = vol(d)
    put("srVolN", nm_, "{:.0f}")
    sigC = sigV / math.sqrt(mbar)
    put("srSigV", 100 * sigV, "{:.4f}"); put("srSigC", sigC, "{:.5f}"); pct("srMeanBill", mbar)   # sigma to five decimals so the Feller ratio can be reproduced from printed inputs
    for key, dt in DATES.items():
        q = d[d.date == dt][COLS].values[0] / 100
        out[key] = dict(q=q)
        for model, sig, nm in [("vasicek", sigV, "Vas"), ("cir", sigC, "Cir")]:
            best = None
            for x0 in ([q[0], 0.3, q[-1]], [q[0], 0.1, q[-1] + 0.01], [q[0], 1.0, q[-1]]):
                x, res = L.fit_short_rate(q, model, sig, x0)
                if best is None or np.sum(res ** 2) < np.sum(best[1] ** 2):
                    best = (x, res)
            x, res = best
            out[key][model] = (x, res, sig)
            pct(f"sr{nm}{key}R", x[0]); put(f"sr{nm}{key}Kap", x[1], "{:.3f}"); pct(f"sr{nm}{key}Rbar", x[2])
            # five decimals for the table, enough for the bond and swap values in the text to reproduce
            pct(f"sr{nm}{key}Rx", x[0], 5); put(f"sr{nm}{key}Kapx", x[1], "{:.5f}"); pct(f"sr{nm}{key}Rbarx", x[2], 5)
            put(f"sr{nm}{key}Rms", math.sqrt(np.mean(res ** 2)), "{:.1f}"); put(f"sr{nm}{key}Max", np.max(np.abs(res)), "{:.1f}")
            if key == "Now" and model == "vasicek":
                put("srHalfNow", 12 * math.log(2) / x[1], "{:.1f}")
            RAW[f"sr{nm}{key}Res"] = [float(v) for v in res]
    N["srDateNow"] = "29 September 2026"; N["srDateInv"] = "31 July 2023"
    for key in DATES:
        qs = []
        for model in ("vasicek", "cir"):
            x, res, sig = out[key][model]
            Pf = L.vas_P if model == "vasicek" else L.cir_P
            qs.append(L.model_quotes(lambda t: Pf(x[0], t, x[1], x[2], sig)))
        put(f"srGap{key}", 1e4 * np.max(np.abs(qs[0] - qs[1])), "{:.3f}")
    lines = ["date,model,r0,kappa,rbar,sigma"]
    for key, dt in DATES.items():
        for model in ("vasicek", "cir"):
            x, _, sig = out[key][model]
            lines.append(f"{dt},{model},{x[0]:.17g},{x[1]:.17g},{x[2]:.17g},{sig:.17g}")
            RAW[f"fit{key}{model}"] = [float(x[0]), float(x[1]), float(x[2]), float(sig)]
    (HERE / "out" / "fit_params.csv").write_text("\n".join(lines) + "\n")
    x = out["Now"]["cir"][0]
    RAW["srFeller"] = 2 * x[1] * x[2] / sigC ** 2; put("srFeller", RAW["srFeller"], "{:.1f}")
    return out


def figure(out):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    for ax, key, title in [(axes[0], "Now", "29 September 2026"), (axes[1], "Inv", "31 July 2023")]:
        q = out[key]["q"]
        ax.scatter(MATS, 100 * q, color="black", s=14, zorder=5, label="quotes")
        for model, col, ls, lab in [("vasicek", BLUE, "-", "Vasicek"), ("cir", RED, "--", "CIR")]:
            x, res, sig = out[key][model]
            Pf = L.vas_P if model == "vasicek" else L.cir_P
            mq = L.model_quotes(lambda t: Pf(x[0], t, x[1], x[2], sig))
            ax.plot(MATS, 100 * mq, color=col, ls=ls, marker="o", ms=2.5, label=lab)
        ax.set_title(title); ax.set_xlabel("maturity (years)")
    axes[0].set_ylabel("yield (%)"); axes[0].legend(loc="lower right")
    save(fig, "fig04_curve_fits")


def negative(out):
    x, _, sig = out["Inv"]["vasicek"]
    for T, nm in [(1, "One"), (10, "Ten")]:
        m, C = L.vas_moments(x[0], T, x[1], x[2], sig)
        pct(f"srNeg{nm}", stats.norm.cdf(-m[0] / math.sqrt(C[0, 0])))
    sci("srNegOneProb", RAW["srNegOne"] / 100, 1)   # the one year probability is too small for a percentage


def cir_mc(out):
    x, _, sig = out["Now"]["cir"]
    T = 10; paths = 100000
    exact = float(L.cir_P(x[0], T, x[1], x[2], sig))
    for steps, nm, k in [(120, "Month", 2), (520, "Week", 3)]:
        m, se = L.cir_sim_bond(x[0], T, x[1], x[2], sig, steps, paths, rng_for(k))
        put(f"srMc{nm}", m, "{:.6f}"); put(f"srMc{nm}SE", se, "{:.6f}"); put(f"srMc{nm}Z", (m - exact) / se, "{:.1f}")
        put(f"srMc{nm}Zabs", abs(m - exact) / se, "{:.1f}"); N[f"srMc{nm}Dir"] = "above" if m > exact else "below"
    put("srMcExact", exact, "{:.6f}"); put("srMcPaths", paths, "{:,.0f}")


def holee():
    sigV = round(RAW["srSigV"], 4) / 100   # the printed sigma, so the adjustments reproduce from the text
    for T1, nm in [(1, "One"), (2, "Two"), (5, "Five"), (10, "Ten")]:
        put(f"hlAdj{nm}", 1e4 * L.holee_adjustment(sigV, T1, T1 + 0.25), "{:.1f}")


def main():
    d = load()
    out = fits(d); figure(out); negative(out); cir_mc(out); holee()
    return finish("short_rates")


if __name__ == "__main__":
    main()
