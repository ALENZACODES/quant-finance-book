"""Section 6.1: price and yield, duration and convexity, the daily repricing test on Government of
Canada benchmark yields, and immunisation by a barbell.

Run on its own:  python code/ch06/bonds_duration.py   (writes code/ch06/out/bonds_duration.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from numbers_ch06 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT  # noqa: F401

DATA = ROOT / "data" / "goc_bills_bonds.csv"


def load():
    return pd.read_csv(DATA, parse_dates=["date"])


def example(d):
    last = d.dropna(subset=["BD.CDN.10YR.DQ.YLD"]).iloc[-1]
    y = last["BD.CDN.10YR.DQ.YLD"] / 100
    N["bdDate"] = last["date"].strftime("%-d %B %Y").replace(" 0", " ")
    pct("bdY", y); D = L.mod_duration(y, y, 10); C = L.convexity(y, y, 10)
    put("bdD", D); put("bdMac", L.macaulay(y, y, 10)); put("bdC", C, "{:.1f}")
    put("bdDpar", L.par_duration(y, 10), "{:.4f}"); put("bdDnum", D, "{:.4f}")
    for dy, nm in [(0.01, "Up"), (-0.01, "Dn"), (0.02, "UpTwo"), (-0.02, "DnTwo")]:
        ex = L.bond_price(y + dy, y, 10)
        put(f"bdEx{nm}", ex); put(f"bdDur{nm}", 100 * (1 - D * dy)); put(f"bdQd{nm}", 100 * (1 - D * dy + 0.5 * C * dy * dy))
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    ys = np.linspace(0.0, 0.09, 400)
    ax.plot(100 * ys, [L.bond_price(v, y, 10) for v in ys], color=BLUE, label="price")
    ax.plot(100 * ys, 100 * (1 - D * (ys - y)), color=GREY, ls="--", lw=1.1, label="duration only")
    ax.plot(100 * ys, 100 * (1 - D * (ys - y) + 0.5 * C * (ys - y) ** 2), color=RED, ls=":", lw=1.5,
            label="duration and convexity")
    ax.scatter([100 * y], [100], color=BLUE, s=14, zorder=5)
    ax.set_xlabel("yield (%, semi-annual)"); ax.set_ylabel("price per 100 face")
    ax.legend(loc="upper right"); ax.set_ylim(55, 145)
    save(fig, "fig01_price_yield")
    return y


def daily(d):
    rows = {}
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharey=True)
    for T, col, ax in [(5, "BD.CDN.5YR.DQ.YLD", axes[0]), (10, "BD.CDN.10YR.DQ.YLD", axes[1])]:
        s = d[["date", col]].dropna()
        y0 = s[col].values[:-1] / 100; y1 = s[col].values[1:] / 100; dy = y1 - y0
        ex = np.array([L.bond_price(b, a, T) / 100 - 1 for a, b in zip(y0, y1)])
        D = np.array([L.mod_duration(a, a, T) for a in y0]); C = np.array([L.convexity(a, a, T) for a in y0])
        e1 = 1e4 * (ex - (-D * dy)); e2 = 1e4 * (ex - (-D * dy + 0.5 * C * dy * dy))
        nz = dy != 0
        sign_ok = np.mean(np.sign(e2[nz]) == -np.sign(dy[nz]))
        k = "Five" if T == 5 else "Ten"
        put(f"bdN{k}", len(dy), "{:,.0f}"); put(f"bdRmsRet{k}", 1e4 * np.sqrt(np.mean(ex ** 2)), "{:.1f}")
        put(f"bdRmsDur{k}", np.sqrt(np.mean(e1 ** 2)), "{:.3f}"); put(f"bdMaxDur{k}", np.max(np.abs(e1)), "{:.2f}")
        put(f"bdRmsQd{k}", np.sqrt(np.mean(e2 ** 2)), "{:.5f}"); put(f"bdMaxQd{k}", np.max(np.abs(e2)), "{:.4f}")
        put(f"bdMaxDy{k}", 1e4 * np.max(np.abs(dy)), "{:.0f}")
        RAW[f"bdSign{k}"] = float(sign_ok)
        rows[T] = (dy, e1, e2)
        ax.scatter(1e4 * dy, e1, s=2, color=GREY, alpha=0.5, label="duration only", rasterized=True)
        ax.scatter(1e4 * dy, 100 * e2, s=2, color=RED, alpha=0.6, label=r"duration and convexity ($\times 100$)", rasterized=True)
        ax.set_title(f"{T} year benchmark"); ax.set_xlabel("daily yield change (bp)")
    axes[0].set_ylabel("error (bp of price)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, markerscale=4, fontsize=8)
    N["bdFirst"] = d["date"].iloc[0].strftime("%-d %B %Y")
    save(fig, "fig02_daily_error")
    return rows


def immunise():
    z, T, t1, t2 = 0.035, 7.0, 2.0, 15.0
    a1, a2 = L.barbell(z, T, t1, t2)
    put("imZ", 100 * z, "{:.1f}"); put("imAone", a1); put("imAtwo", a2)
    single = 100 * math.exp(-z * T) * math.exp(z * t1)
    put("imSingle", single)
    s = np.linspace(-0.03, 0.03, 601)
    bar = np.array([L.surplus_shift(v, z, [t1, t2], [a1, a2], [T], [100]) for v in s])
    sho = np.array([L.surplus_shift(v, z, [t1], [single], [T], [100]) for v in s])
    RAW["imMinBar"] = float(bar.min())
    for v, nm in [(0.01, "Up"), (-0.01, "Dn"), (0.02, "UpTwo"), (-0.02, "DnTwo")]:
        put(f"imBar{nm}", L.surplus_shift(v, z, [t1, t2], [a1, a2], [T], [100]), "{:.3f}")
        put(f"imSho{nm}", L.surplus_shift(v, z, [t1], [single], [T], [100]), "{:.2f}")
    put("imShoLoss", -RAW["imShoDn"], "{:.2f}")   # the loss after a fall, as a positive amount
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.plot(1e4 * s, bar, color=BLUE, label="2 and 15 year zeros, duration matched")
    ax.plot(1e4 * s, sho, color=RED, ls="--", label="2 year zero, present value matched only")
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xlabel("parallel shift of the flat rate (bp)"); ax.set_ylabel("surplus (per 100 owed at 7 years)")
    ax.legend(loc="upper center")
    save(fig, "fig03_immunise")


def main():
    d = load()
    example(d); daily(d); immunise()
    return finish("bonds_duration")


if __name__ == "__main__":
    main()
