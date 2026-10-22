"""Brinson attribution of an illustrative industry-tilted portfolio against the cap-weighted ten
industries, January to August 2026, linked over the months by Carino's factors, and a factor
attribution of the HiTec industry. Table tab7:attrib, figure fig17_waterfall. No random numbers."""
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

TILT = {"HiTec": -0.10, "Hlth": 0.04, "Enrgy": 0.03, "Utils": 0.03}
START = 202601


def main():
    f, vw, ew, cap = L.load()
    months = [m for m in f.index if m >= START]
    b0 = cap.loc[START].values / cap.loc[START].values.sum()
    w = b0 + np.array([TILT.get(k, 0.0) for k in L.INDUSTRIES])
    assert abs(w.sum() - 1) < 1e-12 and w.min() > 0
    Rp, Rb, eff = [], [], []
    for mth in months:
        b = cap.loc[mth].values / cap.loc[mth].values.sum()
        rb = vw.loc[mth].values; rp = ew.loc[mth].values
        a, s, i = L.brinson(w, b, rp, rb)
        Rp.append(w @ rp); Rb.append(b @ rb); eff.append(np.vstack([a, s, i]))
        assert abs(a.sum() + s.sum() + i.sum() - (w @ rp - b @ rb)) < 1e-14
    Rp, Rb, eff = np.array(Rp), np.array(Rb), np.array(eff)        # eff: months x 3 x industries
    k, P, B = L.carino(Rp, Rb)
    linked = np.einsum("t,tej->ej", k, eff)
    R.RAW["atLinkErr"] = float(abs(linked.sum() - (P - B)))
    R.RAW["atArithSum"] = float(eff.sum()); R.RAW["atActive"] = float(P - B)
    R.put("atP", 100 * P); R.put("atB", 100 * B); R.put("atActive", 100 * (P - B))
    R.put("atSumMonthly", 100 * eff.sum()); R.put("atAlloc", 100 * linked[0].sum()); R.put("atSel", 100 * linked[1].sum())
    R.put("atInter", 100 * linked[2].sum()); R.put("atMonths", len(months), "{:.0f}")
    R.N["atMonthsWord"] = R.word(len(months)).lower(); R.N["atLast"] = L.month_name(months[-1])
    R.put("atKmin", k.min(), "{:.3f}"); R.put("atKmax", k.max(), "{:.3f}")
    R.put("atHiTecW", 100 * b0[L.INDUSTRIES.index("HiTec")], "{:.1f}")
    rows = []
    fm = lambda v: (f"{v:.2f}" if v >= -0.005 else f"\\ensuremath{{-}}{-v:.2f}")
    for j, nm in enumerate(L.INDUSTRIES):
        rows.append(f"{nm} & {100 * b0[j]:.1f} & {100 * w[j]:.1f} & {fm(100 * linked[0, j])} & {fm(100 * linked[1, j])} & "
                    f"{fm(100 * linked[2, j])} \\\\")
    rows.append(f"\\midrule\nTotal & 100.0 & 100.0 & {fm(100 * linked[0].sum())} & {fm(100 * linked[1].sum())} & "
                f"{fm(100 * linked[2].sum())} \\\\")
    R.N["atTableRows"] = "\n".join(rows)
    rd = lambda v, d: float(f"{v:.{d}f}")                       # totals of the entries as printed
    R.put("atBSum", sum(rd(100 * v, 1) for v in b0), "{:.1f}"); R.put("atWSum", sum(rd(100 * v, 1) for v in w), "{:.1f}")
    R.put("atEffSum", sum(rd(v, 2) for v in (100 * linked[0].sum(), 100 * linked[1].sum(), 100 * linked[2].sum())))
    hs = L.INDUSTRIES.index("HiTec"); sel = 100 * linked[1]
    R.put("atSelHiTec", sel[hs]); R.put("atAllocHiTec", 100 * linked[0, hs])
    R.N["atBestSel"] = L.INDUSTRIES[int(np.argmax(sel))]; R.put("atBestSelV", sel.max())
    R.N["atWorstSel"] = L.INDUSTRIES[int(np.argmin(sel))]; R.put("atWorstSelV", sel.min())
    # factor attribution of HiTec's average excess return, full sample
    X = vw["HiTec"].values - f.rf.values; F = f[L.FACTORS].values; T = len(X)
    c = L.ols(X, np.c_[np.ones(T), F]); contrib = c["b"][1:] * F.mean(0)
    R.RAW["faErr"] = float(abs(c["b"][0] + contrib.sum() - X.mean()))
    R.put("faMean", 1200 * X.mean()); R.put("faAlpha", 1200 * c["b"][0])
    for nm, v, b in zip(["Mkt", "Smb", "Hml", "Rmw", "Cma", "Mom"], contrib, c["b"][1:]):
        R.put(f"fa{nm}", 1200 * v); R.put(f"faB{nm}", b)
    # what moves the alpha from the CAPM to the six-factor model
    c1 = L.ols(X, np.c_[np.ones(T), F[:, 0]]); fm_ = 1200 * F.mean(0)
    R.put("faCapmPred", c1["b"][1] * fm_[0]); R.put("faSixPred", 1200 * contrib.sum())
    R.put("faMktDrop", (c1["b"][1] - c["b"][1]) * fm_[0]); R.put("faFourNeg", -1200 * contrib[2:].sum())
    R.put("faGap", 1200 * (c["b"][0] - c1["b"][0]))
    R.RAW["faGapErr"] = float(abs((c1["b"][1] - c["b"][1]) * fm_[0] - 1200 * contrib[1:].sum() - 1200 * (c["b"][0] - c1["b"][0])))
    waterfall(1200 * contrib, 1200 * c["b"][0], 1200 * X.mean(), c1["b"][1] * fm_[0])
    return R.finish("attribution")


def waterfall(parts, alpha, total, capm_pred):
    """Factor attribution of the mean excess return of HiTec, from the unrounded estimates."""
    labels = ["market", "size", "value", "profit.", "invest.", "momentum", "alpha"]
    vals = np.append(parts, alpha)
    R.RAW["faWaterErr"] = float(abs(vals.sum() - total))
    R.put("faRoundedSum", sum(float(f"{v:.2f}") for v in vals))       # the printed parts add to this
    R.put("faFourRoundedSum", -sum(float(f"{v:.2f}") for v in vals[2:6]))   # the four printed factor terms
    R.put("faFourRoundedNeg", sum(float(f"{v:.2f}") for v in vals[2:6]))    # the same, signed
    R.put("faFourNegSigned", float(vals[2:6].sum()))                         # unrounded, signed
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    base = 0.0
    for k, (lab, v) in enumerate(zip(labels, vals)):
        ax.bar(k, v, bottom=base, color=(GREY if k == 0 else (BLUE if v >= 0 else RED)) if k < 6 else SKY,
               width=0.6)
        ax.text(k, base + max(v, 0) + 0.15, f"{v:.2f}".replace("-", "\u2212"), ha="center", fontsize=7)
        if k < len(vals) - 1:
            ax.plot([k + 0.3, k + 0.7], [base + v] * 2, color="#777777", lw=0.6)
        base += v
    ax.bar(len(vals), total, color=BLUE, width=0.6)
    ax.text(len(vals), total + 0.15, f"{total:.2f}", ha="center", fontsize=7)
    ax.axhline(capm_pred, color=RED, ls="--", lw=0.9)
    ax.annotate(f"CAPM prediction, {capm_pred:.2f}", (-0.3, capm_pred), fontsize=7, ha="left", va="bottom",
                xytext=(0, 2), textcoords="offset points", color=RED)
    ax.set_xticks(range(len(vals) + 1)); ax.set_xticklabels(labels + ["mean"], fontsize=7.5)
    ax.set_ylabel("% a year"); ax.set_ylim(0, max(total, base) + 1.5); ax.grid(axis="x", visible=False)
    R.save(fig, "fig17_waterfall")


if __name__ == "__main__":
    main()
