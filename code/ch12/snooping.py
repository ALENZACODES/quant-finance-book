"""Section 12.7, data snooping in managed futures: 66 moving-average trend rules, long or short, on six
markets of the Chapter 9 trend programme (WTI crude, the US and Canadian 10 year bonds, and the euro,
yen and Canadian dollar against the US dollar), from month-end prices in the shipped data, tested by White's reality check,
Hansen's SPA test, the StepM of Romano and Wolf, and the Holm, Benjamini-Hochberg and
Benjamini-Yekutieli rules. A second part draws samples in which no rule beats cash and counts how often
each procedure finds one. Writes fig22_snooping.png and out/snoop_p.csv (per-rule p-values for the R
check).

Run on its own:  python code/ch12/snooping.py   (writes code/ch12/out/snooping.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, HERE  # noqa: F401
from numbers_ch12 import ROOT

ASSETS = {"OIL": "WTI crude", "UST": "US 10 year bond", "GOC": "Canada 10 year bond", "EUR": "euro",
          "JPY": "yen", "CAD": "Canadian dollar"}
LOOKS = list(range(2, 13))
COST = 0.001            # one way cost of a change of position, as a share of the position
B, BLOCK, ALPHA = 5000, 6, 0.05
SIM_REPS, SIM_B = 1000, 499


def month_end(s):
    s = s.dropna()
    return s.groupby(s.index.to_period("M")).last()


def par_bond_return(y0, y1, m=10.0):
    """Monthly return of a par bond bought at yield y0 (decimal, compounded twice a year) with m years
    to run, priced a month later at y1, including the month's accrued coupon."""
    n = 2 * (m - 1 / 12); v = (1 + y1 / 2) ** (-n)
    return y0 / y1 * (1 - v) + v - 1 + y0 / 12


def market_returns():
    """Monthly excess returns over bills of the six markets, from month-end prices in the shipped data:
    oil and the three currencies are spot returns (no carry or roll), the bonds are par bond returns."""
    d = pd.read_csv(ROOT / "data" / "fred_markets_daily.csv", index_col=0, parse_dates=True)
    g = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", index_col=0, parse_dates=True)
    ra = pd.read_csv(ROOT / "data" / "real_assets_monthly.csv", index_col=0, parse_dates=True)
    rf = (ra["tbill"] / 1200); rf.index = rf.index.to_period("M")
    y, yc = month_end(d["DGS10"]) / 100, month_end(g["BD.CDN.10YR.DQ.YLD"]) / 100
    bc = month_end(g["TB.CDN.90D.MID"]) / 100
    oil, eur, jpy, cad = (month_end(d[c]) for c in ("DCOILWTICO", "DEXUSEU", "DEXJPUS", "DEXCAUS"))
    out = pd.DataFrame({
        "OIL": oil / oil.shift(1) - 1 - rf.shift(1),
        "UST": par_bond_return(y.shift(1), y) - rf.shift(1),
        "GOC": par_bond_return(yc.shift(1), yc) - bc.shift(1) / 12,
        "EUR": eur / eur.shift(1) - 1, "JPY": jpy.shift(1) / jpy - 1, "CAD": cad.shift(1) / cad - 1})
    out = out[out.index <= pd.Period("2026-09", "M")]
    return out.dropna()


def rule_returns():
    """Monthly returns after costs of every (market, lookback) trend rule: long when the cumulative
    excess return index is above its moving average over the lookback, short when below, decided at a
    month end and held through the next month."""
    x = market_returns()
    cols, names, turn = [], [], []
    for a in ASSETS:
        lp = np.log1p(x[a]).cumsum()
        for Lk in LOOKS:
            ma = lp.rolling(Lk).mean()
            pos = np.sign(lp - ma).where(ma.notna()).shift(1)        # set at the end of the previous month
            sw = pos.diff().abs()
            cols.append(pos * x[a] - COST * sw); names.append((a, Lk)); turn.append(sw / 2)
    X = pd.concat(cols, axis=1); X.columns = [f"{a}_{k}" for a, k in names]
    X = X.dropna()
    T = pd.concat(turn, axis=1).loc[X.index]
    return X, names, float(T.mean().mean() * 12)


def main():
    X, names, turns = rule_returns()
    d = X.values; n, m = d.shape
    put("mtRules", m, "{:.0f}"); put("mtAssets", len(ASSETS), "{:.0f}"); put("mtLookMin", LOOKS[0], "{:.0f}")
    put("mtLookMax", LOOKS[-1], "{:.0f}"); put("mtN", n, "{:.0f}")
    N["mtFirst"] = X.index[0].strftime("%B %Y"); N["mtLast"] = X.index[-1].strftime("%B %Y")
    put("mtB", B, "{:,.0f}"); put("mtBlock", BLOCK, "{:.0f}"); pct("mtCost", COST, 1); put("mtTurn", turns, "{:.1f}")
    idx = L.stationary_bootstrap(n, B, BLOCK, rng_for(8))
    r = L.snooping_tests(d, idx, ALPHA)
    k = int(np.argmax(r["dbar"])); a, Lk = names[k]
    N["mtBestAsset"] = ASSETS[a]; put("mtBestLook", Lk, "{:.0f}")
    pct("mtBestMean", 12 * r["dbar"][k], 2)
    put("mtBestT", r["t"][k], "{:.2f}"); put("mtBestP", r["p_ind"][k], "{:.3f}")
    kt = int(np.argmax(r["t"])); N["mtTopAsset"] = ASSETS[names[kt][0]]; put("mtTopLook", names[kt][1], "{:.0f}")
    put("mtTopT", r["t"][kt], "{:.2f}"); put("mtTopP", r["p_ind"][kt], "{:.3f}")
    put("mtPos", int((r["dbar"] > 0).sum()), "{:.0f}")
    put("mtNaiveFive", int((r["p_ind"] <= ALPHA).sum()), "{:.0f}")
    put("mtNaiveTen", int((r["p_ind"] <= 0.10).sum()), "{:.0f}")
    put("mtRcP", r["p_rc"], "{:.3f}"); put("mtSpaP", r["p_spa_c"], "{:.3f}")
    put("mtSpaPl", r["p_spa_l"], "{:.3f}"); put("mtSpaPu", r["p_spa_u"], "{:.3f}")
    put("mtStepM", int(r["stepm"].sum()), "{:.0f}"); put("mtThr", r["thr"], "{:.2f}")
    for al, nm in [(0.05, "Five"), (0.10, "Ten")]:
        put(f"mtHolm{nm}", int(L.holm(r["p_ind"], al).sum()), "{:.0f}")
        put(f"mtBh{nm}", int(L.bh(r["p_ind"], al).sum()), "{:.0f}")
        put(f"mtBhy{nm}", int(L.bhy(r["p_ind"], al).sum()), "{:.0f}")
    put("mtCm", float(np.sum(1.0 / np.arange(1, m + 1))), "{:.2f}")
    RAW["mtDbar"] = r["dbar"].tolist()
    pd.DataFrame({"rule": X.columns, "mean": r["dbar"], "t": r["t"], "p": r["p_ind"]}).to_csv(
        HERE / "out" / "snoop_p.csv", index=False, float_format="%.12g")
    # samples with no skill: resample the demeaned excess returns, so every rule's mean is zero
    rs = rng_for(81); dz = d - d.mean(0)
    hits = dict(naive=0, rc=0, spa=0, stepm=0, bhy=0)
    for _ in range(SIM_REPS):
        z = dz[L.stationary_bootstrap(n, 1, BLOCK, rs)[0]]
        q = L.snooping_tests(z, L.stationary_bootstrap(n, SIM_B, BLOCK, rs), ALPHA)
        hits["naive"] += q["p_ind"].min() <= ALPHA
        hits["rc"] += q["p_rc"] <= ALPHA
        hits["spa"] += q["p_spa_c"] <= ALPHA
        hits["stepm"] += q["stepm"].any()
        hits["bhy"] += L.bhy(q["p_ind"], ALPHA).any()
    # the prose: no rule survives the search; the studentized procedures run a little above 5%, the
    # reality check below it, and the sample is under 300 months
    assert r["p_rc"] > ALPHA and r["p_spa_c"] > ALPHA and not r["stepm"].any() and n < 300
    assert hits["rc"] / SIM_REPS < ALPHA < min(hits["spa"], hits["stepm"], hits["bhy"]) / SIM_REPS
    put("mtSimReps", SIM_REPS, "{:.0f}"); put("mtSimB", SIM_B, "{:.0f}")
    for key, nm in [("naive", "Naive"), ("rc", "Rc"), ("spa", "Spa"), ("stepm", "StepM"), ("bhy", "Bhy")]:
        pct(f"mtSim{nm}", hits[key] / SIM_REPS, 1)
        RAW[f"mtSim{nm}Se"] = math.sqrt(max(hits[key], 1) / SIM_REPS * (1 - hits[key] / SIM_REPS) / SIM_REPS)
    # figure: the studentized statistics of the rules with the StepM first-step critical value
    tb = math.sqrt(n) * (L.boot_means(d, idx) - r["dbar"]) / r["omega"]
    c1 = float(np.quantile(tb.max(1), 1 - ALPHA)); put("mtCrit", c1, "{:.2f}")
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    colours = [BLUE, RED, GREY, SKY, ROSE, "#7a5c2e"]
    for j, a in enumerate(ASSETS):
        sel = [i for i, (aa, _) in enumerate(names) if aa == a]
        ax.plot(LOOKS, r["t"][sel], marker="o", ms=3, lw=1.0, color=colours[j], label=ASSETS[a])
    ax.axhline(c1, color="#333333", lw=0.8, ls="--"); ax.axhline(1.645, color="#333333", lw=0.6, ls=":")
    ax.axhline(0, color="#333333", lw=0.4)
    ax.set_xlabel("moving average length (months)"); ax.set_ylabel("studentized mean return")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    save(fig, "fig22_snooping")
    return finish("snooping")


if __name__ == "__main__":
    main()
