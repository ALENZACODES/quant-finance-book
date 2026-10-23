"""Section 6.9, case study: can a five year commercial mortgage refinance at maturity? Every month-end
vintage from January 2001 to September 2021, priced off the five year Government of Canada yield,
each sized at origination by a loan to value and a coverage test and refinanced under the same tests,
then a pool of loans with correlated income shocks for the last vintage.

Run on its own:  python code/ch06/case_refinance.py   (writes code/ch06/out/case_refinance.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from numbers_ch06 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT  # noqa: F401

P = dict(ltv_max=0.65, dscr_min=1.25, cap_ref=0.06, y_ref=4.0, beta=0.5, spread=2.25, amort=25, g=0.02)
SIG_M, SIG_E, COST, LOANS = 0.10, 0.15, 0.10, 100
SIG = math.sqrt(SIG_M ** 2 + SIG_E ** 2)


def series():
    d = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", parse_dates=["date"]).set_index("date")
    return d["BD.CDN.5YR.DQ.YLD"].resample("ME").last().dropna()


def main():
    y = series()
    rows = pd.DataFrame(L.refi_vintages(y.values, **P)); rows.index = y.index[:len(rows)]
    put("csN", len(rows), "{:.0f}"); N["csFirst"] = rows.index[0].strftime("%B %Y"); N["csLast"] = rows.index[-1].strftime("%B %Y")
    N["csMatLast"] = y.index[-1].strftime("%B %Y")
    pct("csLtv", P["ltv_max"], 0); pct("csCap", P["cap_ref"], 0); put("csYRef", P["y_ref"], "{:.0f}")
    put("csSpread", P["spread"]); put("csAmort", P["amort"], "{:.0f}")
    pct("csG", P["g"], 0); put("csBeta", P["beta"], "{:.1f}"); put("csDscrMin", P["dscr_min"])
    # origination: both tests are applied, so no vintage starts below the coverage the refinancing needs
    assert (rows.dscr0 >= P["dscr_min"] - 1e-12).all() and (rows.ltv0 <= P["ltv_max"] + 1e-12).all()
    pct("csCapZeroMin", rows.cap0.min(), 2); pct("csCapZeroMax", rows.cap0.max(), 2)
    pct("csLtvZeroMin", rows.ltv0.min(), 1); pct("csLtvZeroMax", rows.ltv0.max(), 1)
    put("csDscrBindsZero", int(rows.dscr_binds0.sum()), "{:.0f}")
    # the earlier design, 65% of value at a fixed 6% cap rate with no coverage test at origination
    old = pd.DataFrame(L.refi_vintages(y.values, **P, size_at_origination=False, fixed_cap=True))
    put("csOldBelow", int((old.dscr0 < P["dscr_min"]).sum()), "{:.0f}")
    ycut = L.optimize.brentq(lambda v: 100 * P["cap_ref"] / (100 * P["ltv_max"] * L.mortgage_constant(v + P["spread"], P["amort"])) - P["dscr_min"], 0.0, 10.0)
    put("csOldYCut", ycut)
    fails = rows[rows.cover < 1]
    put("csFails", len(fails), "{:.0f}")
    if len(fails):
        N["csFailFirst"] = fails.index[0].strftime("%B %Y"); N["csFailLast"] = fails.index[-1].strftime("%B %Y")
        put("csFailYMax", fails.y0.max())
    put("csCoverMed", rows.cover.median(), "{:.3f}"); put("csCoverMin", rows.cover.min(), "{:.3f}")
    N["csCoverMinDate"] = rows.cover.idxmin().strftime("%B %Y")
    put("csLtvBinds", int(rows.ltv_binds.sum()), "{:.0f}"); put("csDscrBindsOne", int((~rows.ltv_binds).sum()), "{:.0f}")
    last = rows.iloc[-1]
    put("csLastYZero", last.y0); put("csLastYOne", last.y1); put("csLastB", last.B); put("csLastCover", last.cover, "{:.3f}")
    put("csLastDscrOne", last.dscr1); put("csLastLtvOne", 100 * last.ltv1, "{:.1f}"); put("csLastCapOne", 100 * last.cap1)
    put("csLastCapZero", 100 * last.cap0); put("csLastLoan", last.loan, "{:.1f}"); put("csLastDscrZero", last.dscr0)
    put("csLastKZero", 100 * L.mortgage_constant(last.y0 + P["spread"], P["amort"])); put("csLastKOne", 100 * last.k1)
    pct("csSigM", SIG_M, 0); pct("csSigE", SIG_E, 0); pct("csSig", SIG, 1); put("csRho", SIG_M ** 2 / SIG ** 2, "{:.3f}")
    rows["pd"] = [L.refi_pd(c, SIG) for c in rows.cover]
    pct("csPdMed", rows.pd.median(), 1); pct("csPdLast", rows.pd.iloc[-1], 1); pct("csPdMax", rows.pd.max(), 1)
    N["csPdMaxDate"] = rows.pd.idxmax().strftime("%B %Y")
    assert rows.pd.idxmax() == rows.cover.idxmin()
    pre = rows[rows.index < "2020-01-01"]; pct("csPdPreMed", pre.pd.median(), 1)
    # compounding: the same loans with the rate read as compounded monthly instead of semi-annually
    rm = pd.DataFrame(L.refi_vintages(y.values, **P, comp="monthly"))
    put("csMonthlyFails", int((rm.cover < 1).sum()), "{:.0f}")
    put("csConvMax", 100 * max(r + P["spread"] - 1200 * L.monthly_rate(r + P["spread"]) for r in y.values), "{:.1f}")
    # sensitivity of the last vintage to the cap rate pass-through
    for b, nm in [(0.0, "Zero"), (1.0, "One")]:
        r = L.refi_vintages(y.values, **dict(P, beta=b))[-1]
        assert r["dscr_binds0"] == (b > 0) and not r["ltv_binds"]   # the prose: which test binds
        put(f"csCoverBeta{nm}", r["cover"], "{:.3f}"); pct(f"csPdBeta{nm}", L.refi_pd(r["cover"], SIG), 1)
    # expected loss for the last vintage, closed form and by simulation of a pool
    a = (1 - COST) * last.V1
    el = L.refi_el(last.B, a, last.cover, SIG)
    put("csEl", 1e4 * el / last.B, "{:.3f}"); pct("csCost", COST, 0)
    rng = rng_for(7); scen = 100000
    M = rng.standard_normal(scen)[:, None]; E = rng.standard_normal((scen, LOANS))
    X = np.exp(SIG_M * M + SIG_E * E - 0.5 * SIG ** 2)
    fail = X < 1 / last.cover
    loss = np.where(fail, np.maximum(last.B - a * X, 0), 0.0) / last.B
    per = loss.mean(axis=1)
    put("csMcEl", 1e4 * per.mean(), "{:.3f}"); put("csMcElSE", 1e4 * per.std(ddof=1) / math.sqrt(scen), "{:.3f}")
    put("csMcElZ", (per.mean() - el / last.B) / (per.std(ddof=1) / math.sqrt(scen)), "{:.1f}")
    nf = fail.sum(axis=1)
    pmf = L.pool_pmf(LOANS, rows.pd.iloc[-1], SIG_M ** 2 / SIG ** 2)
    emp = np.bincount(nf, minlength=LOANS + 1) / scen
    put("csMcScen", scen, "{:,.0f}"); put("csLoans", LOANS, "{:.0f}")
    put("csMcMeanFail", nf.mean(), "{:.2f}"); put("csMcMeanFailSE", nf.std(ddof=1) / math.sqrt(scen), "{:.3f}")
    put("csMeanFail", LOANS * rows.pd.iloc[-1], "{:.2f}")
    pct("csPoolTail", pmf[int(np.ceil(0.25 * LOANS)):].sum(), 1); pct("csPoolTailMc", (nf >= 0.25 * LOANS).mean(), 1)
    # loss on a claim that absorbs pool losses above 1% of the pool balance, up to the whole pool.
    # Its expected loss is a far-tail quantity, so it gets its own, larger simulation (stream 9),
    # run in blocks; the 100,000 scenarios above leave a standard error near 0.06 bp.
    put("csAttach", 1.0, "{:.0f}"); put("csLastVOne", last.V1); put("csEta", a)
    rs = rng_for(9); blocks, bsz = 20, 100000; acc = []
    for _ in range(blocks):
        Mb = rs.standard_normal(bsz)[:, None]; Xb = np.exp(SIG_M * Mb + SIG_E * rs.standard_normal((bsz, LOANS)) - 0.5 * SIG ** 2)
        pl = (np.where(Xb < 1 / last.cover, np.maximum(last.B - a * Xb, 0), 0.0) / last.B).mean(axis=1)
        acc.append(np.minimum(np.maximum(pl - 0.01, 0), 0.99))      # loss as a share of the pool balance
    sn = np.concatenate(acc); ns = len(sn)
    put("csSenScen", ns, "{:,.0f}")
    put("csSenEl", 1e4 * sn.mean(), "{:.2f}"); put("csSenElSE", 1e4 * sn.std(ddof=1) / math.sqrt(ns), "{:.3f}")
    put("csSenElOwn", 1e4 * sn.mean() / 0.99, "{:.2f}")
    put("csSenPHit", 100 * (sn > 0).mean(), "{:.1f}")
    # screener grade of the last vintage at origination, with no cap shift and with the realised one
    deal = dict(loan=last.loan, value=100.0, noi=100 * last.cap0, rate=last.y0 + P["spread"], amort=25, termYrs=5,
                capRate=100 * last.cap0,
                exitCapShift=0.0, noiGrowth=2, fees=0, sponsorScore=3)
    put("csSponsor", deal["sponsorScore"], "{:.0f}")
    t0, g0, m0, _ = L.screener_score(deal)
    t1, g1, m1, _ = L.screener_score(dict(deal, exitCapShift=100 * (last.cap1 - last.cap0)))
    put("csScreenExitLtv", m1["exitLtv"], "{:.1f}"); put("csScore", t0, "{:.0f}"); N["csGrade"] = g0; put("csScoreReal", t1, "{:.0f}"); N["csGradeReal"] = g1
    put("csRateZero", last.y0 + P["spread"])
    # figures
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 4.6), sharex=True)
    axes[0].plot(y.index, y.values, color=BLUE, lw=1.0)
    axes[0].set_ylabel("5 year GoC yield (%)")
    axes[1].plot(rows.index, rows.cover, color=BLUE, lw=1.1, label="refinancing cover (left)")
    axes[1].axhline(1, color="#333333", lw=0.6)
    axes[1].set_ylabel("largest new loan / balloon"); axes[1].set_xlabel("origination month")
    ax2 = axes[1].twinx(); ax2.plot(rows.index, 100 * rows.pd, color=RED, ls="--", lw=1.1, label="failure probability (right)")
    ax2.set_ylabel("probability (%)"); ax2.grid(False); ax2.spines["right"].set_visible(True); ax2.set_ylim(0, 60)
    h1, l1 = axes[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    fig.subplots_adjust(hspace=0.28)
    save(fig, "fig11_refinance")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    k = np.arange(LOANS + 1)
    ax.bar(k, emp, width=0.85, color=PALE, edgecolor=SKY, lw=0.4, label=f"simulation, {scen:,} scenarios")
    pct("csAboveSeventy", pmf[71:].sum(), 1)   # the mass the earlier cut at 70 hid; every count is drawn now
    ax.plot(k, pmf, color=RED, lw=1.2, label=r"one-factor formula, $\rho=%.3f$" % (SIG_M ** 2 / SIG ** 2))
    ax.set_xlim(-1, LOANS + 1)
    ax.set_xlabel(f"loans out of {LOANS} that cannot refinance"); ax.set_ylabel("probability")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)   # above the axes, clear of the bars
    save(fig, "fig12_pool")
    RAW["csChi"] = float(np.max(np.abs(emp - pmf)))
    return finish("case_refinance")


if __name__ == "__main__":
    main()
