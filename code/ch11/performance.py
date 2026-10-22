"""Section 11.2: what a fund's numbers measure. A simulated population of buyout funds gives the
interim J-curve of IRR, DPI and TVPI and how well early IRRs rank funds; the same funds give the
Kaplan and Schoar PME and direct alpha against their own index, and the bias of the PME when a fund's
beta is not one. The CalPERS portfolio of 31 March 2026 gives multiples by vintage and the duration
implied by each fund's IRR and multiple. Writes fig04_jcurve.png, fig05_beta.png, fig06_calpers.png.
Also writes fig17_transition.png, quartile transitions of the same simulated funds, and fig18_duration.png,
lines of equal implied duration."""
import math
import numpy as np
import pandas as pd
from scipy import stats
import numbers_ch11 as R
from numbers_ch11 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, put, pct, finish, rng_for

DT = 0.25; LIFE = 12; NQ = int(LIFE / DT)
MU_M, SIG_M = 0.07, 0.16          # index drift and volatility
ALPHA, BETA, SIG_E = 0.03, 1.2, 0.35
N_DEALS, N_FUNDS = 12, 4000
FEE_INV, FEE_POST, CARRY = 0.02, 0.015, 0.20


def simulate(rng, alpha=ALPHA, beta=BETA, n_funds=N_FUNDS):
    """Quarterly LP cash flows and reported NAV of simulated funds, each with its own index path."""
    out = []
    for _ in range(n_funds):
        lr = (MU_M - 0.5 * SIG_M ** 2) * DT + SIG_M * math.sqrt(DT) * rng.standard_normal(NQ)
        lnI = np.concatenate([[0.0], np.cumsum(lr)])
        q_in = rng.integers(0, 20, N_DEALS)
        q_out = np.minimum(q_in + 4 + rng.geometric(1 / 14, N_DEALS), NQ)
        cost = 100.0 / N_DEALS * 0.9                       # 90% of commitments invested; five years of fees use the rest
        e = np.cumsum(SIG_E * math.sqrt(DT) * rng.standard_normal((N_DEALS, NQ + 1)), axis=1)
        e -= e[np.arange(N_DEALS), q_in][:, None]
        contrib = np.zeros(NQ + 1); gross = np.zeros(NQ + 1); nav = np.zeros(NQ + 1); cost_out = np.zeros(NQ + 1)
        for j in range(N_DEALS):
            contrib[q_in[j]] += cost
            q = np.arange(NQ + 1)
            v = cost * np.exp((alpha - 0.5 * SIG_E ** 2) * DT * (q - q_in[j])
                              + beta * (lnI - lnI[q_in[j]]) - 0.5 * beta * (beta - 1) * SIG_M ** 2 * DT * (q - q_in[j])
                              + e[j])
            gross[q_out[j]] += v[q_out[j]]
            live = (q >= q_in[j]) & (q < q_out[j])
            mark = np.where(q - q_in[j] < 4, cost, v)        # held at cost for its first year
            nav[live] += mark[live]; cost_out[live] += cost
        fees = np.where(np.arange(NQ + 1) < 20, FEE_INV * 100 * DT, FEE_POST * cost_out * DT)
        fees[-1] = 0.0
        out.append(dict(lnI=lnI, contrib=contrib + fees, gross=gross, nav=nav))
    return out


def lp_view(f, age_q):
    """LP flows to quarter age_q, net of fees, with carry of 20% of any profit on distributions plus NAV
    taken pro rata from the distributions and the NAV, and the NAV at age_q as a final flow."""
    C = f["contrib"][: age_q + 1]; D = f["gross"][: age_q + 1].copy(); nav = f["nav"][age_q]
    profit = D.sum() + nav - C.sum()
    carry = CARRY * max(profit, 0.0)
    share = 1 - carry / (D.sum() + nav) if carry > 0 else 1.0   # carry taken pro rata from every payment
    D *= share; nav_net = nav * share
    return C, D, nav_net


def interim(f, age_q):
    C, D, nav = lp_view(f, age_q)
    t = DT * np.arange(age_q + 1); cf = D - C; cf[-1] += nav
    try:
        r = L.irr(t, cf, -0.95, 5.0)
    except ValueError:
        r = np.nan
    return r, D.sum() / C.sum(), (D.sum() + nav) / C.sum()


def jcurve(funds):
    ages = np.arange(2, NQ + 1)
    irr = np.full((len(funds), len(ages)), np.nan); dpi = np.zeros_like(irr); tvpi = np.zeros_like(irr)
    for i, f in enumerate(funds):
        for k, a in enumerate(ages):
            irr[i, k], dpi[i, k], tvpi[i, k] = interim(f, a)
    med = np.nanmedian(irr, 0); lo = np.nanpercentile(irr, 10, 0); hi = np.nanpercentile(irr, 90, 0)
    yrs = DT * ages
    for y, nm in [(1, "One"), (2, "Two"), (4, "Four"), (12, "End")]:
        k = int(np.where(np.isclose(yrs, y))[0][0])
        pct(f"jcIrr{nm}", med[k], 1); put(f"jcDpi{nm}", np.median(dpi[:, k]), "{:.2f}"); put(f"jcTvpi{nm}", np.median(tvpi[:, k]), "{:.2f}")
    pos = yrs[np.argmax(med > 0)]; put("jcCross", pos, "{:.2f}")
    pct("jcIrrLoTwo", lo[int(np.where(np.isclose(yrs, 2))[0][0])], 1); pct("jcIrrHiTwo", hi[int(np.where(np.isclose(yrs, 2))[0][0])], 1)
    final = irr[:, -1]
    for y, nm in [(3, "Three"), (5, "Five"), (8, "Eight")]:
        k = int(np.where(np.isclose(yrs, y))[0][0])
        ok = np.isfinite(irr[:, k]) & np.isfinite(final)
        put(f"jcRank{nm}", stats.spearmanr(irr[ok, k], final[ok])[0], "{:.2f}")
        # share of funds in the top quartile at the end that were in the top quartile at age y
        top_end = final >= np.nanpercentile(final, 75); top_y = irr[:, k] >= np.nanpercentile(irr[:, k], 75)
        pct(f"jcTop{nm}", np.mean(top_y[top_end]), 0)
    put("jcFunds", len(funds), "{:,.0f}"); put("jcDealsPer", N_DEALS, "{:.0f}"); put("jcLife", LIFE, "{:.0f}")
    pct("jcAlpha", ALPHA, 0); put("jcBeta", BETA, "{:.1f}"); pct("jcSigE", SIG_E, 0); pct("jcMuM", MU_M, 0); pct("jcSigM", SIG_M, 0)
    pct("jcMuLessVar", MU_M - SIG_M ** 2, 2)   # positive, so beta above one raises the expected PME here
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9))
    ax[0].fill_between(yrs, 100 * lo, 100 * hi, color=PALE, label="10th to 90th percentile")
    ax[0].plot(yrs, 100 * med, color=BLUE, label="median"); ax[0].axhline(0, color="#333333", lw=0.6)
    ax[0].set_ylim(-60, 60); ax[0].set_xlabel("age of the fund (years)"); ax[0].set_ylabel("interim net IRR (%)")
    ax[0].legend(loc="lower right", fontsize=7.5)
    ax[1].plot(yrs, np.median(tvpi, 0), color=BLUE, label="TVPI"); ax[1].plot(yrs, np.median(dpi, 0), color=RED, ls="--", label="DPI")
    ax[1].axhline(1, color="#333333", lw=0.6); ax[1].set_xlabel("age of the fund (years)"); ax[1].set_ylabel("median multiple")
    ax[1].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); R.save(fig, "fig04_jcurve")
    transitions(irr, yrs)
    return final


def quartile(x):
    """Quartile 1 to 4 by the same 25th, 50th and 75th percentile cuts as the text; 0 when x is missing."""
    q = np.zeros(len(x), int); ok = np.isfinite(x)
    cuts = np.nanpercentile(x, [25, 50, 75])
    q[ok] = 1 + (x[ok] >= cuts[0]) + (x[ok] >= cuts[1]) + (x[ok] >= cuts[2])
    return q


def largest_remainder(X):
    """Round each row of X, which adds to 100, to whole numbers that still add to 100."""
    F = np.floor(X).astype(int)
    for a in range(X.shape[0]):
        short = int(round(X[a].sum())) - F[a].sum()
        F[a, np.argsort(-(X[a] - F[a]), kind="stable")[:short]] += 1
    return F


def transitions(irr, yrs):
    """Share of the funds in each final IRR quartile that were in each interim IRR quartile at three and
    five years, from the same simulated funds as the J-curve."""
    qf = quartile(irr[:, -1])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", BLUE])
    for ax, (y, nm) in zip(axes, ((3, "Three"), (5, "Five"))):
        k = int(np.where(np.isclose(yrs, y))[0][0]); qi = quartile(irr[:, k])
        M = np.array([[np.mean(qi[qf == a] == b) for b in range(1, 5)] for a in range(1, 5)])
        R.RAW[f"trMat{nm}"] = M.tolist()
        P = largest_remainder(100 * M)                 # whole percents, each row adds to 100
        R.RAW[f"trInt{nm}"] = P.tolist()
        put(f"trMiss{nm}", np.sum(qi == 0), "{:.0f}")
        for key, (a, b) in (("TopTop", (3, 3)), ("TopBottom", (3, 0)), ("BottomBottom", (0, 0)),
                            ("BottomTop", (0, 3))):
            put(f"tr{key}{nm}", P[a, b], "{:.0f}")
        ax.imshow(100 * M, origin="lower", cmap=cmap, vmin=0, vmax=75, aspect="auto")
        for a in range(4):
            for b in range(4):
                ax.text(b, a, f"{P[a, b]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[a, b] > 0.45 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(["1", "2", "3", "4"]); ax.set_yticks(range(4))
        ax.set_yticklabels(["1", "2", "3", "4"]); ax.grid(False)
        ax.set_xlabel(f"quartile of the IRR at {y} years"); ax.set_title(f"{y} years", fontsize=8)
    axes[0].set_ylabel("quartile of the final IRR")
    fig.tight_layout(); R.save(fig, "fig17_transition")


def duration_figure():
    """Lines of equal implied duration D = ln(TVPI) / ln(1 + IRR) in the plane of IRR and TVPI."""
    irr = np.linspace(0.005, 0.40, 200)
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    for D, col, ls in ((1, RED, "-"), (2, BLUE, "-"), (4, BLUE, "--"), (8, SKY, "-.")):
        ax.plot(100 * irr, (1 + irr) ** D, color=col, ls=ls, label=rf"$\mathcal{{D}}$ = {D} year" + ("s" if D > 1 else ""))
    ex_irr, ex_tvpi = 0.25, 1.2; d_ex = math.log(ex_tvpi) / math.log1p(ex_irr)
    put("duEx", d_ex, "{:.2f}")
    ax.plot([100 * ex_irr], [ex_tvpi], "o", color=RED, ms=4)
    ax.annotate(rf"IRR 25%, TVPI 1.2: $\mathcal{{D}}$ = {d_ex:.2f}", (100 * ex_irr, ex_tvpi), xytext=(6, -12),
                textcoords="offset points", fontsize=7)
    # an IRR of 10% with a TVPI of 2 needs the money out for longer
    put("duTen", math.log(2.0) / math.log(1.10), "{:.1f}")
    ax.set_ylim(1, 3.2); ax.set_xlim(0, 40)
    ax.set_xlabel("IRR (%)"); ax.set_ylabel("TVPI")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7.5)
    R.save(fig, "fig18_duration")


def pme(funds):
    """KS-PME and direct alpha of each fund at the end of its life against its own index."""
    T = LIFE; ks = []; da = []
    for f in funds:
        C, D, nav = lp_view(f, NQ)
        I = np.exp(f["lnI"]); idx = lambda s: I[int(round(s / DT))]
        t = DT * np.arange(NQ + 1)
        ks.append(L.ks_pme(t, C, D, nav, idx, T)); da.append(L.direct_alpha(t, C, D, nav, idx, T))
    ks = np.array(ks); da = np.array(da)
    agree = np.mean((ks > 1) == (da > 0))
    R.RAW["pkAgree"] = float(agree); assert agree == 1.0
    put("pkMed", np.median(ks), "{:.2f}"); pct("pkAbove", np.mean(ks > 1), 0); pct("pkDaMed", np.median(da), 1)
    put("pkCorr", stats.spearmanr(ks, da)[0], "{:.3f}")
    # with r = 0, Proposition (beta and the PME) adds (beta - 1)(mu - sigma^2) a year to the expected PME exponent
    pct("pkBetaBoost", (BETA - 1) * (MU_M - SIG_M ** 2), 2)
    return ks, da


def beta_bias(rng):
    """E[KS-PME] for one contribution and one exit after T years, alpha zero, against beta."""
    T = 5.0; betas = np.linspace(0, 2, 41)
    m = pd.read_csv(ROOT / "data" / "real_assets_monthly.csv", parse_dates=["date"])
    rf = float(m.loc[m.date == "2026-09-01", "gs10"].iloc[0]) / 100
    erp_t = pd.read_csv(ROOT / "data" / "damodaran_erp.csv")
    erp = float(erp_t.loc[erp_t.year == 2025, "erp_fcfe"].iloc[0])
    exact = np.array([L.pme_beta_mean(0.0, b, MU_M, SIG_M, T, rf) for b in betas])
    sim_b = np.array([0.5, 1.0, 1.5, 2.0]); sims = []; ses = []
    x = (MU_M - 0.5 * SIG_M ** 2) * T + SIG_M * math.sqrt(T) * rng.standard_normal(400_000)
    for b in sim_b:
        y = L.levered_index_payoff(b, x, rf, SIG_M, T) * np.exp(-x)   # distribution over the index return
        sims.append(y.mean()); ses.append(y.std(ddof=1) / math.sqrt(len(y)))
    for b, nm in [(0.5, "Half"), (1.5, "OneHalf"), (2.0, "Two")]:
        put(f"pkBeta{nm}", L.pme_beta_mean(0.0, b, MU_M, SIG_M, T, rf), "{:.3f}")
    pct("pkPrem", MU_M - rf, 2); pct("pkVar", SIG_M ** 2, 2); pct("pkMu", MU_M, 0); pct("pkSig", SIG_M, 0)
    pct("pkErp", erp, 2); put("pkBetaTwoErp", L.pme_beta_mean(0.0, 2.0, rf + erp, SIG_M, T, rf), "{:.3f}")
    R.RAW["pkBetaSimZ"] = [float((s - L.pme_beta_mean(0.0, b, MU_M, SIG_M, T, rf)) / se) if se > 0 else 0.0
                           for s, b, se in zip(sims, sim_b, ses)]
    put("pkBetaSimTwo", sims[-1], "{:.3f}"); put("pkBetaSimTwoSE", ses[-1], "{:.3f}")
    put("pkT", T, "{:.0f}")
    # the alpha that a beta of 1.5 fakes: the PME equals that of a beta one fund with this alpha
    pct("pkFakeAlpha", math.log(L.pme_beta_mean(0.0, 1.5, MU_M, SIG_M, T, rf)) / T, 3)
    # the rate convention: the monthly GS10 yield is used as a continuously compounded rate; for comparison,
    # the same yield read as semiannual bond-equivalent, converted to continuous compounding
    rc = 2 * math.log(1 + rf / 2); pct("pkRfCont", rc, 2)
    put("pkBetaTwoCont", L.pme_beta_mean(0.0, 2.0, MU_M, SIG_M, T, rc), "{:.3f}")
    assert max(abs(z) for z in R.RAW["pkBetaSimZ"]) < 4
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(betas, exact, color=BLUE, label="formula, alpha = 0")
    ax.errorbar(sim_b, sims, yerr=2 * np.array(ses), fmt="o", color=RED, ms=3.5, label="simulation, two standard errors")
    ax.axhline(1, color="#333333", lw=0.6); ax.set_xlabel(r"beta of the fund on the index, $\beta$")
    ax.set_ylabel("expected KS-PME"); ax.legend(loc="lower left")
    R.save(fig, "fig05_beta")


def calpers():
    d = pd.read_csv(ROOT / "data" / "calpers_pe_funds.csv")
    d["tvpi"] = d.total_value / d.cash_in; d["dpi"] = d.cash_out / d.cash_in
    put("cvFunds", len(d), "{:.0f}"); put("cvCommitted", d.committed.sum() / 1e9, "{:.1f}")
    put("cvNav", (d.total_value - d.cash_out).sum() / 1e9, "{:.1f}")
    put("cvFirstVintage", d.vintage.min(), "{:.0f}"); put("cvNm", (d.not_meaningful).sum(), "{:.0f}")
    put("cvNmFrom", d.vintage[d.not_meaningful].min(), "{:.0f}")
    # check: the published multiple is total value over cash in, to one decimal
    m = d.dropna(subset=["multiple"])
    R.RAW["cvMultGap"] = float((m.tvpi - m.multiple).abs().max()); assert R.RAW["cvMultGap"] <= 0.05 + 1e-9
    g = d[d.cash_in > 0].groupby("vintage")
    pooled = g.apply(lambda x: pd.Series(dict(tvpi=x.total_value.sum() / x.cash_in.sum(), dpi=x.cash_out.sum() / x.cash_in.sum(), n=len(x))))
    pooled = pooled[pooled.n >= 5]
    for v, nm in ((2007, "Seven"), (2015, "Fifteen"), (2019, "Nineteen"), (2023, "TwentyThree")):
        put(f"cvTvpi{nm}", pooled.loc[v, "tvpi"], "{:.2f}"); put(f"cvDpi{nm}", pooled.loc[v, "dpi"], "{:.2f}")
        put(f"cvN{nm}", pooled.loc[v, "n"], "{:.0f}")
    old = d[(d.vintage <= 2012)]
    put("cvOldN", len(old), "{:.0f}"); put("cvOldTvpi", old.total_value.sum() / old.cash_in.sum(), "{:.2f}")
    pct("cvOldRvpi", (old.total_value - old.cash_out).sum() / old.total_value.sum(), 0)
    # implied duration: ln(TVPI) / ln(1 + IRR), for funds with a meaningful positive IRR
    e = d[(~d.not_meaningful) & d.net_irr.notna() & (d.net_irr > 0.005) & (d.tvpi > 1)].copy()
    e["dur"] = np.log(e.tvpi) / np.log1p(e.net_irr)
    e["bound"] = 2026.25 - e.vintage
    put("cvDurN", len(e), "{:.0f}"); put("cvDurMed", e.dur.median(), "{:.1f}")
    over = e[e.dur > e.bound]
    pct("cvDurRatioMax", (e.dur / e.bound).max(), 0)
    put("cvDurOver", len(over), "{:.0f}")
    pct("cvDurShareMax", (e.dur / e.bound).median(), 0)
    R.RAW["cvDurOverFunds"] = list(over.fund)
    for lo_, hi_, nm in [(2005, 2008, "Pre"), (2015, 2018, "Mid")]:
        s = e[(e.vintage >= lo_) & (e.vintage <= hi_)]
        put(f"cvDur{nm}", s.dur.median(), "{:.1f}"); put(f"cvDurN{nm}", len(s), "{:.0f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.9))
    ax[0].plot(pooled.index, pooled.tvpi, "o-", color=BLUE, ms=3, label="TVPI")
    ax[0].plot(pooled.index, pooled.dpi, "s--", color=RED, ms=3, label="DPI")
    ax[0].axhline(1, color="#333333", lw=0.6); ax[0].set_xlabel("vintage year"); ax[0].set_ylabel("pooled multiple")
    ax[0].legend(loc="upper right", fontsize=7.5)
    ax[1].scatter(e.vintage + 0.15 * (np.arange(len(e)) % 5 - 2) / 2, e.dur, s=6, color=SKY, label="funds")
    vv = np.arange(e.vintage.min(), e.vintage.max() + 1)
    ax[1].plot(vv, 2026.25 - vv, color=RED, lw=1.0, label="years since 1 January of vintage")
    ax[1].set_xlabel("vintage year"); ax[1].set_ylabel("implied duration (years)"); ax[1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); R.save(fig, "fig06_calpers")


def snippet():
    """The toy fund of snippets/pme_alpha.py, through the library."""
    t = [0.0, 1.0, 2.0, 4.0, 6.0]; C = [40.0, 35.0, 25.0, 0.0, 0.0]; D = [0.0, 0.0, 20.0, 70.0, 90.0]
    I = dict(zip(t, [100.0, 108.0, 101.0, 125.0, 140.0]))
    put("pkSnipPme", L.ks_pme(t, C, D, 30.0, I.get, 6.0), "{:.4f}"); pct("pkSnipAlpha", L.direct_alpha(t, C, D, 30.0, I.get, 6.0), 2)
    pct("pkSnipIrr", L.irr(t + [6.0], list(np.array(D) - np.array(C)) + [30.0]), 2)
    put("pkSnipTvpi", (sum(D) + 30) / sum(C), "{:.2f}")


def main():
    snippet()
    funds = simulate(rng_for(2))
    jcurve(funds); pme(funds); beta_bias(rng_for(3)); calpers(); duration_figure()
    # the same random draws with a beta of one: how far the funds fall without the beta boost
    ks1 = []
    for f in simulate(rng_for(2), beta=1.0):
        C, D, nav = lp_view(f, NQ); I = np.exp(f["lnI"])
        ks1.append(L.ks_pme(DT * np.arange(NQ + 1), C, D, nav, lambda s: I[int(round(s / DT))], LIFE))
    ks1 = np.array(ks1); put("pkMedBetaOne", np.median(ks1), "{:.2f}"); pct("pkAboveBetaOne", np.mean(ks1 > 1), 0)
    return finish("performance")


if __name__ == "__main__":
    main()
