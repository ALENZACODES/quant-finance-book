"""Sections 2.10 to 2.12: hedge fund fees, high-water marks, option and annuity views, fee layering.

Run on its own:  python code/ch02/hedge_fund_fees.py   (writes code/ch02/out/hedge_fund_fees.json)
run_all.py runs every script of the chapter and writes chapters/ch02-numbers.tex.
"""
from __future__ import annotations
import numpy as np
from numbers_ch02 import N, put, pct, word, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, SEED  # noqa: F401


# ============================================================ 2.10 hedge fund fees
def hedge_fund_section():
    g = [0.15, -0.10, 0.08, 0.20]
    r = L.hedge_fund_fees(g)
    for t in range(4):
        w = word(t + 1)
        for k, nm in [("open", "Open"), ("gross_nav", "Gross"), ("mgmt", "Mgmt"), ("pre", "Pre"),
                      ("inc", "Inc"), ("close", "Close"), ("hwm", "Hwm")]:
            put("hf" + nm + w, r[k][t])
        pct("hfNet" + w, r["net"][t])
    put("hfTotalFees", r["mgmt"].sum() + r["inc"].sum())
    gross_end = 100 * np.prod(1 + np.array(g))
    put("hfGrossEnd", gross_end); pct("hfGrossCagr", (gross_end / 100) ** 0.25 - 1); pct("hfNetCagr", (r["close"][-1] / 100) ** 0.25 - 1)
    nh = L.hedge_fund_fees(g, hwm=False)
    put("hfNoHwmIncThree", nh["inc"][2]); put("hfNoHwmTotal", nh["mgmt"].sum() + nh["inc"].sum())
    hh = L.hedge_fund_fees(g, hurdle=0.05, hurdle_type="hard"); hs = L.hedge_fund_fees(g, hurdle=0.05, hurdle_type="soft")
    put("hfHardIncOne", hh["inc"][0]); put("hfSoftIncOne", hs["inc"][0])
    put("hfHardIncFour", hh["inc"][3]); put("hfSoftIncFour", hs["inc"][3])
    # modified HWM, after-management-fee returns
    gm = [-0.25] + [0.12] * 5
    mod_rate, mod_mult = 0.10, 1.25
    trad = L.hedge_fund_fees(gm, mgmt=0.0); mod = L.hedge_fund_fees(gm, mgmt=0.0, modified=(mod_rate, mod_mult))
    target = recovery_target(100.0, 0.25, mod_mult)
    put("modTarget", target)
    for t in range(1, 6):
        put("modTradInc" + word(t + 1), trad["inc"][t]); put("modModInc" + word(t + 1), mod["inc"][t])
        put("modTradClose" + word(t + 1), trad["close"][t]); put("modModClose" + word(t + 1), mod["close"][t])
    put("modTradSum", trad["inc"].sum()); put("modModSum", mod["inc"].sum())
    fig = hwm_figure(trad, mod, target, mod_rate, mod_mult)
    save(fig, "fig07_hwm")


def recovery_target(mark, loss, mult):
    """NAV at which the modified mark reverts: the trough plus mult times the loss."""
    trough = mark * (1 - loss)
    return trough + mult * (mark - trough)


def hwm_figure(trad, mod, target, mod_rate, mod_mult):
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    yrs = np.arange(0, 7)
    ax[0].plot(yrs, np.r_[100, trad["close"]], "o-", ms=3, color=BLUE, label="NAV, traditional HWM")
    ax[0].plot(yrs, np.r_[100, mod["close"]], "s--", ms=3, color=RED, label="NAV, modified HWM")
    ax[0].axhline(100, color=GREY, lw=0.7, ls=":"); ax[0].axhline(target, color=ROSE, lw=0.7, ls=":", gid="target")
    ax[0].text(0.1, 101.5, "old high-water mark", fontsize=7, color=GREY)
    ax[0].text(0.1, target + 1.0, "recovery target", fontsize=7, color=ROSE)
    ax[0].set_xlabel("year"); ax[0].set_ylabel("NAV after fees"); ax[0].set_title("(a) A 25% loss and recovery")
    ax[0].legend(loc="lower right", fontsize=7.2)
    w = 0.38
    ax[1].bar(yrs[1:] - w / 2, trad["inc"], width=w, color=BLUE, label="traditional")
    ax[1].bar(yrs[1:] + w / 2, mod["inc"], width=w, color=RED, label=f"modified ({100 * mod_rate:.0f}% until {100 * mod_mult:.0f}% recovered)")
    ax[1].set_xlabel("year"); ax[1].set_ylabel("incentive fee"); ax[1].set_title("(b) Incentive fee by year")
    top = max(trad["inc"].max(), mod["inc"].max())
    ax[1].set_ylim(0, 1.45 * top)
    ax[1].legend(loc="upper left", fontsize=7.2)
    fig.tight_layout()
    return fig


def hf_fee_vectorised(G, mgmt=0.02, inc=0.20, hwm=True):
    """Terminal NAV after fees for a matrix of gross returns (paths x years), traditional HWM."""
    nav = np.full(G.shape[0], 100.0); mark = nav.copy(); fees = np.zeros(G.shape[0])
    for t in range(G.shape[1]):
        pre = nav * (1 + G[:, t]) - mgmt * nav
        base = mark if hwm else nav
        fee = inc * np.maximum(pre - base, 0.0)
        fees += mgmt * nav + fee
        nav = pre - fee
        mark = np.maximum(mark, nav)
    return nav, fees


def fee_share(rng, n=40_000, years=10, mu=0.08):
    sig = np.array([0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    Z = rng.standard_normal((n, years))
    out = {}
    for hwm in (True, False):
        s_ = []
        for s in sig:
            G = np.exp(np.log1p(mu) - 0.5 * s * s + s * Z) - 1        # mean gross return mu each year
            gross_end = 100 * np.prod(1 + G, axis=1)
            net_end, _ = hf_fee_vectorised(G, hwm=hwm)
            s_.append(1 - (net_end.mean() - 100) / (gross_end.mean() - 100))
        out[hwm] = np.array(s_)
    # deterministic benchmark: constant 8% gross, exact fees
    det = L.hedge_fund_fees([mu] * years)
    det_share = 1 - (det["close"][-1] - 100) / (100 * (1 + mu) ** years - 100)
    pct("shareDet", det_share, 1)
    for s, nm in [(0.10, "Ten"), (0.20, "Twenty"), (0.40, "Forty")]:
        i = int(np.where(np.isclose(sig, s))[0][0])
        pct("shareHwm" + nm, out[True][i], 1); pct("shareNoHwm" + nm, out[False][i], 1)
    put("shareN", n, "{:,.0f}")
    return sig, out, det_share


def option_view(rng):
    r, p = 0.03, 0.20
    m = np.linspace(0.5, 1.5, 201)
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.55))
    for s, col, ls in [(0.10, SKY, "--"), (0.20, BLUE, "-"), (0.40, RED, "-.")]:
        v = p * L.bs_call(100 * m, 100, 1.0, r, s)
        ax[0].plot(m, v, color=col, ls=ls, label=f"$\\sigma$ = {int(100*s)}%")
        ax[1].plot(m, p * L.bs_call_vega(100 * m, 100, 1.0, r, s) / 100, color=col, ls=ls)
    ax[0].set_xlabel("NAV / high-water mark"); ax[0].set_ylabel("fee value per 100 of HWM")
    ax[0].set_title("(a) One-year incentive fee", fontsize=9); ax[0].legend(loc="upper left", fontsize=7)
    ax[1].set_xlabel("NAV / high-water mark"); ax[1].set_ylabel("gain per 1 point of volatility")
    ax[1].set_title("(b) Vega of the fee", fontsize=9)
    for nm, (S, s) in {"AtSigTen": (100, .10), "AtSigTwenty": (100, .20), "DownSigTwenty": (80, .20),
                       "DownSigForty": (80, .40), "UpSigTwenty": (130, .20)}.items():
        put("opt" + nm, p * float(L.bs_call(S, 100, 1.0, r, s)), "{:.2f}")    # exact form: p calls on the gross NAV struck at M + m A, with S = (1 - m) A at the mark
    A = 100 / (1 - 0.02)
    put("optExactAtSigTwenty", p * float(L.bs_call(A, 100 + 0.02 * A, 1.0, r, .2)), "{:.2f}")
    put("optVegaAt", p * float(L.bs_call_vega(100, 100, 1, r, .2)) / 100, "{:.3f}")
    put("optVegaDown", p * float(L.bs_call_vega(80, 100, 1, r, .2)) / 100, "{:.3f}")
    put("optVegaUp", p * float(L.bs_call_vega(130, 100, 1, r, .2)) / 100, "{:.3f}")
    sig, out, det = fee_share(rng)
    ax[2].plot(100 * sig, 100 * out[True], "o-", ms=3, color=BLUE, label="with HWM")
    ax[2].plot(100 * sig, 100 * out[False], "s--", ms=3, color=RED, label="no HWM")
    ax[2].axhline(100 * det, color=GREY, lw=0.7, ls=":")
    ax[2].set_xlabel("volatility (% a year)"); ax[2].set_ylabel("fees as % of expected gross gain")
    ax[2].set_title("(c) Share of gross gain", fontsize=9); ax[2].legend(loc="upper left", fontsize=7)
    fig.tight_layout(w_pad=0.8); save(fig, "fig08_option_view")


def annuity_view():
    A, g, m, p, dsc, T = 1000.0, 0.10, 0.02, 0.20, 0.10, 10
    pv = L.fee_annuity_pv(A, g, m, p, dsc, T)
    pvm = L.fee_annuity_pv(A, g, m, 0.0, dsc, T)   # not the split; compute split by loop
    a, pm, pi = A, 0.0, 0.0
    for t in range(1, T + 1):
        mm = m * a; ii = p * (a * g - mm)
        pm += mm / (1 + dsc) ** t; pi += ii / (1 + dsc) ** t
        a = a * (1 + g) - mm - ii
    put("annIncOne", p * (A * g - m * A), "{:.1f}")   # first year's incentive fee
    put("annPV", pv, "{:.1f}"); put("annPVm", pm, "{:.1f}"); put("annPVi", pi, "{:.1f}")
    pct("annMgmtShare", pm / pv, 1); put("annPhi", 100 * (m + p * (g - m)), "{:.1f}")
    pct("annNetGrowth", (1 - p) * (g - m), 1)
    put("annPVperp", (m + p * (g - m)) * A / (dsc - (1 - p) * (g - m)), "{:,.0f}")
    put("annPVtwenty", L.fee_annuity_pv(A, g, m, p, dsc, 20), "{:.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    hs = np.arange(1, 31)
    ax[0].plot(hs, [L.fee_annuity_pv(A, g, m, p, dsc, h) for h in hs], color=BLUE, label="2% and 20%")
    ax[0].plot(hs, [L.fee_annuity_pv(A, g, m, 0, dsc, h) * 0 + sum(
        m * A * ((1 + g - m) ** (t - 1)) / (1 + dsc) ** t for t in range(1, h + 1)) for h in hs], color=GREY, ls=":",
        label="2% only, 0% incentive")
    ax[0].set_xlabel("years the fund survives"); ax[0].set_ylabel("PV of fees (on 1,000 of AUM)")
    ax[0].set_title("(a) Fees as an annuity"); ax[0].legend(loc="upper left", fontsize=7.2)
    gs = np.linspace(0.0, 0.2, 41)
    pms, pis = [], []
    for gg in gs:
        a, pm_, pi_ = A, 0.0, 0.0
        for t in range(1, T + 1):
            mm = m * a; ii = p * max(0.0, a * gg - mm)
            pm_ += mm / (1 + dsc) ** t; pi_ += ii / (1 + dsc) ** t
            a = a * (1 + gg) - mm - ii
        pms.append(pm_); pis.append(pi_)
    ax[1].stackplot(100 * gs, pms, pis, colors=[BLUE, ROSE], labels=["management fee", "incentive fee"], alpha=0.9)
    ax[1].set_xlabel("constant gross return (% a year)"); ax[1].set_ylabel("PV of ten years of fees")
    ax[1].set_title("(b) Composition of the annuity"); ax[1].legend(loc="upper left", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig09_annuity")


# ============================================================ 2.12 fund of funds layering
def fund_of_funds(rng, n=200_000, k=10, mu=0.10, s=0.15):
    rhos = np.linspace(0, 1, 11)
    res = {"fof": [], "single": [], "under": []}
    for rho in rhos:
        z0 = rng.standard_normal((n, 1)); e = rng.standard_normal((n, k))
        G = mu + s * (np.sqrt(rho) * z0 + np.sqrt(1 - rho) * e)
        under_fee = 0.015 + 0.20 * np.maximum(G - 0.015, 0)       # per unit invested in each fund
        net_u = (G - under_fee).mean(axis=1)                     # equal weights
        fof_fee = 0.01 + 0.10 * np.maximum(net_u - 0.01, 0)
        gross = G.mean(axis=1)
        single_fee = 0.015 + 0.20 * np.maximum(gross - 0.015, 0)
        res["under"].append(under_fee.mean() / mu); res["fof"].append((under_fee.mean() + fof_fee.mean()) / mu)
        res["single"].append(single_fee.mean() / mu)
    for key in res:
        res[key] = np.array(res[key])
    pct("fofTotalZero", res["fof"][0], 1); pct("fofTotalOne", res["fof"][-1], 1)
    pct("fofSingleZero", res["single"][0], 1); pct("fofUnderZero", res["under"][0], 1)
    pct("fofUnderOne", res["under"][-1], 1); pct("fofSingleOne", res["single"][-1], 1)
    # netting example: two funds, +30% and -10% gross
    a, b = 0.30, -0.10
    fa = 0.015 + 0.2 * max(a - 0.015, 0); fb = 0.015 + 0.2 * max(b - 0.015, 0)
    put("netFeeA", 100 * fa); put("netFeeB", 100 * fb); put("netGross", 100 * (a + b) / 2)
    put("netPaid", 100 * (fa + fb) / 2); put("netSingle", 100 * (0.015 + 0.2 * max((a + b) / 2 - 0.015, 0)))
    pct("netEffRate", ((fa + fb) / 2 - 0.015) / ((a + b) / 2 - 0.015), 1)
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    ax.plot(rhos, 100 * res["fof"], "o-", ms=3, color=RED, label="fund of funds: both layers")
    ax.plot(rhos, 100 * res["under"], "s--", ms=3, color=ROSE, label="underlying funds only")
    ax.plot(rhos, 100 * res["single"], "^-", ms=3, color=BLUE, label="one fund, 1.5% and 20% on the total")
    ax.set_xlabel("correlation between underlying funds"); ax.set_ylabel("expected fees as % of expected gross return")
    ax.set_title("Fee layering and netting risk"); ax.legend(loc="center right", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig10_fof")


def main():
    hedge_fund_section()
    option_view(np.random.default_rng(SEED + 2))
    annuity_view()
    fund_of_funds(np.random.default_rng(SEED + 3))
    return finish('hedge_fund_fees')


if __name__ == "__main__":
    main()
