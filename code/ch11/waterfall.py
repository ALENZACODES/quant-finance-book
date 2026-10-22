"""Section 11.1: how a fund pays its managers. A stylised ten year buyout fund with management fees,
its whole fund (European) waterfall year by year, the map from gross to net returns over a range of
gross multiples, and the timing value of deal by deal (American) carry with an interest free clawback
across simulated funds. Writes fig01_waterfall.png, fig02_gross_net.png and fig03_timing.png.
Also writes fig15_catchup.png (catch-up rates) and fig16_carrypaths.png (a toy fund of three deals)."""
import math
import numpy as np
import numbers_ch11 as R
from numbers_ch11 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, put, pct, finish, rng_for

COMMIT = 100.0
FEE_INV, FEE_POST, INV_YEARS = 0.02, 0.015, 5      # fee on commitments, then on unrealised cost
H, CARRY, CATCH = 0.08, 0.20, 1.0
# eight deals of 10 each: year bought, year sold, gross multiple of cost (an illustration)
DEALS = [(0, 3, 2.8), (1, 5, 0.0), (1, 5, 2.2), (2, 6, 1.6), (2, 6, 3.2), (3, 7, 0.8), (3, 8, 2.6), (4, 9, 2.0)]
COST = 10.0
YEARS = np.arange(0, 11)


def fund_flows(scale=1.0, fees=True):
    """Annual LP contributions and gross distributions of the stylised fund, with every deal multiple
    multiplied by scale."""
    C = np.zeros(len(YEARS)); X = np.zeros(len(YEARS)); invested = np.zeros(len(YEARS))
    for b, s, m in DEALS:
        C[b] += COST; X[s] += COST * m * scale
        invested[b:s] += COST
    if fees:
        for y in YEARS[:-1]:
            C[y] += FEE_INV * COMMIT if y < INV_YEARS else FEE_POST * invested[y]
    return C, X, invested


def gross_irr(scale):
    C = np.zeros(len(YEARS)); X = np.zeros(len(YEARS))
    for b, s, m in DEALS:
        C[b] += COST; X[s] += COST * m * scale
    return L.irr(YEARS, X - C)


def net_irr(scale, carry=True, k=CATCH):
    C, X, _ = fund_flows(scale)
    if not carry:
        return L.irr(YEARS, X - C)
    w = L.european_waterfall(YEARS, C, X, H, CARRY, k)
    lp = w["roc"] + w["pref"] + w["lp_split"]
    lp[-1] += w["clawback"]
    return L.irr(YEARS, lp - C)


def stylised():
    C, X, inv = fund_flows()
    w = L.european_waterfall(YEARS, C, X, H, CARRY, CATCH)
    put("wfCommit", COMMIT, "{:.0f}"); put("wfDeals", len(DEALS), "{:.0f}"); put("wfCost", COST, "{:.0f}")
    pct("wfFeeInv", FEE_INV, 0); pct("wfFeePost", FEE_POST, 1); put("wfInvYears", INV_YEARS, "{:.0f}")
    pct("wfH", H, 0); pct("wfCarry", CARRY, 0)
    put("wfPaidIn", C.sum(), "{:.1f}"); put("wfFees", C.sum() - COST * len(DEALS), "{:.1f}")
    put("wfGross", X.sum(), "{:.1f}"); put("wfGrossMoic", X.sum() / (COST * len(DEALS)), "{:.2f}")
    put("wfRoc", w["roc"].sum(), "{:.1f}"); put("wfPref", w["pref"].sum(), "{:.2f}")
    put("wfCatch", w["catch"].sum(), "{:.2f}"); put("wfGpSplit", w["gp_split"].sum(), "{:.2f}")
    put("wfLpSplit", w["lp_split"].sum(), "{:.2f}"); put("wfGpTotal", w["gp_total"], "{:.2f}")
    put("wfLpTotal", w["lp_total"], "{:.1f}"); put("wfClaw", w["clawback"], "{:.2f}")
    profit = X.sum() - C.sum(); put("wfProfit", profit, "{:.1f}")
    pct("wfGpShare", w["gp_total"] / profit, 2)
    first_carry = int(YEARS[np.argmax(w["catch"] + w["gp_split"] > 0)]); put("wfFirstCarry", first_carry, "{:.0f}")
    lp = w["roc"] + w["pref"] + w["lp_split"]
    pct("wfGrossIrr", gross_irr(1.0), 1); pct("wfNetIrrFee", net_irr(1.0, False), 1); pct("wfNetIrr", net_irr(1.0), 1)
    put("wfNetTvpi", lp.sum() / C.sum(), "{:.2f}"); put("wfDpiPaid", lp.sum(), "{:.1f}")
    R.RAW["wfTable"] = {k: [float(v) for v in w[k]] for k in ("roc", "pref", "catch", "lp_split", "gp_split")}
    R.RAW["wfContrib"] = [float(v) for v in C]; R.RAW["wfDist"] = [float(v) for v in X]
    # figure 1: distributions by year and tier, contributions below the axis
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    bottom = np.zeros(len(YEARS))
    for key, col, lab in [("roc", BLUE, "LPs: return of capital"), ("pref", SKY, "LPs: preferred return"),
                          ("catch", RED, "GP: catch-up"), ("lp_split", PALE, "LPs: 80% split"),
                          ("gp_split", ROSE, "GP: 20% split")]:
        ax.bar(YEARS, w[key], bottom=bottom, color=col, edgecolor="#333333", lw=0.3, label=lab, width=0.7)
        bottom += w[key]
    ax.bar(YEARS, -C, color=GREY, width=0.7, label="LP contributions, with fees")
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xlabel("year of the fund"); ax.set_ylabel("cash flow (per 100 committed)"); ax.set_xticks(YEARS)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=3)
    R.save(fig, "fig01_waterfall")
    return w


def gross_to_net():
    scales = np.linspace(0.62, 1.8, 60)
    g = np.array([gross_irr(s) for s in scales]); nf = np.array([net_irr(s, False) for s in scales])
    nc = np.array([net_irr(s) for s in scales])
    moic = scales * sum(m for *_, m in DEALS) / len(DEALS)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.plot(100 * g, 100 * g, color=GREY, lw=0.8, ls=":", label="gross = net")
    ax.plot(100 * g, 100 * nf, color=SKY, ls="--", label="net of fees")
    ax.plot(100 * g, 100 * nc, color=BLUE, label="net of fees and carry")
    ax.set_xlabel("gross IRR of the deals (%)"); ax.set_ylabel("IRR to the LPs (%)"); ax.legend(loc="upper left")
    R.save(fig, "fig02_gross_net")
    # spreads at three gross multiples
    for mo, nm in [(1.2, "Low"), (1.8, "Mid"), (2.5, "High")]:
        s = mo / (sum(m for *_, m in DEALS) / len(DEALS))
        put(f"wfGm{nm}", mo, "{:.1f}"); pct(f"wfGi{nm}", gross_irr(s), 1); pct(f"wfNi{nm}", net_irr(s), 1)
        R.RAW[f"wfSpread{nm}"] = 100 * (gross_irr(s) - net_irr(s)); put(f"wfSpread{nm}", R.RAW[f"wfSpread{nm}"], "{:.1f}")
    # the gross multiple at which the LPs just earn the hurdle
    s_h = __import__("scipy.optimize", fromlist=["brentq"]).brentq(lambda s: net_irr(s) - H, 0.8, 1.6, xtol=1e-12)
    put("wfMoicAtHurdle", s_h * sum(m for *_, m in DEALS) / len(DEALS), "{:.2f}"); pct("wfGrossAtHurdle", gross_irr(s_h), 1)
    R.RAW["wfMoicRange"] = [float(moic[0]), float(moic[-1])]


def timing():
    """Deal by deal carry with a full clawback against whole fund carry, no hurdle, simulated funds."""
    rng = rng_for(1); n_funds, n_deals, r = 20_000, 12, 0.08
    mu, sig = 0.10, 0.45
    excess = []; share = []; claw = []; R.RAW["_pe"] = []
    for _ in range(n_funds):
        si = rng.uniform(0, 4, n_deals); hold = np.clip(rng.exponential(4.0, n_deals), 1.0, 8.0)
        so = si + hold
        P = 10 * np.exp((mu - 0.5 * sig ** 2) * hold + sig * np.sqrt(hold) * rng.standard_normal(n_deals))
        K = np.full(n_deals, 10.0)
        d, E, A, final = L.deal_carry_paths(K, si, P, so, CARRY)
        E = np.array([CARRY * (P[so <= x + 1e-12].sum() - K[si <= x + 1e-12].sum()) for x in d]).clip(0)
        E[-1] = final
        assert np.all(A >= E - 1e-9)
        pa, pe = L.pv_of_path(d, A, r), L.pv_of_path(d, E, r)
        R.RAW["_pe"].append(pe); excess.append(pa - pe); share.append((pa - pe) / pe if pe > 0 else np.nan)
        claw.append((A[-2] - final) / final if final > 0 else np.nan)
    excess = np.array(excess); share = np.array(share); claw = np.array(claw)
    cap = 10.0 * n_deals; ex = 100 * excess / cap
    put("wfTmFunds", n_funds, "{:,.0f}"); put("wfTmDeals", n_deals, "{:.0f}"); pct("wfTmR", r, 0)
    pct("wfTmMu", mu, 0); pct("wfTmSig", sig, 0); put("wfTmCap", cap, "{:.0f}")
    put("wfTmExMed", np.median(ex), "{:.2f}"); put("wfTmExMean", ex.mean(), "{:.2f}")
    put("wfTmExNine", np.percentile(ex, 90), "{:.2f}")
    pct("wfTmRatio", excess.mean() / np.mean(R.RAW.pop("_pe")), 1)
    pct("wfTmClawFreq", np.nanmean(claw > 1e-9), 0); pct("wfTmClawMed", np.nanmedian(claw[claw > 1e-9]), 0)
    pct("wfTmNoCarry", np.mean(np.isnan(share)), 1)
    R.RAW["wfTmMinExcess"] = float(excess.min())
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.hist(np.clip(ex, 0, 6), bins=60, color=PALE, edgecolor=SKY, lw=0.4)
    ax.axvline(np.median(ex), color=RED, lw=1.1, label="median")
    ax.set_xlabel("present value of deal by deal less whole fund carry (% of capital invested)")
    ax.set_ylabel("funds"); ax.legend(loc="upper right")
    R.save(fig, "fig03_timing")


CATCHES = ((1.0, "One"), (0.8, "Eight"), (0.5, "Half"), (0.0, "Zero"))


def catchup_figure():
    """The gross to net map of the stylised fund for catch-up rates k of 1, 0.8 and 0.5 and with no
    catch-up (k = 0, tier (iii) skipped)."""
    scales = np.linspace(0.75, 1.4, 131); mean_m = sum(m for *_, m in DEALS) / len(DEALS)
    g = np.array([gross_irr(s) for s in scales])
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.plot(100 * g, 100 * g, color=GREY, lw=0.8, ls=":", label="gross = net")
    for (k, nm), col, ls in zip(CATCHES, (BLUE, SKY, RED, GREY), ("-", "--", "-.", "-")):
        ax.plot(100 * g, [100 * net_irr(s, k=k) for s in scales], color=col, ls=ls,
                label="no catch-up" if k == 0 else f"catch-up at k = {k:g}")
    ax.axhline(100 * H, color="#333333", lw=0.5, ls=":")
    ax.set_xlim(9, 20); ax.set_ylim(5, 15)         # the stretch where the catch-up rate matters
    ax.set_xlabel("gross IRR of the deals (%)"); ax.set_ylabel("IRR to the LPs (%)"); ax.legend(loc="upper left")
    R.save(fig, "fig15_catchup")
    # the gross multiple at which the GP holds its full share of the profit, for each catch-up rate
    from scipy.optimize import brentq

    def share(s, k):
        C, X, _ = fund_flows(s); w = L.european_waterfall(YEARS, C, X, H, CARRY, k)
        return w["gp_total"] / (X.sum() - C.sum())
    for k, nm in CATCHES[:3]:
        s_full = brentq(lambda s: share(s, k) - (CARRY - 1e-9), 0.8, 3.0, xtol=1e-12)
        put(f"ckFull{nm}", s_full * mean_m, "{:.2f}")
    # with no catch-up the GP's share of the profit stays below the carry rate
    C, X, _ = fund_flows()
    w0 = L.european_waterfall(YEARS, C, X, H, CARRY, 0.0)
    pct("ckShareZero", w0["gp_total"] / (X.sum() - C.sum()), 1)
    pct("ckZeroMid", net_irr(1.8 / mean_m, k=0.0), 1)


TOY = dict(K=[10.0, 10.0, 10.0], s_in=[0.0, 0.0, 1.0], P=[25.0, 18.0, 2.0], s_out=[2.0, 4.0, 6.0])


def carry_paths_figure(r=0.08):
    """Three deals, the loser sold last: cumulative deal by deal carry A(t) and whole fund carry E(t),
    no hurdle, with the clawback at the last exit."""
    d, E, A, final = L.deal_carry_paths(TOY["K"], TOY["s_in"], TOY["P"], TOY["s_out"], CARRY)
    assert np.all(A >= E - 1e-12) and abs(A[-1] - E[-1]) < 1e-12
    pa, pe = L.pv_of_path(d, A, r), L.pv_of_path(d, E, r)
    # Equation (timing): the gap is the sum of (A - E) times the fall in the discount factor
    gap = sum((A[j] - E[j]) * (np.exp(-r * d[j]) - np.exp(-r * d[j + 1])) for j in range(len(d) - 1))
    R.RAW["ctIdErr"] = abs(gap - (pa - pe))
    put("ctPvGap", pa - pe, "{:.3f}"); put("ctClaw", A[-2] - final, "{:.1f}"); put("ctFinal", final, "{:.1f}")
    put("ctAPeak", A[-2], "{:.1f}"); pct("ctR", r, 0)
    for j, nm in enumerate(("One", "Two", "Three")):
        put(f"ctP{nm}", TOY["P"][j], "{:.0f}"); put(f"ctS{nm}", TOY["s_out"][j], "{:.0f}")
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    tt = np.concatenate([[0.0], d, [d[-1] + 0.6]])
    a = np.concatenate([[0.0], A, [A[-1]]]); e = np.concatenate([[0.0], E, [E[-1]]])
    ax.fill_between(tt, e, a, step="post", color=PALE, label="carry paid early, A less E")
    ax.step(tt, a, where="post", color=RED, label="deal by deal, A(t)")
    ax.step(tt, e, where="post", color=BLUE, ls="--", label="whole fund, E(t)")
    ax.annotate("clawback", (d[-1], (A[-2] + final) / 2), xytext=(-52, 0), textcoords="offset points",
                fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlabel("year"); ax.set_ylabel("cumulative carry"); ax.legend(loc="upper left", fontsize=7.5)
    R.save(fig, "fig16_carrypaths")


def main():
    stylised(); gross_to_net(); timing(); catchup_figure(); carry_paths_figure()
    return finish("waterfall")


if __name__ == "__main__":
    main()
