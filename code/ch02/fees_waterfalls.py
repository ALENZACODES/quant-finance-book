"""Section 2.2 to 2.6 and 2.8: structure diagram, management fees, single period waterfall, Fund B.

Run on its own:  python code/ch02/fees_waterfalls.py   (writes code/ch02/out/fees_waterfalls.json)
run_all.py runs every script of the chapter and writes chapters/ch02-numbers.tex.
"""
from __future__ import annotations
import numpy as np
from numbers_ch02 import N, put, pct, word, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, SEED  # noqa: F401
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================ 2.2 structure diagram
def structure_diagram():
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc="white", ec=BLUE, tc=BLUE, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=7.9,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, txt=None, col=GREY, dx=0.0, dy=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
                                     color=col, lw=0.9, linestyle=ls))
        if txt:
            ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, txt, fontsize=7.2, color=col, ha="center")

    box(0.05, 4.7, 2.6, 0.9, "Taxable domestic\ninvestors")
    box(3.7, 4.7, 2.6, 0.9, "Tax-exempt and\nnon-resident investors")
    box(7.35, 4.7, 2.6, 0.9, "General partner\n(fund sponsor)", fc=PALE)
    box(0.05, 2.9, 2.6, 0.9, "Onshore feeder\n(limited partnership)")
    box(3.7, 2.9, 2.6, 0.9, "Offshore feeder\n(corporation)")
    box(3.0, 1.15, 4.0, 0.95, "Master fund\n(holds the portfolio)", fc=PALE, bold=True)
    box(7.35, 2.9, 2.6, 0.9, "Management company\n(investment adviser)", fc=PALE)
    box(0.2, 0.05, 2.3, 0.75, "Prime broker,\ncustodian", ec=GREY, tc=GREY)
    box(7.5, 0.05, 2.3, 0.75, "Administrator,\nauditor", ec=GREY, tc=GREY)
    arrow(1.35, 4.7, 1.35, 3.8, "capital", dx=0.45)
    arrow(5.0, 4.7, 5.0, 3.8, "capital", dx=0.45)
    arrow(1.35, 2.9, 3.6, 2.1, "", col=BLUE)
    arrow(5.0, 2.9, 5.0, 2.1, "", col=BLUE)
    arrow(7.35, 4.95, 6.55, 2.1, "", col=RED, ls="--")
    arrow(8.65, 4.7, 8.65, 3.8, "", col=RED)
    ax.text(8.75, 4.15, "owns", fontsize=7.0, color=RED, ha="left")
    ax.text(6.78, 4.05, "general\npartner,\ncarry", fontsize=6.8, color=RED, ha="center")
    arrow(7.35, 3.1, 7.0, 1.85, "", col=RED)
    ax.text(7.95, 2.35, "advises,\nmanagement fee", fontsize=7.0, color=RED, ha="left")
    arrow(3.0, 1.2, 2.5, 0.6, "", col=GREY)
    arrow(7.0, 1.2, 7.5, 0.6, "", col=GREY)
    save(fig, "fig01_structure")


# ============================================================ fund B: deals, fees, waterfalls
DEALS_B = [dict(name="A", t_in=0.0, cost=30.0, t_out=3.0, proceeds=75.0),
           dict(name="B", t_in=1.0, cost=25.0, t_out=5.0, proceeds=40.0),
           dict(name="C", t_in=1.0, cost=20.0, t_out=6.0, proceeds=0.0),
           dict(name="D", t_in=2.0, cost=10.0, t_out=8.0, proceeds=20.0)]
YEARS_B = 9


def fund_b_flows():
    ft, f = L.fund_fee_flows(DEALS_B, years=YEARS_B)
    ct = np.r_[[d["t_in"] for d in DEALS_B], ft]
    c = np.r_[[d["cost"] for d in DEALS_B], f]
    dt = np.array([d["t_out"] for d in DEALS_B]); d = np.array([d["proceeds"] for d in DEALS_B])
    return ft, f, ct, c, dt, d


def fees_section():
    ft, f, ct, c, dt, d = fund_b_flows()
    put("bFeeIP", f[:5].sum(), "{:.2f}"); put("bFeePost", f[5:].sum(), "{:.2f}")
    put("bFeeTotal", f.sum(), "{:.2f}"); put("bInvested", sum(x["cost"] for x in DEALS_B), "{:.0f}")
    put("bPaidIn", c.sum(), "{:.2f}"); put("bProceeds", d.sum(), "{:.0f}")
    for t in (5, 6, 7):
        put("bFeeYear" + word(t), f[t], "{:.2f}")
    # alternative fee terms over a ten year life for a fund that invests 85 evenly over five years
    # and realises each vintage after five years
    years = 10
    cost_out = np.array([17 * sum(1 for v in range(5) if v <= t < v + 5) for t in range(years)], float)
    terms = {"2% of commitments, 10 years": L.management_fees(100, years, .02, .02, 10, cost_out),
             "2%, then 1.5% of cost": L.management_fees(100, years, .02, .015, 5, cost_out),
             "2% of invested cost": L.management_fees(100, years, .02, .02, 0, cost_out)}
    totals = [v.sum() for v in terms.values()]
    put("altFeeFlat", totals[0], "{:.1f}"); put("altFlatTotal", cost_out.max() + totals[0], "{:.0f}"); put("altFeeStep", totals[1], "{:.2f}"); put("altFeeCost", totals[2], "{:.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    tt = np.arange(YEARS_B)
    unreal = np.array([sum(x["cost"] for x in DEALS_B if x["t_in"] <= t < x["t_out"]) for t in tt])
    ax[0].step(tt, np.where(tt < 5, 100, unreal), where="post", color=BLUE, label="fee base")
    ax[0].step(tt, unreal, where="post", color=GREY, ls=":", label="cost of unrealised deals")
    ax[0].bar(tt + 0.5, 10 * f, width=0.6, color=ROSE, alpha=0.8, label="fee $\\times$ 10")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("amount"); ax[0].set_title("(a) Fund B: fee base and annual fee")
    ax[0].set_ylim(0, 158); ax[0].set_yticks(range(0, 101, 20)); ax[0].legend(loc="upper right", fontsize=6.8, ncol=1)
    cols = [BLUE, RED, GREY]
    for (k, v), col in zip(terms.items(), cols):
        ax[1].step(np.arange(1, years + 1), np.cumsum(v), where="post", color=col, label=k)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("cumulative fees (% of commitment)")
    ax[1].set_title("(b) Fee terms over a ten year life"); ax[1].legend(loc="upper left", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig02_fees")
    return ft, f, ct, c, dt, d


# ============================================================ 2.6 the single period waterfall
def single_period():
    K, T, h, cr = 100.0, 5.0, 0.08, 0.20
    H = K * (1 + h) ** T
    put("spH", H); put("spVstarFull", (H - cr * K) / (1 - cr))
    put("spVstarPart", (0.8 * H - cr * K) / (0.8 - cr))
    V = 180.0
    r = L.european_waterfall([0], [K], [T], [V])
    put("spV", V, "{:.0f}")
    put("spRoc", r.tiers[0, 0]); put("spPref", r.tiers[0, 1]); put("spCatch", r.tiers[0, 2])
    put("spSplitTotal", r.tiers[0, 4] + r.tiers[0, 5]); put("spSplitLP", r.tiers[0, 4]); put("spSplitGP", r.tiers[0, 5]); put("spGP", r.gp[0]); put("spLP", r.lp[0])
    for kw, nm in [(dict(catchup=0.8), "Part"), (dict(hard=True), "Hard")]:
        put("spGP" + nm, L.european_waterfall([0], [K], [T], [V], **kw).gp[0])
    put("spGPNoHurdle", cr * (V - K))
    put("spLPirr", 100 * ((r.lp[0] / K) ** (1 / T) - 1)); put("spGrossIrr", 100 * ((V / K) ** (1 / T) - 1))

    Vs = np.linspace(80, 260, 721)
    alloc = np.array([L.european_waterfall([0], [K], [T], [v]).tiers[0] for v in Vs])
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8))
    lab = ["return of capital", "preferred return", "catch-up to GP", "", "split to LP", "split to GP"]
    colr = [SKY, BLUE, RED, RED, GREY, ROSE]
    ax[0].stackplot(Vs, *[alloc[:, i] for i in (0, 1, 2, 4, 5)], colors=[colr[i] for i in (0, 1, 2, 4, 5)],
                    labels=[lab[i] for i in (0, 1, 2, 4, 5)], alpha=0.9)
    ax[0].set_xlabel("proceeds at year 5, $V$"); ax[0].set_ylabel("amount paid through each tier")
    ax[0].set_title("(a) Where each dollar goes"); ax[0].legend(loc="upper left", fontsize=7.0)
    for kw, lb, col, ls in [(dict(), "full catch-up", BLUE, "-"), (dict(catchup=0.8), "80% catch-up", SKY, "--"),
                            (dict(hard=True), "hard hurdle", RED, "-."), ]:
        ax[1].plot(Vs, L.carry_payoff(Vs, K, H, **kw), color=col, ls=ls, label=lb)
    ax[1].plot(Vs, cr * np.maximum(Vs - K, 0), color=GREY, ls=":", label="no hurdle")
    ax[1].axvline(H, color=GREY, lw=0.6); ax[1].text(H + 2, 0.55, "$H$", color=GREY, fontsize=8, transform=ax[1].get_xaxis_transform())
    ax[1].set_xlabel("proceeds at year 5, $V$"); ax[1].set_ylabel("carried interest")
    ax[1].set_title("(b) GP payoff under four designs"); ax[1].legend(loc="upper left", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig03_tiers")


# ============================================================ 2.8 whole fund vs deal by deal (fund B)
def fund_b_waterfalls(ft, f, ct, c, dt, d):
    wf = L.european_waterfall(ct, c, dt, d)
    dbd = L.deal_by_deal(DEALS_B)
    pool = L.deal_by_deal_pooled(DEALS_B)
    put("bPoolCarry", pool.sum())
    for i, nm in enumerate("ABCD"):
        put("bPool" + nm, pool[i])
    put("bPoolClaw", max(0.0, pool.sum() - wf.gp.sum()))
    for k in range(1, 5):
        r = DEALS_B[:k]
        put("bPoolEnt" + word(k), L.carry_entitlement([x["t_in"] for x in r], [x["cost"] for x in r],
                                                      [x["t_out"] for x in r], [x["proceeds"] for x in r]))
    put("bGrossIrr", 100 * L.irr(np.r_[ct, dt], np.r_[-c, d]))
    put("bTVPIgross", d.sum() / c.sum())
    put("bWfCarry", wf.gp.sum()); put("bDbdCarry", dbd.sum())
    for i, nm in enumerate("ABCD"):
        put("bDeal" + nm + "Carry", dbd[i])
        dl = DEALS_B[i]
        put("bDeal" + nm + "Hurdle", dl["cost"] * 1.08 ** (dl["t_out"] - dl["t_in"]))
        put("bDeal" + nm + "Profit", dl["proceeds"] - dl["cost"], "{:.0f}")
    owed, rec = L.clawback(dbd.sum(), wf.gp.sum(), tax_rate=0.40)
    put("bClawback", owed); put("bClawCap", 0.6 * dbd.sum())
    put("bProfit", d.sum() - c.sum())
    put("bWfShare", 100 * wf.gp.sum() / (d.sum() - c.sum()), "{:.1f}")
    for j, nm in enumerate("ABCD"):
        put("bWfGP" + nm, wf.gp[j]); put("bWfLP" + nm, wf.lp[j])
        put("bWfRoc" + nm, wf.tiers[j, 0]); put("bWfPref" + nm, wf.tiers[j, 1])
        put("bWfCatch" + nm, wf.tiers[j, 2]); put("bWfSplit" + nm, wf.tiers[j, 4] + wf.tiers[j, 5])
    # LP net IRRs: whole fund, deal by deal before and after clawback
    lp_wf = d - wf.gp
    lp_dbd = d - dbd
    put("bLPirrWf", 100 * L.irr(np.r_[ct, dt], np.r_[-c, lp_wf]))
    put("bLPirrDbdNoClaw", 100 * L.irr(np.r_[ct, dt], np.r_[-c, lp_dbd]))
    t_all = np.r_[ct, dt, 8.0]; cf_all = np.r_[-c, lp_dbd, owed]
    put("bLPirrDbdClaw", 100 * L.irr(t_all, cf_all))
    # gross and LP net multiple
    put("bTVPInet", lp_wf.sum() / c.sum())
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    tt = np.arange(0, 9)
    cum_wf = [wf.gp[dt <= t].sum() for t in tt]
    cum_dbd = [dbd[dt <= t].sum() for t in tt]
    cum_dbd_net = [x - (owed if t >= 8 else 0) for x, t in zip(cum_dbd, tt)]
    ax[0].step(tt, cum_dbd_net, where="post", color=RED, label="deal by deal, after clawback")
    ax[0].step(tt, cum_dbd, where="post", color=RED, ls=":", lw=1.0, label="deal by deal, before clawback")
    ax[0].step(tt, [pool[dt <= t].sum() for t in tt], where="post", color=GREY, ls="--",
               label="deal by deal with loss netting, before clawback")
    ax[0].step(tt, cum_wf, where="post", color=BLUE, label="whole fund")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("cumulative carry paid to GP")
    ax[0].set_ylim(-0.5, 21)
    ax[0].set_title("(a) Timing of carry in Fund B"); ax[0].legend(loc="upper left", fontsize=6.9)
    ax[1].bar(ct[ct < 9] + 0.0, -c[ct < 9], width=0.35, color=SKY, label="contributions (deals and fees)")
    ax[1].bar(dt + 0.4, lp_wf, width=0.35, color=BLUE, label="to LP, whole fund")
    ax[1].bar(dt + 0.4, wf.gp, bottom=lp_wf, width=0.35, color=RED, label="to GP, whole fund")
    ax[1].axhline(0, color="#333333", lw=0.6)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("cash flow"); ax[1].set_title("(b) Fund B cash flows")
    ax[1].set_ylim(-35, 95); ax[1].legend(loc="upper right", fontsize=6.9)
    fig.tight_layout(); save(fig, "fig05_fundb")
    return wf, dbd


def main():
    structure_diagram()
    ft, f, ct, c, dt, d = fees_section()
    single_period()
    fund_b_waterfalls(ft, f, ct, c, dt, d)
    return finish('fees_waterfalls')


if __name__ == "__main__":
    main()
