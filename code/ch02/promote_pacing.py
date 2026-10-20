"""Sections 2.13 and 2.14: real estate promote waterfalls and Takahashi-Alexander commitment pacing.

Run on its own:  python code/ch02/promote_pacing.py   (writes code/ch02/out/promote_pacing.json)
run_all.py runs every script of the chapter and writes chapters/ch02-numbers.tex.
"""
from __future__ import annotations
import numpy as np
from numbers_ch02 import N, put, pct, word, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, SEED  # noqa: F401


# ============================================================ 2.13 real estate promote
def promote_section():
    c_t, c = [0.0], [10.0]
    cf = [0.55 * 1.02 ** (t - 1) for t in range(1, 6)]
    def flows(P):
        return np.arange(1, 6, dtype=float), np.array(cf[:-1] + [cf[-1] + P])
    irr_t = [("irr", 0.08, 0.0), ("irr", 0.12, 0.20), ("irr", None, 0.30)]
    mul_t = [("mult", 1.0, 0.0), ("mult", 1.6, 0.20), ("mult", None, 0.30)]
    P0 = 15.0
    dt, d = flows(P0)
    inv, spo, by = L.tiered_waterfall(c_t, c, dt, d, irr_t)
    put("reP", P0, "{:.0f}"); put("reDist", d.sum()); put("reProfit", d.sum() - 10)
    put("reProjIrr", 100 * L.irr(np.r_[0, dt], np.r_[-10, d]))
    put("reInvIrr", 100 * L.irr(np.r_[0, dt], np.r_[-9, inv])); put("reSpoIrr", 100 * L.irr(np.r_[0, dt], np.r_[-1, spo]))
    put("reInvTot", inv.sum()); put("reSpoTot", spo.sum())
    for k, nm in enumerate(["One", "Two", "Three"]):
        put("reTier" + nm + "Inv", by[k, 0]); put("reTier" + nm + "Spo", by[k, 1])
    put("rePromote", spo.sum() - 0.1 * d.sum())
    # worked tier allocation of the year 5 distribution, using the balances of Definition 2.x
    early = inv[:-1]                                  # investor's take from years 1 to 4 (all in tier 1)
    b1 = 9 * 1.08 ** 5 - sum(a * 1.08 ** (5 - u) for u, a in zip(dt[:-1], early))
    l1 = b1                                           # tier 1 pays the investor exactly its balance
    b2 = 9 * 1.12 ** 5 - sum(a * 1.12 ** (5 - u) for u, a in zip(dt[:-1], early)) - l1
    put("reYrFive", d[-1]); put("reEarlyInv", early.sum())
    put("reBalOne", b1); put("reTierOneSize", b1 / 0.9)
    put("reBalTwo", b2); put("reTierTwoSize", b2 / (0.8 * 0.9))
    put("reTierThreeSize", d[-1] - b1 / 0.9 - b2 / 0.72)
    pct("reSpoShareProfit", (spo.sum() - 1) / (d.sum() - 10), 1)
    inv2, spo2, by2 = L.tiered_waterfall(c_t, c, dt, d, mul_t)
    put("reMulSpoTot", spo2.sum()); put("reMulInvIrr", 100 * L.irr(np.r_[0, dt], np.r_[-9, inv2]))
    # slow sale: same total, sale 3 years later (cash flow held at year 5 level)
    Ps = np.linspace(6, 26, 161)
    rows = []
    for P in Ps:
        dt, d = flows(P)
        pi = L.irr(np.r_[0, dt], np.r_[-10, d])
        i1, s1, _ = L.tiered_waterfall(c_t, c, dt, d, irr_t); i2, s2, _ = L.tiered_waterfall(c_t, c, dt, d, mul_t)
        prof = d.sum() - 10
        rows.append((pi, (s1.sum() - 1) / prof if prof > 0 else np.nan, (s2.sum() - 1) / prof if prof > 0 else np.nan))
    R = np.array(rows)
    # timing: same multiple, ten year hold
    cf10 = [0.55 * 1.02 ** (t - 1) for t in range(1, 11)]
    d10 = np.array(cf10[:-1] + [cf10[-1] + P0 + (sum(cf) - sum(cf10))])   # same total distributions
    dt10 = np.arange(1, 11, dtype=float)
    i10, s10, _ = L.tiered_waterfall(c_t, c, dt10, d10, irr_t); m10, ms10, _ = L.tiered_waterfall(c_t, c, dt10, d10, mul_t)
    put("reTenIrr", 100 * L.irr(np.r_[0, dt10], np.r_[-10, d10]))
    put("reTenSpoIrrTiers", s10.sum()); put("reTenSpoMulTiers", ms10.sum())
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    ax[0].plot(100 * R[:, 0], 100 * R[:, 1], color=BLUE, label="IRR hurdles (8%, 12%)")
    ax[0].plot(100 * R[:, 0], 100 * R[:, 2], color=RED, ls="--", label="multiple hurdles (1.0x, 1.6x)")
    ax[0].axhline(10, color=GREY, lw=0.7, ls=":"); ax[0].text(100 * R[-1, 0], 9, "pro rata share (10%)", fontsize=7, color=GREY, ha="right", va="top")
    ax[0].set_xlabel("project IRR (%)"); ax[0].set_ylabel("sponsor share of total profit (%)")
    ax[0].set_ylim(5, 42)
    ax[0].set_title("(a) Promote against project IRR"); ax[0].legend(loc="upper left", fontsize=7.2)
    ax[0].set_xlim(0, R[:, 0].max() * 100)
    tiers = ["tier 1", "tier 2", "tier 3"]
    ax[1].bar(np.arange(3) - 0.18, by[:, 0], width=0.36, color=BLUE, label="investor")
    ax[1].bar(np.arange(3) + 0.18, by[:, 1], width=0.36, color=RED, label="sponsor")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(tiers); ax[1].set_ylabel("amount")
    ax[1].set_title("(b) Base case by tier (IRR hurdles)"); ax[1].legend(loc="upper right", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig11_promote")


# ============================================================ 2.14 commitment pacing
def pacing():
    kw = dict(rc=(0.25, 1 / 3, 0.5), growth=0.12, bow=2.5, life=12)
    C, D, NAV = L.takahashi_alexander(years=15, **kw)
    put("taPeakNav", NAV.max()); put("taPeakYear", int(np.argmax(NAV)), "{:.0f}")
    put("taCalledFive", C[:6].sum()); put("taDistTotal", D.sum()); put("taTVPI", (D.sum() + NAV[-1]) / C.sum())
    put("taCalledTotal", C.sum(), "{:.3f}")
    cum = np.cumsum(D - C); t0 = int(np.argmin(cum))
    put("taBreakeven", t0 + int(np.argmax(cum[t0:] >= 0)), "{:.0f}"); put("taTroughYear", t0, "{:.0f}")
    put("taTroughCash", -cum.min())
    H = 30
    Cp, Dp, NAVp, U = L.programme([1.0] * (H + 1), H, **kw)
    put("progNavSteady", NAVp[-1]); put("progUncalledSteady", U[-1]); put("progNetCash", Dp[-1] - Cp[-1])
    put("progExposure", (NAVp[-1] + U[-1])); put("progOvercommit", (NAVp[-1] + U[-1]) / NAVp[-1])
    # pacing: commitments needed per year for NAV target of 100 in steady state
    put("progCommitForHundred", 100 / NAVp[-1])
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))
    t = np.arange(16)
    ax[0].bar(t - 0.2, C, width=0.4, color=SKY, label="contributions")
    ax[0].bar(t + 0.2, D, width=0.4, color=RED, label="distributions")
    ax[0].plot(t, NAV, color=BLUE, label="NAV")
    ax[0].plot(t, np.cumsum(D - C), color=GREY, ls=":", label="cumulative net cash")
    ax[0].set_xlabel("year"); ax[0].set_ylabel("per unit of commitment")
    ax[0].set_ylim(-1.0, 1.75)
    ax[0].set_title("(a) One fund"); ax[0].legend(loc="upper right", fontsize=6.8, ncol=2)
    tp = np.arange(H + 1)
    ax[1].plot(tp, NAVp, color=BLUE, label="NAV"); ax[1].plot(tp, U, color=RED, ls="--", label="uncalled commitments")
    ax[1].plot(tp, Dp - Cp, color=GREY, ls=":", label="net cash flow")
    ax[1].set_xlabel("year of programme"); ax[1].set_ylabel("per unit committed each year")
    ax[1].set_title("(b) One unit committed every year"); ax[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.68), fontsize=7)
    fig.tight_layout(); save(fig, "fig12_pacing")


def pacing_sensitivity():
    """How the bow and growth parameters shape one fund's NAV and net cash curve."""
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))
    t = np.arange(16)
    for b, col, ls in [(1.5, SKY, "--"), (2.5, BLUE, "-"), (4.0, RED, "-.")]:
        C, D, NAV = L.takahashi_alexander(years=15, rc=(0.25, 1 / 3, 0.5), growth=0.12, bow=b, life=12)
        ax[0].plot(t, NAV, color=col, ls=ls, label=f"bow $B$ = {b}")
        put("taPeak" + {1.5: "BowLow", 2.5: "BowMid", 4.0: "BowHigh"}[b], NAV.max())
    for g, col, ls in [(0.04, SKY, "--"), (0.12, BLUE, "-"), (0.20, RED, "-.")]:
        C, D, NAV = L.takahashi_alexander(years=15, rc=(0.25, 1 / 3, 0.5), growth=g, bow=2.5, life=12)
        ax[1].plot(t, np.cumsum(D - C), color=col, ls=ls, label=f"growth $G$ = {int(100*g)}%")
        put("taTVPI" + {0.04: "GLow", 0.12: "GMid", 0.20: "GHigh"}[g], (D.sum() + NAV[-1]) / C.sum())
    ax[0].set_xlabel("year"); ax[0].set_ylabel("NAV per unit committed"); ax[0].set_title("(a) Bow and the NAV hump")
    ax[0].legend(loc="upper right", fontsize=7.2)
    ax[1].axhline(0, color="#333333", lw=0.6)
    ax[1].set_xlabel("year"); ax[1].set_ylabel("cumulative net cash"); ax[1].set_title("(b) Growth and the J-curve")
    ax[1].legend(loc="upper left", fontsize=7.2)
    fig.tight_layout(); save(fig, "fig13_pacing_sensitivity")


def main():
    promote_section()
    pacing()
    pacing_sensitivity()
    return finish('promote_pacing')


if __name__ == "__main__":
    main()
