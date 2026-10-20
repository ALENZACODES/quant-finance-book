"""Sections 2.8 and 2.9: Monte Carlo of simulated buyout funds under both waterfalls.

Run on its own:  python code/ch02/simulated_funds.py   (writes code/ch02/out/simulated_funds.json)
run_all.py runs every script of the chapter and writes chapters/ch02-numbers.tex.
"""
from __future__ import annotations
import numpy as np
from numbers_ch02 import N, put, pct, word, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, SEED  # noqa: F401


# ============================================================ 2.8 / 2.9 Monte Carlo of simulated funds
def simulated_funds(rng, n=6000):
    rows = []
    nbind = 0
    for _ in range(n):
        deals = L.simulate_fund(rng)
        ft, f = L.fund_fee_flows(deals, cap=True)   # fee calls never take paid-in above 100
        nbind += L.fund_fee_flows.binding
        ct = np.r_[[x["t_in"] for x in deals], ft]; c = np.r_[[x["cost"] for x in deals], f]
        dt = np.array([x["t_out"] for x in deals]); d = np.array([x["proceeds"] for x in deals])
        wf_seq = L.european_waterfall(ct, c, dt, d)
        ent = L.carry_entitlement(ct, c, dt, d)
        over = wf_seq.gp.sum() - ent
        wf = wf_seq
        dbd = L.deal_by_deal(deals)
        pool = L.deal_by_deal_pooled(deals)
        owed = max(0.0, dbd.sum() - ent)
        owed_p = max(0.0, pool.sum() - ent)
        g_irr = L.irr(np.r_[ct, dt], np.r_[-c, d])
        pre_fee = L.irr(np.r_[[x["t_in"] for x in deals], dt], np.r_[[-x["cost"] for x in deals], d])
        n_irr = L.irr(np.r_[ct, dt, dt.max()], np.r_[-c, d - wf.gp, over])
        t_end = dt.max()
        n_dbd = L.irr(np.r_[ct, dt, t_end], np.r_[-c, d - dbd, owed])
        n_pool = L.irr(np.r_[ct, dt, t_end], np.r_[-c, d - pool, owed_p])
        rows.append((ent, dbd.sum(), owed, g_irr, pre_fee, n_irr, n_dbd, c.sum(), d.sum(),
                     pool.sum(), owed_p, n_pool, over))
    A = np.array(rows)
    wfc, dbc, owed, girr, pirr, nirr, ndbd, paid, dist, poc, owp, npool, over = A.T
    put("mcN", n, "{:,.0f}")
    pct("mcWfOverProb", (over > 1e-9).mean(), 1); put("mcWfOverMean", over[over > 1e-9].mean())
    put("mcMeanMult", 0.88 * np.exp(0.50 + 0.5 * 0.75 ** 2))
    put("mcWfCarry", wfc.mean()); put("mcDbdCarry", dbc.mean()); put("mcOwed", owed.mean())
    pct("mcClawProb", (owed > 1e-9).mean(), 1)
    put("mcOwedGivenPos", owed[owed > 1e-9].mean())
    put("mcOwedNinetyFive", np.quantile(owed, 0.95))
    pct("mcZeroWf", (wfc < 1e-9).mean(), 1); pct("mcZeroDbd", (dbc < 1e-9).mean(), 1)
    put("mcPaidIn", paid.mean(), "{:.1f}"); put("mcDist", dist.mean(), "{:.1f}")
    ok = np.isfinite(pirr) & np.isfinite(nirr) & np.isfinite(ndbd)
    put("mcPreFeeIrr", 100 * np.median(pirr[ok])); put("mcGrossIrr", 100 * np.median(girr[ok]))
    put("mcNetIrr", 100 * np.median(nirr[ok]))
    put("mcWedgeFees", 100 * np.median(pirr[ok] - girr[ok])); put("mcWedgeCarry", 100 * np.median(girr[ok] - nirr[ok]))
    put("mcDbdIrrCost", 100 * np.median(nirr[ok] - ndbd[ok]), "{:.2f}")
    okp = ok & np.isfinite(npool)
    put("mcPoolIrrCost", 100 * np.median(nirr[okp] - npool[okp]), "{:.2f}")
    put("mcPoolCarry", poc.mean()); put("mcPoolOwed", owp.mean()); pct("mcPoolClawProb", (owp > 1e-9).mean(), 1)
    put("mcPoolOwedNinetyFive", np.quantile(owp, 0.95))
    fees = paid - 84.0
    pct("mcPoolFeeCluster", (np.abs(owp - 0.2 * fees) < 0.01).mean(), 1)
    put("mcFeesMean", fees.mean())
    put("mcCapBindN", nbind, "{:,.0f}")
    put("mcPaidMax", paid.max(), "{:.2f}"); put("mcOverCommitN", int((paid > 100 + 1e-9).sum()), "{:,.0f}")
    put("mcWfOverAll", over.mean()); put("mcWfPaid", (wfc + over).mean()); put("mcWfOverNinetyFive", np.quantile(over, 0.95))
    pct("mcZeroPool", (poc < 1e-9).mean(), 1)
    pct("mcWfAboveDbd", (wfc > dbc + 1e-9).mean(), 1)
    put("mcTVPI", np.median(dist / paid)); put("mcNetTVPI", np.median((dist - wfc) / paid))
    put("mcGrossNetSd", 100 * np.std(nirr[ok]), "{:.1f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.8))
    bins = np.linspace(0, max(owed.max(), owp.max()) * 0.8, 40)
    ax[0].hist(owed, bins=bins, color=RED, alpha=0.75, label="pure deal by deal")
    ax[0].hist(owp, bins=bins, color=BLUE, alpha=0.6, label="with loss netting")
    ax[0].set_xlabel("clawback owed (% of commitment)"); ax[0].set_ylabel("number of funds")
    ax[0].set_title("(a) Clawback owed at the end"); ax[0].legend(loc="upper right", fontsize=7.2)
    sel = ok & (rng.random(n) < 0.5)
    ax[1].scatter(100 * pirr[sel], 100 * nirr[sel], s=2, color=BLUE, alpha=0.35, lw=0, label="simulated funds")
    xx = np.linspace(-40, 60, 10)
    ax[1].plot(xx, xx, color=GREY, lw=0.8, ls=":", label="no fees or carry")
    ax[1].set_xlim(-40, 60); ax[1].set_ylim(-40, 60)
    ax[1].set_xlabel("deal-level IRR before fees (%)"); ax[1].set_ylabel("LP net IRR (%)")
    ax[1].set_title("(b) Gross to net"); ax[1].legend(loc="upper left", fontsize=7.2, markerscale=4)
    fig.tight_layout(); save(fig, "fig06_simulated_funds")


def main():
    simulated_funds(np.random.default_rng(SEED + 1))
    return finish('simulated_funds')


if __name__ == "__main__":
    main()
