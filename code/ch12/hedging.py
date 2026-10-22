"""Section 12.6: learning a hedge from simulated paths. A short three month call on a lognormal stock,
rebalanced daily. The hedge ratio is a combination of a constant and logistic steps in
z = ln(S/K)/sqrt(tau), with one set of weights for every date; without costs the best weights solve a
least squares problem. With proportional costs the same weights are re-optimised for a mean-variance
hedger, and a policy with memory, a no-trade band around the delta, has its width learned.
Everything is compared with the Black-Scholes delta on fresh paths. Also writes fig18_hedge_path.png,
one fresh path of the band policy, and the markups of the six-policy table.

Run on its own:  python code/ch12/hedging.py   (writes code/ch12/out/hedging.json)
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

S0, K, T, SIG, MU, STEPS, NTR, NTE, KAPPA, GAMMA = 100.0, 100.0, 0.25, 0.20, 0.0, 63, 20000, 20000, 0.002, 1.0
CENTRES, WIDTH = np.linspace(-0.5, 0.5, 11), 0.1
TAU = T - np.arange(STEPS) * T / STEPS


def bs_deltas(paths):
    return L.bs_delta(paths[:, :-1], K, TAU[None, :], SIG)


def band(c, paths, shape):
    S = paths[:, :-1]
    w = c * (L.ww_width(S, K, TAU[None, :], SIG, KAPPA, GAMMA) if shape == "ww" else np.ones_like(S))
    return L.band_policy(bs_deltas(paths), w)


def report(nm, D, paths, H):
    pnl, cost = L.pnl_from_deltas(D, paths, H, 0.0, KAPPA); X = -pnl
    put(f"hdPrice{nm}", L.mv_price(X, GAMMA), "{:.3f}"); put(f"hdCost{nm}", cost.mean(), "{:.3f}")
    put(f"hdCStd{nm}", X.std(), "{:.3f}"); put(f"hdTurn{nm}", L.turnover(D), "{:.2f}")
    return X


def main():
    tr = L.gbm_paths(NTR, STEPS, T, S0, MU, SIG, rng_for(6)); te = L.gbm_paths(NTE, STEPS, T, S0, MU, SIG, rng_for(61))
    Htr, Hte = np.maximum(tr[:, -1] - K, 0), np.maximum(te[:, -1] - K, 0)
    pbs = float(L.bs_call(S0, K, T, SIG))
    put("hdSZero", S0, "{:.0f}"); put("hdK", K, "{:.0f}"); pct("hdSig", SIG, 0); put("hdSteps", STEPS, "{:.0f}")
    put("hdNtr", NTR, "{:,.0f}"); put("hdNte", NTE, "{:,.0f}"); pct("hdKappa", KAPPA, 1); put("hdBs", pbs, "{:.3f}")
    put("hdBumps", len(CENTRES), "{:.0f}"); put("hdWidth", WIDTH, "{:.3f}"); put("hdParams", 1 + 2 * len(CENTRES), "{:.0f}")
    put("hdCentre", CENTRES[-1], "{:.1f}"); put("hdGamma", GAMMA, "{:.0f}")
    # no costs: least squares
    p0, th0 = L.regression_hedge(tr, Htr, K, T, CENTRES, WIDTH)
    put("hdPriceLearn", p0, "{:.3f}"); put("hdUnhedged", Hte.std(), "{:.2f}")
    D0 = L.hedge_deltas(th0, te, K, T, CENTRES, WIDTH); Db = bs_deltas(te)
    pn_bs, _ = L.pnl_from_deltas(Db, te, Hte, pbs, 0.0); pn_l, _ = L.pnl_from_deltas(D0, te, Hte, p0, 0.0)
    pn_tr, _ = L.pnl_from_deltas(L.hedge_deltas(th0, tr, K, T, CENTRES, WIDTH), tr, Htr, p0, 0.0)
    put("hdStdBs", pn_bs.std(), "{:.3f}"); put("hdStdLearn", pn_l.std(), "{:.3f}"); put("hdStdLearnTrain", pn_tr.std(), "{:.3f}")
    put("hdMeanBs", pn_bs.mean(), "{:.3f}"); put("hdMeanLearn", pn_l.mean(), "{:.3f}")
    # the mean-variance price of the delta hedge with no costs, for comparison with the Black-Scholes price
    put("hdPriceFree", L.mv_price(-L.pnl_from_deltas(Db, te, Hte, 0.0, 0.0)[0], GAMMA), "{:.3f}")
    put("hdDeltaGap", np.sqrt(np.mean((D0 - Db) ** 2)), "{:.3f}")
    # costs: the delta, the no-cost weights, the weights re-optimised, and the no-trade bands
    out = {"Bs": report("Bs", Db, te, Hte), "Zero": report("Zero", D0, te, Hte)}
    th1, res = L.cost_hedge(th0, tr, Htr, K, T, CENTRES, WIDTH, KAPPA, GAMMA)
    put("hdIters", res.nit, "{:.0f}")
    D1 = L.hedge_deltas(th1, te, K, T, CENTRES, WIDTH); out["Cost"] = report("Cost", D1, te, Hte)
    for shape, nm, hi in [("ww", "Ww", 3.0), ("flat", "Flat", 0.5)]:
        loss = lambda c: L.mv_price(-L.pnl_from_deltas(band(c, tr, shape), tr, Htr, 0.0, KAPPA)[0], GAMMA)
        o = minimize_scalar(loss, bounds=(0.0, hi), method="bounded", options=dict(xatol=1e-4))
        put(f"hdBand{nm}", o.x, "{:.2f}" if shape == "ww" else "{:.3f}")
        out[nm] = report(nm, band(o.x, te, shape), te, Hte)
    put("hdPriceWwOne", L.mv_price(-L.pnl_from_deltas(band(1.0, te, "ww"), te, Hte, 0.0, KAPPA)[0], GAMMA), "{:.3f}")
    out["WwOne"] = report("WwOne", band(1.0, te, "ww"), te, Hte)      # the same price, and the other columns
    for nm in ("Bs", "Zero", "Cost", "WwOne", "Ww", "Flat"):
        put(f"hdMarkup{nm}", RAW[f"hdPrice{nm}"] - pbs, "{:.3f}")
    path_figure(te, Db, band(RAW["hdBandWw"], te, "ww"))
    pct("hdPriceCutWw", 1 - (RAW["hdPriceWw"] - pbs) / (RAW["hdPriceBs"] - pbs), 0)
    pct("hdPriceCutCost", 1 - (RAW["hdPriceCost"] - pbs) / (RAW["hdPriceBs"] - pbs), 0)
    put("hdWwAtm", float(L.ww_width(S0, K, T, SIG, KAPPA, GAMMA)), "{:.3f}")
    # figure 11: hedge ratios against the stock price at two dates
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True)
    s = np.linspace(80, 120, 200)
    for a, k in [(ax[0], 20), (ax[1], 55)]:
        tau = TAU[k]; dl = L.bs_delta(s, K, tau, SIG)
        wd = RAW["hdBandWw"] * L.ww_width(s, K, tau, SIG, KAPPA, GAMMA)
        a.fill_between(s, dl - wd, dl + wd, color=ROSE, alpha=0.35, lw=0, label=f"no-trade band, costs of {100 * KAPPA:.1f}%")
        a.plot(s, dl, color=GREY, lw=1.0, ls=":", label="Black-Scholes delta")
        a.plot(s, L.hedge_features(s, np.full_like(s, tau), K, CENTRES, WIDTH) @ th0, color=BLUE, lw=1.2, label="learned, no costs")
        a.set_xlabel(f"stock price after {k} trading days")
    ax[0].set_ylabel("hedge ratio"); ax[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.0), ncol=3, fontsize=7.3)
    save(fig, "fig11_hedge_ratio")
    # figure 12: the hedger's loss with costs on fresh paths
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bins = np.linspace(2.5, 8, 89)
    ax.hist(out["Bs"], bins=bins, color=ROSE, alpha=0.7, label="Black-Scholes delta")
    ax.hist(out["Ww"], bins=bins, histtype="step", color=BLUE, lw=1.3, label="no-trade band, learned width")
    ax.set_xlabel("hedger's loss at expiry, costs included, per call on a stock at 100"); ax.set_ylabel("paths")
    ax.legend(loc="upper right", fontsize=7.8)
    save(fig, "fig12_hedge_pnl")
    return finish("hedging")


def path_figure(te, Db, Dw, i=0):
    """One of the fresh paths: the delta, the edges of the learned band and the band policy's position,
    with its trades marked."""
    S = te[i, :-1]; days = np.arange(STEPS)
    w = RAW["hdBandWw"] * L.ww_width(S, K, TAU, SIG, KAPPA, GAMMA)
    pos = Dw[i]; prev = np.concatenate([[0.0], pos[:-1]]); trade = np.abs(pos - prev) > 1e-12
    put("hpTrades", trade.sum(), "{:.0f}"); put("hpDays", STEPS, "{:.0f}")
    put("hpTurnBand", np.abs(np.diff(np.concatenate([[0.0], pos]))).sum(), "{:.2f}")
    put("hpTurnDelta", np.abs(np.diff(np.concatenate([[0.0], Db[i]]))).sum(), "{:.2f}")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.fill_between(days, Db[i] - w, Db[i] + w, color=ROSE, alpha=0.35, lw=0, label="no-trade band")
    ax.plot(days, Db[i], color=GREY, ls=":", lw=1.0, label="Black-Scholes delta, traded daily")
    ax.step(days, pos, where="post", color=BLUE, lw=1.2, label="band policy")
    ax.plot(days[trade], pos[trade], "o", color=RED, ms=3, label="trade")
    ax.set_xlabel("trading day"); ax.set_ylabel("shares held per call")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.3)
    save(fig, "fig18_hedge_path")


if __name__ == "__main__":
    import time; t = time.time(); main(); print(f"{time.time() - t:.1f}s")
