"""Section 5.10: the profit and loss of a delta-hedged option, discrete rebalancing, the
gamma-theta relation, hedging at the wrong volatility, and transaction costs.

Run on its own:  python code/ch05/hedging_pnl.py   (writes code/ch05/out/hedging_pnl.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch05 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

S0, K, R, SIG, T, MU = 100.0, 100.0, 0.05, 0.20, 1.0, 0.10
NPATH = 20_000
STEPS = [13, 52, 252, 1008]
WORDS = {13: "Thirteen", 52: "FiftyTwo", 252: "Daily", 1008: "Quarter"}


def correct_vol():
    sds, th, out = [], [], {}
    for i, n in enumerate(STEPS):
        pnl, gt = L.hedge_sim(S0, K, R, SIG, SIG, T, n, NPATH, rng_for(10 + i), mu=MU)
        sd, t = float(pnl.std(ddof=1)), L.hedge_sd_theory(S0, K, R, SIG, T, n, mu=MU)
        sds.append(sd); th.append(t); out[n] = pnl
        w = WORDS[n]
        put(f"hedgeSd{w}", sd, "{:.3f}"); put(f"hedgeTh{w}", t, "{:.3f}")
        put(f"hedgeMean{w}", float(pnl.mean()), "{:.3f}"); put(f"hedgeMeanSE{w}", sd / math.sqrt(NPATH), "{:.3f}")
        RAW[f"hedgeCorrGT{w}"] = float(np.corrcoef(pnl, gt)[0, 1])
        put(f"hedgeSdPct{w}", 100 * sd / float(L.bs(S0, K, R, SIG, T)), "{:.1f}")
    put("hedgeSlope", L.ols_slope(STEPS, sds), "{:.3f}")
    put("hedgeCorrGTFiftyTwo", RAW["hedgeCorrGTFiftyTwo"], "{:.3f}")
    put("hedgeCorrGTDaily", RAW["hedgeCorrGTDaily"], "{:.3f}")
    put("hedgePaths", NPATH, "{:,d}"); put("hedgeMu", 100 * MU, "{:.0f}")
    for n in STEPS:
        put(f"hedgeN{WORDS[n]}", n, "{:,d}")
    # how close the leading-order formula is at daily hedging, and the largest mean in standard errors
    put("hedgeThGapDaily", 100 * abs(RAW["hedgeThDaily"] / RAW["hedgeSdDaily"] - 1), "{:.1f}")
    put("hedgeMeanMaxZ", max(abs(RAW[f"hedgeMean{WORDS[n]}"]) / RAW[f"hedgeMeanSE{WORDS[n]}"] for n in STEPS), "{:.1f}")
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    bins = np.linspace(-4, 4, 121)
    for n, col in zip([13, 52, 252], [ROSE, SKY, BLUE]):
        ax.hist(out[n], bins=bins, density=True, histtype="step", color=col, lw=1.3,
                label=f"{n} rebalances, sd {out[n].std():.2f}")
    ax.set_xlabel("discounted profit and loss of the hedged short call"); ax.set_ylabel("density")
    ax.legend(loc="upper left"); fig.tight_layout(); save(fig, "fig13_hedge_hist")


def one_path(rng, steps=252, sig_true=SIG, sig_h=SIG):
    """Per-step mark-to-model P&L of a hedged short call along one path, and the gamma-theta term."""
    dt = T / steps; S = S0; t = 0.0
    V = float(L.bs(S, K, R, sig_h, T)); D = float(L.greeks(S, K, R, sig_h, T)["delta"]); Pi = V
    pnl, gt, rets = [], [], []
    for k in range(steps):
        G = float(L.greeks(S, K, R, sig_h, T - t)["gamma"])
        Sn = S * math.exp((MU - 0.5 * sig_true ** 2) * dt + sig_true * math.sqrt(dt) * rng.standard_normal())
        Pin = D * Sn + (Pi - D * S) * math.exp(R * dt); tn = t + dt
        Vn = float(L.bs(Sn, K, R, sig_h, T - tn)) if k < steps - 1 else max(Sn - K, 0.0)
        pnl.append(math.exp(-R * tn) * (Pin - Vn) - math.exp(-R * t) * (Pi - V))
        ret = Sn / S - 1; rets.append(ret)
        gt.append(math.exp(-R * t) * 0.5 * G * S ** 2 * (sig_h ** 2 * dt - ret ** 2))
        S, V, Pi, t = Sn, Vn, Pin, tn
        if k < steps - 1:
            D = float(L.greeks(S, K, R, sig_h, T - t)["delta"])
    return np.array(pnl), np.array(gt), np.array(rets)


def wrong_vol():
    # sold at 20%, the stock realises 15%: hedge at 20% (the implied) or at 15% (the realised)
    lo, hi = 0.15, 0.20
    gap = float(L.bs(S0, K, R, hi, T) - L.bs(S0, K, R, lo, T)); put("volGap", gap, "{:.4f}")
    pA, _ = L.hedge_sim(S0, K, R, lo, hi, T, 252, NPATH, rng_for(20))
    pB, _ = L.hedge_sim(S0, K, R, lo, lo, T, 252, NPATH, rng_for(20), sell_sig=hi)
    pA2, _ = L.hedge_sim(S0, K, R, lo, hi, T, 252, NPATH, rng_for(21), mu=MU)
    put("wrongImpMean", pA.mean(), "{:.3f}"); put("wrongImpSd", pA.std(ddof=1), "{:.3f}")
    put("wrongImpSE", pA.std(ddof=1) / math.sqrt(NPATH), "{:.3f}")
    put("wrongImpLoss", 100 * float(np.mean(pA < 0)), "{:.2f}")
    put("wrongRealMean", pB.mean(), "{:.3f}"); put("wrongRealSd", pB.std(ddof=1), "{:.3f}")
    put("wrongRealLoss", 100 * float(np.mean(pB < 0)), "{:.2f}")
    put("wrongImpMeanDrift", pA2.mean(), "{:.3f}"); put("wrongImpSdDrift", pA2.std(ddof=1), "{:.3f}")
    put("wrongImpSEDrift", pA2.std(ddof=1) / math.sqrt(NPATH), "{:.3f}")
    # the seller who undercharges: realised 25%, sold and hedged at 20%
    pC, _ = L.hedge_sim(S0, K, R, 0.25, hi, T, 252, NPATH, rng_for(22))
    put("underMean", pC.mean(), "{:.3f}"); put("underGap", float(L.bs(S0, K, R, hi, T) - L.bs(S0, K, R, 0.25, T)), "{:.4f}")
    put("underLoss", -pC.mean(), "{:.3f}"); put("underGapAbs", float(L.bs(S0, K, R, 0.25, T) - L.bs(S0, K, R, hi, T)), "{:.4f}")
    put("underSE", pC.std(ddof=1) / math.sqrt(NPATH), "{:.3f}")
    # one path for the gamma-theta picture
    pnl, gt, rets = one_path(rng_for(23))
    put("gtStepCorr", float(np.corrcoef(pnl, gt)[0, 1]), "{:.4f}")
    put("gtTotal", pnl.sum(), "{:.3f}"); put("gtApprox", gt.sum(), "{:.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ax[0].scatter(100 * rets, pnl, s=5, color=BLUE, alpha=0.7, lw=0, label="daily P&L, one path")
    ax[0].scatter(100 * rets, gt, s=5, color=RED, alpha=0.7, lw=0, marker="x",
                  label=r"$\frac{1}{2}\Gamma S^2(\sigma^2\Delta t-R^2)$")
    ax[0].set_xlabel("daily stock return $R$ (%)"); ax[0].set_ylabel("discounted P&L")
    ax[0].set_title("(a) gamma against theta"); ax[0].legend(loc="lower center", fontsize=7)
    bins = np.linspace(-3, 6, 121)
    ax[1].hist(pA, bins=bins, density=True, histtype="stepfilled", color=PALE, ec=BLUE, lw=1.0,
               label=r"hedged at 20%")
    ax[1].hist(pB, bins=bins, density=True, histtype="step", color=RED, lw=1.3, label=r"hedged at 15%")
    ax[1].axvline(gap, color=GREY, ls=":", lw=1)
    ax[1].set_xlabel("discounted P&L"); ax[1].set_title("(b) sold at 20%, realised 15%")
    ax[1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig12_gamma_theta")


def costs():
    k = 0.002; put("costRate", 100 * k, "{:.1f}")
    d0 = float(L.greeks(S0, K, R, SIG, T)["delta"]); init = k * d0 * S0
    put("costInit", init, "{:.3f}")
    rebal = []
    for i, n in enumerate([13, 52, 252]):
        p0, _ = L.hedge_sim(S0, K, R, SIG, SIG, T, n, NPATH, rng_for(30 + i))
        p1, _ = L.hedge_sim(S0, K, R, SIG, SIG, T, n, NPATH, rng_for(30 + i), cost=k)
        c = float(np.mean(p0 - p1)); rebal.append(c - init)
        put(f"cost{WORDS[n]}", c, "{:.3f}"); put(f"costRebal{WORDS[n]}", c - init, "{:.3f}")
    put("costSlope", L.ols_slope([13, 52, 252], rebal), "{:.2f}")
    # Leland: sell and hedge at the adjusted volatility; k_rt is the round trip cost 2k
    n = 52; dt = T / n
    sL = SIG * math.sqrt(1 + math.sqrt(2 / math.pi) * 2 * k / (SIG * math.sqrt(dt)))
    put("lelandVol", 100 * sL, "{:.2f}"); put("lelandPrice", float(L.bs(S0, K, R, sL, T)), "{:.4f}")
    put("lelandMarkup", float(L.bs(S0, K, R, sL, T) - L.bs(S0, K, R, SIG, T)), "{:.3f}")
    p, _ = L.hedge_sim(S0, K, R, SIG, sL, T, n, NPATH, rng_for(33), cost=k)
    put("lelandMean", p.mean(), "{:.3f}"); put("lelandSE", p.std(ddof=1) / math.sqrt(NPATH), "{:.3f}")
    put("lelandSd", p.std(ddof=1), "{:.3f}")
    dL = float(L.greeks(S0, K, R, sL, T)["delta"]); put("lelandExInit", p.mean() + k * dL * S0, "{:.3f}")


def main():
    correct_vol(); wrong_vol(); costs()
    return finish("hedging_pnl")


if __name__ == "__main__":
    main()
