"""Black-Litterman for the ten industries at the end of August 2026: equilibrium returns from market
weights, two views, the posterior and the implied weights, against unconstrained mean-variance
weights from sample means. Figure fig10_bl. No random numbers."""
import math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

W = 60


def main():
    f, vw, ew, cap = L.load()
    X = vw.sub(f.rf, axis=0).values; m = f.mkt_rf.values
    S = np.cov(X[-W:].T); wm = cap.iloc[-1].values / cap.iloc[-1].values.sum()
    delta = m.mean() / m.var(ddof=1)                     # long-run premium over long-run variance
    pi = delta * S @ wm; tau = 1 / W
    n = len(wm); P = np.zeros((2, n)); iH, iT, iE = (L.INDUSTRIES.index(k) for k in ("Hlth", "HiTec", "Enrgy"))
    P[0, iH], P[0, iT] = 1, -1; P[1, iE] = 1
    q = np.array([0.02, 0.05]) / 12
    Om = np.diag(np.diag(P @ (tau * S) @ P.T))
    mu = L.bl_posterior(pi, S, P, q, Om, tau); mu2 = L.bl_posterior_info(pi, S, P, q, Om, tau)
    R.RAW["blFormErr"] = float(np.max(np.abs(mu - mu2)))
    w_bl = np.linalg.solve(delta * S, mu)
    lam = np.linalg.lstsq(P.T, w_bl - wm, rcond=None)[0]
    R.RAW["blSpanErr"] = float(np.max(np.abs(P.T @ lam - (w_bl - wm))))
    R.RAW["blPiWeightsErr"] = float(np.max(np.abs(np.linalg.solve(delta * S, pi) - wm)))
    w_mv = np.linalg.solve(delta * S, X[-W:].mean(0))
    R.put("blDelta", delta, "{:.2f}"); R.put("blTau", tau, "{:.4f}"); R.put("blW", W, "{:.0f}")
    R.N["blTauFrac"] = f"1/{W}"
    for k, nm in ((iH, "Hlth"), (iT, "HiTec"), (iE, "Enrgy")):
        R.put(f"blPi{nm}", 1200 * pi[k]); R.put(f"blMu{nm}", 1200 * mu[k])
        R.put(f"blWm{nm}", 100 * wm[k], "{:.1f}"); R.put(f"blWbl{nm}", 100 * w_bl[k], "{:.1f}")
        R.put(f"blWmv{nm}", 100 * w_mv[k], "{:.0f}")
    R.put("blViewOneBefore", 1200 * (pi[iH] - pi[iT])); R.put("blViewOneAfter", 1200 * (mu[iH] - mu[iT]))
    R.put("blViewTwoBefore", 1200 * pi[iE]); R.put("blViewTwoAfter", 1200 * mu[iE])
    R.put("blLamOne", lam[0], "{:.3f}"); R.put("blLamTwo", lam[1], "{:.3f}")
    R.put("blMvGross", np.abs(w_mv).sum(), "{:.1f}"); R.put("blMvSum", w_mv.sum(), "{:.2f}")
    R.put("blMvMax", 100 * np.abs(w_mv).max(), "{:.0f}"); R.N["blMvMaxName"] = L.INDUSTRIES[int(np.argmax(np.abs(w_mv)))]
    R.put("blBlSum", w_bl.sum(), "{:.3f}")
    R.put("blPiMin", 1200 * pi.min()); R.put("blPiMax", 1200 * pi.max())
    R.N["blPiMinName"] = L.INDUSTRIES[int(np.argmin(pi))]; R.N["blPiMaxName"] = L.INDUSTRIES[int(np.argmax(pi))]
    R.RAW["blPi"] = list(pi); R.RAW["blMu"] = list(mu); R.RAW["blWm"] = list(wm); R.RAW["blWbl"] = list(w_bl)
    R.RAW["blWmv"] = list(w_mv)
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(n)
    axs[0].bar(x - 0.2, 100 * wm, 0.4, color=SKY, label="market")
    axs[0].bar(x + 0.2, 100 * w_bl, 0.4, color=RED, label="Black-Litterman")
    axs[0].set_xticks(x); axs[0].set_xticklabels(L.INDUSTRIES, rotation=90, fontsize=7)
    axs[0].set_ylabel("weight, %"); axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, frameon=False); axs[0].axhline(0, color="#333333", lw=0.5)
    axs[1].bar(x, 100 * w_mv, 0.6, color=BLUE)
    axs[1].set_xticks(x); axs[1].set_xticklabels(L.INDUSTRIES, rotation=90, fontsize=7)
    axs[1].set_title("sample means", fontsize=9); axs[1].axhline(0, color="#333333", lw=0.5)
    R.save(fig, "fig10_bl")
    return R.finish("black_litterman")


if __name__ == "__main__":
    main()
