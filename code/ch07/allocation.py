"""Risk contributions and the equal-risk-contribution portfolio of the market and five long-short
factor premia, in sample and with rolling estimates, and the diversification return of rebalancing
the ten industries to equal weights. Figure fig11_risk. No random numbers."""
import math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

NAMES = ["Mkt", "SMB", "HML", "RMW", "CMA", "Mom"]
W = 60


def risk_parity(f):
    F = f[L.FACTORS].values; T, n = F.shape; S = np.cov(F.T)
    w_eq = np.ones(n) / n; w_erc = L.erc(S)
    rc_eq = L.risk_contrib(w_eq, S); rc_erc = L.risk_contrib(w_erc, S)
    R.RAW["rpEulerErr"] = float(abs(rc_eq.sum() - math.sqrt(w_eq @ S @ w_eq)))
    R.RAW["rpErcSpread"] = float(rc_erc.max() - rc_erc.min())
    R.put("rpEqMktShare", 100 * rc_eq[0] / rc_eq.sum(), "{:.0f}")
    R.put("rpErcMkt", 100 * w_erc[0], "{:.1f}"); R.put("rpErcMom", 100 * w_erc[5], "{:.1f}")
    R.put("rpErcCma", 100 * w_erc[4], "{:.1f}"); R.put("rpErcHml", 100 * w_erc[2], "{:.1f}")
    R.put("rpErcMax", 100 * w_erc.max(), "{:.1f}"); R.N["rpErcMaxName"] = NAMES[int(np.argmax(w_erc))]
    R.put("rpErcMin", 100 * w_erc.min(), "{:.1f}"); R.N["rpErcMinName"] = NAMES[int(np.argmin(w_erc))]
    sr = lambda x: math.sqrt(12) * x.mean() / x.std(ddof=1)
    R.put("rpSRmkt", sr(F[:, 0])); R.put("rpSRerc", sr(F @ w_erc)); R.put("rpSReq", sr(F @ w_eq))
    R.put("rpVolErc", 100 * math.sqrt(12 * w_erc @ S @ w_erc), "{:.1f}")
    R.put("rpLev", f.mkt_rf.std() / (F @ w_erc).std(), "{:.1f}")
    # rolling: weights from the previous W months, held one month
    out = []
    for t in range(W, T):
        St = np.cov(F[t - W:t].T); out.append(F[t] @ L.erc(St))
    out = np.array(out); R.put("rpSRrolling", sr(out)); R.put("rpSRmktSame", sr(F[W:, 0]))
    R.RAW["rpWerc"] = list(w_erc); R.RAW["rpRcErc"] = list(rc_erc / rc_erc.sum()); R.RAW["rpRcEq"] = list(rc_eq / rc_eq.sum())
    return rc_eq, rc_erc, w_erc


def rebalancing(f, vw):
    G = 1 + vw.values; T, n = G.shape; w = np.ones(n) / n
    reb = G @ w
    hold = np.cumprod(G, axis=0) @ w        # buy and hold from equal weights at the start
    g_reb = 12 * np.mean(np.log(reb)); g_hold = 12 * math.log(hold[-1]) / T
    g_assets = 12 * np.mean(np.log(G), axis=0)
    R.put("rbGrowReb", 100 * g_reb); R.put("rbGrowHold", 100 * g_hold); R.put("rbGrowAvg", 100 * g_assets @ w)
    S = np.cov(vw.values.T, bias=True)
    approx = 0.5 * (w @ np.diag(S) - w @ S @ w) * 12
    R.put("rbDivApprox", 100 * approx); R.put("rbDivActual", 100 * (g_reb - g_assets @ w))
    R.put("rbWealthReb", np.prod(reb), "{:,.0f}"); R.put("rbWealthHold", hold[-1], "{:,.0f}")
    last = np.cumprod(G, axis=0)[-1] / n; R.put("rbHoldTop", 100 * last.max() / last.sum(), "{:.0f}")
    R.N["rbHoldTopName"] = L.INDUSTRIES[int(np.argmax(last))]


def main():
    f, vw, ew, cap = L.load()
    rc_eq, rc_erc, w_erc = risk_parity(f)
    rebalancing(f, vw)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(NAMES))
    ax.bar(x - 0.2, 100 * rc_eq / rc_eq.sum(), 0.4, color=SKY, label="equal weights")
    ax.bar(x + 0.2, 100 * rc_erc / rc_erc.sum(), 0.4, color=RED, label="equal risk contributions")
    for i in x:
        ax.annotate(f"{100 * w_erc[i]:.0f}%", (i + 0.2, 100 * rc_erc[i] / rc_erc.sum()), ha="center", fontsize=7,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("share of portfolio volatility, %")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False); R.save(fig, "fig11_risk")
    return R.finish("allocation")


if __name__ == "__main__":
    main()
