"""The CAPM as a statement about the tangency portfolio, the security market line of the ten
industries, the covariance of a factor model and the principal components of industry returns.
Figures fig04_sml (market and tangency betas), fig05_pca and fig15_corrgap. No random numbers."""
import math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE


def main():
    f, vw, ew, cap = L.load()
    X = vw.sub(f.rf, axis=0).values; m = f.mkt_rf.values; T, n = X.shape
    mx = X.mean(0); S = np.cov(X.T)
    # beta relation holds exactly against the in-sample tangency portfolio
    wt = L.tangency(mx, S); rt = X @ wt
    beta_t = np.array([np.cov(X[:, i], rt)[0, 1] for i in range(n)]) / rt.var(ddof=1)
    R.RAW["cpTanBetaErr"] = float(np.max(np.abs(mx - beta_t * rt.mean())))
    # CAPM betas against the market factor and the cross-section
    betas = np.array([L.ols(X[:, i], np.c_[np.ones(T), m])["b"][1] for i in range(n)])
    cs = L.ols(mx, np.c_[np.ones(n), betas])
    R.put("smSlope", 1200 * cs["b"][1], "{:.2f}"); R.put("smIcpt", 1200 * cs["b"][0], "{:.2f}")
    R.put("smSlopeT", cs["t"][1], "{:.2f}"); R.put("smRsq", cs["r2"], "{:.2f}")
    R.put("smBetaMin", betas.min(), "{:.2f}"); R.put("smBetaMax", betas.max(), "{:.2f}")
    R.N["smBetaMinName"] = L.INDUSTRIES[int(np.argmin(betas))]; R.N["smBetaMaxName"] = L.INDUSTRIES[int(np.argmax(betas))]
    R.RAW["smBetas"] = list(betas)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.1), sharey=True)
    ax = axs[0]
    bb = np.linspace(0.4, 1.35, 10)
    ax.plot(bb, 1200 * m.mean() * bb, color=RED, ls="--", lw=1.1, label="slope equal to the premium")
    ax.plot(bb, 1200 * (cs["b"][0] + cs["b"][1] * bb), color=GREY, lw=1.0, label="least-squares line")
    ax.scatter(betas, 1200 * mx, color=BLUE, s=14, zorder=3)
    for i, nm in enumerate(L.INDUSTRIES):
        ax.annotate(nm, (betas[i], 1200 * mx[i]), fontsize=6.5, xytext={"Hlth": (-16, 5), "Enrgy": (4, -2), "NoDur": (-24, -1), "Manuf": (-40, -12), "Other": (4, -7), "HiTec": (-30, 3)}.get(nm, (3, 2)), textcoords="offset points")
    ax.set_xlabel("beta on the market factor"); ax.set_ylabel("mean excess return, % a year")
    ax.set_title("market factor", fontsize=9); ax.legend(loc="upper left", fontsize=7)
    ax = axs[1]
    bt = np.linspace(0.2, 1.3, 10)
    ax.plot(bt, 1200 * rt.mean() * bt, color=RED, ls="--", lw=1.1, label="slope equal to its mean")
    ax.scatter(beta_t, 1200 * mx, color=BLUE, s=14, zorder=3)
    for i, nm in enumerate(L.INDUSTRIES):
        if nm in ("HiTec", "Telcm", "Utils", "Other"):            # the rest crowd the middle of the line
            ax.annotate(nm, (beta_t[i], 1200 * mx[i]), fontsize=6.5, xytext=(4, -6), textcoords="offset points")
    ax.set_xlabel("beta on the tangency portfolio"); ax.set_title("in-sample tangency portfolio", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    R.put("cpTanBetaMin", beta_t.min(), "{:.2f}"); R.put("cpTanBetaMax", beta_t.max(), "{:.2f}")
    R.save(fig, "fig04_sml")
    # factor model covariance: six factors, diagonal residuals
    F = f[L.FACTORS].values; K = F.shape[1]
    Xf = np.c_[np.ones(T), F]; coef = np.linalg.lstsq(Xf, X, rcond=None)[0]; Bm = coef[1:].T
    E = X - Xf @ coef; Dg = np.diag(E.var(0, ddof=K + 1)); Om = np.cov(F.T)
    Sf = Bm @ Om @ Bm.T + Dg
    iD = np.diag(1 / np.diag(Dg))
    wood = iD - iD @ Bm @ np.linalg.inv(np.linalg.inv(Om) + Bm.T @ iD @ Bm) @ Bm.T @ iD
    R.RAW["fmWoodburyErr"] = float(np.max(np.abs(wood @ Sf - np.eye(n))))
    vol = lambda w, M: 100 * math.sqrt(12 * w @ M @ w)
    g_s, g_f = L.gmv(S), L.gmv(Sf)
    R.put("fmGmvSample", vol(g_s, S), "{:.2f}"); R.put("fmGmvFactor", vol(g_f, Sf), "{:.2f}")
    R.put("fmGmvFactorTrue", vol(g_f, S), "{:.2f}")
    R.put("fmCorrErr", np.max(np.abs(np.corrcoef(X.T) - Sf / np.sqrt(np.outer(np.diag(Sf), np.diag(Sf))))), "{:.2f}")
    corr_gap_figure(np.corrcoef(X.T) - Sf / np.sqrt(np.outer(np.diag(Sf), np.diag(Sf))), E)
    # principal components of the correlation matrix
    Cr = np.corrcoef(X.T); ev, V = np.linalg.eigh(Cr); ev, V = ev[::-1], V[:, ::-1]
    if V[:, 0].sum() < 0:
        V[:, 0] = -V[:, 0]
    share = ev / ev.sum()
    R.put("pcOne", 100 * share[0], "{:.0f}"); R.put("pcThree", 100 * share[:3].sum(), "{:.0f}")
    R.put("pcTwo", 100 * share[1], "{:.0f}")
    z = (X - X.mean(0)) / X.std(0, ddof=1); pc1 = z @ V[:, 0]
    R.put("pcMktCorr", np.corrcoef(pc1, m)[0, 1], "{:.2f}")
    R.RAW["pcEig"] = list(ev)
    pc2 = z @ V[:, 1]
    R.put("pcTwoHml", abs(np.corrcoef(pc2, f.hml.values)[0, 1]), "{:.2f}")
    R.put("pcTwoMkt", abs(np.corrcoef(pc2, m)[0, 1]), "{:.2f}")
    v2 = V[:, 1] * (1 if V[L.INDUSTRIES.index("HiTec"), 1] > 0 else -1)
    R.put("pcTwoHmlSigned", np.corrcoef(z @ v2, f.hml.values)[0, 1], "{:.2f}")
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.8))
    axs[0].bar(range(1, n + 1), 100 * share, color=BLUE); axs[0].set_xticks(range(1, n + 1))
    axs[0].set_xlabel("component"); axs[0].set_ylabel("share of total variance, %")
    x = np.arange(n); axs[1].bar(x - 0.2, V[:, 0], 0.4, color=BLUE, label="first")
    axs[1].bar(x + 0.2, v2, 0.4, color=RED, label="second")
    axs[1].set_xticks(x); axs[1].set_xticklabels(L.INDUSTRIES, rotation=90, fontsize=7)
    axs[1].set_ylabel("loading"); axs[1].legend(loc="lower left", fontsize=7.5); axs[1].axhline(0, color="#333333", lw=0.5)
    R.save(fig, "fig05_pca")
    return R.finish("capm_factors")


def corr_gap_figure(G, E):
    """Sample correlation of the industries minus the correlation implied by six factors with a diagonal
    residual covariance. The diagonal is zero by construction."""
    n = len(G); iu = np.triu_indices(n, 1)
    k = int(np.argmax(np.abs(G[iu]))); i, j = iu[0][k], iu[1][k]
    R.N["fmGapPair"] = f"{L.INDUSTRIES[i]} and {L.INDUSTRIES[j]}"; R.put("fmGapMax", G[i, j], "{:.2f}")
    R.put("fmGapPos", int(np.sum(G[iu] > 0)), "{:.0f}"); R.put("fmGapPairs", len(iu[0]), "{:.0f}")
    R.RAW["fmGapResCorr"] = float(np.corrcoef(E.T)[i, j])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    lim = np.max(np.abs(G))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bwr_book", [BLUE, "#FFFFFF", RED])
    U = np.ma.masked_where(~np.triu(np.ones((n, n), bool), 1), G)   # upper triangle: each pair once
    cmap.set_bad("white")
    im = ax.imshow(U, cmap=cmap, vmin=-lim, vmax=lim)
    ax.set_xticks(range(1, n)); ax.set_xticklabels(L.INDUSTRIES[1:], rotation=90, fontsize=7)
    ax.set_yticks(range(n - 1)); ax.set_yticklabels(L.INDUSTRIES[:-1], fontsize=7); ax.grid(False)
    ax.set_xlim(0.5, n - 0.5); ax.set_ylim(n - 1.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for a, b in zip(*iu):
        g = G[a, b]
        v = f"{g:+.2f}" if abs(g) >= 0.005 else f"{g:+.3f}"            # near-zero cells keep their sign
        ax.text(b, a, v.replace("-", "\u2212"), ha="center", va="center", fontsize=5.0,
                color="white" if abs(g) > 0.6 * lim else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    R.save(fig, "fig15_corrgap")


if __name__ == "__main__":
    main()
