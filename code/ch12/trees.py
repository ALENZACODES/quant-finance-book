"""Section 12.3: regression trees, bagging, random forests and gradient boosting on Friedman's
benchmark function, the correlation that limits a forest, and a one-dimensional picture of a tree
and of boosting. Also writes fig16_shrink.png, the analytic shrinkage of boosting and ridge.

Run on its own:  python code/ch12/trees.py   (writes code/ch12/out/trees.json)
"""
from __future__ import annotations
import math
import numpy as np
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

NTR, NTE, P, SIG, B, M, NU = 500, 5000, 10, 1.0, 300, 1000, 0.1


def friedman(X):
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]


def main():
    rng = rng_for(3)
    X = rng.uniform(size=(NTR, P)); y = friedman(X) + SIG * rng.standard_normal(NTR)
    Xt = rng.uniform(size=(NTE, P)); ft = friedman(Xt)
    put("tbN", NTR, "{:.0f}"); put("tbNte", NTE, "{:,.0f}"); put("tbP", P, "{:.0f}"); put("tbB", B, "{:.0f}")
    put("tbM", M, "{:,.0f}"); put("tbNu", NU, "{:.1f}")
    err = lambda f: float(np.mean((f - ft) ** 2))       # error against the true function: add SIG^2 for new data
    put("tbVarF", ft.var(), "{:.1f}")
    # the variance of f under the uniform law, by quadrature: the four terms are independent
    from scipy import integrate
    m1 = integrate.dblquad(lambda u, v: 10 * np.sin(np.pi * u * v), 0, 1, 0, 1)[0]
    m2 = integrate.dblquad(lambda u, v: (10 * np.sin(np.pi * u * v)) ** 2, 0, 1, 0, 1)[0]
    var_pop = (m2 - m1 ** 2) + (400 / 80 - (20 / 12) ** 2) + 100 / 12 + 25 / 12
    put("tbVarFPop", var_pop, "{:.1f}")
    b = L.ols_fit(X, y); put("tbOls", err(b[0] + Xt @ b[1:]), "{:.2f}")
    # single trees: depth chosen by 5-fold cross-validation
    folds = L.kfold_blocks(NTR, 5); perm = rng.permutation(NTR)
    cv = {}
    for d in range(1, 13):
        s = 0.0
        for f in folds:
            te = perm[f]; tr = np.setdiff1d(perm, te)
            s += np.sum((L.Tree(d, 5).fit(X[tr], y[tr]).predict(X[te]) - y[te]) ** 2)
        cv[d] = s / NTR
    dbest = min(cv, key=cv.get)
    put("tbDepth", dbest, "{:.0f}"); t1 = L.Tree(dbest, 5).fit(X, y); put("tbTree", err(t1.predict(Xt)), "{:.2f}")
    put("tbTreeLeaves", t1.leaves, "{:.0f}")
    deep = L.Tree(30, 5).fit(X, y); put("tbDeep", err(deep.predict(Xt)), "{:.2f}"); put("tbDeepLeaves", deep.leaves, "{:.0f}")
    bag = L.forest(X, y, B, rng_for(31), None); pb = L.forest_predict(bag, Xt, cumulative=True)
    rf = L.forest(X, y, B, rng_for(32), 3); pf = L.forest_predict(rf, Xt, cumulative=True)
    eb = ((pb - ft) ** 2).mean(1); ef = ((pf - ft) ** 2).mean(1)
    put("tbBag", eb[-1], "{:.2f}"); put("tbRf", ef[-1], "{:.2f}"); put("tbMtry", 3, "{:.0f}")
    f0, trees, path = L.boost(X, y, M, NU, 3, 5, Xtest=Xt); ebo = ((path - ft) ** 2).mean(1)
    put("tbBoost", ebo[-1], "{:.2f}"); put("tbBoostBest", ebo.min(), "{:.2f}"); put("tbBoostAt", int(np.argmin(ebo)) + 1, "{:.0f}")
    put("tbBoostHundred", ebo[99], "{:.2f}")
    _, _, path1 = L.boost(X, y, 200, 1.0, 3, 5, Xtest=Xt); ebo1 = ((path1 - ft) ** 2).mean(1)
    put("tbBoostOne", ebo1.min(), "{:.2f}"); put("tbBoostOneAt", int(np.argmin(ebo1)) + 1, "{:.0f}")
    put("tbBoostOneEnd", ebo1[-1], "{:.2f}")
    # the forest variance identity: rho sigma^2 + (1 - rho) sigma^2 / B over training sets, at fixed points
    R, Bv, x0 = 40, 50, rng.uniform(size=(200, P))
    rv = rng_for(33); preds = np.empty((R, Bv, len(x0)))
    for r in range(R):
        Xr = rv.uniform(size=(NTR, P)); yr = friedman(Xr) + SIG * rv.standard_normal(NTR)
        preds[r] = np.array([t.predict(x0) for t in L.forest(Xr, yr, Bv, rv, 3)])
    s2 = preds.reshape(R * Bv, -1).var(0)                       # variance of one tree, over data and randomisation
    pairs = np.array([np.mean((preds[:, i] - preds[:, i].mean(0)) * (preds[:, j] - preds[:, j].mean(0)), 0)
                      for i, j in zip(range(0, Bv, 2), range(1, Bv, 2))]).mean(0)
    rho = pairs / s2
    vavg = preds.mean(1).var(0)
    pred_formula = rho * s2 + (1 - rho) * s2 / Bv
    put("tbRho", pairs.mean() / s2.mean(), "{:.3f}");               # pooled: average covariance over average variance
    assert abs(s2.mean() * (pairs.mean() / s2.mean() + (1 - pairs.mean() / s2.mean()) / Bv) - np.mean(pred_formula)) < 1e-12
    put("tbSigTwo", np.mean(s2), "{:.2f}")
    put("tbVarAvg", np.mean(vavg), "{:.2f}"); put("tbVarFormula", np.mean(pred_formula), "{:.2f}")
    put("tbVarR", R, "{:.0f}"); put("tbVarB", Bv, "{:.0f}")
    RAW["tbVarRatio"] = float(np.mean(vavg) / np.mean(pred_formula))
    # figure 5: one predictor, a tree of depth 2 and 5, and boosting
    r5 = rng_for(34); x = np.sort(r5.uniform(size=200)); g = lambda x: np.sin(2 * np.pi * x) + (x > 0.7)
    yy = g(x) + 0.3 * r5.standard_normal(200); xx = np.linspace(0, 1, 600)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True)
    ax[0].scatter(x, yy, s=4, color=PALE, edgecolor=SKY, lw=0.3)
    ax[0].plot(xx, g(xx), color=GREY, lw=0.9, ls=":", label="true function")
    for d, col, ls in [(2, RED, "--"), (5, BLUE, "-")]:
        ax[0].plot(xx, L.Tree(d, 5).fit(x[:, None], yy).predict(xx[:, None]), color=col, ls=ls, lw=1.1, label=f"tree, depth {d}")
    ax[1].scatter(x, yy, s=4, color=PALE, edgecolor=SKY, lw=0.3)
    ax[1].plot(xx, g(xx), color=GREY, lw=0.9, ls=":")
    for m, col, ls in [(10, RED, "--"), (200, BLUE, "-")]:
        f0b, tr, _ = L.boost(x[:, None], yy, m, NU, 1, 5)
        ax[1].plot(xx, L.boost_predict(f0b, tr, xx[:, None], NU), color=col, ls=ls, lw=1.1, label=f"boosted stumps, {m} rounds")
    for a in ax:
        a.set_xlabel("$x$"); a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=7.2)
    save(fig, "fig05_tree_boost")
    # figure 6: test error against the number of trees
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    k = np.arange(1, B + 1)
    ax.plot(k, eb, color=GREY, lw=1.1, label="bagging")
    ax.plot(k, ef, color=BLUE, lw=1.3, label=f"random forest, {3} of {P} variables per split")
    kb = np.arange(1, M + 1)
    ax.plot(kb, ebo, color=RED, lw=1.3, label=rf"boosting, depth 3, $\nu={NU}$")
    ax.plot(np.arange(1, 201), ebo1, color=ROSE, lw=1.0, ls="--", label=r"boosting, depth 3, $\nu=1$")
    ax.axhline(RAW["tbTree"], color=GREY, ls=":", lw=0.9); ax.text(600, RAW["tbTree"] + 0.3, f"single tree, depth {dbest}", fontsize=7.5, color=GREY)
    ax.axhline(RAW["tbOls"], color=SKY, ls=":", lw=0.9); ax.text(600, RAW["tbOls"] + 0.3, "linear regression", fontsize=7.5, color=SKY)
    ax.set_xscale("log"); ax.set_xlabel("number of trees"); ax.set_ylabel("test error against the true function")
    ax.set_ylim(0, max(RAW["tbTree"], RAW["tbOls"]) * 1.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.3)
    save(fig, "fig06_ensembles")
    shrink_figure()
    return finish("trees")


def shrink_figure():
    """Analytic: the share of one eigencomponent fitted after M rounds of boosting a linear smoother with
    eigenvalue s at rate nu, 1 - (1 - nu s)^M, against the ridge factor d^2/(d^2 + lambda)."""
    M = np.arange(1, 501)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8), sharey=True)
    for s, col in ((0.9, BLUE), (0.3, SKY), (0.05, RED)):
        for nu, ls in ((1.0, "--"), (0.1, "-")):
            ax[0].semilogx(M, 1 - (1 - nu * s) ** M, color=col, ls=ls, lw=1.1,
                           label=rf"$s={s}$, $\nu={nu:g}$")
    for s, nm in ((0.05, "Small"), (0.9, "Big")):
        put(f"skHalf{nm}", math.ceil(np.log(0.5) / np.log(1 - 0.1 * s)), "{:.0f}")  # whole rounds to fit half, nu = 0.1
    lam = np.logspace(-2, 3, 200)
    for d2, col in ((90.0, BLUE), (3.0, SKY), (0.05, RED)):
        ax[1].semilogx(lam, d2 / (d2 + lam), color=col, label=rf"$d^2={d2:g}$")
    ax[0].set_xlabel("rounds of boosting $M$"); ax[0].set_ylabel("share of the component fitted")
    ax[1].set_xlabel(r"ridge penalty $\lambda$")
    ax[0].legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax[1].legend(fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig16_shrink")


if __name__ == "__main__":
    import time; t = time.time(); main(); print(f"{time.time() - t:.1f}s")
