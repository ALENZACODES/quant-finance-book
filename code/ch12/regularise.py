"""Section 12.1: bias, variance and regularisation. A sparse linear model with correlated predictors,
fitted by least squares, ridge and the lasso on many training samples, with the test error split into
squared bias and variance; and the coefficient paths of one sample. Also writes fig15_cv_oracle.png, a
separate simulation (random stream 13) of 10-fold cross-validation against the oracle penalty.

Run on its own:  python code/ch12/regularise.py   (writes code/ch12/out/regularise.json)
"""
from __future__ import annotations
import numpy as np
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

NTR, P, RHO, SIG, REPS, NTE = 100, 40, 0.6, 2.0, 300, 4000
BETA = np.zeros(P); BETA[[0, 8, 16, 24, 32]] = [1.0, -0.8, 0.6, -0.4, 0.2]
RIDGE = np.logspace(-1, 4, 41)
LASSO = np.logspace(-3, 0, 31)


def draw(n, rng, C):
    return rng.standard_normal((n, P)) @ C.T


def main():
    rng = rng_for(1)
    C = np.linalg.cholesky(L.toeplitz_corr(P, RHO))
    Xte = draw(NTE, rng, C); fte = Xte @ BETA
    put("bvN", NTR, "{:.0f}"); put("bvP", P, "{:.0f}"); put("bvRho", RHO, "{:.1f}"); put("bvSig", SIG, "{:.0f}")
    put("bvReps", REPS, "{:.0f}"); put("bvNte", NTE, "{:,.0f}"); put("bvNonzero", int((BETA != 0).sum()), "{:.0f}")
    put("bvSnr", BETA @ L.toeplitz_corr(P, RHO) @ BETA / SIG ** 2, "{:.2f}")
    pr = np.empty((REPS, len(RIDGE), NTE)); pl = np.empty((REPS, len(LASSO), NTE)); po = np.empty((REPS, NTE))
    nz = np.empty((REPS, len(LASSO))); kkt = 0.0
    for r in range(REPS):
        X = draw(NTR, rng, C); y = X @ BETA + SIG * rng.standard_normal(NTR)
        mx, my = X.mean(0), y.mean(); Xc, yc = X - mx, y - my
        Xt = Xte - mx
        po[r] = Xt @ np.linalg.lstsq(Xc, yc, rcond=None)[0] + my
        pr[r] = (L.ridge_path(Xc, yc, RIDGE) @ Xt.T) + my
        B = L.lasso_path(Xc, yc, LASSO)
        pl[r] = B @ Xt.T + my; nz[r] = (B != 0).sum(1)
        if r == 0:
            kkt = max(L.kkt_gap(Xc, yc, B[i], LASSO[i]) for i in range(len(LASSO)))
    def split(P_):
        m = P_.mean(0)
        return ((m - fte) ** 2).mean(-1), P_.var(0).mean(-1)
    bo, vo = split(po); br, vr = split(pr); bl, vl = split(pl)
    er, el = br + vr, bl + vl
    put("bvOlsBias", bo, "{:.3f}"); put("bvOlsVar", vo, "{:.3f}"); put("bvOlsErr", bo + vo, "{:.3f}")
    i = int(np.argmin(er)); j = int(np.argmin(el))
    put("bvRidgeLam", RIDGE[i], "{:.1f}"); put("bvRidgeErr", er[i], "{:.3f}"); put("bvRidgeBias", br[i], "{:.3f}")
    put("bvRidgeVar", vr[i], "{:.3f}")
    X0 = draw(NTR, rng_for(11), C); X0c = X0 - X0.mean(0)
    put("bvRidgeDf", L.ridge_df(X0c, RIDGE[i]), "{:.1f}")
    put("bvLassoLam", LASSO[j], "{:.3f}"); put("bvLassoErr", el[j], "{:.3f}"); put("bvLassoBias", bl[j], "{:.3f}")
    put("bvLassoVar", vl[j], "{:.3f}"); put("bvLassoNz", nz[:, j].mean(), "{:.1f}")
    pct("bvRidgeCut", 1 - er[i] / (bo + vo), 0); pct("bvLassoCut", 1 - el[j] / (bo + vo), 0)
    RAW["bvKkt"] = kkt
    put("bvRidgeSmall", er[0], "{:.3f}"); put("bvRidgeSmallLam", RIDGE[0], "{:.1f}")
    # figure 1: the split of the test error along the ridge path, with the lasso's total
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)
    for a, lam, b, v, nm in [(ax[0], RIDGE, br, vr, "ridge"), (ax[1], LASSO, bl, vl, "lasso")]:
        a.semilogx(lam, b, color=RED, ls="--", label="squared bias")
        a.semilogx(lam, v, color=SKY, ls=":", label="variance")
        a.semilogx(lam, b + v, color=BLUE, label="sum")
        a.axhline(bo + vo, color=GREY, lw=0.8); a.set_xlabel(r"penalty $\lambda$ (%s)" % nm)
    ax[0].text(RIDGE[1], (bo + vo) * 1.04, "least squares", color=GREY, fontsize=7.5)
    ax[0].set_ylabel("excess test error"); ax[0].set_ylim(0, 1.25 * (bo + vo))
    ax[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.0), ncol=3, fontsize=7.5)
    save(fig, "fig01_bias_variance")
    # figure 2: coefficient paths for one training sample
    X = draw(NTR, rng_for(11), C); y = X @ BETA + SIG * rng_for(12).standard_normal(NTR)
    Xc, yc = X - X.mean(0), y - y.mean()
    lam_r = np.logspace(-1, 5, 80); lam_l = np.logspace(-3, np.log10(L.lasso_lam_max(Xc, yc)) + 0.05, 80)
    Br = L.ridge_path(Xc, yc, lam_r); Bl = L.lasso_path(Xc, yc, lam_l)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)
    for a, lam, B, nm in [(ax[0], lam_r, Br, "ridge"), (ax[1], lam_l, Bl, "lasso")]:
        for k in range(P):
            on = BETA[k] != 0
            a.semilogx(lam, B[:, k], color=RED if on else PALE if not on else GREY, lw=1.1 if on else 0.6, zorder=3 if on else 1)
        a.set_xlabel(r"penalty $\lambda$ (%s)" % nm); a.axhline(0, color="#333333", lw=0.5)
    ax[0].set_ylabel("coefficient")
    save(fig, "fig02_paths")
    put("bvPathNzOne", int((Bl[np.argmin(np.abs(lam_l - LASSO[j]))] != 0).sum()), "{:.0f}")
    cv_figure(C)
    return finish("regularise")


NCV, KCV = 100, 10


def cv_figure(C):
    """A separate simulation on random stream 13: 100 training samples of the same model, each scored by
    10-fold cross-validation along the ridge and lasso grids, against the excess error of the full
    sample fit at each penalty, (b - beta)' Sigma (b - beta) + (intercept error)^2, which needs no test set."""
    rng = rng_for(13); S = L.toeplitz_corr(P, RHO)
    folds = L.kfold_blocks(NTR, KCV)
    out = {}
    for nm, grid, path in (("Ridge", RIDGE, L.ridge_path), ("Lasso", LASSO, L.lasso_path)):
        cv = np.empty((NCV, len(grid))); ex = np.empty((NCV, len(grid)))
        out[nm] = (cv, ex)
    for r in range(NCV):
        X = draw(NTR, rng, C); y = X @ BETA + SIG * rng.standard_normal(NTR)
        for nm, grid, path in (("Ridge", RIDGE, L.ridge_path), ("Lasso", LASSO, L.lasso_path)):
            cv, ex = out[nm]; err = np.zeros(len(grid))
            for te in folds:
                tr = np.setdiff1d(np.arange(NTR), te); mx, my = X[tr].mean(0), y[tr].mean()
                B = path(X[tr] - mx, y[tr] - my, grid)
                err += (((X[te] - mx) @ B.T + my - y[te][:, None]) ** 2).sum(0)
            cv[r] = err / NTR - SIG ** 2                     # less the noise, an estimate of the excess error
            mx, my = X.mean(0), y.mean(); B = path(X - mx, y - my, grid)
            D = B - BETA
            ex[r] = np.einsum("ij,jk,ik->i", D, S, D) + (my - mx @ B.T) ** 2
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.9), sharey=True)
    for a, (nm, grid) in zip(ax, (("Ridge", RIDGE), ("Lasso", LASSO))):
        cv, ex = out[nm]
        k_or = int(np.argmin(ex.mean(0))); k_cv = np.argmin(cv, axis=1)
        chosen = ex[np.arange(NCV), k_cv]
        fmt = "{:.1f}" if nm == "Ridge" else "{:.3f}"
        put(f"cv{nm}Oracle", grid[k_or], fmt); put(f"cv{nm}Med", np.median(grid[k_cv]), fmt)
        put(f"cv{nm}Lo", np.percentile(grid[k_cv], 10), fmt); put(f"cv{nm}Hi", np.percentile(grid[k_cv], 90), fmt)
        put(f"cv{nm}ErrOracle", ex[:, k_or].mean(), "{:.3f}"); put(f"cv{nm}ErrCv", chosen.mean(), "{:.3f}")
        pct(f"cv{nm}Extra", chosen.mean() / ex[:, k_or].mean() - 1, 0)
        a.fill_betweenx([0, 4], np.percentile(grid[k_cv], 10), np.percentile(grid[k_cv], 90), color=PALE, lw=0,
                        label="10th to 90th percentile of the chosen penalty")
        a.semilogx(grid, ex.mean(0), color=BLUE, label="excess error of the fit")
        a.semilogx(grid, cv.mean(0), color=RED, ls="--", label="cross-validation estimate")
        a.axvline(grid[k_or], color=GREY, lw=0.8, ls=":")
        a.set_xlabel(r"penalty $\lambda$ (%s)" % nm.lower())
    put("cvReps", NCV, "{:.0f}"); put("cvK", KCV, "{:.0f}")
    ax[0].set_ylabel("excess test error"); ax[0].set_ylim(0, 3.5)
    ax[0].legend(loc="lower center", bbox_to_anchor=(1.05, 1.0), ncol=3, fontsize=7.2)
    save(fig, "fig15_cv_oracle")


if __name__ == "__main__":
    import time; t = time.time(); main(); print(f"{time.time() - t:.1f}s")
