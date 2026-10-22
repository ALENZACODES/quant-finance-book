"""Section 12.2: cross-validation when observations are ordered in time. A predictor with no power
over overlapping twelve month returns, scored by shuffled K-fold, by blocked folds and by purged and
embargoed folds; and the leak from choosing a predictor on the whole sample before cross-validating.

Run on its own:  python code/ch12/cv_time.py   (writes code/ch12/out/cv_time.json)
"""
from __future__ import annotations
import numpy as np
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE  # noqa: F401

T, H, K, PHI, MU, VOL, REPS, EMB = 480, 12, 10, 0.98, 0.005, 0.045, 1000, 12
NSEL, TSEL = 100, 480


def ar1(n, phi, rng, m=1):
    e = rng.standard_normal((n + 200, m)); x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def one(rng):
    r = MU + VOL * rng.standard_normal(T)
    x = ar1(T, PHI, rng)[:, 0]
    n = T - H
    y = np.array([r[t + 1:t + 1 + H].sum() for t in range(n)])
    X = x[:n, None]
    perm = rng.permutation(n)
    shuf = [np.sort(perm[f]) for f in L.kfold_blocks(n, K)]
    blocks = L.kfold_blocks(n, K)
    out = dict(
        shuffled=L.cv_r2(X, y, shuf, lambda te: np.setdiff1d(np.arange(n), te)),
        blocked=L.cv_r2(X, y, blocks, lambda te: np.setdiff1d(np.arange(n), te)),
        purged=L.cv_r2(X, y, blocks, lambda te: L.purged_train(n, te, H, EMB)))
    # a fresh sample from the same model, scored with the full-sample fit
    r2 = MU + VOL * rng.standard_normal(T); x2 = ar1(T, PHI, rng)[:, 0]
    y2 = np.array([r2[t + 1:t + 1 + H].sum() for t in range(n)])
    b = L.ols_fit(X, y)
    f = b[0] + b[1] * x2[:n]
    out["fresh"] = 1 - np.sum((y2 - f) ** 2) / np.sum((y2 - y.mean()) ** 2)
    return out


def selection(rng):
    """Monthly labels, no overlap; NSEL noise predictors. Choose the best on all the data, or inside each fold."""
    r = MU + VOL * rng.standard_normal(TSEL + 1)
    Xall = ar1(TSEL, PHI, rng, NSEL)
    y = r[1:]
    blocks = L.kfold_blocks(TSEL, K)
    def best(idx):
        xc = Xall[idx] - Xall[idx].mean(0); yc = y[idx] - y[idx].mean()
        return int(np.argmax(np.abs(xc.T @ yc) / np.sqrt((xc ** 2).sum(0))))
    j = best(np.arange(TSEL))
    outside = L.cv_r2(Xall[:, [j]], y, blocks, lambda te: np.setdiff1d(np.arange(TSEL), te))
    sse = sst = 0.0
    for te in blocks:
        tr = np.setdiff1d(np.arange(TSEL), te); k = best(tr)
        b = L.ols_fit(Xall[tr, [k]], y[tr]); f = b[0] + b[1] * Xall[te, k]
        sse += np.sum((y[te] - f) ** 2); sst += np.sum((y[te] - y[tr].mean()) ** 2)
    return outside, 1 - sse / sst


def main():
    rng = rng_for(2)
    res = [one(rng) for _ in range(REPS)]
    put("kfT", T, "{:.0f}"); put("kfH", H, "{:.0f}"); put("kfK", K, "{:.0f}"); put("kfPhi", PHI, "{:.2f}")
    put("kfReps", REPS, "{:,.0f}"); put("kfEmb", EMB, "{:.0f}"); put("kfN", T - H, "{:.0f}")
    pct("kfMu", MU, 1); pct("kfVol", VOL, 1)
    names = dict(shuffled="Shuf", blocked="Block", purged="Purge", fresh="Fresh")
    R = {k: np.array([d[k] for d in res]) for k in names}
    for k, nm in names.items():
        pct(f"kf{nm}Mean", R[k].mean(), 1); pct(f"kf{nm}Pos", (R[k] > 0).mean(), 0)
        pct(f"kf{nm}Med", np.median(R[k]), 1)
    pct("kfShufHigh", (R["shuffled"] > 0.05).mean(), 0)
    # correlation of overlapping labels, simulated against (h - k)/h
    r = rng_for(21).standard_normal(200000)
    y = np.convolve(r, np.ones(H), "valid")
    for k, nm in [(1, "One"), (6, "Six"), (11, "Eleven")]:
        put(f"kfCorr{nm}", np.corrcoef(y[:-k], y[k:])[0, 1], "{:.3f}"); put(f"kfCorrTh{nm}", (H - k) / H, "{:.3f}")
    # selection leak
    rs = rng_for(22); sel = np.array([selection(rs) for _ in range(400)])
    put("kfSelN", NSEL, "{:.0f}"); put("kfSelReps", 400, "{:.0f}"); put("kfSelT", TSEL, "{:.0f}")
    pct("kfSelOut", sel[:, 0].mean(), 2); pct("kfSelIn", sel[:, 1].mean(), 2)
    pct("kfSelOutPos", (sel[:, 0] > 0).mean(), 0); pct("kfSelInPos", (sel[:, 1] > 0).mean(), 0)
    # figure 3: the folds of the purged scheme
    n = T - H; blocks = L.kfold_blocks(n, K)
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    for i, te in enumerate(blocks[:5]):
        tr = L.purged_train(n, te, H, EMB)
        row = np.full(n, 0)
        row[tr] = 1; row[te] = 2
        for state, col in [(1, PALE), (2, RED)]:
            m = row == state
            ax.scatter(np.arange(n)[m], np.full(m.sum(), -i), marker="|", s=40, color=col, lw=0.8)
        gap = np.setdiff1d(np.arange(n), np.concatenate([tr, te]))
        ax.scatter(gap, np.full(len(gap), -i), marker="|", s=40, color=GREY, lw=0.8)
    ax.set_yticks(-np.arange(5)); ax.set_yticklabels([f"fold {i + 1}" for i in range(5)])
    ax.set_xlabel("month of the label"); ax.grid(False)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=c, marker="|", ls="", ms=9, mew=1.5) for c in (PALE, RED, GREY)]
    ax.legend(hs, ["training", "test", "purged or embargoed"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    save(fig, "fig03_purged_folds")
    # figure 4: the distributions of the cross-validated R^2
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bins = np.linspace(-0.25, 0.45, 57)
    ax.hist(100 * R["shuffled"], bins=100 * bins, color=ROSE, alpha=0.75, label="shuffled folds")
    ax.hist(100 * R["purged"], bins=100 * bins, histtype="step", color=BLUE, lw=1.2, label="purged and embargoed folds")
    ax.hist(100 * R["fresh"], bins=100 * bins, histtype="step", color=GREY, lw=1.0, ls="--", label="a fresh sample")
    ax.axvline(0, color="#333333", lw=0.6); ax.set_xlabel(r"out-of-sample $R^2$ (%)"); ax.set_ylabel("runs")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    save(fig, "fig04_cv_leak")
    return finish("cv_time")


if __name__ == "__main__":
    import time; t = time.time(); main(); print(f"{time.time() - t:.1f}s")
