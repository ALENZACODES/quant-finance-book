"""Diversification, the mean-variance frontier of the ten industry portfolios, the tangency portfolio,
long-only portfolios and the effect of estimation error on the sample tangency portfolio.
Figures fig01_diversify, fig02_frontier, fig03_estimation, fig13_weights and fig14_convergence; table
tab7:weights. Random streams 1 and 11."""
import itertools, math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

STREAM = 1


def data():
    f, vw, ew, cap = L.load()
    X = vw.sub(f.rf, axis=0).values
    return f, vw, cap, X


def diversification(X):
    S = np.cov(X.T); n = len(S)
    v = np.diag(S); c = S[np.triu_indices(n, 1)]
    rows = []
    for k in range(1, n + 1):
        vals = []
        for s in itertools.combinations(range(n), k):
            w = np.zeros(n); w[list(s)] = 1 / k; vals.append(w @ S @ w)
        vals = np.array(vals); rows.append((k, vals.mean(), vals.min(), vals.max()))
    rows = np.array(rows)
    theory = v.mean() / rows[:, 0] + (1 - 1 / rows[:, 0]) * c.mean()
    assert np.allclose(theory, rows[:, 1], rtol=1e-12)
    R.RAW["dvTheoryMaxErr"] = float(np.max(np.abs(theory / rows[:, 1] - 1)))
    ann = lambda x: 100 * np.sqrt(12 * x)
    R.put("dvOne", ann(v.mean()), "{:.1f}"); R.put("dvTen", ann(rows[-1, 1]), "{:.1f}")
    R.put("dvLimit", ann(c.mean()), "{:.1f}")
    R.put("dvShare", 100 * (1 - rows[-1, 1] / v.mean()), "{:.0f}")
    R.put("dvRhoBar", (c / np.sqrt(np.outer(v, v))[np.triu_indices(n, 1)]).mean(), "{:.2f}")
    R.put("dvShareInf", 100 * (1 - c.mean() / v.mean()), "{:.0f}")
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.fill_between(rows[:, 0], ann(rows[:, 2]), ann(rows[:, 3]), color=PALE, label="range over subsets")
    ax.plot(rows[:, 0], ann(rows[:, 1]), "o-", color=BLUE, ms=4, label="root of average variance")
    ax.axhline(ann(c.mean()), color=RED, ls="--", lw=1.1, label="limit, average covariance")
    ax.set_xlabel("number of industries held in equal weights"); ax.set_ylabel("volatility, % a year")
    ax.set_xticks(range(1, 11)); ax.legend(loc="upper right")
    R.save(fig, "fig01_diversify")


def frontier(f, cap, X):
    mx = X.mean(0); S = np.cov(X.T); n = len(mx)
    A, B, C, D = L.abcd(mx, S)
    w_g = L.gmv(S); w_t = L.tangency(mx, S); w_lt = L.long_only_tangency(mx, S); w_lg = L.long_only_gmv(S)
    sr = lambda w: math.sqrt(12) * (w @ mx) / math.sqrt(w @ S @ w)
    R.RAW["frA"], R.RAW["frB"], R.RAW["frC"], R.RAW["frD"] = A, B, C, D
    R.put("frGmvMu", 1200 * B / A, "{:.2f}"); R.put("frGmvVol", 100 * math.sqrt(12 / A), "{:.1f}")
    R.put("frTanMu", 1200 * w_t @ mx, "{:.2f}"); R.put("frTanVol", 100 * math.sqrt(12 * w_t @ S @ w_t), "{:.1f}")
    R.put("frTanSR", math.sqrt(12 * L.sharpe2(mx, S)), "{:.2f}"); R.RAW["frTanSRcheck"] = sr(w_t)
    R.put("frLoSR", sr(w_lt), "{:.2f}"); R.put("frLoGmvVol", 100 * math.sqrt(12 * w_lg @ S @ w_lg), "{:.1f}")
    R.put("frLoN", int(np.sum(w_lt > 1e-9)), "{:.0f}"); R.N["frLoNword"] = R.word(int(np.sum(w_lt > 1e-9))).lower()
    m = f.mkt_rf.values
    R.put("frMktSR", math.sqrt(12) * m.mean() / m.std(ddof=1), "{:.2f}")
    R.put("frMktMu", 1200 * m.mean(), "{:.2f}"); R.put("frMktVol", 100 * math.sqrt(12) * m.std(ddof=1), "{:.1f}")
    R.put("frGross", np.abs(w_t).sum(), "{:.2f}"); R.put("frShortMax", -100 * w_t.min(), "{:.0f}")
    R.N["frShortName"] = L.INDUSTRIES[int(np.argmin(w_t))]
    capw = cap.iloc[-1].values / cap.iloc[-1].values.sum()
    R.RAW["wTables"] = dict(gmv=list(w_g), tan=list(w_t), lotan=list(w_lt), logmv=list(w_lg), cap=list(capw))
    fm = lambda v: (f"{v:.1f}" if v >= -0.05 else f"\\ensuremath{{-}}{-v:.1f}")
    rows = [f"{nm} & {fm(100 * w_g[k])} & {fm(100 * w_t[k])} & {fm(100 * w_lg[k])} & {fm(100 * w_lt[k])} & "
            f"{fm(100 * capw[k])} \\\\" for k, nm in enumerate(L.INDUSTRIES)]
    vol = lambda w: 100 * math.sqrt(12 * w @ S @ w); mu_ = lambda w: 1200 * w @ mx
    rows.append("\\midrule")
    rows.append("Mean, \\% a year & " + " & ".join(f"{mu_(w):.2f}" for w in (w_g, w_t, w_lg, w_lt, capw)) + " \\\\")
    rows.append("Volatility, \\% a year & " + " & ".join(f"{vol(w):.1f}" for w in (w_g, w_t, w_lg, w_lt, capw)) + " \\\\")
    rows.append("Sharpe ratio & " + " & ".join(f"{sr(w):.2f}" for w in (w_g, w_t, w_lg, w_lt, capw)) + " \\\\")
    R.N["frTableRows"] = "\n".join(rows)
    R.put("frCapSR", sr(capw), "{:.2f}")
    # KKT certificate of the long-only tangency, through the scaled problem
    y, lam, nu = L.qp_enum(S, mx[None, :], np.array([1.0]))
    R.RAW["frLoKktMinNu"] = float(nu.min()); R.RAW["frLoKktComp"] = float(np.max(np.abs(nu * y)))
    # figure
    ann_m = lambda x: 1200 * x; ann_s = lambda v: 100 * np.sqrt(12 * v)
    ms = np.linspace(min(B / A, mx.min()) - 0.002, mx.max() + 0.004, 300)
    fv = np.array([L.frontier_var(mx, S, q) for q in ms])
    lo_m = np.linspace(w_lg @ mx, mx.max(), 40)
    lo_v = np.array([(lambda w: w @ S @ w)(L.long_only_frontier_w(mx, S, q)) for q in lo_m])
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(ann_s(fv), ann_m(ms), color=BLUE, label="frontier, short sales allowed")
    ax.plot(ann_s(lo_v), ann_m(lo_m), color=RED, label="frontier, long only")
    sv = np.linspace(0, 26, 50); ax.plot(sv, sv * math.sqrt(12 * L.sharpe2(mx, S)) , color=GREY, lw=0.9, ls="--",
                                         label="capital market line")
    ax.scatter(ann_s(np.diag(S)), ann_m(mx), s=14, color=SKY, zorder=3)
    for i, nm in enumerate(L.INDUSTRIES):
        ax.annotate(nm, (ann_s(S[i, i]), ann_m(mx[i])), fontsize=7, xytext=(3, -2), textcoords="offset points")
    ax.scatter([ann_s(w_t @ S @ w_t)], [ann_m(w_t @ mx)], color=BLUE, marker="D", s=22, zorder=4, label="tangency")
    ax.scatter([100 * math.sqrt(12) * m.std(ddof=1)], [ann_m(m.mean())], color=RED, marker="s", s=20, zorder=4,
               label="market factor")
    ax.set_xlim(0, 27); ax.set_ylim(0, 13)
    ax.set_xlabel("volatility, % a year"); ax.set_ylabel("mean excess return, % a year"); ax.legend(loc="lower right")
    R.save(fig, "fig02_frontier")
    weights_figure(mx, S, A, B, C)
    return mx, S


def weights_figure(mx, S, A, B, C):
    """Weights of the unconstrained frontier against its mean: each is a straight line in m."""
    ms = np.linspace(0.0, 0.20 / 12, 81)
    Wm = np.array([L.frontier_w(mx, S, q) for q in ms])
    # affine in m: second differences vanish
    R.RAW["fwAffineErr"] = float(np.max(np.abs(np.diff(Wm, 2, axis=0))))
    m_g, m_t = B / A, C / B
    w_g, w_t = L.frontier_w(mx, S, m_g), L.frontier_w(mx, S, m_t)
    R.RAW["fwGmvErr"] = float(np.max(np.abs(w_g - L.gmv(S)))); R.RAW["fwTanErr"] = float(np.max(np.abs(w_t - L.tangency(mx, S))))
    m2 = 2 * m_t; w2 = L.frontier_w(mx, S, m2)
    R.put("fwDoubleMu", 1200 * m2, "{:.1f}"); R.put("fwDoubleGross", np.abs(w2).sum(), "{:.1f}")
    R.put("fwDoubleShort", -100 * w2.min(), "{:.0f}"); R.N["fwDoubleShortName"] = L.INDUSTRIES[int(np.argmin(w2))]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    cols = [BLUE, RED, GREY, SKY, ROSE, BLUE, RED, GREY, SKY, ROSE]
    styles = ["-"] * 5 + ["--"] * 5
    for i, nm in enumerate(L.INDUSTRIES):
        ax.plot(1200 * ms, 100 * Wm[:, i], color=cols[i], ls=styles[i], lw=1.0, label=nm)
    # name each line at its right end, spreading names that would collide
    ends = 100 * Wm[-1]; order = np.argsort(ends); gap = 0.045 * (100 * Wm.max() - 100 * Wm.min() + 20)
    ypos = ends[order].copy()
    for k in range(1, len(ypos)):
        ypos[k] = max(ypos[k], ypos[k - 1] + gap)
    ypos -= max(0.0, ypos[-1] - (100 * Wm.max() + 10 - gap / 2))
    for k in range(len(ypos) - 2, -1, -1):
        ypos[k] = min(ypos[k], ypos[k + 1] - gap)
    for k, i in enumerate(order):
        ax.annotate(L.INDUSTRIES[i], (20, ends[i]), xytext=(20.4, ypos[k]), textcoords="data", fontsize=7,
                    color=cols[i], va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=cols[i], lw=0.5, shrinkA=0, shrinkB=0))
    lo, hi = 100 * Wm.min() - 10, 100 * Wm.max() + 10
    for q, lab in ((m_g, "minimum variance"), (m_t, "tangency")):
        ax.axvline(1200 * q, color="#999999", lw=0.8, ls=":")
        ax.annotate(lab, (1200 * q, lo), fontsize=7, rotation=90, ha="right", va="bottom", xytext=(-2, 3),
                    textcoords="offset points")
    ax.axhline(0, color="#333333", lw=0.5)
    ax.annotate("zero weight", (13.2, 0), xytext=(0, 2), textcoords="offset points", fontsize=7, color="#333333")
    ax.set_xlim(0, 20); ax.set_ylim(lo, hi)
    ax.set_xlabel("mean excess return of the frontier portfolio, % a year"); ax.set_ylabel("weight, %")
    R.save(fig, "fig13_weights")


def estimation(mx, S, rng):
    n = len(mx); th2 = L.sharpe2(mx, S); Lc = np.linalg.cholesky(S)
    R.put("esTheta", math.sqrt(12 * th2), "{:.2f}")
    res = {}
    M = 4000
    for T in (60, 120, 240):
        ins = np.empty(M); oos = np.empty(M)
        for j in range(M):
            Z = mx + rng.standard_normal((T, n)) @ Lc.T
            m_hat = Z.mean(0); S_hat = np.cov(Z.T)
            y = np.linalg.solve(S_hat, m_hat)
            ins[j] = m_hat @ y; oos[j] = (y @ mx) / math.sqrt(y @ S @ y)
        res[T] = (ins, oos)
        nm = {60: "Sixty", 120: "OneTwenty", 240: "TwoForty"}[T]
        R.put(f"esIn{nm}", math.sqrt(12 * np.mean(ins)), "{:.2f}")
        R.put(f"esInTheory{nm}", math.sqrt(12 * L.sharpe2_bias(th2, n, T)), "{:.2f}")
        R.put(f"esOut{nm}", math.sqrt(12) * np.mean(oos), "{:.2f}")
        R.put(f"esNeg{nm}", 100 * np.mean(oos < 0), "{:.0f}")
        R.RAW[f"esInSe{nm}"] = float(np.std(ins, ddof=1) / math.sqrt(M))
        R.RAW[f"esInMean{nm}"] = float(np.mean(ins)); R.RAW[f"esInTheoryRaw{nm}"] = L.sharpe2_bias(th2, n, T)
    R.put("esM", M, "{:,.0f}")
    fig, axs = plt.subplots(1, 2, figsize=(5.9, 2.9), sharey=True)
    for ax, T in zip(axs, (60, 240)):
        ins, oos = res[T]; bins = np.linspace(-1.5, 3.5, 51)
        ax.hist(np.sqrt(12 * ins), bins=bins, color=SKY, alpha=0.8, label="in sample")
        ax.hist(math.sqrt(12) * oos, bins=bins, color=ROSE, alpha=0.8, label="true Sharpe ratio")
        ax.axvline(math.sqrt(12 * th2), color=BLUE, lw=1.2); ax.set_title(f"{T} months")
        ax.set_xlabel("Sharpe ratio, a year")
    axs[0].set_ylabel("simulations"); axs[0].legend(loc="upper right", fontsize=7.5)
    R.save(fig, "fig03_estimation")
    convergence_figure(mx, S)


def convergence_figure(mx, S):
    """The in-sample root mean square Sharpe ratio, its exact value from the bias formula, and the
    average true Sharpe ratio of the estimated tangency direction, against the length of the sample.
    Its own random stream, so the numbers of estimation() are unchanged."""
    rng = R.rng_for(11)
    n = len(mx); th2 = L.sharpe2(mx, S); Lc = np.linalg.cholesky(S)
    Ts = np.array([36, 48, 60, 84, 120, 180, 240, 300, 360, 480, 600]); M = 2000
    ins, oos, theo = [], [], []
    for T in Ts:
        a = np.empty(M); b = np.empty(M)
        for j in range(M):
            Z = mx + rng.standard_normal((T, n)) @ Lc.T
            m_hat = Z.mean(0); y = np.linalg.solve(np.cov(Z.T), m_hat)
            a[j] = m_hat @ y; b[j] = (y @ mx) / math.sqrt(y @ S @ y)
        ins.append(math.sqrt(12 * a.mean())); oos.append(math.sqrt(12) * b.mean())
        theo.append(math.sqrt(12 * L.sharpe2_bias(th2, n, int(T))))
    ins, oos, theo = map(np.array, (ins, oos, theo))
    R.RAW["cvMaxRelErr"] = float(np.max(np.abs(ins / theo - 1)))
    R.put("cvM", M, "{:,.0f}"); R.put("cvTmax", Ts[-1], "{:.0f}")
    R.put("cvInMax", ins[-1], "{:.2f}"); R.put("cvOutMax", oos[-1], "{:.2f}")
    R.put("cvShareMax", 100 * oos[-1] / math.sqrt(12 * th2), "{:.0f}")
    R.RAW["cvOut"] = list(map(float, oos)); R.RAW["cvTs"] = list(map(int, Ts))
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    tt = np.arange(30, 601)
    ax.plot(tt, [math.sqrt(12 * L.sharpe2_bias(th2, n, int(T))) for T in tt], color=BLUE, lw=1.0,
            label="in sample, exact")
    ax.plot(Ts, ins, "o", color=BLUE, ms=3.5, label="in sample, simulated")
    ax.plot(Ts, oos, "o-", color=RED, ms=3.5, lw=1.0, label="true Sharpe ratio of the estimate")
    ax.axhline(math.sqrt(12 * th2), color=GREY, lw=1.0, ls="--", label="best available")
    ax.set_xlabel("months of data, $T$"); ax.set_ylabel("Sharpe ratio, a year")
    ax.set_xlim(30, 600); ax.set_ylim(0, 3.0); ax.legend(loc="upper right")
    R.save(fig, "fig14_convergence")


def main():
    rng = R.rng_for(STREAM)
    f, vw, cap, X = data()
    R.put("fqT", len(X), "{:.0f}"); R.N["fqFirst"] = L.month_name(int(f.index[0])); R.N["fqLast"] = L.month_name(int(f.index[-1]))
    diversification(X)
    mx, S = frontier(f, cap, X)
    estimation(mx, S, rng)
    return R.finish("frontier")


if __name__ == "__main__":
    main()
