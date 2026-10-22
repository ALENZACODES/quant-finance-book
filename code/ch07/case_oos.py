"""Case study: would an investor who estimated the frontier from the previous five years of data have
beaten equal weights across the ten industries? Out-of-sample returns from July 1968 to August 2026
for seven rules, their Sharpe ratios, Memmel tests against equal weights, turnover and CAPM alphas.
Figures fig12_case; table tab7:oos. No random numbers. Writes out/case_returns.csv for the R check."""
import math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

W = 60
RULES = [("ew", "Equal weights"), ("cap", "Market weights"), ("gmv", "Minimum variance"),
         ("gmvlw", "Minimum variance, shrunk"), ("logmv", "Minimum variance, long only"),
         ("tan", "Tangency, scaled"), ("lotan", "Tangency, long only")]


def weights(rule, Xw, capw):
    n = Xw.shape[1]
    if rule == "ew":
        return np.ones(n) / n
    if rule == "cap":
        return capw
    S = np.cov(Xw.T); mx = Xw.mean(0)
    if rule == "gmv":
        return L.gmv(S)
    if rule == "gmvlw":
        return L.gmv(L.lw_identity(Xw)[0])
    if rule == "logmv":
        return L.long_only_gmv(S)
    if rule == "tan":
        # the direction of the sample tangency portfolio, scaled to the in-window volatility of equal
        # weights; the rest is held in bills, so the weights need not add to one
        y = np.linalg.solve(S, mx); e = np.ones(n) / n
        return y * math.sqrt(e @ S @ e) / math.sqrt(y @ S @ y)
    if rule == "lotan":
        if mx.max() <= 0:
            return L.long_only_gmv(S)
        return L.long_only_tangency(mx, S)
    raise ValueError(rule)


def run(f, vw, cap, W=W):
    X = vw.sub(f.rf, axis=0).values; Rt = vw.values; T, n = X.shape
    out = {r: [] for r, _ in RULES}; turn = {r: [] for r, _ in RULES}; prev = {r: None for r, _ in RULES}
    lotan_fallback = 0; shrink = []
    for t in range(W, T):
        capw = cap.iloc[t].values / cap.iloc[t].values.sum()
        for r, _ in RULES:
            w = weights(r, X[t - W:t], capw)
            if r == "lotan" and X[t - W:t].mean(0).max() <= 0:
                lotan_fallback += 1
            if r == "gmvlw":
                shrink.append(L.lw_identity(X[t - W:t])[1])
            out[r].append(w @ X[t])
            if prev[r] is not None:
                turn[r].append(np.abs(w - prev[r]).sum())
            rf = f.rf.values[t]; port = 1 + w @ Rt[t] + (1 - w.sum()) * rf
            prev[r] = w * (1 + Rt[t]) / port
    return {r: np.array(v) for r, v in out.items()}, {r: np.mean(v) for r, v in turn.items()}, lotan_fallback, np.array(shrink)


def main():
    f, vw, ew, cap = L.load()
    out, turn, fb, shrink = run(f, vw, cap)
    m = f.mkt_rf.values[W:]; T = len(m)
    R.put("csW", W, "{:.0f}"); R.put("csTT", T, "{:.0f}"); R.N["csStart"] = L.month_name(int(f.index[W]))
    R.N["csEnd"] = L.month_name(int(f.index[-1])); R.put("csFallback", fb, "{:.0f}")
    R.put("csMktMean", 1200 * m.mean())
    R.put("csShrinkMed", np.median(shrink), "{:.2f}")
    sr = lambda x: math.sqrt(12) * x.mean() / x.std(ddof=1)
    rows = []; tags = dict(ew="Ew", cap="Cap", gmv="Gmv", gmvlw="Lw", logmv="Logmv", tan="Tan", lotan="Lotan")
    fm = lambda v, d=2: (f"{v:.{d}f}" if v >= 0 else f"\\ensuremath{{-}}{-v:.{d}f}")
    for r, label in RULES:
        x = out[r]; c = L.ols(x, np.c_[np.ones(T), m], 6)
        z, p = L.memmel(x, out["ew"]) if r != "ew" else (float("nan"), float("nan"))
        tg = tags[r]
        R.put(f"cs{tg}Mean", 1200 * x.mean()); R.put(f"cs{tg}Vol", 100 * math.sqrt(12) * x.std(ddof=1), "{:.1f}")
        R.put(f"cs{tg}SR", sr(x)); R.put(f"cs{tg}Turn", 100 * turn[r], "{:.0f}")
        R.put(f"cs{tg}A", 1200 * c["b"][0]); R.put(f"cs{tg}At", c["b"][0] / c["nw"][0]); R.put(f"cs{tg}B", c["b"][1])
        if r in ("gmv", "gmvlw", "logmv"):    # residual share of variance and mean excess return levered to beta one
            R.put(f"cs{tg}Res", 100 * (1 - c["r2"]), "{:.0f}"); R.put(f"cs{tg}Lev", 1200 * x.mean() / c["b"][1], "{:.1f}")
        if r != "ew":
            R.put(f"cs{tg}Z", z); R.put(f"cs{tg}P", p)
        R.put(f"cs{tg}Worst", -100 * x.min(), "{:.1f}")
        zs = "" if r == "ew" else fm(z)
        rows.append(f"{label} & {fm(1200 * x.mean())} & {100 * math.sqrt(12) * x.std(ddof=1):.1f} & {fm(sr(x))} & {zs} & "
                    f"{100 * turn[r]:.0f} & {fm(1200 * c['b'][0])} & {c['b'][1]:.2f} \\\\")
    R.N["csTableRows"] = "\n".join(rows)
    best = max((sr(out[r]), r) for r, _ in RULES)[1]; R.N["csBest"] = dict(RULES)[best].lower()
    np.savetxt(R.HERE / "out" / "case_returns.csv", np.column_stack([out[r] for r, _ in RULES]), delimiter=",",
               header=",".join(r for r, _ in RULES), comments="", fmt="%.12g")
    # second window as a robustness check
    out2, turn2, _, _ = run(f, vw, cap, 120)
    for r in ("ew", "gmvlw", "tan", "lotan"):
        R.put(f"csTen{tags[r]}SR", sr(out2[r]))
    R.put("csTenTanZ", L.memmel(out2["tan"], out2["ew"])[0])
    R.put("csTenLwZ", L.memmel(out2["gmvlw"], out2["ew"])[0])
    # figure: rolling ten year Sharpe ratio minus that of equal weights
    yrs = (f.index.values[W:] // 100 + (f.index.values[W:] % 100 - 0.5) / 12)
    fig, axs = plt.subplots(2, 1, figsize=(5.9, 4.6), sharex=True)
    for r, col in (("ew", GREY), ("cap", SKY), ("gmvlw", BLUE), ("lotan", RED), ("tan", ROSE)):
        axs[0].plot(yrs, np.cumsum(np.log1p(out[r])), color=col, lw=1.1, label=dict(RULES)[r])
    axs[0].set_ylabel("cumulative log excess return"); axs[0].legend(loc="upper left", fontsize=7.2, ncol=2)
    K = 120
    for r, col in (("cap", SKY), ("gmvlw", BLUE), ("lotan", RED), ("tan", ROSE)):
        d = [sr(out[r][t - K:t]) - sr(out["ew"][t - K:t]) for t in range(K, T + 1)]
        axs[1].plot(yrs[K - 1:], d, color=col, lw=1.0)
    axs[1].axhline(0, color="#333333", lw=0.6); axs[1].set_ylabel("ten year Sharpe ratio\nminus equal weights")
    axs[1].set_xlabel("end of window")
    R.save(fig, "fig12_case")
    return R.finish("case_oos")


if __name__ == "__main__":
    main()
