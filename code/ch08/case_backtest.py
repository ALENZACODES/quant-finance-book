"""Section 8.10, case study: one day 99% VaR and 97.5% ES for the US market, July 1967 to August
2026, by seven models, each using only data up to the day before; the same models on USD/CAD.
Writes fig09_case.png, fig16_crisis.png, out/case_hits.csv and the numbers registered below."""
import math
import numpy as np, pandas as pd
from scipy import stats
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, put, pct, finish

A, AES = 0.99, 0.975
WARM, REFIT = 1000, 250
MODELS = [("nor", "Normal, 250 days"), ("hs", "Historical, 250 days"), ("hsl", "Historical, 1,000 days"),
          ("ew", "EWMA, normal"), ("gn", "GARCH, normal"), ("gt", "GARCH, $t$"),
          ("fhs", "Filtered historical")]


def forecasts(x):
    """x: returns in percent. Returns dict model -> (VaR99, ES975) arrays for days WARM..n-1."""
    n = len(x); T = np.arange(WARM, n)
    out = {m: (np.empty(len(T)), np.empty(len(T)), np.empty(len(T))) for m, _ in MODELS}
    zq = stats.norm.ppf(1 - A); zr = stats.norm.pdf(stats.norm.ppf(AES)) / (1 - AES); z2 = stats.norm.ppf(1 - AES)
    he = L.ewma(x, 0.94)
    fit_n = fit_t = None; hn = ht = None
    for j, t in enumerate(T):
        w = x[t - 250:t]; wl = x[t - 1000:t]
        m, s = w.mean(), w.std(ddof=1)
        out["nor"][0][j] = -(m + s * zq); out["nor"][1][j] = -m + s * zr; out["nor"][2][j] = -(m + s * z2)
        out["hs"][0][j] = L.var_hist(w / 100, A) * 100; out["hs"][1][j] = L.es_hist(w / 100, AES) * 100
        out["hs"][2][j] = L.var_hist(w / 100, AES) * 100
        out["hsl"][0][j] = L.var_hist(wl / 100, A) * 100; out["hsl"][1][j] = L.es_hist(wl / 100, AES) * 100
        out["hsl"][2][j] = L.var_hist(wl / 100, AES) * 100
        se = math.sqrt(he[t])
        out["ew"][0][j] = -se * zq; out["ew"][1][j] = se * zr; out["ew"][2][j] = -se * z2
        if (t - WARM) % REFIT == 0:
            fit_n = L.garch_fit(x[:t], "normal"); fit_t = L.garch_fit(x[:t], "t")
            hn = L.garch_filter(x - fit_n["mu"], fit_n["omega"], fit_n["alpha"], fit_n["beta"], s0=x[:t].var())
            ht = L.garch_filter(x - fit_t["mu"], fit_t["omega"], fit_t["alpha"], fit_t["beta"], s0=x[:t].var())
            zres = np.sort((x[:t] - fit_n["mu"]) / np.sqrt(hn[:t]))
        sn = math.sqrt(hn[t]); st = math.sqrt(ht[t]); nu = fit_t["nu"]
        out["gn"][0][j] = -(fit_n["mu"] + sn * zq); out["gn"][1][j] = -fit_n["mu"] + sn * zr
        out["gn"][2][j] = -(fit_n["mu"] + sn * z2)
        out["gt"][0][j] = L.var_t(fit_t["mu"], st, nu, A); out["gt"][1][j] = L.es_t(fit_t["mu"], st, nu, AES)
        out["gt"][2][j] = L.var_t(fit_t["mu"], st, nu, AES)
        k = max(1, math.ceil(len(zres) * (1 - A))); k2 = max(1, math.ceil(len(zres) * (1 - AES)))
        out["fhs"][0][j] = -(fit_n["mu"] + sn * zres[k - 1]); out["fhs"][1][j] = -(fit_n["mu"] + sn * zres[:k2].mean())
        out["fhs"][2][j] = -(fit_n["mu"] + sn * zres[k2 - 1])
    forecasts.last = dict(nu=nu, zres=zres)          # the laws assumed at the last refit, for the Z2 nulls
    return T, out


def evaluate(x, T, out):
    loss = -x[T]; res = {}
    for m, _ in MODELS:
        v, e, v2 = out[m]
        hits = (loss > v).astype(int)
        xk = int(hits.sum()); n = len(hits)
        lr, pk = L.kupiec(xk, n, 1 - A)
        lc, pc, _ = L.christoffersen(hits)
        roll = np.convolve(hits, np.ones(250), "valid")
        red = float(np.mean(roll >= 10)); green = float(np.mean(roll <= 4))
        # ES backtest: Acerbi-Szekely Z2 at 97.5%, zero if ES is right, positive if ES is too small
        res[m] = dict(x=xk, n=n, pk=pk, pc=pc, red=red, green=green, hits=hits, roll=roll,
                      z2=L.es_z2(loss, v2, e, AES), x975=int(np.sum(loss > v2)),
                      avgvar=float(np.mean(v)))
    return res


def main():
    r, _ = L.load_daily()
    x = 100 * r.values
    T, out = forecasts(x)
    res = evaluate(x, T, out)
    R.N["caFirst"] = f"{r.index[T[0]]:%-d %B %Y}"; put("caN", len(T), "{:,.0f}")
    put("caExpected", len(T) * (1 - A), "{:.0f}"); put("caExpectedEs", len(T) * (1 - AES), "{:.0f}")
    # null critical values of Z2 for this record under the law each model assumes, by the central
    # limit theorem: normal innovations, t innovations with the last fitted nu, and the empirical law
    # of the normal GARCH residuals (losses are minus the residuals) used by filtered historical simulation
    nT = len(T); last = forecasts.last
    for nm, mom in (("Nor", L.z2_moments("normal", AES)), ("T", L.z2_moments("t", AES, nu=last["nu"])),
                    ("Emp", L.z2_moments("empirical", AES, sample=-last["zres"]))):
        zm, zs, zc = L.z2_null(*mom, nT, AES)
        put(f"caCrit{nm}", zc, "{:.3f}"); put(f"caNullSd{nm}", zs, "{:.3f}"); put(f"caNullMean{nm}", zm, "{:.3f}")
    put("caNuLast", last["nu"], "{:.1f}")
    fx = L.load_usdcad(); y = 100 * fx.values
    Ty, outy = forecasts(y); resy = evaluate(y, Ty, outy)
    R.N["cyFirst"] = f"{fx.index[Ty[0]]:%-d %B %Y}"; R.N["cyLast"] = f"{fx.index[-1]:%-d %B %Y}"
    put("cyN", len(Ty), "{:,.0f}"); put("cyExpected", len(Ty) * (1 - A), "{:.0f}")
    rows = []
    for m, name in MODELS:
        a, b = res[m], resy[m]
        cap = m.capitalize()
        put(f"ca{cap}X", a["x"], "{:.0f}"); put(f"ca{cap}Pk", a["pk"], "{:.3f}")
        put(f"ca{cap}Pc", a["pc"], "{:.3f}"); pct(f"ca{cap}Red", a["red"], 0)
        put(f"ca{cap}Es", a["z2"]); put(f"ca{cap}Xs", a["x975"], "{:.0f}"); put(f"cy{cap}X", b["x"], "{:.0f}"); put(f"cy{cap}Pk", b["pk"], "{:.3f}")
        put(f"cy{cap}Pc", b["pc"], "{:.3f}")
        pct(f"ca{cap}Rate", a["x"] / a["n"], 2)
        rows.append(f"{name} & \\ca{cap}X & \\ca{cap}Pk & \\ca{cap}Pc & \\ca{cap}Red & \\ca{cap}Es & "
                    f"\\cy{cap}X & \\cy{cap}Pk & \\cy{cap}Pc \\\\")
    R.N["caTableRows"] = "\n".join(rows)
    # small p-values typeset as < 0.001
    for k in list(R.N):
        if (k.startswith("ca") or k.startswith("cy")) and (k.endswith("Pk") or k.endswith("Pc")) and R.RAW[k] < 0.001:
            R.N[k] = "\\ensuremath{<}0.001"
    # worst cluster: the largest 250-day count for each model
    for m in ("hs", "gt", "fhs", "nor"):
        put(f"caMax{m.capitalize()}", res[m]["roll"].max(), "{:.0f}")
    # loss on 19 October 1987 relative to each model's VaR
    i87 = int(np.searchsorted(r.index.values, np.datetime64("1987-10-19"))) - T[0]
    for m in ("hs", "gt", "fhs"):
        put(f"caOct{m.capitalize()}", -x[T[0] + i87] / out[m][0][i87], "{:.1f}")
    pd.DataFrame({m: res[m]["hits"] for m, _ in MODELS}, index=r.index[T]).to_csv(R.HERE / "out" / "case_hits.csv")
    pd.DataFrame({m: out[m][0] for m, _ in MODELS} | {"loss": -x[T]}, index=r.index[T]).round(6) \
        .to_csv(R.HERE / "out" / "case_var.csv")
    # figure: rolling 250 day exception counts, historical 250 vs filtered historical
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    t = r.index[T][249:]
    ax.axhspan(0, 4.5, color="#EEF1F6", lw=0); ax.axhspan(9.5, 40, color="#F6E9E9", lw=0)
    ax.plot(t, res["hs"]["roll"], color=RED, lw=0.8, label="historical, 250 days")
    ax.plot(t, res["fhs"]["roll"], color=BLUE, lw=0.8, label="filtered historical")
    ax.set_ylim(0, max(res["hs"]["roll"].max(), res["fhs"]["roll"].max()) + 2)
    ax.set_ylabel("exceptions in the last 250 days"); ax.legend(loc="upper center")
    fig.tight_layout(); R.save(fig, "fig09_case"); plt.close(fig)
    crisis_figure(pd.read_csv(R.HERE / "out" / "case_var.csv", index_col=0, parse_dates=True))
    return finish("case_backtest")


def crisis_figure(v):
    """Daily losses of 2007 to 2009 with the 99% VaR of historical simulation on 250 days and of filtered
    historical simulation, exceptions of each marked. Reads the saved series, so it plots what was scored."""
    w = v.loc["2007-01-01":"2009-12-31"]
    put("crDays", len(w), "{:.0f}"); put("crExpected", 0.01 * len(w), "{:.1f}")
    put("crHs", int((w.loss > w.hs).sum()), "{:.0f}"); put("crFhs", int((w.loss > w.fhs).sum()), "{:.0f}")
    au = w.loc["2008-09-01":"2008-11-30"]
    put("crAutDays", len(au), "{:.0f}")
    put("crAutHs", int((au.loss > au.hs).sum()), "{:.0f}"); put("crAutFhs", int((au.loss > au.fhs).sum()), "{:.0f}")
    R.N["crHsPeak"] = f"{w.hs.idxmax():%-d %B %Y}"; R.N["crFhsPeak"] = f"{w.fhs.idxmax():%-d %B %Y}"
    # the window VaR holds its peak until the day whose loss set it leaves the last 250 days
    after = v.loc[w.hs.idxmax():]
    i = v.index.get_loc(after.index[after.hs < w.hs.max()][0])
    R.N["crHsDrop"] = f"{v.index[i]:%-d %B %Y}"; R.N["crDropDate"] = f"{v.index[i - 251]:%-d %B %Y}"
    pct("crDropLoss", v.loss.iloc[i - 251] / 100, 2); pct("crHsMax", w.hs.max() / 100, 2)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.vlines(w.index, 0, w.loss.clip(lower=0), color="#C9CED8", lw=0.6)
    ax.plot(w.index, w.hs, color=RED, lw=0.9, label="historical, 250 days")
    ax.plot(w.index, w.fhs, color=BLUE, lw=0.9, label="filtered historical")
    for col, c, y in (("hs", RED, -0.6), ("fhs", BLUE, -1.2)):
        e = w.index[w.loss > w[col]]
        ax.plot(e, np.full(len(e), y), "|", color=c, ms=6, mew=1.0)
    ax.set_ylim(-1.6, max(w.loss.max(), w.fhs.max()) + 0.5); ax.set_yticks(np.arange(0, 13, 2))
    ax.set_ylabel("daily loss and 99% VaR, %"); ax.legend(loc="upper left")
    fig.tight_layout(); R.save(fig, "fig16_crisis"); plt.close(fig)


if __name__ == "__main__":
    import time; t0 = time.time(); out = main()
    print({k: v for k, v in out["N"].items() if k != "caTableRows"}); print(out["N"]["caTableRows"]); print(time.time() - t0)
