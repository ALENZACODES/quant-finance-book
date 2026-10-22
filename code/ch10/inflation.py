"""Section 10.7: inflation hedging. Calendar year log price changes of commercial property, houses,
equity REITs, farm products, timber, lumber, crude oil and US equities, and the return of rolling 3 month
bills, regressed on CPI inflation, and on expected and unexpected inflation in the manner of Fama and
Schwert with the bill yield at the start of the year as the expected part; real returns over the
inflation of 2021 and 2022; the bill and real asset mix with the least real variance; and slopes of
overlapping 3, 5 and 10 year sums on inflation over the same years. Writes fig11_betas.png and
fig13_horizon.png.
Also writes fig18_crescatter.png, and simulates the lagged price model of Exercise 10.16 (random stream 3)."""
import numpy as np
import pandas as pd
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, put, pct, finish

ASSETS = [("bill", "Bill", "3 month bills"), ("cre", "Cre", "commercial property"), ("hpi", "Hpi", "houses"),
          ("reit", "Reit", "equity REITs"), ("mkt", "Mkt", "US equities"), ("ppi_farm", "Farm", "farm products"),
          ("ppi_timber", "Timber", "logs and timber"), ("wti", "Oil", "crude oil")]
FIRST, LAST = 1971, 2025


def annual():
    m = L.load_monthly(); q = L.load_quarterly(); r = L.quarterly_returns()
    dec = m[m.index.month == 12].copy(); dec.index = dec.index.year
    A = pd.DataFrame(index=dec.index)
    A["infl"] = np.log(dec.cpi).diff()
    A["expected"] = np.log1p(dec.tbill.shift(1) / 100)
    bill = np.log1p(m.tbill / 1200); A["bill"] = bill.groupby(bill.index.year).sum().where(bill.groupby(bill.index.year).count() == 12)
    for c in ("ppi_farm", "ppi_timber", "wti"):
        A[c] = np.log(dec[c]).diff()
    q4 = q[q.index.month == 10].copy(); q4.index = q4.index.year
    A["cre"] = np.log(q4.cre).diff(); A["hpi"] = np.log(q4.hpi).diff()
    rr = r["reit"].dropna(); n = rr.groupby(rr.index.year).count()
    A["reit"] = rr.groupby(rr.index.year).sum().where(n == 4)
    try:
        mk = L.load_market_quarterly(); nn = mk.groupby(mk.index.year).count()
        A["mkt"] = mk.groupby(mk.index.year).sum().where(nn == 4)
    except FileNotFoundError:
        A["mkt"] = np.nan
    return A.loc[FIRST:LAST]


def main():
    A = annual()
    R.N["ihFirst"] = str(FIRST); R.N["ihLast"] = str(LAST)
    pct("ihInflMean", A.infl.mean(), 1); pct("ihInflSd", A.infl.std(ddof=1), 1)
    rows = []
    for col, tag, lab in ASSETS:
        d = A[["infl", "expected", col]].dropna()
        b, se, _ = L.ols(d[col], d[["infl"]])
        bf, sef, _ = L.ols(d[col], d[["expected"]].assign(u=d.infl - d.expected))
        real = d[col] - d.infl
        put(f"ihB{tag}", b[1]); put(f"ihSe{tag}", se[1]); put(f"ihU{tag}", bf[2]); put(f"ihUse{tag}", sef[2])
        put(f"ihE{tag}", bf[1]); put(f"ihEse{tag}", sef[1])
        pct(f"ihReal{tag}", real.mean(), 1); pct(f"ihVol{tag}", real.std(ddof=1), 1)
        put(f"ihN{tag}", len(d), "{:.0f}"); R.N[f"ihStart{tag}"] = str(d.index[0])
        put(f"ihCorr{tag}", real.corr(d.infl))
        rows.append((lab, b[1], se[1], d.index[0]))
        if {2021, 2022} <= set(d.index):
            pct(f"ihTwo{tag}", real.loc[2021:2022].sum(), 1); pct(f"ihTwo{tag}Abs", abs(real.loc[2021:2022].sum()), 1)
    pct("ihInflTwo", A.infl.loc[2021:2022].sum(), 1)
    # Proposition 10.14: weight in property, rest in bills, that minimises real variance
    d = A[["infl", "cre", "bill"]].dropna()
    x = d.cre - d.bill; pi = d.infl - d.bill
    w = np.cov(x, pi)[0, 1] / x.var(ddof=1)
    put("ihWcre", w, "{:.2f}")
    v0 = (d.bill - d.infl).var(ddof=1); v1 = (d.bill + w * x - d.infl).var(ddof=1)
    pct("ihVarCut", 1 - v1 / v0, 0)
    pct("ihRealVolBill", (d.bill - d.infl).std(ddof=1), 1)
    # figure
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, b, s, _) in zip(y, rows):
        col = RED if lab == "crude oil" else BLUE
        ax.errorbar(b, yy, xerr=2 * s, fmt="o", color=col, ms=4, capsize=2, lw=1)
    ax.axvline(0, color="#333333", lw=0.6); ax.axvline(1, color=GREY, lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([f"{lab}, from {s0}" for lab, *_, s0 in rows])
    ax.set_xlabel("slope on CPI inflation, with two standard errors"); ax.grid(axis="y", visible=False)
    ax.set_xlim(-6, 19)
    fig.tight_layout(); R.save(fig, "fig11_betas"); plt.close(fig)
    horizons(A)
    scatter_figure(A)
    lag_simulation(A)
    return finish("inflation")


def scatter_figure(A):
    """Calendar year log changes in the property price index against CPI inflation, with the fitted line
    of Table 10.3 and a one for one line."""
    d = A[["infl", "cre"]].dropna()
    b, se, _ = L.ols(d.cre, d[["infl"]])
    res = d.cre - b[0] - b[1] * d.infl
    pct("scResSd", res.std(ddof=1), 1)   # ddof=1 as for the volatilities of Table 10.3
    put("scYears", len(d), "{:.0f}")
    big = res.abs().idxmax(); R.N["scBigYear"] = str(big); pct("scBigRes", res[big], 0)
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.scatter(100 * d.infl, 100 * d.cre, s=9, color=SKY)
    xx = np.linspace(d.infl.min(), d.infl.max(), 50)
    ax.plot(100 * xx, 100 * (b[0] + b[1] * xx), color=BLUE, label=f"fitted, slope {b[1]:.2f}")
    ax.plot(100 * xx, 100 * xx, color=GREY, ls=":", label="one for one")
    for yr in (1974, 1980, 2021, 2022, big):
        if yr in d.index:
            off = (-22, -9) if yr == 1980 else (4, 2)      # 1980 sits on the one-for-one line
            ax.annotate(str(yr), (100 * d.infl[yr], 100 * d.cre[yr]), xytext=off, textcoords="offset points",
                        fontsize=7)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xlabel("CPI inflation, %"); ax.set_ylabel("property price change, %"); ax.legend(loc="lower right")
    fig.tight_layout(); R.save(fig, "fig18_crescatter"); plt.close(fig)


NSIM, TSIM = 10_000, 55


def nw_slope(y, x, lags):
    """Slope of y on x with a constant and its Newey-West standard error, Bartlett weights over `lags`
    lags and no small sample factor (the statsmodels default used by horizon_beta)."""
    X = np.column_stack([np.ones_like(x), x]); XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y; g = X * (y - X @ beta)[:, None]
    S = g.T @ g
    for j in range(1, lags + 1):
        G = g[j:].T @ g[:-j]; S += (1 - j / (lags + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    return beta[1], np.sqrt(V[1, 1])


def lag_simulation(A):
    """Exercise 10.16 by simulation: inflation independent and normal with the mean and standard deviation
    of 1971 to 2025, an asset whose log price change is last year's inflation, 55 years per sample and
    10,000 samples from random stream 3. Slopes of overlapping k year sums on inflation, whose true value
    is 1 - 1/k, with Newey-West standard errors over k lags."""
    rng = R.rng_for(3)
    mu, sd = A.infl.mean(), A.infl.std(ddof=1)
    put("lsN", NSIM, "{:,.0f}"); put("lsT", TSIM, "{:.0f}")
    out = {}
    for k, kw in ((1, "One"), (5, "Five"), (10, "Ten")):
        b = np.empty(NSIM); se = np.empty(NSIM)
        rk = np.random.default_rng(rng.integers(2 ** 32))
        for i in range(NSIM):
            pi = mu + sd * rk.standard_normal(TSIM + 1)
            y, x = pi[:-1], pi[1:]
            sy = np.convolve(y, np.ones(k), "valid"); sx = np.convolve(x, np.ones(k), "valid")
            b[i], se[i] = nw_slope(sy, sx, k)
        true = 1 - 1 / k
        put(f"lsMean{kw}", b.mean()); put(f"lsSd{kw}", b.std(ddof=1)); put(f"lsSe{kw}", np.median(se))
        pct(f"lsRej{kw}", np.mean(np.abs((b - true) / se) > 1.96), 0)
        out[k] = (b, se)
    return out


HORIZONS = ((1, "One"), (3, "Three"), (5, "Five"), (10, "Ten"))
HASSETS = [a for a in ASSETS if a[0] != "wti"]


def horizons(A):
    """Proposition 10.15: slopes of k year sums on k year inflation."""
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    style = {"bill": (GREY, "s"), "cre": (BLUE, "o"), "hpi": (SKY, "o"), "reit": (PALE, "v"), "mkt": (RED, "D"),
             "ppi_farm": (ROSE, "^"), "ppi_timber": ("#333333", "^")}
    for col, tag, lab in HASSETS:
        bs = []
        for k, kw in HORIZONS:
            b, se, n = L.horizon_beta(A[col], A.infl, k)
            bs.append(b)
            if k > 1:
                put(f"ihH{tag}{kw}", b); put(f"ihHs{tag}{kw}", se)
                if col == "cre":
                    put(f"ihHn{kw}", n, "{:.0f}")
        if col != "reit":
            c, mk = style[col]
            ax.plot([k for k, _ in HORIZONS], bs, marker=mk, color=c, lw=1.1, ms=4, label=lab)
    ax.axhline(1, color=GREY, lw=0.8, ls=":"); ax.axhline(0, color="#333333", lw=0.6)
    ax.set_xticks([1, 3, 5, 10]); ax.set_xlabel("horizon, years"); ax.set_ylabel("slope on inflation")
    ax.legend(fontsize=7, ncol=2, loc="lower right", frameon=True, framealpha=1, edgecolor="none").set_zorder(5)
    fig.tight_layout(); R.save(fig, "fig13_horizon"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
