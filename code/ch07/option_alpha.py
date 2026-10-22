"""A fund that sells one month puts on the market each month, its CAPM alpha, the Henriksson-Merton
regression and an alpha measured against traded options. Black-Scholes prices use the trailing twelve
month realised volatility of the market, with and without a markup. Figure fig09_puts. No random numbers."""
import math
import numpy as np
import numbers_ch07 as R
from numbers_ch07 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE

MONEY = 0.95


def strategy(f, markup):
    m = f.mkt_rf.values; rf = f.rf.values; G = 1 + m + rf
    y = []; p_atm = []; p_sold = []
    for t in range(12, len(m)):
        s0 = np.std(np.log(G[t - 12:t]), ddof=1) * math.sqrt(12); r = 12 * math.log1p(rf[t])
        p = L.bs_put(1.0, MONEY, 1 / 12, r, s0 * (1 + markup))
        y.append((p * (1 + rf[t]) - max(MONEY - G[t], 0.0)) / MONEY)   # excess return on collateral MONEY
        p_atm.append(L.bs_put(1.0, 1 + rf[t], 1 / 12, r, s0 * (1 + markup)) * (1 + rf[t]))
        p_sold.append(p)
    return np.array(y), m[12:], np.array(p_atm), np.array(p_sold)


def main():
    f, vw, ew, cap = L.load()
    R.put("poMoney", 100 * MONEY, "{:.0f}")
    for mk, nm in ((0.0, "Zero"), (0.2, "Twenty")):
        y, x, pa, ps = strategy(f, mk); T = len(y)
        c = L.ols(y, np.c_[np.ones(T), x], 6)
        hm = L.ols(y, np.c_[np.ones(T), x, np.maximum(-x, 0)], 6)
        tm = L.ols(y, np.c_[np.ones(T), x, x * x], 6)
        R.put(f"po{nm}Mean", 1200 * y.mean()); R.put(f"po{nm}SR", math.sqrt(12) * y.mean() / y.std(ddof=1))
        R.put(f"po{nm}Vol", 100 * math.sqrt(12) * y.std(ddof=1), "{:.1f}")
        R.put(f"po{nm}A", 1200 * c["b"][0]); R.put(f"po{nm}T", c["b"][0] / c["nw"][0]); R.put(f"po{nm}B", c["b"][1])
        R.put(f"po{nm}HmA", 1200 * hm["b"][0]); R.put(f"po{nm}HmB", hm["b"][1]); R.put(f"po{nm}HmG", hm["b"][2])
        R.put(f"po{nm}HmR", hm["r2"]); R.put(f"po{nm}CapmR", c["r2"])
        R.put(f"po{nm}TmG", tm["b"][2], "{:.2f}"); R.put(f"po{nm}TmT", tm["b"][2] / tm["nw"][2])
        adj = hm["b"][0] + hm["b"][2] * pa.mean()
        R.put(f"po{nm}Adj", 1200 * adj); R.RAW[f"po{nm}AtmPrice"] = float(pa.mean())
        R.put(f"po{nm}Prem", 1200 * ps.mean() / MONEY)
        R.put(f"po{nm}Worst", -100 * y.min(), "{:.1f}")
        R.N[f"po{nm}WorstWhen"] = L.month_name(int(f.index[12 + int(np.argmin(y))]))
        if mk > 0:
            keep = (y, x, c, hm)
            flat = np.abs(x) < 0.005                        # months when the market moved under 0.5%
            R.put("poTwentyFlat", 1200 * y[flat].mean()); R.put("poTwentyFlatN", int(flat.sum()), "{:.0f}")
    # what the markup does to each alpha, from the unrounded values
    R.put("poMarkupAdj", R.RAW["poTwentyAdj"] - R.RAW["poZeroAdj"])
    R.put("poMarkupCapm", R.RAW["poTwentyA"] - R.RAW["poZeroA"])
    R.put("poT", T, "{:.0f}"); R.N["poFirst"] = L.month_name(int(f.index[12]))
    R.put("poAtmPrice", 100 * R.RAW["poZeroAtmPrice"], "{:.2f}")
    R.put("poAtmPayoff", 100 * np.maximum(-keep[1], 0).mean(), "{:.2f}")
    y, x, c, hm = keep
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.scatter(100 * x, 100 * y, s=5, color=SKY, alpha=0.7, label="months")
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(100 * xs, 100 * (c["b"][0] + c["b"][1] * xs), color=GREY, lw=1.1, label="CAPM line")
    ax.plot(100 * xs, 100 * (hm["b"][0] + hm["b"][1] * xs + hm["b"][2] * np.maximum(-xs, 0)), color=RED, lw=1.3,
            label="Henriksson-Merton fit")
    ax.set_xlabel("market excess return, % a month"); ax.set_ylabel("fund excess return, % a month")
    ax.legend(loc="lower right"); R.save(fig, "fig09_puts")
    return R.finish("option_alpha")


if __name__ == "__main__":
    main()
