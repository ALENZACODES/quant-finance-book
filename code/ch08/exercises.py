"""Answers to the Chapter 8 exercises, computed so that the text and the code cannot disagree."""
import math
from scipy import stats
import numbers_ch08 as R
from numbers_ch08 import L, put, pct, finish


def main():
    # 1. normal VaR and ES of a 10 million position, daily volatility 1.2%
    sd = 0.012; pos = 10.0
    put("xaVar", pos * L.var_normal(0, sd, 0.99), "{:.3f}"); put("xaEs", pos * L.es_normal(0, sd, 0.975), "{:.3f}")
    put("xaVarTen", pos * math.sqrt(10) * L.var_normal(0, sd, 0.99), "{:.3f}")
    # 2. two independent bonds, default probability 4%, loss 100 each, 95% level
    p = 0.04
    put("xbJoint", 100 * (1 - (1 - p) ** 2), "{:.2f}"); put("xbEsOne", 100 * p / 0.05, "{:.0f}")
    p2 = p * p
    put("xbEsTwo", (p2 * 200 + (0.05 - p2) * 100) / 0.05, "{:.1f}"); put("xbBoth", 100 * p2, "{:.2f}")
    # 3. GARCH forecasts
    om, al, be = 0.02, 0.08, 0.90
    lr = om / (1 - al - be); put("xcLr", lr, "{:.1f}"); put("xcLrVol", math.sqrt(252 * lr), "{:.1f}")
    h10 = L.garch_forecast(4.0, lr, al + be, 10); put("xcTen", float(h10), "{:.2f}")
    put("xcHalf", math.log(0.5) / math.log(al + be), "{:.1f}")
    avg = sum(L.garch_forecast(4.0, lr, al + be, k) for k in range(1, 11))
    put("xcTenDay", math.sqrt(avg), "{:.2f}"); put("xcTenSqrt", math.sqrt(10 * 4.0), "{:.2f}")
    # 4. Kupiec with 8 exceptions in 250 days
    lrk, pk = L.kupiec(8, 250, 0.01); put("xdLR", lrk); put("xdP", pk, "{:.4f}")
    g, y, r = L.zones(250, 0.01)
    put("xdTail", 100 * stats.binom.sf(7, 250, 0.01), "{:.2f}")
    # 5. GPD tail
    u, xi, b, n, nu = 0.02, 0.25, 0.007, 10000, 500
    v, e = L.gpd_var_es(u, xi, b, n, nu, 0.999)
    pct("xeVar", v, 2); pct("xeEs", e, 2)
    # 6. Sharpe standard error
    th, T, s, k = 0.15, 120, -1.0, 3.0
    put("xfIid", L.sharpe_se(th, T), "{:.4f}"); put("xfNn", L.sharpe_se(th, T, s, k), "{:.4f}")
    put("xfAnn", math.sqrt(12) * th, "{:.2f}"); put("xfAnnSe", math.sqrt(12) * L.sharpe_se(th, T, s, k), "{:.2f}")
    put("xfT", th / L.sharpe_se(th, T, s, k), "{:.2f}")
    # 7. smoothing
    phi = 0.5
    put("xgVol", 16 * math.sqrt((1 - phi) / (1 + phi)), "{:.2f}"); put("xgInfl", math.sqrt((1 + phi) / (1 - phi)), "{:.2f}")
    # 8. composite
    V = [10, 30, 60]; Rr = [0.05, 0.02, -0.01]
    aw = sum(v_ * r_ for v_, r_ in zip(V, Rr)) / sum(V); ew = sum(Rr) / 3
    pct("xhAw", aw, 2); pct("xhEw", ew, 2)
    import numpy as np
    pct("xhDisp", float(np.std(Rr, ddof=0)), 2)
    pct("xhDispAw", math.sqrt(sum(v_ * (r_ - aw) ** 2 for v_, r_ in zip(V, Rr)) / sum(V)), 2)
    # 9. drawdown law
    mu, sg = 0.06, 0.15
    mean = sg * sg / (2 * mu); pct("xiMean", mean, 2)
    pct("xiThirty", math.exp(-(-math.log(0.7)) / mean), 1); put("xiLogThirty", -math.log(0.7), "{:.4f}")
    # 10. M-squared and Treynor
    mp, sp, bp, mm, sm = 0.08, 0.20, 1.1, 0.06, 0.15
    pct("xjMsq", mp * sm / sp, 1); pct("xjTreynor", mp / bp, 2)
    return finish("exercises")


if __name__ == "__main__":
    print(main()["N"])
