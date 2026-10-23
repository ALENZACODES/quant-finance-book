"""Computing projects for Chapter 5. Run:  python code/ch05/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch05lib as L  # noqa: E402


def project_1(window=21, tenor=21):
    """Daily delta hedging of one month at the money USD/CAD calls, written every 21 trading days from
    2008, priced by Garman and Kohlhagen at the realised volatility of the previous 21 days, with the
    one month bill rates of the two countries. Returns the hedged profit of the writer per 100 US dollars
    of notional, its mean and spread, and the share of positive months."""
    u = D.usdcad()[["usdcad", "cad_tbill_1m", "us_tbill_1m"]].dropna()
    u = u.loc["2008":]
    S = u["usdcad"].values; rd = u["cad_tbill_1m"].values / 100; rf = u["us_tbill_1m"].values / 100
    lr = np.diff(np.log(S))
    pnl = []
    for t0 in range(window, len(S) - tenor, tenor):
        sig = float(np.std(lr[t0 - window:t0], ddof=1) * math.sqrt(252))
        K = S[t0]; T = tenor / 252
        prem = float(L.gk(S[t0], K, rd[t0], rf[t0], sig, T))
        cash = prem; delta_prev = 0.0
        for k in range(tenor):
            t = t0 + k; tau = T - k / 252
            d = float(L.greeks(S[t], K, rd[t], sig, tau, "call", q=rf[t])["delta"])
            cash -= (d - delta_prev) * S[t]; delta_prev = d
            cash *= math.exp(rd[t] / 252)
            # the US dollars held earn the US rate
            cash += delta_prev * S[t + 1] * (math.exp(rf[t] / 252) - 1)
        tT = t0 + tenor
        cash += delta_prev * S[tT] - max(S[tT] - K, 0.0)
        pnl.append(100 * cash / S[t0])
    p = np.array(pnl)
    return dict(months=len(p), mean=float(p.mean()), sd=float(p.std(ddof=1)), positive=float((p > 0).mean()),
                worst=float(p.min()), best=float(p.max()))


def project_2():
    """Implied volatility by Newton's method from the inflection point and by Brent's method, on a grid of
    strikes, maturities and volatilities, quoting the out of the money option at each strike: the largest
    error in volatility and the iterations each method needs. Prices below 1e-6 are left out, since
    there the volatility is barely determined by the price."""
    S, r, q = 100.0, 0.03, 0.01
    worst = 0.0; it_n = []; it_b = []; fails = 0
    for T in (0.05, 0.25, 1.0, 5.0):
        F = S * math.exp((r - q) * T)
        for K in (60, 80, 95, 100, 105, 125, 160):
            kind = "call" if K >= F else "put"
            for sig in (0.08, 0.25, 0.6):
                p = float(L.bs(S, K, r, sig, T, kind, q))
                if p < 1e-6:
                    continue
                sb, b_it, _ = L.iv_brent(p, S, K, r, T, kind, q); it_b.append(b_it)
                try:
                    sn, n_it, _ = L.iv_newton(p, S, K, r, T, kind, q)
                except RuntimeError:
                    fails += 1; continue
                worst = max(worst, abs(sn - sig), abs(sb - sig)); it_n.append(n_it)
    return dict(cases=len(it_b), newton_fails=fails, worst_error=worst, newton_mean=float(np.mean(it_n)),
                newton_max=int(max(it_n)), brent_mean=float(np.mean(it_b)), brent_max=int(max(it_b)))


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
