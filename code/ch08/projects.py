"""Computing projects for Chapter 8. Run:  python code/ch08/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch08lib as L  # noqa: E402
from scipy import stats  # noqa: E402


def project_1(window=1000, a=0.99):
    """One day 99% VaR of a long US dollar position against the Canadian dollar from 2012: rolling normal
    and Student t GARCH(1,1) forecasts refitted each year on the previous 1,000 days, the EWMA forecast,
    and historical simulation, with the number of exceptions and the Kupiec and Christoffersen tests."""
    r = 100 * L.load_usdcad()
    r = r[r.index >= "2007-01-01"]
    vals = r.values; dates = r.index
    out = {}
    start = max(window, int(np.searchsorted(dates, np.datetime64("2012-01-01"))))
    hits = {k: [] for k in ("garch_n", "garch_t", "ewma", "hist")}
    fit_n = fit_t = None; year = None
    h_e = L.ewma(vals, 0.94)
    for t in range(start, len(vals)):
        if dates[t].year != year:
            year = dates[t].year; past = vals[t - window:t]
            fit_n = L.garch_fit(past, "normal"); fit_t = L.garch_fit(past, "t")
        past = vals[t - window:t]
        for key, fit in (("garch_n", fit_n), ("garch_t", fit_t)):
            e = past - fit["mu"]
            h = L.garch_filter(e, fit["omega"], fit["alpha"], fit["beta"])[-1]
            sd = math.sqrt(h)
            v = L.var_normal(fit["mu"], sd, a) if key == "garch_n" else L.var_t(fit["mu"], sd, fit["nu"], a)
            hits[key].append(-vals[t] > v)
        hits["ewma"].append(-vals[t] > L.var_normal(0.0, math.sqrt(h_e[t]), a))
        hits["hist"].append(-vals[t] > L.var_hist(past, a))
    n = len(vals) - start
    for k, h in hits.items():
        x = int(np.sum(h)); lr, p = L.kupiec(x, n, 1 - a); lr2, p2 = L.christoffersen(np.array(h, int))[:2]
        out[k] = dict(exceptions=x, expected=n * (1 - a), kupiec_p=float(p), christoffersen_p=float(p2))
    out["days"] = n
    return out


def project_2():
    """Drawdowns and the Sharpe ratio of the six markets of the shipped data: for each, the annualised
    Sharpe ratio of monthly excess returns, its standard error with and without the correction for
    skewness and excess kurtosis, and the largest drawdown."""
    x = D.market_returns(); out = {}
    for c in x:
        r = x[c].values; T = len(r); th = r.mean() / r.std(ddof=1)
        s, k = float(stats.skew(r)), float(stats.kurtosis(r))
        out[c] = dict(sr_annual=float(math.sqrt(12) * th), se_iid=float(math.sqrt(12) * L.sharpe_se(th, T)),
                      se_moments=float(math.sqrt(12) * L.sharpe_se(th, T, s, k)), maxdd=L.max_drawdown(r))
    return out


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
