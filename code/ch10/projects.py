"""Computing projects for Chapter 10. Run:  python code/ch10/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch10lib as L  # noqa: E402


def project_1():
    """Smoothing in two appraisal and transaction based price series: the first autocorrelation of
    quarterly log changes in the Z.1 commercial real estate index and in the FHFA house price index from
    1996, when the Z.1 method changes, each series unsmoothed with its own coefficient, and the standard
    deviations before and after."""
    q = D.real_assets_quarterly()[["cre", "hpi"]].dropna()
    q = q[q.index >= "1996-01-01"]
    r = np.log(q).diff().dropna(); out = dict(quarters=len(r))
    for c in ("cre", "hpi"):
        a = L.ar1(r[c]); u = L.unsmooth(r[c].values, a)
        out[c] = dict(ar1=a, sd=float(r[c].std()), sd_unsmoothed=float(np.std(u, ddof=1)), ar1_unsmoothed=L.ar1(u))
    return out


def project_2():
    """Inflation hedging: regressions of the annual log change of farm and timber producer prices, WTI
    crude and the Z.1 property index on the annual log change of the CPI, with Newey-West standard errors,
    using December values."""
    m = D.real_assets_monthly(); q = D.real_assets_quarterly()
    dec = m[m.index.month == 12]
    cre = q["cre"][q.index.month == 10]                      # the fourth quarter
    cre.index = cre.index.year; dec.index = dec.index.year
    df = pd.concat([np.log(dec[["cpi", "ppi_farm", "ppi_timber", "wti"]]).diff(), np.log(cre).diff().rename("cre")], axis=1)
    out = {}
    for c in ("ppi_farm", "ppi_timber", "wti", "cre"):
        d = df[["cpi", c]].dropna()
        b, se, _ = L.ols(d[c].values, d[["cpi"]].values)
        out[c] = dict(years=len(d), beta=float(b[1]), se=float(se[1]))
    return out


def project_3():
    """The net convenience yield of crude oil from the first and second futures (monthly averages to April
    2024) and the 3 month bill rate, against inventories: the correlation of the yield with the deviation
    of crude stocks from their level a year earlier."""
    e = D.eia()[["cl1", "cl2", "crude_stocks"]].dropna(); m = D.real_assets_monthly()["tbill"] / 100
    r = m.reindex(e.index).ffill()
    cy = L.net_convenience_yield(e["cl1"], e["cl2"], 1 / 12, r)
    dev = np.log(e["crude_stocks"]).diff(12)
    ok = pd.concat([cy, dev], axis=1).dropna()
    return dict(months=len(ok), mean_cy=float(cy.mean()), corr=float(np.corrcoef(ok.iloc[:, 0], ok.iloc[:, 1])[0, 1]))


def main():
    for f in (project_1, project_2, project_3):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
