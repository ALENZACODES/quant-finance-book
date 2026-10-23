"""Computing projects for Chapter 1, on data that ship with the code. Run:  python code/ch01/projects.py
Each function solves one project and returns its results; test_projects.py runs them all."""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch01lib as L  # noqa: E402


def project_1():
    """Canadian 90 day bills: the quoted yield as a simple rate on an actual/365 basis, its effective
    annual and continuous equivalents, and the monthly holding returns of rolling bills."""
    y = D.month_end(D.goc()["TB.CDN.90D.MID"]) / 100
    eff = (1 + y * 91 / 365) ** (365 / 91) - 1
    cont = np.log1p(eff)
    hold = (y.shift(1) * 1 / 12).dropna()                  # a month's accrual on a bill rolled monthly, to first order
    arith = 12 * hold.mean(); geo = (1 + hold).prod() ** (12 / len(hold)) - 1
    return dict(months=len(y), mean_quote=float(y.mean()), mean_eff=float(eff.mean()), mean_cont=float(cont.mean()),
                max_gap_bp=float(1e4 * (eff - y).max()), max_gap_month=str((eff - y).idxmax()),
                arith=float(arith), geo=float(geo))


def project_2(start=None):
    """The Z.1 commercial real estate price index: quarterly log changes, their moments and the
    Jarque-Bera test, the first autocorrelation, and the series unsmoothed with that coefficient."""
    q = D.real_assets_quarterly()["cre"].dropna()
    if start is not None:
        q = q[q.index >= start]
    r = np.diff(np.log(q.values))
    m = L.sample_moments(r); jb, p = L.jarque_bera(r); rho = float(L.acf(r, 1)[1])
    u = L.unsmooth(r, rho)
    return dict(n=len(r), sd=m["sd"], skew=m["skew"], exkurt=m["exkurt"], jb=jb, p=p, rho=rho,
                sd_unsmoothed=float(np.std(u, ddof=1)), ratio=float(np.std(u, ddof=1) / m["sd"]))


def main():
    for name, f in (("project_1", project_1), ("project_2", project_2), ("project_2 from 1996", lambda: project_2("1996-01-01"))):
        print(name, {k: round(float(v), 4) if isinstance(v, float) else v for k, v in f().items()})


if __name__ == "__main__":
    main()
