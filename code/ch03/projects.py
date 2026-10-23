"""Computing projects for Chapter 3. Run:  python code/ch03/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch03lib as L  # noqa: E402


def project_1():
    """An Ornstein-Uhlenbeck process fitted to month-end Canadian 90 day bill yields over two windows,
    with the half-life of the pull towards the mean and its standard error by the delta method."""
    y = D.month_end(D.goc()["TB.CDN.90D.MID"])
    out = {}
    for lab, s in (("2000_2026", y["2000":]), ("2010_2026", y["2010":])):
        f = L.ou_fit(s.values / 100, 1 / 12)
        hl = math.log(2) / f["kappa"]
        se_hl = hl / (f["phi"] * abs(math.log(f["phi"]))) * f["se_phi"]
        out[lab] = dict(n=len(s), kappa=f["kappa"], theta=f["theta"], half_life=hl, se_half_life=se_hl)
    return out


def project_2():
    """Realised variance of the daily log USD/CAD rate by calendar year, against the variance of the
    monthly log changes in the same year: if the log rate were a Brownian motion with drift the two
    would estimate the same quadratic variation."""
    s = D.usdcad()["usdcad"].dropna()
    x = np.log(s)
    rows = []
    for yr, g in x.groupby(x.index.year):
        if len(g) < 200:
            continue
        rv_d = float(L.quadratic_variation(g.values))
        m = g.groupby(g.index.to_period("M")).last().values
        rv_m = float(np.sum(np.diff(np.concatenate([[g.values[0]], m])) ** 2))
        rows.append((yr, rv_d, rv_m))
    r = np.array([(a, b) for _, a, b in rows])
    return dict(years=len(rows), mean_daily=float(r[:, 0].mean()), mean_monthly=float(r[:, 1].mean()),
                ratio=float(r[:, 1].mean() / r[:, 0].mean()), corr=float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]))


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
