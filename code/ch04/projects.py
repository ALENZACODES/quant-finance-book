"""Computing projects for Chapter 4. Run:  python code/ch04/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch04lib as L  # noqa: E402


def project_1():
    """The crude oil curve from the EIA monthly averages (to April 2024): the share of months in
    contango, the mean annualised basis between the first and second contracts, and a test of whether the
    second contract forecasts the first one month later, the regression of ln(F1 next month / F1) on
    ln(F2 / F1), whose slope is 1 if the futures price is an unbiased forecast."""
    e = D.eia()[["cl1", "cl2"]].dropna()
    b = np.log(e["cl2"] / e["cl1"])
    chg = np.log(e["cl1"].shift(-1) / e["cl1"])
    ok = pd.concat([b, chg], axis=1).dropna()
    x, y = ok.iloc[:, 0].values, ok.iloc[:, 1].values
    X = np.column_stack([np.ones_like(x), x]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; s2 = res @ res / (len(y) - 2)
    se = math.sqrt(s2 / np.sum((x - x.mean()) ** 2))
    return dict(months=len(e), first=str(e.index[0].date()), last=str(e.index[-1].date()),
                contango=float((b > 0).mean()), mean_basis_pa=float(12 * b.mean()),
                slope=float(beta[1]), se=se, t_one=float((beta[1] - 1) / se))


def project_2():
    """Covered interest parity forwards for one month USD/CAD from the one month bill rates of the two
    countries: the forward points in pips, their range, and the share of days the US dollar is at a
    forward discount."""
    u = D.usdcad()[["usdcad", "cad_tbill_1m", "us_tbill_1m"]].dropna()
    F = L.cip_forward(u["usdcad"], u["cad_tbill_1m"] / 100, u["us_tbill_1m"] / 100, 1 / 12)
    pts = 1e4 * (F - u["usdcad"])
    return dict(days=len(u), first=str(u.index[0].date()), mean_pips=float(pts.mean()),
                min_pips=float(pts.min()), max_pips=float(pts.max()), discount_share=float((pts < 0).mean()))


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
