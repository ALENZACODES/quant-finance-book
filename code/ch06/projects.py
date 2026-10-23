"""Computing projects for Chapter 6. Run:  python code/ch06/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch06lib as L  # noqa: E402


def project_1():
    """The refinancing case study under other lending standards and other links between the cap rate
    and the five year yield: the number of vintages whose loan cannot be refinanced in full at maturity
    (cover below 1), for three loan-to-value limits and three coverage tests, applied at origination and
    at refinancing, and for three values of the cap rate sensitivity beta."""
    import case_refinance as C
    y = C.series()
    out = {}
    for beta in (0.0, 0.5, 1.0):
        for ltv in (0.60, 0.65, 0.75):
            for dscr in (1.20, 1.25, 1.35):
                P = dict(C.P, ltv_max=ltv, dscr_min=dscr, beta=beta)
                v = pd.DataFrame(L.refi_vintages(y.values, **P))
                out[f"beta{beta:.1f}_ltv{int(round(100 * ltv))}_dscr{int(round(100 * dscr))}"] = dict(
                    fail=int((v["cover"] < 1).sum()), median_cover=round(float(v["cover"].median()), 4))
    return out


def project_2():
    """Upfront payments on standard coupons along the hazard curve of the chapter: at each quoted
    maturity, the upfront per 100 of notional on a 100 basis point coupon from the flat hazard that
    reprices that quote and from the bootstrapped curve."""
    import hazard_cds
    P = hazard_cds.curve()
    knots, lams = L.bootstrap_hazard(hazard_cds.MATS, hazard_cds.QUOTES, hazard_cds.REC, P)
    out = {}
    for T, q in zip(hazard_cds.MATS, hazard_cds.QUOTES):
        lc = L.flat_hazard(T, q, hazard_cds.REC, P)
        uf, _ = L.cds_upfront(T, 0.01, [np.inf], [lc], hazard_cds.REC, P)
        uc, _ = L.cds_upfront(T, 0.01, knots, lams, hazard_cds.REC, P)
        out[T] = dict(quote_bp=1e4 * q, flat=-100 * uf, curve=-100 * uc)
    return out


def project_3():
    """Duration and convexity against the exact price change of a 10 year Government of Canada par bond
    over each month since 2000: the root mean square error of each approximation, in cents per 100."""
    g = D.goc()["BD.CDN.10YR.DQ.YLD"]
    y = (D.month_end(g) / 100)["2000":]
    e1, e2 = [], []
    months = y.index[1:]
    for y0, y1 in zip(y.values[:-1], y.values[1:]):
        p0 = L.bond_price(y0, y0, 10)
        p1 = L.bond_price(y1, y0, 10)
        D_, C_ = L.mod_duration(y0, y0, 10), L.convexity(y0, y0, 10)
        dy = y1 - y0
        a1 = p0 * (-D_ * dy); a2 = a1 + p0 * 0.5 * C_ * dy ** 2
        e1.append(p1 - p0 - a1); e2.append(p1 - p0 - a2)
    e1, e2 = np.array(e1), np.array(e2)
    return dict(months=len(e1), rmse_duration=float(100 * np.sqrt(np.mean(e1 ** 2))),
                rmse_convexity=float(100 * np.sqrt(np.mean(e2 ** 2))), worst_duration=float(100 * np.abs(e1).max()),
                worst_month=str(months[int(np.argmax(np.abs(e1)))]))


def main():
    for f in (project_1, project_2, project_3):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
