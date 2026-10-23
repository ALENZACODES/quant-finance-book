"""Computing projects for Chapter 2. Run:  python code/ch02/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch02lib as L  # noqa: E402


def project_1():
    """The J-curve of the Takahashi and Alexander model: the year in which cumulative net cash flow to
    the limited partners is lowest, and the year it turns positive, for three rates of growth."""
    out = {}
    for g in (0.06, 0.12, 0.18):
        C, Dd, NAV = L.takahashi_alexander(growth=g)[:3]
        cum = np.cumsum(Dd - C)
        pos = np.where(cum > 0)[0]
        out[f"g{int(100 * g)}"] = dict(trough_year=int(np.argmin(cum)), trough=float(cum.min()),
                                       breakeven=int(pos[0]) if len(pos) else None)
    return out


def project_2():
    """The share of a hedge fund's gross return over its life that goes in fees, from the growing annuity
    of fees, for 2 and 20 at gross returns from 4% to 12% and a discount rate equal to the gross return."""
    res = {}
    for gross in (0.04, 0.06, 0.08, 0.10, 0.12):
        pv = L.fee_annuity_pv(100.0, gross, 0.02, 0.20, gross, 10)
        res[f"{int(100 * gross)}"] = float(pv / 100.0)
    return res


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
