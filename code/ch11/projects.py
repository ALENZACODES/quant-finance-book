"""Computing projects for Chapter 11. Run:  python code/ch11/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch11lib as L  # noqa: E402


def project_1():
    """The buyout of the chapter under caps on deductible interest from 10% to 40% of EBITDA, with and
    without a carryforward, and debt of 4, 5 and 6 times EBITDA: the equity IRR of each."""
    import lbo
    out = {}
    for lev in (4.0, 5.0, 6.0):
        base = lbo.deal(lev=lev)["irr"]
        for cap in (0.10, 0.20, 0.30, 0.40):
            for cf in (True, False):
                p = dict(lbo.BASE, int_cap=cap, carry_forward=cf)
                out[f"lev{lev:.0f}_cap{int(100 * cap)}_{'cf' if cf else 'nocf'}"] = round(1e4 * (base - lbo.deal(p, lev=lev)["irr"]), 1)
    return out


def project_2():
    """Three marketability discounts, the protective put at a zero rate, Finnerty's average-strike put and
    Longstaff's lookback value, for volatilities from 20% to 80% and restriction periods of one to five
    years, with the cases in which the lookback value exceeds the share price."""
    out = {}; over = 0
    for sig in (0.2, 0.4, 0.6, 0.8):
        for T in (1, 2, 3, 5):
            c, f, l = L.chaffe_dlom(T, sig), L.finnerty_dlom(T, sig), L.longstaff_bound(T, sig)
            over += l >= 1
            out[f"s{int(100 * sig)}_T{T}"] = (round(c, 4), round(f, 4), round(l, 4))
    out["lookback_over_one"] = int(over)
    return out


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
