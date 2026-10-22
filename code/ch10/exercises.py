"""Answers to the numerical exercises of Chapter 10."""
import math
import numpy as np
import numbers_ch10 as R
from numbers_ch10 import L, put, pct, finish


def main():
    # 10.1 pro forma and cap rate
    egi = 1.2e6 * 0.95; noi = egi - 0.40e6
    put("xaNoi", noi / 1e3, "{:,.0f}"); pct("xaCap", L.cap_rate(0.08, 0.02, 0.08), 2)
    put("xaValue", L.gordon_value(noi, 0.08, 0.02, 0.08) / 1e6, "{:.2f}")
    # 10.2 share of value in the sale and an exit cap 50 bp higher
    pct("xbShare", (1.02 / 1.07) ** 10, 1)
    v0 = L.dcf_value(1.0, 0.02, 0.07, 10, 0.05); v1 = L.dcf_value(1.0, 0.02, 0.07, 10, 0.055)
    pct("xbDrop", v1 / v0 - 1, 1); pct("xbApprox", -(1.02 / 1.07) ** 10 * math.log(0.055 / 0.05), 1)
    pct("xbApproxPct", math.exp(-(1.02 / 1.07) ** 10 * math.log(0.055 / 0.05)) - 1, 1)
    # 10.3 IRR at constant cap rate, then a sale at a higher cap
    pct("xcIrr", 0.06 + 0.02, 0)
    t = np.arange(1, 8); cf = 0.06 * 1.02 ** (t - 1); f = np.concatenate([[-1.0], cf]); f[-1] += 0.06 * 1.02 ** 7 / 0.065
    pct("xcIrrHigh", L.irr(f), 2)
    # 10.4 mortgage constant and DSCR
    k = L.mortgage_constant(0.055, 30); pct("xdK", k, 3); put("xdDscr", L.dscr(0.06, 0.70, k), "{:.2f}")
    # 10.5 the binding test
    k5 = L.mortgage_constant(0.06, 25); loan, bind = L.max_loan(2.0e6, 32e6, k5, 1.30, 0.70)
    put("xeLoan", loan / 1e6, "{:.2f}"); put("xeLtvLoan", 0.70 * 32, "{:.1f}"); pct("xeK", k5, 3)
    assert bind == "DSCR"
    pct("xeLtv", loan / 32e6, 1)
    # 10.6 balloon
    pct("xfBal", L.balance(0.06, 25, 10), 1)
    # 10.7 leverage
    pct("xgRet", L.levered_return(0.07, 0.055, 0.6), 2); pct("xgEqVol", 0.10 / 0.4, 0)
    pct("xgRetLow", L.levered_return(0.04, 0.055, 0.6), 2)
    # 10.8 smoothing
    pct("xhVol", 0.10 * math.sqrt(0.3 / 1.7), 1); put("xhAcf", 0.7 ** 4, "{:.2f}")
    # 10.9 unsmoothing a short series
    s = np.array([1.0, 1.5, 0.5, -1.0]); u = L.unsmooth(s, 0.6)
    for v, tag in zip(u, ("One", "Two", "Three")):
        put(f"xi{tag}", v, "{:.2f}")
    # 10.10 Dimson sum
    put("xjSum", sum(0.5 * 0.5 ** j for j in range(4)), "{:.4f}")
    # 10.11 rotation
    put("xkF", L.faustmann_rotation(40.0, 0.05, 1500.0), "{:.1f}"); put("xkS", L.single_rotation(40.0, 0.05), "{:.1f}")
    put("xkLev", L.land_value(L.faustmann_rotation(40.0, 0.05, 1500.0), 40.0, 0.05, 1500.0), "{:,.0f}")
    # 10.12 concession
    v = L.concession_value(50, 0.08, 0.03, 25)
    put("xlV", v, "{:.0f}"); pct("xlShare", v / (50 / 0.05), 1); put("xlDur", L.concession_duration(0.08, 0.03, 25), "{:.1f}")
    # 10.13 cost of carry
    put("xmBound", 70 * math.exp(0.06 * 0.5), "{:.2f}"); pct("xmCy", L.net_convenience_yield(70.0, 71.0, 0.5, 0.04), 2)
    pct("xmGross", L.net_convenience_yield(70.0, 71.0, 0.5, 0.04) + 0.02, 2)
    # 10.14 inflation hedge
    R.sci("xnCov", L.real_hedge_cov(0.6, 0.03 ** 2), 1)
    put("xnW", L.min_real_variance_weight(0.6 * 0.03 ** 2, 0.10 ** 2), "{:.3f}")
    # 10.15 intrinsic value of storage
    F, t = [3.00, 3.60, 3.20], [1 / 12, 4 / 12, 7 / 12]
    v, i, j = L.intrinsic_storage(F, t, 0.04, 0.05, 0.05)
    assert (i, j) == (0, 1)
    put("xoV", v, "{:.3f}")
    # the listing's twelve-month seasonal curve, filled at once and at a third a month
    import contextlib, importlib, io, pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "snippets"))
    with contextlib.redirect_stdout(io.StringIO()):
        lp = importlib.import_module("storage_lp")
    put("xoSeasOne", lp.v1, "{:.3f}"); put("xoSeasThird", lp.v3, "{:.3f}")
    put("xoLast", math.exp(-0.04 * 7 / 12) * (3.20 - 0.05) - math.exp(-0.04 / 12) * (3.00 + 0.05), "{:.3f}")
    return finish("exercises")


if __name__ == "__main__":
    print(main()["N"])
