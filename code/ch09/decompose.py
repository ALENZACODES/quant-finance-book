"""Section 9.6: where the trend programme's return comes from. An exact split of its mean return into
the seven market sleeves, and two regressions of its monthly return: on the seven market returns, and on
those returns together with the size of each month's move, the absolute return, which a straddle on the
market would roughly pay. Checks the attribution identity of least squares with an intercept."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, put, pct, sci, finish
from trend import program

TAGS = dict(EQ="Eq", UST="Ust", GOC="Goc", OIL="Oil", EUR="Eur", JPY="Jpy", CAD="Cad")


def main():
    r, rf = L.load_markets()
    w, sl, p = program(r)
    y = p.values
    Xall = r.loc[p.index]
    full = Xall.notna().all(axis=1).values             # months with all seven returns
    pct("dcMean", 12 * y.mean(), 2)
    for m in L.MARKETS:
        pct(f"dcSl{TAGS[m]}", 12 * sl[m].mean(), 2)
    put("dcSumGap", 1e4 * abs(12 * sl.sum(axis=1).mean() - 12 * y.mean()), "{:.1f}")
    # regression on the markets, then on markets and absolute moves, over the complete months only
    put("dcRegMonths", full.sum(), "{:.0f}")
    y = y[full]; X = Xall[full]
    b1, e1, se1 = L.ols(y, X.values)
    A = np.column_stack([X.values, np.abs(X.values)])
    b2, e2, se2 = L.ols(y, A)
    r2 = lambda e: 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))
    put("dcRtwoA", r2(e1)); put("dcRtwoB", r2(e2))
    pct("dcAlphaA", 12 * b1[0], 1); put("dcTA", b1[0] / se1[0], "{:.1f}")
    pct("dcAlphaB", 12 * b2[0], 1); put("dcTB", b2[0] / se2[0], "{:.1f}")
    # the identity: mean = intercept + sum of betas times factor means, to rounding
    gap = abs(y.mean() - (b2[0] + b2[1:] @ A.mean(axis=0)))
    sci("dcIdGap", max(gap, 1e-17), 0)
    pct("dcLoadPart", 12 * (b2[1:] @ A.mean(axis=0)), 1)
    pct("dcLoadPartA", 12 * (b1[1:] @ X.values.mean(axis=0)), 1)
    pct("dcLoadShift", 12 * (b2[1:] @ A.mean(axis=0) - b1[1:] @ X.values.mean(axis=0)), 1)
    for i, m in enumerate(L.MARKETS):
        put(f"dcBeta{TAGS[m]}", b1[1 + i]); put(f"dcAbs{TAGS[m]}", b2[8 + i])
        put(f"dcAbsT{TAGS[m]}", b2[8 + i] / se2[8 + i], "{:.1f}")
    nsig = sum(abs(b2[8 + i] / se2[8 + i]) > 2 for i in range(7))
    put("dcAbsSig", nsig, "{:.0f}")
    put("dcAbsPos", sum(b2[8 + i] > 0 for i in range(7)), "{:.0f}")
    put("dcMonths", len(p), "{:.0f}")
    return finish("decompose")


if __name__ == "__main__":
    print(main()["N"])
