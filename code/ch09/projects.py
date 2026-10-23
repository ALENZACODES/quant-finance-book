"""Computing projects for Chapter 9. Run:  python code/ch09/projects.py
The chapter's own programme adds US equities from a file that is not shipped; these projects use the
six markets that are."""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch09lib as L  # noqa: E402


def _months(idx):
    return np.array([p.year * 100 + p.month for p in idx])


def project_1():
    """The time series momentum programme on six markets, for lookbacks of 3, 6, 12 and 24 months and
    trading costs of 0, 5 and 20 basis points per unit traded: the annualised mean, volatility and Sharpe
    ratio of each, and the largest drawdown, from January 2003 once every lookback has a signal."""
    x = D.market_returns(); out = {}
    for lb in (3, 6, 12, 24):
        w = L.tsmom_weights(x, lookback=lb)
        for c in (0.0, 0.0005, 0.002):
            _, p = L.tsmom_returns(x, w, cost=c)
            p = p[p.index >= pd.Period("2003-01", "M")]
            out[f"L{lb}_c{int(1e4 * c)}"] = dict(mean=float(12 * p.mean()), vol=float(math.sqrt(12) * p.std(ddof=1)),
                                               sharpe=L.sharpe(p.values), maxdd=L.max_drawdown(p.values))
    return out


def project_2():
    """Fees on the twelve month programme: net excess returns after a 2% management fee and a 20%
    incentive fee over a high-water mark crystallised each December, against the gross, with the bill
    rate from the shipped monthly file. Reports the share of the gross excess return taken in fees."""
    x = D.market_returns(); w = L.tsmom_weights(x, lookback=12)
    _, p = L.tsmom_returns(x, w, cost=0.0005); p = p[p.index >= pd.Period("2003-01", "M")]
    ra = D.real_assets_monthly(); rf = ra["tbill"] / 1200; rf.index = rf.index.to_period("M")
    rf = rf.reindex(p.index).ffill().values
    net = L.fees_hwm(p.values, rf, months=_months(p.index))
    g, n = float(12 * p.mean()), float(12 * np.mean(net))
    return dict(months=len(p), gross=g, net=n, fee_share=(g - n) / g if g > 0 else None,
                sharpe_gross=L.sharpe(p.values), sharpe_net=L.sharpe(net))


def project_3():
    """The effective fee schedule of each strategy in the Form PF statistics: the fit of drag = a +
    b max(gross, 0) strategy by strategy, with the annual management fee 4a, the incentive share b and
    their robust standard errors, and the mean annual drag against a 2 and 20 schedule on the same gross
    returns with no high-water mark."""
    import hedge_data as H
    d = H.load_ofr(); out = {}
    for s in H.STRATS:
        p = H.fee_panel(d, [s])
        a, b, sa, sb, r2, n = H.fee_fit(p)
        two20 = 0.5 + 0.2 * np.maximum(p.gross.values - 0.5, 0)
        out[s] = dict(quarters=n, mgmt=4 * a, mgmt_se=4 * sa, inc=b, inc_se=sb, r2=r2,
                      drag=4 * float(p.drag.mean()), drag_2_20=4 * float(two20.mean()))
    return out


def main():
    for f in (project_1, project_2, project_3):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
