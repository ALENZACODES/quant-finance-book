"""The computing projects of Chapter 6 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    assert r["beta0.5_ltv65_dscr125"]["fail"] == 3            # the case study of the chapter
    assert all(v["fail"] >= 0 for v in r.values())


def test_project_2():
    r = P.project_2()
    assert abs(r[1]["flat"] - r[1]["curve"]) < 1e-12          # one quote: the flat hazard is the curve
    assert all(v["flat"] > 0 for v in r.values())             # every quote is inside the 100 bp coupon


def test_project_3():
    r = P.project_3()
    assert r["rmse_convexity"] < 0.1 * r["rmse_duration"]


def test_project_4():
    r = P.project_4()
    assert len(r) >= 15
    for t, v in r.items():
        assert v["quarters"] >= 8 and v["vol"] > 0
        assert -1 < v["phi"] < 1 and v["vol_desmoothed"] > 0
    # the de-smoothed series of one lender, rebuilt by hand from its returns
    import direct_lending as DL
    g = DL.bdc_panel(DL.load_bdc()); r = g[g.ticker == "ARCC"].roe.dropna().values
    phi = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    ds = (r[1:] - phi * r[:-1]) / (1 - phi) if phi > 0 else r
    assert abs(2 * np.std(ds, ddof=1) - P.project_4()["ARCC"]["vol_desmoothed"]) < 1e-12
