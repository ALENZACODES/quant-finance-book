"""The computing projects of Chapter 9 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    for lb in (3, 6, 12, 24):
        assert r[f"L{lb}_c0"]["mean"] > r[f"L{lb}_c5"]["mean"] > r[f"L{lb}_c20"]["mean"]


def test_project_2():
    r = P.project_2()
    assert r["net"] < r["gross"] and r["sharpe_net"] < r["sharpe_gross"]


def test_project_3():
    # every strategy has its full record, most pay less than 2 and 20 would take on their average gross
    # returns, and the pooled fit of the chapter lies within the range of the strategy fits
    import hedge_data as H
    r = P.project_3()
    assert set(r) == set(H.STRATS)
    for s, v in r.items():
        assert v["quarters"] >= 50 and 0 < v["mgmt"] < 3 and 0 < v["inc"] < 0.25
    assert sum(v["drag"] < v["drag_2_20"] for v in r.values()) >= 6
    a, b = H.fee_fit(H.fee_panel(H.load_ofr(), [s for s in H.STRATS if s != "other"]))[:2]
    incs = [v["inc"] for s, v in r.items() if s != "other"]
    assert min(incs) <= b <= max(incs)
