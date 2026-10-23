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
