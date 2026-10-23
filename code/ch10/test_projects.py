"""The computing projects of Chapter 10 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    for c in ("cre", "hpi"):
        assert r[c]["ar1"] > 0 and r[c]["sd_unsmoothed"] > r[c]["sd"] and abs(r[c]["ar1_unsmoothed"]) < abs(r[c]["ar1"])


def test_project_2():
    r = P.project_2()
    assert all(v["years"] >= 35 and v["se"] > 0 for v in r.values())


def test_project_3():
    r = P.project_3()
    assert r["months"] > 100 and -1 < r["corr"] < 1
