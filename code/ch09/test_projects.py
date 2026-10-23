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
