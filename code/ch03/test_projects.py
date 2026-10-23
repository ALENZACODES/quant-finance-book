"""The computing projects of Chapter 3 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    for v in r.values():
        assert v["kappa"] > 0 and v["half_life"] > 0 and v["se_half_life"] > 0


def test_project_2():
    r = P.project_2()
    assert r["years"] >= 15 and 0.7 < r["ratio"] < 1.4 and r["corr"] > 0.8
