"""The computing projects of Chapter 1 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    assert r["months"] > 300 and r["mean_cont"] < r["mean_quote"] < r["mean_eff"] and 0 < r["max_gap_bp"] < 50


def test_project_2():
    r = P.project_2()
    assert r["p"] < 0.01 and 0 < r["rho"] < 1 and r["ratio"] > 1
