"""The computing projects of Chapter 5 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    assert r["months"] > 150 and abs(r["mean"]) < 4 * r["sd"] / math.sqrt(r["months"]) + 0.2 and 0.2 < r["positive"] < 0.8


def test_project_2():
    r = P.project_2()
    assert r["newton_fails"] == 0 and r["worst_error"] < 1e-9 and r["newton_mean"] < r["brent_mean"]
