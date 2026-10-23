"""The computing projects of Chapter 4 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    assert r["last"] <= "2024-04-01" and 0.3 < r["contango"] < 0.8 and abs(r["t_one"]) < 3


def test_project_2():
    r = P.project_2()
    assert r["days"] > 4000 and r["min_pips"] < 0 < r["max_pips"] and 0 < r["discount_share"] < 1
