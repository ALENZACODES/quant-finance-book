"""The computing projects of Chapter 2 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    for v in r.values():
        assert v["trough"] < 0 and v["breakeven"] > v["trough_year"]


def test_project_2():
    r = P.project_2()
    v = [r[k] for k in ("4", "6", "8", "10", "12")]
    assert all(0 < a < b < 1 for a, b in zip(v, v[1:]))
