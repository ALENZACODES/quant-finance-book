"""The computing projects of Chapter 11 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    for lev in (4, 5, 6):
        d = [r[f"lev{lev}_cap{c}_nocf"] for c in (10, 20, 30, 40)]
        assert all(a >= b - 1e-9 for a, b in zip(d, d[1:]))     # a tighter cap costs more
        for c in (10, 20, 30, 40):
            assert r[f"lev{lev}_cap{c}_nocf"] >= r[f"lev{lev}_cap{c}_cf"] - 1e-9


def test_project_2():
    r = P.project_2()
    assert r["lookback_over_one"] > 0
    for k, v in r.items():
        if k != "lookback_over_one":
            assert v[1] < v[0] < v[2]
