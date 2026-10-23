"""The computing projects of Chapter 8 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    assert r["days"] > 3000
    for k in ("garch_n", "garch_t", "ewma", "hist"):
        assert 0 <= r[k]["kupiec_p"] <= 1 and r[k]["exceptions"] > 0


def test_project_2():
    r = P.project_2()
    for v in r.values():
        assert 0 < v["maxdd"] < 1 and 0.1 < v["se_iid"] < 0.3
