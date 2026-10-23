"""The computing projects of Chapter 12 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1(B=500)
    assert len(r) == 9 and all(0 <= v["rc"] <= 1 and 0 <= v["spa"] <= 1 for v in r.values())
    assert r["c50_b6"]["rc"] >= r["c0_b6"]["rc"]               # higher costs leave less to find


def test_project_2():
    r = P.project_2()
    assert r["dsr"] < r["psr"] and r["bench_annual"] > 0
