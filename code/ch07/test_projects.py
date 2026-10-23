"""The computing projects of Chapter 7 run on the shipped data and give sensible answers."""
import math
import numpy as np
import projects as P


def test_project_1():
    r = P.project_1()
    assert 0 <= r["shrink"] <= 1 and r["tan_sample_in"] >= r["gmv_sample_in"] - 1e-12


def test_project_2():
    r = P.project_2(reps=500)
    assert r["mean_est_sr2"] > r["true_sr2"]
    assert abs(r["mean_est_sr2"] - r["theory_mean"]) < 0.15 * r["theory_mean"]
