"""Computing projects for Chapter 12. Run:  python code/ch12/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch12lib as L  # noqa: E402


def project_1(B=2000, seed=12):
    """The data snooping tests of the chapter under other choices: costs of 0, 10 and 50 basis points per
    change of position and mean block lengths of 3, 6 and 12 months. Reports the reality check and SPA
    p-values and the number of StepM rejections for each."""
    import snooping as S
    out = {}
    for cost in (0.0, 0.001, 0.005):
        S.COST = cost
        X, names, _ = S.rule_returns(); d = X.values
        for blk in (3, 6, 12):
            idx = L.stationary_bootstrap(len(d), B, blk, np.random.default_rng(seed))
            r = L.snooping_tests(d, idx)
            out[f"c{int(1e4 * cost)}_b{blk}"] = dict(rc=round(float(r["p_rc"]), 3), spa=round(float(r["p_spa_c"]), 3),
                                                   stepm=int(np.sum(r["stepm"])))
    S.COST = 0.001
    return out


def project_2():
    """The best of the 66 trend rules judged one at a time: its Sharpe ratio, the probabilistic Sharpe
    ratio against zero, and the deflated Sharpe ratio that allows for the 66 trials, with the skewness and
    excess kurtosis of its returns."""
    import snooping as S
    from scipy import stats
    X, names, _ = S.rule_returns(); d = X.values; T = len(d)
    sr = d.mean(0) / d.std(0, ddof=1); j = int(np.argmax(sr)); r = d[:, j]
    s, k = float(stats.skew(r)), float(stats.kurtosis(r))
    p = L.psr(sr[j], T, s, k); q, bench = L.dsr(sr[j], T, len(sr), float(np.var(sr, ddof=1)), s, k)
    return dict(rule=X.columns[j], sr_annual=float(math.sqrt(12) * sr[j]), psr=p, dsr=q, bench_annual=float(math.sqrt(12) * bench))


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
