"""Computing projects for Chapter 7. Run:  python code/ch07/projects.py"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "common"))
import shipped as D  # noqa: E402
import ch07lib as L  # noqa: E402


def project_1(split="2014-12"):
    """Mean-variance portfolios of six markets (WTI crude, two 10 year bonds and three currencies)
    estimated on the months to December 2014 and held from 2015: the in-sample and out of sample Sharpe
    ratios of the tangency, minimum variance and equal risk contribution portfolios, with the sample
    covariance and with Ledoit-Wolf shrinkage."""
    x = D.market_returns()
    ins, out = x[x.index <= pd.Period(split, "M")], x[x.index > pd.Period(split, "M")]
    mu = ins.mean().values; S = np.cov(ins.values.T); Sl, rho = L.lw_identity(ins.values)
    res = dict(n_in=len(ins), n_out=len(out), shrink=float(rho))
    def sr(w, X):
        p = X.values @ w; return float(math.sqrt(12) * p.mean() / p.std(ddof=1))
    for lab, C in (("sample", S), ("lw", Sl)):
        for nm, w in (("tan", L.tangency(mu, C)), ("gmv", L.gmv(C)), ("erc", L.erc(C))):
            res[f"{nm}_{lab}_in"] = sr(w, ins); res[f"{nm}_{lab}_out"] = sr(w, out)
    return res


def project_2(reps=2000, seed=7):
    """How much of the in-sample Sharpe ratio of the tangency portfolio is estimation error: a parametric
    bootstrap that draws samples of the same length from a normal law with the sample mean and covariance,
    refits the tangency portfolio on each and records its in-sample squared Sharpe ratio against the true
    one."""
    x = D.market_returns(); mu = x.mean().values; S = np.cov(x.values.T); T, N = x.shape
    true = L.sharpe2(mu, S); rng = np.random.default_rng(seed); est = []
    for _ in range(reps):
        z = rng.multivariate_normal(mu, S, T)
        est.append(L.sharpe2(z.mean(0), np.cov(z.T)))
    est = np.array(est)
    return dict(T=T, N=N, true_sr2=true, mean_est_sr2=float(est.mean()), theory_mean=float(L.sharpe2_bias(true, N, T)),
                se_mean=float(est.std(ddof=1) / math.sqrt(reps)))


def main():
    for f in (project_1, project_2):
        print(f.__name__, f())


if __name__ == "__main__":
    main()
