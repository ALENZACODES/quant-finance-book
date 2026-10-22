import numpy as np
import pandas as pd

names = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
         "Telcm", "Shops", "Hlth", "Utils", "Other"]
f = pd.read_csv("data/ff_factors_monthly.csv", index_col=0)
ind = pd.read_csv("data/industry10_monthly.csv", index_col=0)
X = ind[names].sub(f["rf"], axis=0).values / 100


def shrunk(Xw):
    """Ledoit and Wolf (2004): covariance shrunk towards a multiple of the identity."""
    T, N = Xw.shape
    Z = Xw - Xw.mean(0); S = Z.T @ Z / T
    m = np.trace(S) / N; d2 = np.sum((S - m * np.eye(N)) ** 2) / N
    b2 = sum(np.sum((np.outer(z, z) - S) ** 2) for z in Z) / N / T ** 2
    rho = min(b2, d2) / d2                              # weight on the target
    return rho * m * np.eye(N) + (1 - rho) * S


def gmv(S):
    w = np.linalg.solve(S, np.ones(len(S)))
    return w / w.sum()


W = 60                                                  # five years of months
out = {"sample": [], "shrunk": []}
for t in range(W, len(X)):
    Xw = X[t - W:t]                                     # only months before t
    out["sample"].append(gmv(np.cov(Xw.T)) @ X[t])
    out["shrunk"].append(gmv(shrunk(Xw)) @ X[t])
for k, x in out.items():
    x = np.array(x)
    sr = np.sqrt(12) * x.mean() / x.std(ddof=1)
    print(f"{k} minimum variance: Sharpe ratio {sr:.2f}")
