import numpy as np
import pandas as pd

names = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
         "Telcm", "Shops", "Hlth", "Utils", "Other"]
f = pd.read_csv("data/ff_factors_monthly.csv", index_col=0)
ind = pd.read_csv("data/industry10_monthly.csv", index_col=0)
X = ind[names].sub(f["rf"], axis=0).values / 100       # monthly excess returns
mu, S = X.mean(0), np.cov(X.T)
one = np.ones(len(mu))
Si1, Sim = np.linalg.solve(S, one), np.linalg.solve(S, mu)
A, B, C = one @ Si1, one @ Sim, mu @ Sim                # the constants of the frontier
D = A * C - B * B
w_g = Si1 / A                                           # global minimum variance
w_t = Sim / B                                           # tangency, fully invested
for label, w in (("minimum variance", w_g), ("tangency", w_t)):
    m, v = w @ mu, w @ S @ w
    print(f"{label}: mean {1200 * m:.2f}%, volatility {100 * np.sqrt(12 * v):.1f}%, "
          f"Sharpe {np.sqrt(12) * m / np.sqrt(v):.2f}")
print(f"check: B/A = {1200 * B / A:.2f}%, C/B = {1200 * C / B:.2f}%, "
      f"sqrt(12 C) = {np.sqrt(12 * C):.2f}, D > 0: {D > 0}")
