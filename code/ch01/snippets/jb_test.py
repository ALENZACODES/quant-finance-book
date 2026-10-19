import numpy as np
from scipy import stats

def jarque_bera(x):
    d = x - x.mean()
    m2, m3, m4 = (d**2).mean(), (d**3).mean(), (d**4).mean()
    S, K = m3 / m2**1.5, m4 / m2**2 - 3.0     # skewness, excess kurtosis
    jb = len(x) / 6 * (S**2 + K**2 / 4)
    return jb, stats.chi2.sf(jb, df=2)

rng = np.random.default_rng(101)
for name, x in [("normal", rng.standard_normal(500)), ("t5", rng.standard_t(5, 500))]:
    jb, p = jarque_bera(x)
    print(f"{name}: JB = {jb:.2f}, p = {p:.4f}")
