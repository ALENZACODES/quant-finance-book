import numpy as np
from scipy import integrate
from scipy.stats import norm

def tranche_el(p, rho, R, a, d):
    """Expected loss of the tranche (a, d] of a large pool, as a share of its size."""
    c = norm.ppf(p)
    def loss(m):
        L = (1 - R) * norm.cdf((c - np.sqrt(rho) * m) / np.sqrt(1 - rho))
        return min(max(L - a, 0.0), d - a) * norm.pdf(m)
    return integrate.quad(loss, -10, 10, limit=200)[0] / (d - a)

for a, d in [(0, .03), (.03, .07), (.07, .10), (.10, .15), (.15, .30), (.30, 1)]:
    el = tranche_el(0.05753, 0.3, 0.4, a, d)
    print(f"{100 * a:3.0f} to {100 * d:3.0f}%: {100 * el:8.4f}%")
