"""The best split of one predictor by cumulative sums (Proposition 12.5),
checked by brute force."""
import numpy as np


def best_split(x, y, min_leaf=1):
    """Threshold and fall in the sum of squares from the best split of y on x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or min_leaf < 1:
        raise ValueError("x and y must match and min_leaf be at least 1")
    o = np.argsort(x, kind="stable"); xs, ys = x[o], y[o]
    n = len(y); nl = np.arange(1, n); sl = np.cumsum(ys)[:-1]
    gain = sl**2 / nl + (ys.sum() - sl)**2 / (n - nl) - ys.sum()**2 / n
    ok = (nl >= min_leaf) & (n - nl >= min_leaf) & (xs[:-1] < xs[1:])
    if not ok.any():
        return None, 0.0
    j = np.flatnonzero(ok)[np.argmax(gain[ok])]
    return (xs[j] + xs[j + 1]) / 2, gain[j]


def sse(v):
    return ((v - v.mean())**2).sum() if len(v) else 0.0


rng = np.random.default_rng(12042)
x = rng.uniform(size=200)
y = np.where(x < 0.45, -1.0, 2.0) + 0.3 * rng.standard_normal(200)
t, g = best_split(x, y)
u = np.unique(x); cuts = (u[:-1] + u[1:]) / 2
brute = [sse(y) - sse(y[x <= c]) - sse(y[x > c]) for c in cuts]
print(f"split at {t:.4f}, fall in SSE {g:.2f}; "
      f"brute force {cuts[np.argmax(brute)]:.4f}, {max(brute):.2f}")
