import math
import numpy as np
from scipy.stats import chi2


def xlogy(k, q):
    """k log q with 0 log 0 = 0, so empty cells of the count table are allowed."""
    return k * math.log(q) if k > 0 else 0.0


def kupiec(hits, p):
    """Unconditional coverage: x exceptions in T days against the rate p."""
    h = np.asarray(hits)
    if h.size == 0 or not np.isin(h, (0, 1)).all():
        raise ValueError("hits must be a nonempty sequence of zeros and ones")
    T, x = h.size, int(h.sum())
    lr = -2 * (xlogy(T - x, 1 - p) + xlogy(x, p)
               - xlogy(T - x, 1 - x / T) - xlogy(x, x / T))
    return lr, chi2.sf(lr, 1)


def christoffersen(hits):
    """Independence against a first order Markov chain of hits."""
    h = np.asarray(hits)
    if h.size < 2 or not np.isin(h, (0, 1)).all():
        raise ValueError("hits must be zeros and ones, at least two of them")
    h = h.astype(int); a, b = h[:-1], h[1:]
    n = {(i, j): int(np.sum((a == i) & (b == j))) for i in (0, 1) for j in (0, 1)}
    p01 = n[0, 1] / max(1, n[0, 0] + n[0, 1])
    p11 = n[1, 1] / max(1, n[1, 0] + n[1, 1])             # zero when n11 = 0
    p1 = (n[0, 1] + n[1, 1]) / (len(h) - 1)
    chain = (xlogy(n[0, 0], 1 - p01) + xlogy(n[0, 1], p01)
             + xlogy(n[1, 0], 1 - p11) + xlogy(n[1, 1], p11))
    indep = xlogy(n[0, 0] + n[1, 0], 1 - p1) + xlogy(n[0, 1] + n[1, 1], p1)
    lr = -2 * (indep - chain)
    return lr, chi2.sf(lr, 1), n


if __name__ == "__main__":
    hits = np.zeros(250, int)
    hits[[40, 90, 160, 161, 220]] = 1                     # five exceptions, one pair
    print("Kupiec LR %.3f, p %.3f" % kupiec(hits, 0.01))
    print("Christoffersen LR %.3f, p %.3f" % christoffersen(hits)[:2])
    hits[161] = 0                                         # four apart: n11 = 0
    print("n11 = 0: LR %.3f, p %.3f" % christoffersen(hits)[:2])
