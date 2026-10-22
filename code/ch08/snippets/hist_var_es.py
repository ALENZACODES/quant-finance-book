import math
import numpy as np


def hist_var_es(returns, alpha):
    """Historical VaR and ES at confidence alpha, both as positive losses.
    VaR is the k-th largest loss, k = ceil(n(1 - alpha)). ES is the mean of the
    worst n(1 - alpha) observations, so when that is not a whole number the
    k-th loss enters with the fraction of a weight that falls inside the tail."""
    loss = np.sort(-np.asarray(returns, float).ravel())[::-1]  # largest loss first
    if loss.size == 0 or not np.all(np.isfinite(loss)):
        raise ValueError("returns must be a nonempty array of finite numbers")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    m = round(len(loss) * (1 - alpha), 9)                 # observations in the tail
    if m <= 0:
        raise ValueError("alpha leaves no observation in the tail")
    k = max(1, math.ceil(m))
    w = np.ones(k)
    w[-1] = m - (k - 1)                                   # weight of the boundary loss
    return loss[k - 1], float(w @ loss[:k]) / m


if __name__ == "__main__":
    r = np.array([-0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.01, 0.02, 0.02, 0.03])
    for a in (0.90, 0.85):                                # n(1 - a) = 1 and 1.5
        v, e = hist_var_es(r, a)
        print(f"alpha {a:.2f}: VaR {v:.4f}, ES {e:.4f}")
