# Intrinsic value of storage as a linear programme: inject x_i and withdraw y_i at
# each delivery date, inventory between 0 and 1, start and end empty (Exercise 10.15).
# The LP allows injection and withdrawal at the same date; with positive costs of
# filling and emptying that is never optimal, so it does not change the value.
import numpy as np
from scipy.optimize import linprog


def intrinsic_lp(F, t, r, c_in, c_out, rate=1.0):
    n = len(F); d = np.exp(-r * np.asarray(t))
    cost = np.concatenate([d * (np.asarray(F) + c_in), -d * (np.asarray(F) - c_out)])
    tri = np.tril(np.ones((n, n)))                     # cumulative sums
    A = np.block([[tri, -tri], [-tri, tri]])            # 0 <= inventory <= 1
    b = np.concatenate([np.ones(n), np.zeros(n)])
    eq = np.concatenate([np.ones(n), -np.ones(n)])[None, :]     # ends empty
    res = linprog(cost, A_ub=A, b_ub=b, A_eq=eq, b_eq=[0], bounds=[(0, rate)] * 2 * n)
    return -res.fun, res.x[:n], res.x[n:]


v, x, y = intrinsic_lp([3.00, 3.60, 3.20], [1 / 12, 4 / 12, 7 / 12], 0.04, 0.05, 0.05)
print(f"Exercise 10.15: intrinsic value {v:.3f}, injected {x.round(2) + 0}, "
      f"withdrawn {y.round(2) + 0}")
# a stylised seasonal curve, twelve months, filled or emptied at most a third a month;
# month 0 is the first delivery, a month ahead; the curve peaks in the fifth (m = 4)
m = np.arange(12)
F = 3.20 + 0.60 * np.cos(2 * np.pi * (m - 4) / 12)
v1, *_ = intrinsic_lp(F, (m + 1) / 12, 0.04, 0.05, 0.05)
v3, x, y = intrinsic_lp(F, (m + 1) / 12, 0.04, 0.05, 0.05, rate=1 / 3)
print(f"twelve months: {v1:.3f} filling at once, {v3:.3f} at a third a month")
