import numpy as np

def irr_newton(cfs, r=0.10, tol=1e-12, max_iter=100):
    """Newton's method on NPV(r) = sum_t CF_t (1+r)^(-t)."""
    t = np.arange(len(cfs))
    for _ in range(max_iter):
        f = np.sum(cfs * (1 + r) ** -t)
        df = np.sum(-t * cfs * (1 + r) ** (-t - 1))
        if df == 0:
            raise ZeroDivisionError("NPV'(r) = 0: Newton step undefined")
        step = f / df
        r -= step
        if r <= -1:
            raise ValueError("trial rate at or below -100%")
        if abs(step) < tol:
            return r
    raise RuntimeError("no convergence")

fund_a = np.array([-22, -24, -22, -17, -12, 38.23, 42.25, 38.23, 28.17, 20.11])
print(f"IRR of Fund A: {irr_newton(fund_a):.4%}")
