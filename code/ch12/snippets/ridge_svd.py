"""Ridge through the singular value decomposition: coefficients, the shrinkage of
each singular direction and the effective degrees of freedom, checked against the
normal equations."""
import numpy as np


def ridge_svd(X, y, lam):
    """Minimise ||y - Xb||^2 + lam ||b||^2 for centred X of full column rank and centred y."""
    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    U, d, Vt = np.linalg.svd(X, full_matrices=False)
    ok = d > max(X.shape) * np.finfo(float).eps * d.max()   # numerically zero: factor 0
    den = np.where(ok, d**2 + lam, 1.0)
    shrink = np.where(ok, d**2 / den, 0.0)
    b = Vt.T @ (np.where(ok, d / den, 0.0) * (U.T @ y))
    return b, shrink, shrink.sum()


rng = np.random.default_rng(12016)
X = rng.standard_normal((100, 6)); X -= X.mean(0)
y = X @ [1, -0.5, 0, 0.25, 0, 0] + rng.standard_normal(100)
y -= y.mean()
for lam in (0.0, 25.0, 250.0):
    b, shrink, df = ridge_svd(X, y, lam)
    ref = np.linalg.solve(X.T @ X + lam * np.eye(6), X.T @ y)
    print(f"lambda {lam:3.0f}: df {df:.2f}, shrinkage {np.round(shrink, 2)}, "
          f"gap to normal equations {np.abs(b - ref).max():.0e}")
