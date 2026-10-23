import numpy as np

def lasso(X, y, lam, sweeps=500):
    """Minimise (1/2n)|y - Xb|^2 + lam*|b|_1 by coordinate descent.
    A fixed number of sweeps keeps the listing short; working code also stops
    when b stops changing."""
    n, p = X.shape
    b, r = np.zeros(p), y.copy()            # r is the residual y - Xb
    c = (X ** 2).sum(axis=0) / n
    for _ in range(sweeps):
        for j in range(p):
            if c[j] == 0:                   # zero column: coefficient remains zero
                continue
            z = X[:, j] @ r / n + c[j] * b[j]
            new = np.sign(z) * max(abs(z) - lam, 0.0) / c[j]
            r -= X[:, j] * (new - b[j])
            b[j] = new
    return b

rng = np.random.default_rng(12022)
X = rng.standard_normal((200, 6)); X -= X.mean(axis=0)
y = X @ [1.0, -0.5, 0, 0, 0.25, 0] + rng.standard_normal(200); y -= y.mean()
for lam in (0.01, 0.1, 0.3):
    print(lam, np.round(lasso(X, y, lam), 3))
