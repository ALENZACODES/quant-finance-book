"""Backpropagation for one hidden tanh layer and squared error, checked on the
network of Exercise 12.5 and by finite differences."""
import numpy as np


def loss_grad(W1, b1, w2, b2, X, y):
    """Loss 0.5 mean (f - y)^2 of f = tanh(X W1 + b1) w2 + b2, its gradients and f."""
    H = np.tanh(X @ W1 + b1); f = H @ w2 + b2               # forward pass
    df = (f - y) / len(y)                                   # backward: dL/df
    dZ = np.outer(df, w2) * (1 - H**2)                      # through tanh
    return 0.5 * np.mean((f - y)**2), [X.T @ dZ, dZ.sum(0), H.T @ df, df.sum()], f


# Exercise 12.5: x = 1, w1 = 0.5, w2 = 2, no biases, target 0
X, y = np.ones((1, 1)), np.zeros(1)
L, g, f = loss_grad(np.array([[0.5]]), np.zeros(1), np.array([2.0]), 0.0, X, y)
print(f"Exercise 12.5: y hat {f[0]:.4f}, dL/dw2 {g[2][0]:.4f}, "
      f"dL/dw1 {g[0][0, 0]:.4f}")

rng = np.random.default_rng(12058)                    # 2 inputs, 8 hidden units
X = rng.standard_normal((64, 2)); y = np.sin(X[:, 0]) * X[:, 1]
P = [0.5 * rng.standard_normal((2, 8)), np.zeros(8), 0.5 * rng.standard_normal(8), 0.0]
_, G, _ = loss_grad(*P, X, y)
gap = 0.0
for i in range(4):                                # central differences, one by one
    for j in np.ndindex(np.shape(P[i])):
        up = [np.array(p, float) for p in P]; dn = [np.array(p, float) for p in P]
        up[i][j] += 1e-6; dn[i][j] -= 1e-6
        fd = (loss_grad(*up, X, y)[0] - loss_grad(*dn, X, y)[0]) / 2e-6
        gap = max(gap, abs(fd - np.asarray(G[i])[j]))
n = sum(np.size(p) for p in P)
print(f"{n} parameters: largest gap to finite differences {gap:.1e}")
