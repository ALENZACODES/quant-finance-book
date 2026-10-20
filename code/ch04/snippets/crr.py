import numpy as np

def crr(S, K, r, sigma, T, n, american=False):
    h = T / n; u = np.exp(sigma * np.sqrt(h)); d = 1 / u
    q = (np.exp(r * h) - d) / (u - d); disc = np.exp(-r * h)
    S_T = S * u ** np.arange(n + 1) * d ** np.arange(n, -1, -1)
    V = np.maximum(K - S_T, 0)                       # a put
    for i in range(n - 1, -1, -1):
        V = disc * (q * V[1:] + (1 - q) * V[:-1])
        if american:
            S_i = S * u ** np.arange(i + 1) * d ** np.arange(i, -1, -1)
            V = np.maximum(V, K - S_i)
    return V[0]

print(f"European {crr(100, 100, 0.05, 0.2, 1, 1000):.4f}, "
      f"American {crr(100, 100, 0.05, 0.2, 1, 1000, american=True):.4f}")
