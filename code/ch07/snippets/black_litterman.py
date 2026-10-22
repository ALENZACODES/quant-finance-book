import numpy as np

def black_litterman(S, w_mkt, delta, tau, P, q, Omega):
    """Posterior mean returns and the weights they imply."""
    pi = delta * S @ w_mkt                     # equilibrium excess returns
    M = P @ (tau * S) @ P.T + Omega
    mu = pi + tau * S @ P.T @ np.linalg.solve(M, q - P @ pi)
    w = np.linalg.solve(delta * S, mu)         # equals w_mkt + P.T @ lam
    return mu, w

S = np.diag([0.04, 0.09]); w_mkt = np.array([0.5, 0.5])
P = np.array([[1.0, 0.0]]); q = np.array([0.03]); Omega = np.array([[0.05 * 0.04]])
mu, w = black_litterman(S, w_mkt, 2.5, 0.05, P, q, Omega)
print("posterior means", mu.round(4), "weights", w.round(3))
