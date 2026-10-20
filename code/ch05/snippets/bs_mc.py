import numpy as np
from scipy.stats import norm

def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sigma * np.sqrt(T))

S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
z = np.random.default_rng(1).standard_normal(500_000)
def disc_payoff_and_control(z):
    ST = S * np.exp((r - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * z)
    return np.exp(-r * T) * np.maximum(ST - K, 0), np.exp(-r * T) * ST
(Y1, X1), (Y2, X2) = disc_payoff_and_control(z), disc_payoff_and_control(-z)
Y, X = (Y1 + Y2) / 2, (X1 + X2) / 2           # antithetic pairs
b = np.cov(Y, X)[0, 1] / np.var(X, ddof=1)     # control: E[X] = S
est = Y - b * (X - S)
m, se = est.mean(), est.std(ddof=1) / np.sqrt(len(est))
print(f"formula {bs_call(S, K, r, sigma, T):.4f}, Monte Carlo {m:.4f}, se {se:.4f}")
