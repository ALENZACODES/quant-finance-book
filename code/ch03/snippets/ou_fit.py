import numpy as np

rng = np.random.default_rng(1)
# 50 years of monthly data
kappa, theta, sigma, dt, n = 1.2, 0.03, 0.02, 1 / 12, 600
phi = np.exp(-kappa * dt)
sd = sigma * np.sqrt((1 - phi**2) / (2 * kappa))   # exact one-step s.d.
x = np.empty(n + 1)
x[0] = theta
for k in range(n):
    x[k + 1] = theta + phi * (x[k] - theta) + sd * rng.standard_normal()
phi_hat, c_hat = np.polyfit(x[:-1], x[1:], 1)     # x_{k+1} on x_k
kappa_hat = -np.log(phi_hat) / dt
half = np.log(2) / kappa_hat
print(f"kappa_hat = {kappa_hat:.3f}, half-life = {half:.3f} years")
