import numpy as np
from scipy.optimize import brentq

t = np.array([0.0, 1.0, 2.0, 4.0, 6.0])            # years
calls = np.array([40.0, 35.0, 25.0, 0.0, 0.0])      # paid in
dists = np.array([0.0, 0.0, 20.0, 70.0, 90.0])      # paid out
nav = 30.0                                          # value left at t = 6
index = np.array([100.0, 108.0, 101.0, 125.0, 140.0])

grow = index[-1] / index                            # carry each flow to t = 6
pme = (dists @ grow + nav) / (calls @ grow)
flows = np.append((dists - calls) * grow, nav)
times = np.append(t, t[-1])
alpha = brentq(lambda a: flows @ np.exp(-a * times), -1, 1)
print(f"KS-PME {pme:.4f}, direct alpha {alpha:.2%}")
