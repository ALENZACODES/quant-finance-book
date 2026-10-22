# Pro forma of the building in Table 10.1: Gordon value, a ten year DCF with a sale
# at an exit cap a quarter point higher less 2% costs, and the IRR at the Gordon price.
import numpy as np
from scipy.optimize import brentq

noi = 2.40e6 * (1 - 0.07) + 0.05e6 - 0.82e6            # net operating income, year 1
r, g, capex, T = 0.075, 0.025, 0.10, 10
cap = (r - g) / (1 - capex)                             # going-in cap rate
value = noi * (1 - capex) / (r - g)                     # Gordon value
t = np.arange(1, T + 1)
cf = noi * (1 + g) ** (t - 1) * (1 - capex)             # cash flow after the reserve


def dcf(exit_cap, sale_cost=0.0):
    sale = noi * (1 + g) ** T / exit_cap * (1 - sale_cost)
    return np.sum(cf / (1 + r) ** t) + sale / (1 + r) ** T


def irr(exit_cap, sale_cost=0.0):
    flows = np.concatenate([[-value], cf])
    flows[-1] += noi * (1 + g) ** T / exit_cap * (1 - sale_cost)
    return brentq(lambda x: np.sum(flows / (1 + x) ** np.arange(T + 1)), -0.9, 1.0)


print(f"Gordon value {value / 1e6:.1f}m, "
      f"DCF at the going-in cap {dcf(cap) / 1e6:.1f}m")
print(f"DCF at exit cap {100 * (cap + 0.0025):.2f}% with costs "
      f"{dcf(cap + 0.0025, 0.02) / 1e6:.1f}m")
print(f"IRR {100 * irr(cap + 0.0025, 0.02):.2f}% against {100 * irr(cap):.2f}%")
