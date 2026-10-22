# Loan-to-value of a loan written at 65% of value, at the 10 year Treasury yield plus
# 1.8 points, amortised over 25 years, marked each quarter against the Z.1 index.
import sys
import numpy as np
sys.path.insert(0, "code/ch10")
import ch10lib as L

cre = L.load_quarterly()["cre"].dropna()
cre = cre[cre.index > L.SPLICE]                         # the repeat-sales era
gs10 = L.load_monthly()["gs10"] / 100
quarters = np.arange(21)                                # five years of quarters
peak = {}
for i, t0 in enumerate(cre.index[:len(cre) - 20]):
    rate = gs10.loc[t0] + 0.018
    bal = np.array([L.balance(rate, 25, q / 4) for q in quarters])
    px = cre.iloc[i:i + 21].to_numpy() / cre.iloc[i]
    peak[t0] = (0.65 * bal / px).max()          # lambda_t = lambda_0 B_t / (P_t/P_0)
worst = max(peak, key=peak.get)
share = np.mean([v > 0.80 for v in peak.values()])
print(f"{len(peak)} vintages; worst {worst.year}Q{worst.quarter} peaks at "
      f"{100 * peak[worst]:.0f}%; {100 * share:.0f}% of vintages pass 80%")
