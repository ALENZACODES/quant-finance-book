import sys
import numpy as np
from tsmom import programme, sharpe

sys.path.insert(0, "code/ch09")
from ch09lib import load_markets

r, _ = load_markets()
actual = sharpe(programme(r))
rng = np.random.default_rng([9473, 2])      # the chapter's stream for this test
live = np.flatnonzero(r.notna().all(axis=1).values)   # months with all seven
N, shuffled = 400, []
for _ in range(N):
    order = np.arange(len(r))
    order[live] = rng.permutation(live)       # whole months move; earlier ones stay
    rp = r.iloc[order].copy()
    rp.index = r.index
    shuffled.append(sharpe(programme(rp)))
k = int(np.sum(np.array(shuffled) >= actual))
print(f"actual {actual:.2f}; {k} of {N} shuffles at least as high; "
      f"p = (1 + k) / (N + 1) = {(1 + k) / (N + 1):.3f}")
