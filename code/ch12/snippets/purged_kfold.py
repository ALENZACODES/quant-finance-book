"""Purged and embargoed K-fold splits for labels that use the next h returns: label s
uses returns s+1..s+h, so a training label must end before the test block starts or
begin after the last test label's window ends, plus the embargo."""
import numpy as np


def purged_kfold(n, k=10, h=12, embargo=12):
    if not (1 <= k <= n and h >= 1 and embargo >= 0):
        raise ValueError("need 1 <= k <= n, h >= 1 and embargo >= 0")
    idx = np.arange(n)
    edges = np.linspace(0, n, k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:] - 1):          # test labels a..b
        train = idx[(idx <= a - h) | (idx >= b + h + embargo)]
        assert np.all((train + h <= a) | (train >= b + h))  # no shared return
        yield train, np.arange(a, b + 1)


for i, (train, test) in enumerate(purged_kfold(468), 1):
    if i <= 3 or i == 10:
        print(f"fold {i:2d}: test {test[0]:3d}-{test[-1]:3d}, train {len(train)}")
# Exercise 12.4: test labels 40-45, h = 12, no embargo
a, b = 40, 45
kept = [s for s in range(100) if s <= a - 12 or s >= b + 12]
print("Exercise 12.4: labels excluded", 100 - len(kept))
