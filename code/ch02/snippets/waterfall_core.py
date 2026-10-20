"""Waterfall for one contribution K at time 0 and one distribution V at T."""


def waterfall(V, K=100.0, T=5.0, h=0.08, c=0.20, k=1.0, hard=False):
    assert k > c, "catch-up rate must exceed the carry rate"
    H = K * (1 + h) ** T                     # capital plus preferred return
    lp = min(V, H)                           # tiers 1 and 2: LPs up to H
    x = V - lp
    if hard:                                 # no catch-up: carry above H
        return lp + (1 - c) * x, c * x
    y = min(x, max(c * (H - K) / (k - c), 0))  # tier 3: catch-up
    lp, gp, x = lp + (1 - k) * y, k * y, x - y
    return lp + (1 - c) * x, gp + c * x      # tier 4: the split


if __name__ == "__main__":
    for V in (140, 150, 180):
        lp, gp = waterfall(V)
        print(f"V = {V}: LP {lp:.2f}, GP {gp:.2f}")
    print(f"Hard hurdle, V = 180: GP {waterfall(180, hard=True)[1]:.2f}")
