# Whole fund waterfall of Definition 11.1 for a fund that calls 100 at year 0 and
# pays out 50 at year 3 and 140 at year 6, with an 8% hurdle and 20% carry.
h, c, k = 0.08, 0.20, 1.0
dates, calls, cash = [0, 3, 6], [100.0, 0.0, 0.0], [0.0, 50.0, 140.0]
hurdle = unreturned = called = lp = gp = 0.0
prev = 0
for t, C, X in zip(dates, calls, cash):
    hurdle *= (1 + h) ** (t - prev); prev = t            # the preferred return accrues
    hurdle += C; unreturned += C; called += C
    pay = min(X, hurdle); roc = min(pay, unreturned)     # tiers (i) and (ii)
    hurdle -= pay; unreturned -= roc; lp += pay; X -= pay
    catch = split = 0.0
    if X > 0 and hurdle <= 1e-12:
        y = max((c * (lp - called + gp) - gp) / (k - c), 0.0)   # Equation (11.1)
        y = min(y, X); catch = k * y; gp += k * y; lp += (1 - k) * y; X -= y
        split = c * X; gp += split; lp += X - split          # tier (iv)
    print(f"year {t}: capital {roc:.2f}, preferred {pay - roc:.2f}, "
          f"catch-up {catch:.2f}, GP split {split:.2f}")
print(f"GP {gp:.2f}, LPs {lp:.2f}, GP share of profit {gp / (lp + gp - called):.2f}")
