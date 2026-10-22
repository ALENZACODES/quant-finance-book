import pandas as pd

g = pd.read_csv("data/goc_bills_bonds.csv", index_col=0, parse_dates=True)
m = g.groupby(g.index.to_period("M")).last() / 100     # month-end yields and bill
y10, y7, bill = m["BD.CDN.10YR.DQ.YLD"], m["BD.CDN.7YR.DQ.YLD"], m["TB.CDN.90D.MID"]
ok = y10.notna() & y7.notna() & bill.notna() & (m.index <= pd.Period("2026-08"))
y10, y7, bill = y10[ok], y7[ok], bill[ok]


def price(c, y, years):
    """Price per unit face: coupon rate c paid twice a year, yield y."""
    v = (1 + y / 2) ** (-2 * years)
    return c / y * (1 - v) + v


y_roll = y10 - (y10 - y7) / 3 / 12   # 9 years 11 months, linear between 7 and 10
carry = (y10 - bill) / 12            # coupon accrual less financing
roll = price(y10, y_roll, 10 - 1 / 12) - 1   # slide down an unchanged curve
real = (price(y10.shift(1), y_roll, 10 - 1 / 12) - 1
        + y10.shift(1) / 12 - bill.shift(1) / 12)
d = pd.DataFrame(dict(carry=carry.shift(1), roll=roll.shift(1), real=real)).dropna()
d["curve"] = d.real - d.carry - d.roll    # realised less expected: the curve's move
print(f"{d.index[0]} to {d.index[-1]}, % a year: " +
      ", ".join(f"{k} {1200 * v:.2f}" for k, v in d.mean().items()))
