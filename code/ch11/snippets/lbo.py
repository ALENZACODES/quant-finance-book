# The five year buyout of Table 11.1, in millions: every dollar after interest, tax,
# capital spending and working capital repays debt; cash, if any, earns i(1 - tax).
def buyout(debt_multiple):
    rev, ebitda0, i, tax = 250.0, 50.0, 0.08, 0.25
    growth, margin = [0.06, 0.06, 0.05, 0.05, 0.04], [0.205, 0.21, 0.215, 0.22, 0.22]
    debt = debt_multiple * ebitda0; cash = 0.0
    equity_in = 10.0 * ebitda0 - debt + 0.02 * 10.0 * ebitda0   # price - debt + fees
    for g, m in zip(growth, margin):
        cash *= 1 + i * (1 - tax)
        prev, rev = rev, rev * (1 + g)
        ebitda = m * rev; interest = i * debt
        taxes = max(ebitda - 0.03 * rev - interest, 0.0) * tax
        fcf = ebitda - taxes - 0.035 * rev - 0.10 * (rev - prev) - interest
        repay = min(max(fcf, 0.0), debt); debt -= repay; cash += fcf - repay
    ev = 10.5 * ebitda; equity = ev - (debt - cash)
    moic = equity / equity_in
    return ebitda, ev, debt - cash, equity, moic, moic ** (1 / 5) - 1


e, ev, nd, eq, moic, irr = buyout(5.0)
print(f"EBITDA {e:.1f}, value {ev:.1f}, net debt {nd:.1f}, equity {eq:.1f}")
print(f"multiple {moic:.2f}, IRR {100 * irr:.1f}%")
*_, moic_u, irr_u = buyout(0.0)
print(f"without debt: multiple {moic_u:.2f}, IRR {100 * irr_u:.1f}%")
