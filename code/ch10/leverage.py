"""Section 10.2: leverage. The mortgage constant, the debt service coverage and loan-to-value tests
for the building of Section 10.1, the loan they allow, the return on equity and its volatility, and a
simulation of ten year unlevered and levered IRRs with random rent growth and a random exit cap rate.
Writes fig02_levered.png.
Also writes fig14_loanrate.png, the largest loan against the mortgage rate at three cap rates."""
import numpy as np
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, PALE, put, pct, finish
import proforma as P

RATE, AMORT, TERM = 0.0625, 25, 10
LTV_MAX, DSCR_MIN = 0.65, 1.25
NSIM = 20000
G_SD, CAP_SD = 0.025, 0.01   # sd of annual rent growth shocks and of the exit cap rate


def simulate(rng, value, noi, loan, cap, n=NSIM):
    """Unlevered and levered ten year IRRs and the lowest DSCR on each simulated path."""
    k = L.mortgage_constant(RATE, AMORT)
    bal = L.balance(RATE, AMORT, TERM)
    j = RATE / 12
    ds = k * ((1 + j) ** 12 - 1) / (12 * j)   # the year's monthly payments carried to its end at the loan rate
    g = P.G + G_SD * rng.standard_normal((n, TERM))
    path = noi * np.cumprod(1 + np.concatenate([np.zeros((n, 1)), g[:, :-1]], axis=1), axis=1)
    noi_next = path[:, -1] * (1 + g[:, -1])
    exitc = np.maximum(cap + P.SPREAD_EXIT + CAP_SD * rng.standard_normal(n), 0.03)
    sale = noi_next / exitc * (1 - P.SALE_COST)
    cf = path * (1 - P.CAPEX)
    iu, il = np.empty(n), np.empty(n)
    for m in range(n):
        f = np.concatenate([[-value], cf[m]]); f[-1] += sale[m]; iu[m] = L.irr(f)
        e = np.concatenate([[-(value - loan)], cf[m] - ds * loan]); e[-1] += sale[m] - bal * loan; il[m] = L.irr(e)
    dmin = (path / (k * loan)).min(axis=1)
    return iu, il, dmin


def main():
    noi = P.GPR * (1 - P.VAC) + P.OTHER - P.OPEX
    cap = L.cap_rate(P.RREQ, P.G, P.CAPEX); value = L.gordon_value(noi, P.RREQ, P.G, P.CAPEX)
    k = L.mortgage_constant(RATE, AMORT)
    pct("lvRate", RATE, 2); put("lvAmort", AMORT, "{:.0f}"); put("lvTerm", TERM, "{:.0f}")
    pct("lvK", k, 2); pct("lvLtvMax", LTV_MAX, 0); put("lvDscrMin", DSCR_MIN, "{:.2f}")
    put("lvDscrAtMax", L.dscr(noi / value, LTV_MAX, k), "{:.2f}")
    loan, bind = L.max_loan(noi, value, k, DSCR_MIN, LTV_MAX)
    assert bind == "DSCR"
    put("lvLoan", loan / 1e6, "{:.1f}"); pct("lvLtv", loan / value, 1)
    pct("lvDebtYield", noi / loan, 1)
    bal = L.balance(RATE, AMORT, TERM)
    pct("lvBal", bal, 1)
    # rate at which the LTV test would bind instead
    from scipy import optimize
    rstar = optimize.brentq(lambda x: noi / (DSCR_MIN * L.mortgage_constant(x, AMORT)) - LTV_MAX * value, 0.001, 0.2)
    pct("lvRateSwitch", rstar, 2)
    # leverage arithmetic: positive leverage needs the unlevered return above the cost of debt
    lam = loan / value
    pct("lvLevRet", L.levered_return(P.RREQ, RATE, lam), 2)
    eff = (1 + RATE / 12) ** 12 - 1                      # the quoted rate paid monthly, as an annual cost
    pct("lvRateEff", eff, 2); pct("lvLevRetEff", L.levered_return(P.RREQ, eff, lam), 2)
    put("lvVolMult", 1 / (1 - lam), "{:.2f}")
    # the cash on cash yield in year one is below the cap rate: negative leverage at the start
    pct("lvCashYield", (noi * (1 - P.CAPEX) - k * loan) / (value - loan), 2)
    pct("lvCfYield", noi * (1 - P.CAPEX) / value, 2)
    # simulation
    rng = R.rng_for(1)
    iu, il, dmin = simulate(rng, value, noi, loan, cap)
    put("lvNsim", NSIM, "{:,.0f}"); pct("lvGsd", G_SD, 1); pct("lvCapSd", CAP_SD, 2)
    pct("lvIuMean", iu.mean(), 2); pct("lvIlMean", il.mean(), 2)
    pct("lvIuSd", iu.std(ddof=1), 2); pct("lvIlSd", il.std(ddof=1), 2)
    put("lvSdRatio", il.std(ddof=1) / iu.std(ddof=1), "{:.2f}")
    pct("lvIuQ", np.quantile(iu, 0.05), 2); pct("lvIlQ", np.quantile(il, 0.05), 2)
    pct("lvPLoss", np.mean(il < 0), 1); pct("lvPLossU", np.mean(iu < 0), 1)
    pct("lvPBelow", np.mean(il < iu), 1)
    assert abs(np.mean(il < iu) - np.mean(iu < eff)) < 1e-3   # levered below unlevered exactly when the property earns less than the loan costs
    # figure
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    b = np.linspace(-0.06, 0.18, 61)
    ax.hist(iu, bins=b, color=PALE, edgecolor=BLUE, lw=0.5, label="unlevered")
    ax.hist(il, bins=b, histtype="step", color=RED, lw=1.2, label=f"levered at {100 * lam:.0f}% of value")
    ax.axvline(eff, color=GREY, lw=0.9, ls=":")
    ax.set_xlabel("ten year IRR"); ax.set_ylabel("simulated paths")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); ax.grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig02_levered"); plt.close(fig)
    loan_rate_figure(noi / value, rstar)
    return finish("leverage")


def loan_rate_figure(cap, rstar):
    """Largest loan as a share of value, min(LTV_MAX, cap / (DSCR_MIN k)), against the mortgage rate."""
    rates = np.linspace(0.03, 0.09, 241)
    share = lambda c, x: min(LTV_MAX, c / (DSCR_MIN * L.mortgage_constant(x, AMORT)))
    for x, tag in ((0.03, "Three"), (0.09, "Nine")):
        pct(f"lrLtv{tag}", share(cap, x), 0)
    for c, tag in ((0.05, "Low"), (0.065, "High")):
        from scipy import optimize
        pct(f"lrSwitch{tag}", optimize.brentq(lambda x: c / (DSCR_MIN * L.mortgage_constant(x, AMORT)) - LTV_MAX,
                                             0.001, 0.2), 2)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for c, col, ls in ((0.05, SKY, "--"), (cap, BLUE, "-"), (0.065, RED, "-.")):
        ax.plot(100 * rates, [100 * share(c, x) for x in rates], color=col, ls=ls,
                label=f"cap rate {100 * c:.2f}%")
    ax.axvline(100 * rstar, color=GREY, lw=0.8, ls=":")
    ax.annotate(f"DSCR binds above {100 * rstar:.2f}%", (100 * rstar, 40), xytext=(4, 0),
                textcoords="offset points", fontsize=7)
    ax.set_xlabel("mortgage rate, %"); ax.set_ylabel("largest loan, % of value")
    ax.set_ylim(35, 70); ax.legend(loc="lower left")
    fig.tight_layout(); R.save(fig, "fig14_loanrate"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
