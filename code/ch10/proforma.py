"""Section 10.1: from rents to value. A stylised building's pro forma, the cap rate implied by a
required return, growth and a capital reserve, a ten year DCF with a sale at an exit cap rate, the IRR
at the asking price, and the sensitivity of value to the exit cap rate. Writes fig01_exitcap.png."""
import math
import numpy as np
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, put, pct, finish

GPR = 2.40e6          # gross potential rent, dollars a year
VAC = 0.07            # vacancy and credit loss, share of potential rent
OTHER = 0.05e6        # parking and other income
OPEX = 0.82e6         # operating expenses not recovered from tenants
CAPEX = 0.10          # capital reserve (leasing commissions, tenant improvements, roof), share of NOI
RREQ, G, T = 0.075, 0.025, 10
SPREAD_EXIT = 0.0025  # exit cap above the going-in cap, for an older building at sale
SALE_COST = 0.02


def main():
    egi = GPR * (1 - VAC) + OTHER
    noi = egi - OPEX
    put("pfGpr", GPR / 1e6, "{:.2f}"); pct("pfVac", VAC, 0); put("pfOther", OTHER / 1e6, "{:.2f}")
    put("pfEgi", egi / 1e6, "{:.3f}"); put("pfOpex", OPEX / 1e6, "{:.2f}"); put("pfNoi", noi / 1e6, "{:.3f}")
    put("pfVacAmt", GPR * VAC / 1e6, "{:.3f}"); put("pfCapexAmt", noi * CAPEX / 1e6, "{:.3f}")
    pct("pfOpexShare", OPEX / egi, 1); pct("pfCapex", CAPEX, 0); put("pfCf", noi * (1 - CAPEX) / 1e6, "{:.3f}")
    pct("pfR", RREQ, 1); pct("pfG", G, 1); put("pfT", T, "{:.0f}")
    cap = L.cap_rate(RREQ, G, CAPEX); value = L.gordon_value(noi, RREQ, G, CAPEX)
    pct("pfCap", cap, 2); put("pfValue", value / 1e6, "{:.1f}"); pct("pfCapNoCapex", RREQ - G, 2)
    put("pfValueNoCapex", noi / (RREQ - G) / 1e6, "{:.1f}")
    # DCF with exit at the going-in cap gives the Gordon value; with a higher exit cap and costs, less
    v_same = L.dcf_value(noi, G, RREQ, T, cap, CAPEX)
    assert abs(v_same - value) < 1e-6 * value
    exitc = cap + SPREAD_EXIT
    v_dcf = L.dcf_value(noi, G, RREQ, T, exitc, CAPEX, SALE_COST)
    pct("pfExit", exitc, 2); pct("pfSaleCost", SALE_COST, 0); put("pfValueDcf", v_dcf / 1e6, "{:.1f}")
    pct("pfDcfGap", v_dcf / value - 1, 1); pct("pfDcfGapAbs", 1 - v_dcf / value, 1)
    share = ((1 + G) / (1 + RREQ)) ** T
    pct("pfSaleShare", share, 1)
    # IRR when paying the Gordon value but selling at the higher exit cap with costs
    t = np.arange(1, T + 1)
    cf = noi * (1 + G) ** (t - 1) * (1 - CAPEX)
    sale = noi * (1 + G) ** T / exitc * (1 - SALE_COST)
    flows = np.concatenate([[-value], cf]); flows[-1] += sale
    pct("pfIrr", L.irr(flows), 2)
    flows0 = np.concatenate([[-value], cf]); flows0[-1] += noi * (1 + G) ** T / cap
    pct("pfIrrSame", L.irr(flows0), 2)
    # elasticity of value to the exit cap, and a 100 basis point exit cap shock
    el = L.exit_cap_elasticity(RREQ, G, T, cap, CAPEX)
    assert abs(el + share) < 1e-12
    put("pfElast", el, "{:.3f}")
    # a rise of one tenth in the exit cap rate: first order in log value (the convention of Answer 10.2),
    # that log change as a change in value, and exact
    pct("pfCapTenth", 1.1 * cap, 2); pct("pfTenthLog", -el * math.log(1.1), 1)
    pct("pfTenthLogVal", 1 - math.exp(el * math.log(1.1)), 1)
    pct("pfTenthExact", 1 - L.dcf_value(noi, G, RREQ, T, 1.1 * cap, CAPEX) / value, 1)
    v_up = L.dcf_value(noi, G, RREQ, T, cap + 0.01, CAPEX)
    pct("pfUpHundred", v_up / value - 1, 1); pct("pfUpHundredAbs", 1 - v_up / value, 1)
    # same shock to the going-in cap through the discount rate: value falls far more
    pct("pfCapUp", cap / (cap + 0.01) - 1, 1); pct("pfCapUpAbs", 1 - cap / (cap + 0.01), 1)
    # figure: value against exit cap for three holding periods, and the share of value in the sale
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    xs = np.linspace(cap - 0.01, cap + 0.02, 61)
    for TT, col in ((5, SKY), (10, BLUE), (20, RED)):
        ax[0].plot(100 * xs, [L.dcf_value(noi, G, RREQ, TT, x, CAPEX) / 1e6 for x in xs], color=col, label=f"{TT} years")
    ax[0].axvline(100 * cap, color=GREY, lw=0.8, ls=":")
    ax[0].set_xlabel("exit cap rate, %"); ax[0].set_ylabel("value, $ million")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    TT = np.arange(1, 41)
    for g, col in ((0.0, SKY), (G, BLUE), (0.04, RED)):
        ax[1].plot(TT, 100 * ((1 + g) / (1 + RREQ)) ** TT, color=col, label=f"g = {100 * g:.1f}%")
    ax[1].set_xlabel("holding period, years"); ax[1].set_ylabel("share of value in the sale, %")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig.tight_layout(); R.save(fig, "fig01_exitcap"); plt.close(fig)
    return finish("proforma")


if __name__ == "__main__":
    print(main()["N"])
