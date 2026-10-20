"""Section 6.7: the screener's metrics and grade for its three example loans, the identities that tie
the metrics together, and expected losses of senior and mezzanine slices under a lognormal value.

Run on its own:  python code/ch06/private_credit.py   (writes code/ch06/out/private_credit.json)
"""
from __future__ import annotations
import json, math
import numpy as np
from numbers_ch06 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, HERE  # noqa: F401

# The three example deals shipped with ALENZACODES/private-credit-screener (app.js), all made up.
DEALS = {
    "Cal": dict(loan=6e6, value=10e6, noi=700000, rate=9.5, amort=0, termYrs=2, capRate=7, exitCapShift=0.25,
                noiGrowth=2, fees=1.5, sponsorScore=4),
    "Tor": dict(loan=14e6, value=18.5e6, noi=900000, rate=11.5, amort=0, termYrs=1, capRate=5, exitCapShift=0.75,
                noiGrowth=0, fees=2, sponsorScore=3),
    "Ott": dict(loan=4.2e6, value=6e6, noi=430000, rate=10.25, amort=25, termYrs=3, capRate=6.75, exitCapShift=0.5,
                noiGrowth=1, fees=1.75, sponsorScore=3),
}
V0, T, G, COST, RATE, AMORT = 100.0, 5, 0.02, 0.10, 6.0, 25


def screener():
    out = {}
    for k, d in DEALS.items():
        total, grade, m, s = L.screener_score(d)
        out[k] = dict(total=total, grade=grade, **m)
        put(f"pc{k}Score", total, "{:.0f}"); N[f"pc{k}Grade"] = grade
        put(f"pc{k}Ltv", m["ltv"], "{:.1f}"); put(f"pc{k}Dscr", m["dscr"]); put(f"pc{k}Dy", m["dy"], "{:.2f}")
        put(f"pc{k}ExitLtv", m["exitLtv"], "{:.1f}")
        put(f"pc{k}Cap", d["noi"] / d["value"] * 100, "{:.2f}")
        k_const = m["ds"] / d["loan"]
        RAW[f"pc{k}IdOne"] = m["ltv"] * m["dy"] / 100 - d["noi"] / d["value"] * 100
        RAW[f"pc{k}IdTwo"] = m["dscr"] - m["dy"] / 100 / k_const
        put(f"pc{k}Const", 100 * k_const, "{:.2f}")
    (HERE / "out" / "screener_py.json").write_text(json.dumps(out, indent=1))


def slices():
    s_face = lambda ltv: L.balloon(ltv * V0, RATE, AMORT, 12 * T)
    put("slT", T, "{:.0f}"); pct("slG", G, 0); pct("slCost", COST, 0); put("slRate", RATE, "{:.0f}")
    put("slAmort", AMORT, "{:.0f}")
    for sig, nm in [(0.10, "Ten"), (0.15, "Fifteen"), (0.20, "Twenty")]:
        Bs, Bm = s_face(0.60), s_face(0.75)
        els = L.slice_el(0, Bs, V0, G, sig, T, COST); elm = L.slice_el(Bs, Bm, V0, G, sig, T, COST)
        put(f"slSen{nm}", 100 * els / Bs, "{:.3f}"); put(f"slMez{nm}", 100 * elm / (Bm - Bs), "{:.2f}")
        m = (1 - COST) * V0 * (1 + G) ** T; s = sig * math.sqrt(T)
        pd_s = L.ncdf((math.log(Bs / m) + 0.5 * s * s) / s); pd_m = L.ncdf((math.log(Bm / m) + 0.5 * s * s) / s)
        pct(f"slPdSen{nm}", pd_s, 2); pct(f"slPdMez{nm}", pd_m, 1)
    put("slBs", s_face(0.60)); put("slBm", s_face(0.75))
    ratio = lambda lo, hi, a, b: (L.slice_el(lo, hi, V0, G, b, T, COST) / L.slice_el(lo, hi, V0, G, a, T, COST))
    put("slSenRatio", ratio(0, s_face(0.60), 0.10, 0.20), "{:.0f}")
    put("slMezRatio", ratio(s_face(0.60), s_face(0.75), 0.10, 0.20), "{:.1f}")
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    ltvs = np.linspace(0.40, 0.90, 101)
    for sig, col, ls in [(0.10, BLUE, "-"), (0.15, RED, "--"), (0.20, GREY, ":")]:
        sen = [100 * L.slice_el(0, s_face(x), V0, G, sig, T, COST) / s_face(x) for x in ltvs]
        axes[0].plot(100 * ltvs, sen, color=col, ls=ls, label=f"$\\sigma_V={int(100*sig)}\\%$")
        mez = [100 * L.slice_el(s_face(x - 0.15), s_face(x), V0, G, sig, T, COST) / (s_face(x) - s_face(x - 0.15))
               for x in ltvs]
        axes[1].plot(100 * ltvs, mez, color=col, ls=ls)
    axes[0].set_title("whole loan from 0"); axes[1].set_title("15 point slice below the given LTV")
    for ax in axes:
        ax.set_xlabel("going-in LTV of the top of the claim (%)")
    axes[0].set_ylabel("five year expected loss (% of face)"); axes[0].legend(loc="upper left")
    save(fig, "fig10_slice_el")


def main():
    screener(); slices()
    return finish("private_credit")


if __name__ == "__main__":
    main()
