"""Section 11.3: the arithmetic of a buyout. When leverage raises the equity multiple, a five year
model with a cash sweep for a stylised company, the bridge of the equity gain into EBITDA growth,
the multiple and debt repayment, the IRR across entry and exit multiples, and a simulation of equity
returns with and without debt. Writes fig07_leverage.png, fig08_bridge.png, fig09_grid.png and
fig10_lbo_mc.png."""
import math
import numpy as np
import numbers_ch11 as R
from numbers_ch11 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, put, pct, finish, rng_for

BASE = dict(rev0=250.0, years=5, growth=[0.06, 0.06, 0.05, 0.05, 0.04], margin=[0.205, 0.21, 0.215, 0.22, 0.22],
            da=0.03, capex=0.035, nwc=0.10, tax=0.25, rate=0.08, sweep=1.0)
M0, MT, LEV, FEES = 10.0, 10.5, 5.0, 0.02      # entry and exit EV/EBITDA, debt/EBITDA, fees as share of EV
CAP = 0.30                                      # deductible net interest as a share of EBITDA


def deal(p=None, m0=M0, mT=MT, lev=LEV, fees=FEES):
    p = dict(BASE if p is None else p)
    e0 = p["rev0"] * 0.20; ev0 = m0 * e0; d0 = lev * e0; eq0 = ev0 - d0 + fees * ev0
    p["debt0"] = d0
    rows = L.lbo_schedule(p)
    last = rows[-1]; evT = mT * last["ebitda"]; ndT = last["debt"] - last["cash"]
    eqT = max(evT - ndT, 0.0)
    return dict(rows=rows, e0=e0, ev0=ev0, d0=d0, eq0=eq0, eT=last["ebitda"], evT=evT, ndT=ndT, eqT=eqT,
                moic=eqT / eq0, irr=(eqT / eq0) ** (1 / p["years"]) - 1 if eqT > 0 else -1.0)


def leverage_figure():
    i, tau, T = 0.08, 0.25, 5
    k = (1 + i * (1 - tau)) ** T; put("lxK", k, "{:.3f}"); pct("lxI", i, 0); pct("lxTau", tau, 0); put("lxT", T, "{:.0f}")
    lam = np.linspace(0, 0.8, 81)
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    for U, col, ls in [(2.0, BLUE, "-"), (1.6, SKY, "--"), (k, GREY, ":"), (1.1, RED, "-.")]:
        lab = rf"$U={U:.2f}$" if abs(U - k) > 1e-9 else rf"$U=\kappa={k:.3f}$"
        ax.plot(100 * lam, L.lbo_moic(U, lam, i, tau, T), color=col, ls=ls, label=lab)
    ax.set_xlabel(r"debt as a share of entry value, $\lambda$ (%)"); ax.set_ylabel("equity multiple")
    ax.legend(loc="upper left"); R.save(fig, "fig07_leverage")
    for U, nm in [(2.0, "Hi"), (1.1, "Lo")]:
        put(f"lxU{nm}", U, "{:.1f}"); put(f"lxM{nm}", L.lbo_moic(U, 0.6, i, tau, T), "{:.2f}")
    pct("lxLamEx", 0.6, 0)


def base_case():
    d = deal(); rows = d["rows"]
    put("lxRev", BASE["rev0"], "{:.0f}"); put("lxEz", d["e0"], "{:.0f}"); put("lxMz", M0, "{:.1f}"); put("lxMt", MT, "{:.1f}")
    put("lxLev", LEV, "{:.1f}"); pct("lxFees", FEES, 0); put("lxEvz", d["ev0"], "{:.0f}"); put("lxDz", d["d0"], "{:.0f}")
    put("lxEqz", d["eq0"], "{:.0f}"); pct("lxRate", BASE["rate"], 0); pct("lxDa", BASE["da"], 0); pct("lxCapex", BASE["capex"], 1)
    pct("lxNwc", BASE["nwc"], 0); pct("lxLamBase", d["d0"] / d["ev0"], 0)
    put("lxEt", d["eT"], "{:.1f}"); put("lxEvt", d["evT"], "{:.1f}"); put("lxNdt", d["ndT"], "{:.1f}"); put("lxEqt", d["eqT"], "{:.1f}")
    put("lxMoic", d["moic"], "{:.2f}"); pct("lxIrr", d["irr"], 1)
    for k_ in ("rev", "ebitda", "interest", "tax", "capex", "dnwc", "fcf", "debt"):
        for r in rows:
            put(f"lxT{k_.capitalize()}{['One', 'Two', 'Three', 'Four', 'Five'][r['year'] - 1]}", r[k_], "{:.1f}")
    put("lxLevExit", d["ndT"] / d["eT"], "{:.2f}")
    # unlevered: same deal with no debt, fees paid by equity, cash kept
    u = deal(lev=0.0); put("lxMoicU", u["moic"], "{:.2f}"); pct("lxIrrU", u["irr"], 1)
    # the asset multiple U and the debt factor of the proposition, from the unlevered run
    U = u["eqT"] / u["eq0"]; R.RAW["lxUbase"] = U
    lam = d["d0"] / u["eq0"]; k = (1 + BASE["rate"] * (1 - BASE["tax"])) ** BASE["years"]
    assert abs(L.lbo_moic(U, lam, BASE["rate"], BASE["tax"], BASE["years"]) - d["moic"]) < 1e-9   # the proposition, exactly
    put("lxUb", U, "{:.2f}"); pct("lxLamB", lam, 1)
    # bridge
    g, m, dl = L.shapley_bridge(d["e0"], M0, d["eT"], MT, d["d0"], d["ndT"])
    fee = FEES * d["ev0"]
    put("lxBrG", g, "{:.1f}"); put("lxBrM", m, "{:.1f}"); put("lxBrD", dl, "{:.1f}"); put("lxBrF", fee, "{:.1f}")
    gain = d["eqT"] - d["eq0"]; put("lxGain", gain, "{:.1f}")
    assert abs(g + m + dl - fee - gain) < 1e-9
    pct("lxBrGs", g / gain, 0); pct("lxBrMs", m / gain, 0); pct("lxBrDs", dl / gain, 0)
    # the order dependence that the Shapley split removes: growth first or multiple first
    put("lxGfirst", M0 * (d["eT"] - d["e0"]), "{:.1f}"); put("lxMfirst", MT * (d["eT"] - d["e0"]), "{:.1f}")
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    labels = ["equity in", "EBITDA growth", "multiple", "debt repaid", "fees", "equity out"]
    vals = [d["eq0"], g, m, dl, -fee, d["eqT"]]
    base = 0.0
    for j, (lab, v) in enumerate(zip(labels, vals)):
        if j in (0, len(vals) - 1):
            ax.bar(j, v, color=BLUE, width=0.6); base = v if j == 0 else base
        else:
            ax.bar(j, v, bottom=base if v >= 0 else base + v, color=SKY if v >= 0 else RED, width=0.6); base += v
        ax.text(j, (v if j in (0, len(vals) - 1) else base if v >= 0 else base - v) + 6, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("value (millions)")
    R.save(fig, "fig08_bridge")
    return d


def interest_cap(base):
    """The base deal when deductible net interest is capped at 30% of EBITDA, the fixed ratio in force in
    Canada for taxation years from 2024, with the excess carried forward."""
    p = dict(BASE); p["int_cap"] = CAP
    c = deal(p); rows = c["rows"]; b = base["rows"]
    pct("lxCap", CAP, 0)
    put("lxCapIntOne", rows[0]["interest"], "{:.1f}"); put("lxCapDedOne", rows[0]["deduct"], "{:.1f}")
    pct("lxCapIntShareOne", rows[0]["interest"] / rows[0]["ebitda"], 0)
    binds = sum(r["deduct"] < r["interest"] - 1e-9 for r in rows)
    put("lxCapBinds", binds, "{:.0f}")
    put("lxCapCarryPeak", max(r["carry"] for r in rows), "{:.1f}")
    early = sum(r["tax"] - q["tax"] for r, q in zip(rows[:binds], b[:binds]))
    extra = sum(r["tax"] for r in rows) - sum(r["tax"] for r in b)
    put("lxCapTaxEarly", early, "{:.1f}"); put("lxCapTaxExtra", extra, "{:.2f}")
    put("lxCapMoic", c["moic"], "{:.3f}"); put("lxMoicThree", base["moic"], "{:.3f}")
    put("lxCapIrrDropBp", 1e4 * (base["irr"] - c["irr"]), "{:.0f}")
    assert 0 < 1e4 * (base["irr"] - c["irr"]) < 2          # the text: lower by less than two basis points
    # the carryforward is used up before exit, so the cap moves tax earlier and costs only the time value
    assert binds >= 1 and early > 0 and rows[-1]["carry"] < 1e-9 and abs(extra) < 0.1 * early and c["moic"] < base["moic"]
    # without a carryforward the excess interest is lost for good
    p2 = dict(p); p2["carry_forward"] = False; n = deal(p2)
    lost = sum(r["tax"] for r in n["rows"]) - sum(r["tax"] for r in b)
    put("lxCapNoCfTax", lost, "{:.1f}"); put("lxCapNoCfMoic", n["moic"], "{:.3f}")
    put("lxCapNoCfIrrDropBp", 1e4 * (base["irr"] - n["irr"]), "{:.0f}")
    assert lost > 0 and n["moic"] < c["moic"]
    # a heavier deal: seven times EBITDA at 9%
    p3 = dict(p); p3["rate"] = 0.09; h = deal(p3, lev=7.0); p4 = dict(BASE); p4["rate"] = 0.09; h0 = deal(p4, lev=7.0)
    put("lxCapHvBinds", sum(r["deduct"] < r["interest"] - 1e-9 for r in h["rows"]), "{:.0f}")
    put("lxCapHvCarry", h["rows"][-1]["carry"], "{:.1f}")
    put("lxCapHvIrrDropBp", 1e4 * (h0["irr"] - h["irr"]), "{:.0f}"); pct("lxCapHvIrr", h["irr"], 1); pct("lxCapHvIrrNo", h0["irr"], 1)
    R.RAW["lxCapHvCarryRaw"] = h["rows"][-1]["carry"]
    # the debt at entry, in turns of entry EBITDA, at which year one interest just meets the cap
    put("lxCapLevFree", CAP * rows[0]["ebitda"] / (BASE["rate"] * base["e0"]), "{:.1f}")


def grid():
    ms = np.arange(8.0, 12.01, 0.5); Z = np.zeros((len(ms), len(ms)))
    for a, m0 in enumerate(ms):
        for b, mT in enumerate(ms):
            Z[a, b] = deal(m0=m0, mT=mT)["irr"]
    pct("lxGridLoLo", Z[0, 0], 1); pct("lxGridHiHi", Z[-1, -1], 1); pct("lxGridHiLo", Z[-1, 0], 1); pct("lxGridLoHi", Z[0, -1], 1)
    put("lxGridMin", ms[0], "{:.0f}"); put("lxGridMax", ms[-1], "{:.0f}")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    im = ax.imshow(100 * Z, origin="lower", cmap="Blues", vmin=0, vmax=40, aspect="auto",
                   extent=[ms[0] - 0.25, ms[-1] + 0.25, ms[0] - 0.25, ms[-1] + 0.25])
    for a, m0 in enumerate(ms):
        for b, mT in enumerate(ms):
            ax.text(mT, m0, f"{100 * Z[a, b]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if 100 * Z[a, b] > 24 else "black")
    ax.set_xlabel("exit EV/EBITDA"); ax.set_ylabel("entry EV/EBITDA"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax); cb.set_label("equity IRR (%)")
    R.save(fig, "fig09_grid")


def monte_carlo(rng, n=20_000):
    res = {0.0: [], LEV: []}
    shocks = []
    for _ in range(n):
        p = dict(BASE); z = rng.standard_normal(5)
        p["growth"] = list(0.05 + 0.06 * z); p["margin"] = list(np.clip(0.21 + 0.015 * rng.standard_normal(5) + 0.01 * np.cumsum(z) / 3, 0.05, 0.4))
        mT = M0 * math.exp(0.18 * rng.standard_normal() - 0.5 * 0.18 ** 2)
        for lev in res:
            res[lev].append(deal(p, mT=mT, lev=lev)["moic"])
    u = np.array(res[0.0]); v = np.array(res[LEV])
    iu = u ** (1 / 5) - 1; iv = np.where(v > 0, v ** (1 / 5) - 1, -1.0)
    put("lxMcN", n, "{:,.0f}")
    pct("lxMcMedU", np.median(iu), 1); pct("lxMcMedL", np.median(iv), 1)
    pct("lxMcSdU", iu.std(), 1); pct("lxMcSdL", np.std(iv[v > 0]), 1)
    pct("lxMcLossU", np.mean(u < 1), 1); pct("lxMcLossL", np.mean(v < 1), 1); pct("lxMcWipe", np.mean(v <= 0), 2)
    pct("lxMcQU", np.percentile(iu, 5), 1); pct("lxMcQL", np.percentile(iv, 5), 1)
    # the proposition's test, path by path: leverage raises the multiple exactly when U beats the debt factor
    k = (1 + BASE["rate"] * (1 - BASE["tax"])) ** 5
    pct("lxMcBeat", np.mean(v > u), 1); pct("lxMcUAbove", np.mean(u > k), 1)
    R.RAW["lxMcMismatch"] = int(np.sum((v > u) != (u > k)))
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    bins = np.linspace(-40, 60, 81)
    ax.hist(100 * iu, bins=bins, color=PALE, edgecolor=SKY, lw=0.4, label="no debt")
    ax.hist(100 * np.clip(iv, -0.4, None), bins=bins, histtype="step", color=RED, lw=1.1, label=f"debt of {LEV:.0f} times EBITDA")
    ax.set_xlabel("equity IRR over five years (%)"); ax.set_ylabel("simulated deals"); ax.legend(loc="upper left")
    R.save(fig, "fig10_lbo_mc")
    R.RAW["lxMcU"] = u.tolist()[:0]


def main():
    leverage_figure(); d = base_case(); interest_cap(d); grid(); monte_carlo(rng_for(4))
    return finish("lbo")


if __name__ == "__main__":
    main()
