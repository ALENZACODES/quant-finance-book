"""Section 11.4: valuing a business. The normalised earnings of a stylised private manufacturer, a
cost of capital built from the September 2026 Treasury yield, an implied equity risk premium and an
industry beta unlevered two ways, a five year DCF with a value driver terminal value, the same value
by adjusted present value, the sensitivity of value to each input, and the value driver multiple set
against industry multiples. Writes fig11_multiples.png, fig12_erp.png and fig13_tornado.png.
Also writes fig19_driver.png, fig20_wacc.png, the rows of the DCF table and out/multiple_medians.csv for
the R listing."""
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.ticker as mticker
import numbers_ch11 as R
from numbers_ch11 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, put, pct, finish

TAX = 0.25                 # marginal tax rate used to unlever and relever, and in the forecast
TARGET_DV = 0.30           # target debt to value of the company, rebalanced each year
SPREAD = 0.0175            # spread of the company's debt over the Treasury yield
G_TERM, ROC_TERM = 0.025, 0.12
GROWTH = [0.05, 0.05, 0.04, 0.04, 0.03]
REV0, DA, CAPEX, NWC = 120.0, 0.03, 0.035, 0.12
NET_DEBT = 25.0
# normalisation of the reported year (millions): label, amount added to reported EBITDA
REPORTED = 18.4
ADJ = [("owner's pay above a market salary", 1.2), ("one-off legal settlement", 0.9),
       ("rent paid to the owner below market", -0.5), ("inventory write-down not expected to recur", 0.6),
       ("gain on the sale of equipment", -0.4)]


def normalise():
    put("nmRep", REPORTED, "{:.1f}")
    for (lab, a), nm in zip(ADJ, ["Pay", "Legal", "Rent", "Stock", "Gain"]):
        put(f"nm{nm}", a, "{:.1f}")
    e = REPORTED + sum(a for _, a in ADJ); put("nmNorm", e, "{:.1f}"); put("nmRev", REV0, "{:.0f}")
    pct("nmMargin", e / REV0, 1); pct("nmChange", e / REPORTED - 1, 0)
    return e / REV0


def build_wacc():
    m = pd.read_csv(ROOT / "data" / "real_assets_monthly.csv", parse_dates=["date"])
    rf = float(m.loc[m.date == "2026-09-01", "gs10"].iloc[0]) / 100
    erp_t = pd.read_csv(ROOT / "data" / "damodaran_erp.csv")
    erp = float(erp_t.loc[erp_t.year == 2025, "erp_fcfe"].iloc[0])
    ind = pd.read_csv(ROOT / "data" / "damodaran_industry.csv").set_index("industry")
    row = ind.loc["Machinery"]
    bl = float(row.beta); de_ind = float(row.debt_weight / row.equity_weight)
    pct("wkRf", rf, 2); pct("wkErp", erp, 2); put("wkBetaL", bl, "{:.4f}"); pct("wkDvInd", row.debt_weight, 1)
    put("wkFirms", row.n_firms, "{:.0f}"); pct("wkTax", TAX, 0); pct("wkTarget", TARGET_DV, 0); pct("wkSpread", SPREAD, 2)
    rd = rf + SPREAD; pct("wkRd", rd, 2)
    de_t = TARGET_DV / (1 - TARGET_DV)
    out = {}
    for rule, nm in [("hamada", "Ham"), ("harris_pringle", "Hp")]:
        bu = L.unlever_beta(bl, de_ind, TAX, rule=rule); br = L.relever_beta(bu, de_t, TAX, rule=rule)
        re_ = rf + br * erp; w = (1 - TARGET_DV) * re_ + TARGET_DV * rd * (1 - TAX)
        put(f"wkBu{nm}", bu, "{:.3f}"); put(f"wkBr{nm}", br, "{:.3f}"); pct(f"wkRe{nm}", re_, 2); pct(f"wkW{nm}", w, 2)
        out[rule] = dict(bu=bu, re=re_, w=w)
    # the cost of capital used below: unlevered by Harris and Pringle (debt rebalanced), discounted with
    # the Miles and Ezzell WACC for annual rebalancing
    rU = rf + out["harris_pringle"]["bu"] * erp; pct("wkRu", rU, 2)
    W = L.miles_ezzell_wacc(rU, rd, TAX, TARGET_DV); pct("wkWme", W, 2)
    # the textbook formula with the Miles and Ezzell cost of equity gives the same number
    re_me = L.lever_cost_equity(rU, rd, de_t, TAX, "miles_ezzell")
    w_text = (1 - TARGET_DV) * re_me + TARGET_DV * rd * (1 - TAX)
    assert abs(w_text - W) < 1e-12
    pct("wkReMe", re_me, 2)
    # equity risk premium history for the figure
    put("wkErpMean", 100 * erp_t.erp_fcfe.mean(), "{:.2f}")
    lo = erp_t.loc[erp_t.erp_fcfe.idxmin()]; hi = erp_t.loc[erp_t.erp_fcfe.idxmax()]
    pct("wkErpMin", lo.erp_fcfe, 2); put("wkErpMinYear", lo.year, "{:.0f}"); pct("wkErpMax", hi.erp_fcfe, 2); put("wkErpMaxYear", hi.year, "{:.0f}")
    pct("wkTbond", float(erp_t.loc[erp_t.year == 2025, "tbond"].iloc[0]), 2)
    put("wkErpFirst", erp_t.year[erp_t.erp_fcfe.notna()].min(), "{:.0f}")
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(erp_t.year, 100 * erp_t.erp_fcfe, color=BLUE, label="implied equity risk premium")
    ax.plot(erp_t.year, 100 * erp_t.tbond, color=RED, ls="--", label="ten year Treasury yield")
    ax.set_xlabel("end of year"); ax.set_ylabel("percent"); ax.legend(loc="upper right")
    R.save(fig, "fig12_erp")
    return dict(rf=rf, erp=erp, rd=rd, rU=rU, W=W, ind=ind, wHam=out["hamada"]["w"], wHp=out["harris_pringle"]["w"])


def forecast(margin, growth=GROWTH, capex=CAPEX, da=DA, nwc=NWC):
    rev = REV0; fcf = []; nopat = []
    for g in growth:
        prev = rev; rev *= 1 + g
        ebit = rev * (margin - da); n = ebit * (1 - TAX)
        fcf.append(n + da * rev - capex * rev - nwc * (rev - prev)); nopat.append(n)
    nopat_next = nopat[-1] * (1 + G_TERM)
    return np.array(fcf), nopat_next


def schedule(margin, W):
    """The rows of the five year DCF: revenue, EBITDA, depreciation, EBIT, NOPAT, capital spending, change
    in working capital, free cash flow, discount factor and present value."""
    rev = REV0; rows = []
    for t, g in enumerate(GROWTH, 1):
        prev = rev; rev *= 1 + g
        ebitda = margin * rev; da = DA * rev; ebit = ebitda - da; nopat = ebit * (1 - TAX)
        capex = CAPEX * rev; dwc = NWC * (rev - prev); fcf = nopat + da - capex - dwc
        df = (1 + W) ** -t
        rows.append(dict(Rev=rev, Ebitda=ebitda, Da=da, Ebit=ebit, Nopat=nopat, Capex=capex, Dwc=dwc,
                         Fcf=fcf, Df=df, Pv=fcf * df))
    return rows


def company(margin, c):
    W = c["W"]; fcf, nn = forecast(margin)
    rows = schedule(margin, W)
    assert max(abs(r["Fcf"] - f) for r, f in zip(rows, fcf)) < 1e-12
    for r, yr in zip(rows, ["One", "Two", "Three", "Four", "Five"]):
        for key, v in r.items():
            put(f"vd{key}{yr}", v, "{:.3f}" if key == "Df" else "{:.2f}")
    put("vdPvSum", sum(r["Pv"] for r in rows), "{:.2f}")
    ev, tv = L.dcf_value(fcf, W, G_TERM, nopat_next=nn, roc=ROC_TERM)
    put("vlEv", ev, "{:.1f}"); put("vlTv", tv, "{:.1f}"); pct("vlTvShare", tv * (1 + W) ** -5 / ev, 0)
    put("vdPvTv", tv * (1 + W) ** -5, "{:.2f}")
    put("vlEq", ev - NET_DEBT, "{:.1f}"); put("vlNetDebt", NET_DEBT, "{:.0f}"); pct("vlG", G_TERM, 1); pct("vlRoc", ROC_TERM, 0)
    for j, f in enumerate(fcf):
        put(f"vlFcf{['One', 'Two', 'Three', 'Four', 'Five'][j]}", f, "{:.2f}")
    put("vlNopatSix", nn, "{:.2f}"); pct("vlReinv", G_TERM / ROC_TERM, 1)
    # adjusted present value with the same financing policy gives the same number
    tv_fcf = nn * (1 - G_TERM / ROC_TERM)
    apv = L.apv_rebalanced(fcf, tv_fcf, G_TERM, c["rU"], c["rd"], TAX, TARGET_DV)
    R.RAW["vlApvGap"] = abs(apv - ev); assert R.RAW["vlApvGap"] < 1e-9 * ev
    # the unlevered value and the value of the tax shields
    vu, _ = L.dcf_value(fcf, c["rU"], G_TERM, nopat_next=nn, roc=ROC_TERM)
    put("vlVu", vu, "{:.1f}"); put("vlShield", ev - vu, "{:.1f}"); pct("vlShieldShare", (ev - vu) / ev, 1)
    # the naive constant debt shortcut tau D on the same debt overstates the shield
    put("vlTauD", TAX * TARGET_DV * ev, "{:.1f}")
    # value driver multiple and the enterprise value over EBITDA it implies
    e0 = margin * REV0
    put("vlEvEbitda", ev / e0, "{:.2f}")
    put("vlTvNopat", L.ev_nopat_multiple(W, G_TERM, ROC_TERM), "{:.2f}")
    put("vlTvNopatHiRoc", L.ev_nopat_multiple(W, G_TERM, 1e9), "{:.2f}"); put("vlTvNopatAtW", 1 / W, "{:.2f}")
    # market approach with the industry multiple
    mult = float(c["ind"].loc["Machinery", "ev_ebitda_pos"]); put("vlIndMult", mult, "{:.2f}")
    put("vlMktEv", mult * e0, "{:.1f}"); pct("vlMktGap", mult * e0 / ev - 1, 0)
    # capitalised earnings: the one stage version with the year one cash flow
    f1 = fcf[0]; put("vlCapEarn", f1 / (W - G_TERM), "{:.1f}")
    return ev


def tornado(margin, c):
    W = c["W"]; base = company_quiet(margin, W)
    cases = [("cost of capital, 1 point", dict(W=W - 0.01), dict(W=W + 0.01)),
             ("terminal growth, 0.5 point", dict(g=G_TERM + 0.005), dict(g=G_TERM - 0.005)),
             ("return on new capital, 9% to 15%", dict(roc=0.15), dict(roc=0.09)),
             ("EBITDA margin, 1 point", dict(margin=margin + 0.01), dict(margin=margin - 0.01)),
             ("growth in years 1 to 5, 2 points", dict(growth=[g + 0.02 for g in GROWTH]), dict(growth=[g - 0.02 for g in GROWTH])),
             ("capital spending, 0.5 point of revenue", dict(capex=CAPEX - 0.005), dict(capex=CAPEX + 0.005))]
    rows = []
    for lab, up, dn in cases:
        vu = company_quiet(**{**dict(margin=margin, W=W), **up}) - base
        vd = company_quiet(**{**dict(margin=margin, W=W), **dn}) - base
        rows.append((lab, vu, vd))
    rows.sort(key=lambda r: abs(r[1] - r[2]))
    names = ["Wacc", "Growth", "Roc", "Margin", "Near", "Capex"]
    for (lab, up, dn), nm in zip(cases, names):
        r_ = [x for x in rows if x[0] == lab][0]
        nd = 1 if nm == "Growth" else 0      # terminal growth: one decimal, so the asymmetry shows
        pct(f"vlTo{nm}Up", r_[1] / base, nd); pct(f"vlTo{nm}Dn", r_[2] / base, nd)
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    y = np.arange(len(rows))
    ax.barh(y, [100 * r[1] / base for r in rows], color=BLUE, height=0.6, label="favourable")
    ax.barh(y, [100 * r[2] / base for r in rows], color=RED, height=0.6, label="unfavourable")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5); ax.axvline(0, color="#333333", lw=0.6)
    ax.set_xlabel("change in enterprise value (%)"); ax.legend(loc="lower right", fontsize=7.5)
    R.save(fig, "fig13_tornado")


def company_quiet(margin, W, g=G_TERM, roc=ROC_TERM, growth=GROWTH, capex=CAPEX):
    fcf, nn = forecast(margin, growth=growth, capex=capex)
    nn = nn / (1 + G_TERM) * (1 + g)
    return L.dcf_value(fcf, W, g, nopat_next=nn, roc=roc)[0]


def two_stage(W, roc, g1, n=5):
    """Value over this year's NOPAT with growth g1 (held between 0 and 20%) for n years and G_TERM after,
    reinvesting g/roc of NOPAT throughout."""
    g1 = min(max(g1, 0.0), 0.20); v = 0.0; x = 1.0
    for t in range(1, n + 1):
        x *= 1 + g1; v += x * (1 - g1 / roc) / (1 + W) ** t
    return v + x * (1 + G_TERM) * L.ev_nopat_multiple(W, G_TERM, roc) / (1 + W) ** n


def multiples(c):
    d = c["ind"].reset_index()
    d = d[(d.industry != "Total Market (without financials)") & (d.industry != "Total Market")]
    ok = d.roc.notna() & (d.roc > 0.04) & (d.wacc > G_TERM + 0.01) & d.ev_ebit1t_pos.notna() & (d.ev_ebit1t_pos > 0)
    d = d[ok].copy()
    d["implied"] = (1 + G_TERM) * L.ev_nopat_multiple(d.wacc, G_TERM, d.roc)
    rho = stats.spearmanr(d.implied, d.ev_ebit1t_pos)[0]
    put("mkN", len(d), "{:.0f}"); put("mkRho", rho, "{:.2f}")
    ratio = d.ev_ebit1t_pos / d.implied
    put("mkRatioMed", ratio.median(), "{:.2f}"); pct("mkAbove", (ratio > 1).mean(), 0)
    put("mkRhoW", stats.spearmanr(d.wacc, d.ev_ebit1t_pos)[0], "{:.2f}")
    put("mkRhoG", stats.spearmanr(d.growth, d.ev_ebit1t_pos)[0], "{:.2f}")
    two = np.array([two_stage(w, r_, g1) for w, r_, g1 in zip(d.wacc, d.roc, d.growth)])
    put("mkRhoTwo", stats.spearmanr(two, d.ev_ebit1t_pos)[0], "{:.2f}")
    put("mkImpMed", d.implied.median(), "{:.1f}"); put("mkObsMed", d.ev_ebit1t_pos.median(), "{:.1f}")
    # the growth that would reconcile the median industry: solve the value driver for g
    from scipy.optimize import brentq
    w_m, roc_m, obs = d.wacc.median(), d.roc.median(), d.ev_ebit1t_pos.median()
    g_star = brentq(lambda g: (1 + g) * L.ev_nopat_multiple(w_m, g, roc_m) - obs, -0.05, w_m - 1e-4)
    pct("mkGstar", g_star, 2); pct("mkWmed", w_m, 2); pct("mkRocMed", roc_m, 1)
    (R.HERE / "out" / "multiple_medians.csv").write_text(
        f"wacc,roc,multiple\n{float(w_m)!r},{float(roc_m)!r},{float(obs)!r}\n")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(d.implied, d.ev_ebit1t_pos, s=9, color=SKY, edgecolor=BLUE, lw=0.3)
    lim = [10, 70]
    assert d.implied.between(*lim).all() and d.ev_ebit1t_pos.between(*lim).all()
    ax.plot(lim, lim, color=RED, lw=0.9, ls="--", label="observed = value driver")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(mticker.FixedLocator([10, 20, 30, 50, 70]))
        a.set_major_formatter(mticker.ScalarFormatter()); a.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r"value driver multiple, $g$ = 2.5% (log scale)"); ax.set_ylabel(r"observed EV/EBIT$(1-\tau)$ (log scale)")
    ax.legend(loc="lower right", frameon=True, framealpha=1, edgecolor="none").set_zorder(5)
    R.save(fig, "fig11_multiples")


def driver_figure(W):
    """Terminal value over next year's NOPAT, (1 - g/ROC)/(W - g), against g for four returns on new
    capital, one of them equal to W."""
    g = np.linspace(0.0, 0.06, 121)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for roc, col, ls, lab in ((0.20, RED, "-", "ROC = 20%"), (ROC_TERM, BLUE, "-", "ROC = 12%"),
                              (W, GREY, ":", f"ROC = W = {100 * W:.2f}%"), (0.06, SKY, "--", "ROC = 6%")):
        ax.plot(100 * g, L.ev_nopat_multiple(W, g, roc), color=col, ls=ls, label=lab)
    ax.plot([100 * G_TERM], [L.ev_nopat_multiple(W, G_TERM, ROC_TERM)], "o", color=BLUE, ms=4)
    for roc, nm in ((0.20, "Twenty"), (ROC_TERM, "Twelve"), (0.06, "Six")):
        put(f"drFive{nm}", L.ev_nopat_multiple(W, 0.05, roc), "{:.1f}")
    put("drZero", 1 / W, "{:.1f}")
    ax.set_xlabel("terminal growth g (%)"); ax.set_ylabel("terminal value / NOPAT next year")
    ax.legend(loc="upper left", fontsize=7.5)
    R.save(fig, "fig19_driver")


def wacc_figure(c):
    """The cost of capital against target debt to value under the three treatments of the text, each
    from the machinery beta: Hamada and Harris and Pringle with the usual WACC formula, and Miles and
    Ezzell with the Harris and Pringle unlevered cost."""
    ind = c["ind"].loc["Machinery"]; bl = float(ind.beta); de_ind = float(ind.debt_weight / ind.equity_weight)
    rf, erp, rd, rU = c["rf"], c["erp"], c["rd"], c["rU"]
    L_ = np.linspace(0.0, 0.6, 61)

    def usual(rule, l):
        bu = L.unlever_beta(bl, de_ind, TAX, rule=rule); br = L.relever_beta(bu, l / (1 - l), TAX, rule=rule)
        return (1 - l) * (rf + br * erp) + l * rd * (1 - TAX)
    lines = {"Ham": [usual("hamada", l) for l in L_], "Hp": [usual("harris_pringle", l) for l in L_],
             "Me": [L.miles_ezzell_wacc(rU, rd, TAX, l) for l in L_]}
    # at the 30% target these are the three numbers of the text
    j = int(np.argmin(np.abs(L_ - TARGET_DV)))
    R.RAW["wfTargetGap"] = max(abs(lines["Ham"][j] - c["wHam"]), abs(lines["Hp"][j] - c["wHp"]), abs(lines["Me"][j] - c["W"]))
    for nm in lines:
        pct(f"wr{nm}Sixty", lines[nm][-1], 2); pct(f"wr{nm}Zero", lines[nm][0], 2)
    pct("wrSpreadSixty", max(v[-1] for v in lines.values()) - min(v[-1] for v in lines.values()), 2)
    pct("wrSpreadTarget", max(v[j] for v in lines.values()) - min(v[j] for v in lines.values()), 2)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for nm, col, ls, lab in (("Ham", SKY, "--", "Hamada, constant debt"), ("Hp", BLUE, "-", "Harris and Pringle"),
                             ("Me", RED, "-.", "Miles and Ezzell, annual reset")):
        ax.plot(100 * L_, 100 * np.array(lines[nm]), color=col, ls=ls, label=lab)
    ax.axvline(100 * TARGET_DV, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("target debt to value (%)"); ax.set_ylabel("cost of capital (%)"); ax.legend(loc="lower left")
    R.save(fig, "fig20_wacc")


def main():
    margin = normalise(); c = build_wacc(); company(margin, c); tornado(margin, c); multiples(c)
    driver_figure(c["W"]); wacc_figure(c)
    return finish("valuation")


if __name__ == "__main__":
    main()
