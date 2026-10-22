"""Section 10.5: timberland, farmland and infrastructure. The Faustmann rotation for a stand with a
Chapman-Richards growth curve, set against the single rotation and the rotation of greatest average
growth; the value of an infrastructure concession of finite life against a perpetuity, and its duration;
and the real prices of timber, lumber and farm products from the BLS producer price indexes.
Writes fig07_rotation.png and fig08_ppi.png.
Also writes fig16_marginal.png, the marginal rule at 4%."""
import numpy as np
import pandas as pd
from scipy import optimize
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish

PRICE, COST = 40.0, 1500.0   # stumpage price, dollars per cubic metre, and planting cost per hectare
RATES = (0.03, 0.04, 0.06)
CF1, RCON, GCON = 100.0, 0.07, 0.025   # concession: first year cash flow, discount rate, growth


def main():
    put("tbPrice", PRICE, "{:.0f}"); put("tbCost", COST, "{:,.0f}")
    put("tbA", 400, "{:.0f}"); put("tbK", 0.06, "{:.2f}"); put("tbC", 3, "{:.0f}")
    tmai = optimize.minimize_scalar(lambda t: -L.volume(t) / t, bounds=(5, 150), method="bounded", options={"xatol": 1e-10}).x
    put("tbMai", tmai, "{:.1f}"); put("tbVolMai", L.volume(tmai), "{:.0f}")
    for r, tag in zip(RATES, ("Three", "Four", "Six")):
        tf = L.faustmann_rotation(PRICE, r, COST); ts = L.single_rotation(PRICE, r)
        put(f"tbF{tag}", tf, "{:.1f}"); put(f"tbS{tag}", ts, "{:.1f}")
        put(f"tbLev{tag}", L.land_value(tf, PRICE, r, COST), "{:,.0f}")
        # land value at the single rotation, for comparison
        put(f"tbLevS{tag}", L.land_value(ts, PRICE, r, COST), "{:,.0f}")
        pct(f"tbGrowth{tag}", L.dvolume(tf) / L.volume(tf), 2)
    # the price of timber rising at a real rate a acts like a lower discount rate
    # with the planting cost fixed, only the harvests are discounted at r - a
    lev_a = lambda T: PRICE * L.volume(T) * np.exp(-0.03 * T) / (1 - np.exp(-0.03 * T)) - COST / (1 - np.exp(-0.04 * T))
    tf_a = optimize.minimize_scalar(lambda T: -lev_a(T), bounds=(10, 80), method="bounded", options={"xatol": 1e-10}).x
    put("tbFPriceGrowth", tf_a, "{:.1f}")
    put("tbFPriceBoth", L.faustmann_rotation(PRICE, 0.04 - 0.01, COST), "{:.1f}")   # planting cost rising at 1% too
    pct("tbRcross", 1 / tmai, 1)   # the single harvest rotation is shorter than tmai exactly when r > 1/tmai
    # concession against a perpetuity
    perp = CF1 / (RCON - GCON)
    put("tbPerp", perp, "{:,.0f}"); pct("tbRcon", RCON, 0); pct("tbGcon", GCON, 1); put("tbCfOne", CF1, "{:.0f}")
    for n in (30, 50, 99):
        v = L.concession_value(CF1, RCON, GCON, n)
        put(f"tbCon{n}".replace("30", "Thirty").replace("50", "Fifty").replace("99", "NinetyNine"), v, "{:,.0f}")
        pct(f"tbShare{n}".replace("30", "Thirty").replace("50", "Fifty").replace("99", "NinetyNine"), v / perp, 1)
        put(f"tbDur{n}".replace("30", "Thirty").replace("50", "Fifty").replace("99", "NinetyNine"), L.concession_duration(RCON, GCON, n), "{:.1f}")
    put("tbDurPerp", (1 + RCON) / (RCON - GCON), "{:.1f}")
    # a one point rise in the discount rate
    for n, tag in ((30, "Thirty"), (10 ** 4, "Perp")):
        v0 = L.concession_value(CF1, RCON, GCON, n); v1 = L.concession_value(CF1, RCON + 0.01, GCON, n)
        pct(f"tbShock{tag}", v1 / v0 - 1, 1); pct(f"tbShock{tag}Abs", 1 - v1 / v0, 1)
    # real commodity prices
    m = L.load_monthly()
    dec = m[m.index.month == 12]
    real = np.log(dec[["ppi_farm", "ppi_timber", "ppi_lumber"]].div(dec["cpi"], axis=0))
    first = {c: real[c].dropna().index[0].year for c in real}
    R.N["tbFirstTimber"] = str(first["ppi_timber"]); R.N["tbFirstFarm"] = str(first["ppi_farm"])
    yrs = {}
    for c, tag in (("ppi_farm", "Farm"), ("ppi_timber", "Timber"), ("ppi_lumber", "Lumber")):
        x = real[c].dropna(); n = x.index[-1].year - x.index[0].year
        pct(f"tbReal{tag}", (x.iloc[-1] - x.iloc[0]) / n, 2); pct(f"tbReal{tag}Abs", abs(x.iloc[-1] - x.iloc[0]) / n, 2)
        pct(f"tbRealVol{tag}", x.diff().std(ddof=1), 1)
        yrs[tag] = n
    put("tbYearsTimber", yrs["Timber"], "{:.0f}")
    pct("tbCpiAnn", (np.log(dec.cpi.iloc[-1] / dec.cpi.iloc[0])) / (dec.index[-1].year - dec.index[0].year), 2)
    R.N["tbLastYear"] = str(dec.index[-1].year)
    mm = m.loc["2020":"2022"]["ppi_lumber"]; pct("tbLumberPeak", mm.max() / m.loc["2019-12", "ppi_lumber"].iloc[0] - 1, 0)
    R.N["tbLumberPeakDate"] = mm.idxmax().strftime("%B %Y")
    # figures
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    T = np.linspace(12, 90, 300)
    for r, col in zip(RATES, (BLUE, SKY, RED)):
        lev = [L.land_value(t, PRICE, r, COST) for t in T]
        ax.plot(T, np.array(lev) / 1e3, color=col, label=f"r = {100 * r:.0f}%")
        tf = L.faustmann_rotation(PRICE, r, COST); ax.plot([tf], [L.land_value(tf, PRICE, r, COST) / 1e3], "o", color=col, ms=3.5)
    ax.axhline(0, color="#333333", lw=0.6)
    ax.set_ylim(-3, None); ax.set_xlabel("rotation length, years"); ax.set_ylabel("land value, $000 a hectare")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    fig.tight_layout(); R.save(fig, "fig07_rotation"); plt.close(fig)
    rm = np.log(m[["ppi_farm", "ppi_timber", "ppi_lumber"]].div(m["cpi"], axis=0)).dropna()
    rm = 100 * np.exp(rm - rm.iloc[0])
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for c, col, lab in (("ppi_farm", BLUE, "farm products"), ("ppi_timber", GREY, "logs and timber"), ("ppi_lumber", RED, "softwood lumber")):
        ax.plot(rm.index, rm[c], color=col, lw=1.0, label=lab)
    ax.set_yscale("log"); ax.set_ylabel(f"real price, {rm.index[0]:%b %Y} = 100"); ax.set_yticks([40, 60, 100, 150, 250])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}")); ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout(); R.save(fig, "fig08_ppi"); plt.close(fig)
    marginal_figure()
    return finish("timber")


def marginal_figure(r=0.04):
    """The marginal rule of the first order condition at r: the value of a year's growth p Q'(T) against
    the cost of waiting, r p Q(T) + r LEV*, and against r p Q(T) alone (one harvest)."""
    tf, ts = L.faustmann_rotation(PRICE, r, COST), L.single_rotation(PRICE, r)
    lev = L.land_value(tf, PRICE, r, COST)
    R.RAW["mgFocErr"] = float(abs(PRICE * L.dvolume(tf) - r * PRICE * L.volume(tf) - r * lev))
    put("mgRent", r * lev, "{:.0f}"); put("mgGrowthValue", PRICE * L.dvolume(tf), "{:,.0f}")
    T = np.linspace(10, 50, 401)
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.plot(T, PRICE * L.dvolume(T), color=BLUE, label="value of a year's growth")
    ax.plot(T, r * PRICE * L.volume(T) + r * lev, color=RED, label="interest on timber and land")
    ax.plot(T, r * PRICE * L.volume(T), color=RED, ls="--", lw=0.9, label="interest on timber only")
    for x, lab, y, dx, ha in ((tf, "Faustmann", 60, -3, "right"), (ts, "one harvest", 60, 3, "left")):
        ax.axvline(x, color=GREY, lw=0.6, ls=":")
        ax.annotate(f"{lab}, {x:.1f}", (x, y), xytext=(dx, 0), textcoords="offset points", fontsize=7, ha=ha)
    ax.set_ylim(0, 700)
    ax.set_xlabel("rotation, years"); ax.set_ylabel("dollars per hectare a year")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    fig.tight_layout(); R.save(fig, "fig16_marginal"); plt.close(fig)


if __name__ == "__main__":
    print(main()["N"])
