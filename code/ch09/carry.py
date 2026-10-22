"""Section 9.4: carry and roll yield. Splits the excess return of holding Canadian dollar bills funded in
US dollars into spot, carry and cross terms, and the excess return of the Canada 10 year bond into the
carry and roll-down it would have earned on an unchanged curve and the effect of yield changes. Regresses
realised returns on the carry known in advance. Also the illustrative oil curve of the text.
Writes fig13_carry.png."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, put, pct, finish
from trend import label as plain_label


def main():
    c = L.load_cad_carry()
    R.N["cadFirst"] = plain_label(c.index[0]); R.N["cadLast"] = plain_label(c.index[-1])
    put("cadMonths", len(c), "{:.0f}")
    R.sci("cadIdGap", max((c.excess - (c.spot + c.carry + c.cross)).abs().max(), 1e-16), 0)
    for col, tag in (("excess", "Ex"), ("spot", "Spot"), ("carry", "Carry"), ("cross", "Cross")):
        pct(f"cad{tag}", 12 * c[col].mean(), 2)
    pct("cadVol", math.sqrt(12) * c.excess.std(), 1)
    pct("cadCarryPos", (c.carry > 0).mean(), 0)
    b, e, se = L.ols(c.excess, c.carry)
    put("cadSlope", b[1], "{:.1f}"); put("cadSlopeSe", se[1], "{:.1f}")
    pct("cadCarryMax", 12 * c.carry.max(), 1); pct("cadCarryMin", 12 * c.carry.min(), 1)
    g = L.goc_carry()
    R.N["gcFirst"] = plain_label(g.index[0]); R.N["gcLast"] = plain_label(g.index[-1])
    for col, tag in (("carry", "Carry"), ("roll", "Roll"), ("expected", "Exp"), ("real", "Real")):
        pct(f"gc{tag}", 12 * g[col].mean(), 2)
    pct("gcRest", 12 * (g.real - g.expected).mean(), 2)
    pct("gcCarryShare", g.carry.mean() / g.real.mean(), 0); pct("gcExpShare", g.expected.mean() / g.real.mean(), 0)
    import pandas as pd
    gg = pd.read_csv(R.ROOT / "data" / "goc_bills_bonds.csv", index_col=0, parse_dates=True)
    y10 = L.month_end(gg["BD.CDN.10YR.DQ.YLD"])
    put("gcYStart", y10.loc[g.index[0] - 1 if g.index[0] % 100 != 1 else g.index[0] - 89], "{:.2f}")
    put("gcYEnd", y10.loc[g.index[-1]], "{:.2f}")
    pct("gcExpPos", (g.expected > 0).mean(), 0)
    b2, _, se2 = L.ols(g.real, g.expected)
    put("gcSlope", b2[1], "{:.1f}"); put("gcSlopeSe", se2[1], "{:.1f}")
    pct("gcVolReal", math.sqrt(12) * g.real.std(), 1); pct("gcVolExp", math.sqrt(12) * g.expected.std(), 2)
    # illustrative oil curve: spot 70, one and two month futures in contango or backwardation
    put("oilS", 70, "{:.0f}"); put("oilFOne", 70.7, "{:.1f}")
    pct("oilRoll", 12 * math.log(70 / 70.7), 1)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    yr = lambda idx: (idx // 100) + ((idx % 100) - 0.5) / 12
    ax[0].plot(yr(c.index), 100 * c.excess.cumsum(), color=BLUE, label="excess return")
    ax[0].plot(yr(c.index), 100 * c.carry.cumsum(), color=RED, label="carry")
    ax[0].plot(yr(c.index), 100 * c.spot.cumsum(), color=GREY, lw=0.8, label="spot")
    ax[0].axhline(0, color=GREY, lw=0.5)
    ax[0].set_ylabel("cumulative, %"); ax[0].legend(loc="upper right")
    ax[1].plot(yr(g.index), 100 * g.real.cumsum(), color=BLUE, label="excess return")
    ax[1].plot(yr(g.index), 100 * g.carry.cumsum(), color=RED, label="carry")
    ax[1].plot(yr(g.index), 100 * g.expected.cumsum(), color=SKY, label="carry and roll-down")
    ax[1].set_ylabel("cumulative, %"); ax[1].legend(loc="lower right", fontsize=7, frameon=True, framealpha=1, edgecolor="none").set_zorder(5)
    lo = 5 * math.ceil(min(yr(c.index).min(), yr(g.index).min()) / 5)
    hi = max(yr(c.index).max(), yr(g.index).max())
    for a in ax:                                        # the same five-year ticks on both panels
        a.set_xticks(np.arange(lo, hi + 0.01, 5))
    fig.tight_layout(); R.save(fig, "fig13_carry"); plt.close(fig)
    return finish("carry")


if __name__ == "__main__":
    print(main()["N"])
