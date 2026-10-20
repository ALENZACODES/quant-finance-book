"""Section 4.2: discount factors, zero and forward rates, bootstrapping, the forward rate agreement,
and the Government of Canada curve at the last month end in the data.

Run on its own:  python code/ch04/term_structure.py   (writes code/ch04/out/term_structure.json)
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from numbers_ch04 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

BILLS = {0.25: "TB.CDN.90D.MID", 0.5: "TB.CDN.180D.MID", 1.0: "TB.CDN.1Y.MID"}
BONDS = {2.0: "BD.CDN.2YR.DQ.YLD", 3.0: "BD.CDN.3YR.DQ.YLD", 5.0: "BD.CDN.5YR.DQ.YLD",
         7.0: "BD.CDN.7YR.DQ.YLD", 10.0: "BD.CDN.10YR.DQ.YLD"}


def load_month_end():
    df = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", parse_dates=["date"]).set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any", subset=list(BILLS.values()) + list(BONDS.values()))
    me = df.groupby(df.index.to_period("M")).tail(1)
    last_month = df.index.max().to_period("M")
    if df.index.max() < (last_month.to_timestamp(how="end") - pd.Timedelta(days=5)):
        me = me[me.index.to_period("M") < last_month]            # drop an incomplete final month
    return me


def curve_on(row, tenors_bonds=(2.0, 3.0, 5.0, 7.0, 10.0)):
    bills = {t: row[s] / 100 for t, s in BILLS.items()}
    bonds = {T: row[BONDS[T]] / 100 for T in tenors_bonds}
    return L.bootstrap(bills, bonds)


def hand_example():
    c = L.bootstrap({0.5: 0.040, 1.0: 0.042}, {2.0: 0.045, 3.0: 0.047})
    for t, nm in [(0.5, "Half"), (1.0, "One"), (2.0, "Two"), (3.0, "Three")]:
        put(f"hxD{nm}", float(c.D(t)), "{:.6f}")
        pct(f"hxZ{nm}", float(c.zero(t)), 3)
    put("hxDOneHalf", float(c.D(1.5)), "{:.6f}")
    pct("hxFwdOneTwo", float(c.fwd_simple(1.0, 2.0)), 3)
    pct("hxFwdTwoThree", float(c.fwd_simple(2.0, 3.0)), 3)
    pct("hxFwdInstTwo", float(c.fwd_inst(2.5)), 3)
    put("hxCheckTwo", 100 * L.bond_price(c, 0.045, 2.0), "{:.6f}")
    # a 4% annual coupon three year bond priced off the curve
    times = np.array([1.0, 2.0, 3.0])
    put("hxBondFour", 100 * float(0.04 * c.D(times).sum() + c.D(3.0)), "{:.4f}")
    # FRA: borrow 1,000,000 from year 1 to year 2 at the forward rate
    f = float(c.fwd_simple(1.0, 2.0)); notional = 1_000_000
    put("fraRepay", notional * (1 + f), "{:,.0f}")
    put("fraBuy", notional * float(c.D(1.0)), "{:,.0f}")
    put("fraSell", notional * (1 + f) * float(c.D(2.0)), "{:,.0f}")
    return c


def goc_curve(me):
    row = me.iloc[-1]; date = me.index[-1]
    c = curve_on(row)
    N["gocDate"] = f"{date.day} {date.strftime('%B %Y')}"
    put("gocOneY", row[BILLS[1.0]], "{:.2f}"); put("gocTwoY", row[BONDS[2.0]], "{:.2f}")
    put("gocTenY", row[BONDS[10.0]], "{:.2f}"); put("gocThreeM", row[BILLS[0.25]], "{:.2f}")
    put("gocSixM", row[BILLS[0.5]], "{:.2f}")
    pct("gocZeroFive", float(c.zero(5.0)), 2); pct("gocZeroTen", float(c.zero(10.0)), 2)
    pct("gocFwdOneTwo", float(c.fwd_simple(1.0, 2.0)), 2)
    pct("gocFwdSevenTen", float(c.fwd_simple(7.0, 10.0)), 2)
    pct("gocFwdOneTwoCont", float(np.log(c.D(1.0) / c.D(2.0))), 2)
    pct("gocFwdSevenTenCont", float(np.log(c.D(7.0) / c.D(10.0)) / 3), 2)
    # repricing error of the input bonds (should be zero to rounding)
    err = max(abs(L.bond_price(c, row[BONDS[T]] / 100, T) - 1) for T in BONDS)
    RAW["gocRepriceErr"] = float(err)
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    x = np.linspace(0.02, 10, 800)
    ax.step(np.r_[c.t[:-1], 10], 100 * np.r_[c.fwd_inst(c.t[:-1] + 1e-9), c.fwd_inst(9.99)], where="post",
            color=RED, lw=1.2, label="instantaneous forward rate")
    ax.plot(x, 100 * c.zero(x), color=BLUE, lw=1.4, label="zero rate (continuous)")
    ax.plot(list(BILLS), [row[s] for s in BILLS.values()], "o", ms=4, mfc="white", mec=GREY, label="bill yields")
    ax.plot(list(BONDS), [row[s] for s in BONDS.values()], "s", ms=4, mfc="white", mec=BLUE, label="bond yields")
    ax.set_xlabel("maturity (years)"); ax.set_ylabel("percent")
    ax.set_title(f"Government of Canada curve, {N['gocDate']}")
    ax.legend(loc="lower right", fontsize=7.6); ax.set_xlim(0, 10.2)
    fig.tight_layout(); save(fig, "fig01_goc_curve")
    return c


def main():
    hand_example()
    me = load_month_end()
    goc_curve(me)
    return finish("term_structure")


if __name__ == "__main__":
    main()
