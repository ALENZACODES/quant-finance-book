"""Section 10.2, case study: loan-to-value through the property cycle. A stylised loan is written at the
start of each quarter from 1996Q2 to 2021Q2 at 65% of value, at the 10 year Treasury yield of the
quarter's first month plus a fixed spread, amortised over 25 years with a five year term. Its
loan-to-value ratio is then marked each quarter against the Z.1 commercial property price index, and the
script reports the highest ratio over the term, the ratio at maturity and the equity a borrower would
need to refinance at 65%. Writes fig03_cycle.png."""
import numpy as np
import pandas as pd
import numbers_ch10 as R
from numbers_ch10 import L, plt, BLUE, RED, GREY, SKY, PALE, put, pct, finish

LTV0, SPREAD, AMORT, TERM = 0.65, 0.018, 25, 5


def vintages(start=L.SPLICE):
    q = L.load_quarterly()["cre"].dropna()
    m = L.load_monthly()["gs10"] / 100
    q = q[q.index > start]
    rows, paths = [], {}
    for i, t0 in enumerate(q.index):
        if i + 4 * TERM >= len(q):
            break
        rate = m.loc[t0] + SPREAD
        h = np.arange(0, 4 * TERM + 1)
        bal = np.array([L.balance(rate, AMORT, j / 4) for j in h])
        px = q.iloc[i:i + 4 * TERM + 1].values / q.iloc[i]
        ltv = LTV0 * bal / px
        paths[t0] = ltv
        rows.append(dict(start=t0, rate=rate, peak=ltv.max(), end=ltv[-1], bal=bal[-1], px=px[-1],
                         gap=max(bal[-1] * LTV0 - LTV0 * LTV0 * px[-1], 0.0) / LTV0))
    return pd.DataFrame(rows).set_index("start"), paths


def lab(t):
    return f"{t.year}Q{t.quarter}"


def main():
    v, paths = vintages()
    put("lcN", len(v), "{:.0f}"); R.N["lcFirst"] = lab(v.index[0]); R.N["lcLastStart"] = lab(v.index[-1])
    pct("lcSpread", SPREAD, 2); pct("lcLtvZero", LTV0, 0)
    pct("lcShareEighty", np.mean(v.peak > 0.80), 0); pct("lcShareHundred", np.mean(v.peak > 1.0), 0)
    pct("lcShareRefi", np.mean(v.end > LTV0), 0)
    j = v.peak.idxmax(); R.N["lcWorst"] = lab(j)
    pct("lcWorstPeak", v.peak[j], 0); pct("lcWorstEnd", v.end[j], 0); pct("lcWorstRate", v.rate[j], 2)
    pct("lcWorstPx", v.px[j] - 1, 0); pct("lcWorstPxAbs", 1 - v.px[j], 0)
    k = v.end.idxmax(); R.N["lcWorstEndVintage"] = lab(k); pct("lcWorstEndLtv", v.end[k], 0)
    # equity needed to refinance at 65%, as a share of the original value
    pct("lcGapMax", (v.bal * LTV0 - LTV0 * v.px).clip(lower=0).max(), 1)
    late = v[(v.index >= "2016-01-01")]
    pct("lcLateEnd", late.end.mean(), 0); R.N["lcLateFirst"] = lab(late.index[0])
    pct("lcMedPeak", v.peak.median(), 0)
    # the appraisal era: the same loans written in 1985 to 1990, marked against the appraisal index
    va, _ = vintages(start="1970-01-01")
    a = va[(va.index >= "1985-01-01") & (va.index < "1991-01-01")]
    pct("lcApprPeak", a.peak.max(), 0); R.N["lcApprWorst"] = lab(a.peak.idxmax())
    pct("lcApprShareEighty", np.mean(a.peak > 0.80), 0)
    v.assign(start=[lab(t) for t in v.index]).to_csv(R.HERE / "out" / "cycle.csv", index=False)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    yrs = np.arange(0, 4 * TERM + 1) / 4
    for t0, col, lw in ((pd.Timestamp("2005-01-01"), SKY, 1.0), (j, RED, 1.4), (pd.Timestamp("2012-01-01"), BLUE, 1.0),
                        (pd.Timestamp("2020-01-01"), GREY, 1.0)):
        ax[0].plot(yrs, 100 * paths[t0], color=col, lw=lw, label=lab(t0))
    ax[0].axhline(100 * LTV0, color="#333333", lw=0.6, ls=":")
    ax[0].set_xlabel("years since the loan was written"); ax[0].set_ylabel("loan-to-value, %")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=7.5)
    ax[1].plot(v.index, 100 * v.peak, color=RED, lw=1.2, label="highest over the term")
    ax[1].plot(v.index, 100 * v.end, color=BLUE, lw=1.2, label="at maturity")
    ax[1].axhline(100 * LTV0, color="#333333", lw=0.6, ls=":")
    ax[1].set_xlabel("quarter the loan was written"); ax[1].set_ylabel("loan-to-value, %")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7.5)
    fig.tight_layout(); R.save(fig, "fig03_cycle"); plt.close(fig)
    return finish("cycle")


if __name__ == "__main__":
    print(main()["N"])
