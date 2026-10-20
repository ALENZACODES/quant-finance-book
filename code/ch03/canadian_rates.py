"""Canadian rate examples for Chapter 3: realised quadratic variation of the five year Government of
Canada yield, a reflection-principle check on its one-year maximum, and the convexity drift of a five
year zero from Ito's lemma. Data: Bank of Canada daily yields, 2001-01-02 to 2026-09-29."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats
from numbers_ch03 import N, RAW, put, pct, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, PALE, ROOT  # noqa: F401

SAMPLE_END = "2026-09-30"
SERIES = "BD.CDN.5YR.DQ.YLD"


def load():
    df = pd.read_csv(ROOT / "data" / "goc_bills_bonds.csv", parse_dates=["date"]).set_index("date")
    y = df[SERIES].dropna()
    return y[y.index <= SAMPLE_END]


def realised(y):
    """Annualised realised volatility (yield points) by calendar year from daily, weekly and month-end data."""
    rows = []
    for yr in range(2001, 2026):
        d = y[y.index.year == yr]
        prev = y[y.index < f"{yr}-01-01"]
        d = pd.concat([prev.iloc[-1:], d]) if len(prev) else d
        qd = np.sum(np.diff(d.values) ** 2)
        w = d.resample("W-FRI").last().dropna(); qw = np.sum(np.diff(w.values) ** 2)
        m = d.resample("ME").last().dropna(); qm = np.sum(np.diff(m.values) ** 2)
        rows.append((yr, math.sqrt(qd), math.sqrt(qw), math.sqrt(qm), len(d) - 1))
    return pd.DataFrame(rows, columns=["year", "daily", "weekly", "monthly", "n"]).set_index("year")


def first_passage(y, a=0.5, days=252):
    """Share of start dates at which the yield moves at least a points up (or down) within `days` trading days,
    measured on the running maximum, on month-end checks only, and at the end of the window only."""
    v = y.values; n = len(v) - days
    up = dn = upm = dnm = upe = dne = eit = 0
    for i in range(n):
        w = v[i:i + days + 1] - v[i]
        up += w.max() >= a; dn += w.min() <= -a; eit += (w.max() >= a) or (w.min() <= -a)
        mm = w[::21]
        upm += mm.max() >= a; dnm += mm.min() <= -a
        upe += w[-1] >= a; dne += w[-1] <= -a
    return dict(n=n, eit=eit / n, up=up / n, dn=dn / n, upm=upm / n, dnm=dnm / n, upe=upe / n, dne=dne / n)


def main():
    y = load()
    put("canN", len(y), "{:,d}"); N["canStart"] = y.index[0].strftime("%B %Y"); N["canEnd"] = y.index[-1].strftime("%B %Y")
    # quadratic variation by sampling frequency
    rv = realised(y)
    put("canRvDailyMed", rv["daily"].median()); put("canRvWeeklyMed", rv["weekly"].median())
    put("canRvMonthlyMed", rv["monthly"].median())
    put("canRvMin", rv["daily"].min()); put("canRvMax", rv["daily"].max())
    N["canRvMinYear"] = str(int(rv["daily"].idxmin())); N["canRvMaxYear"] = str(int(rv["daily"].idxmax()))
    put("canRvRatioMed", (rv["monthly"] / rv["daily"]).median())
    # sampling noise alone: twelve iid normal changes give sqrt(chi2_12 / 12) times the true volatility
    put("canRvNoiseLo", math.sqrt(stats.chi2.ppf(0.1, 12) / 12)); put("canRvNoiseHi", math.sqrt(stats.chi2.ppf(0.9, 12) / 12))
    put("canRvRatioLo", (rv["monthly"] / rv["daily"]).quantile(0.1)); put("canRvRatioHi", (rv["monthly"] / rv["daily"]).quantile(0.9))
    RAW["canRvTable"] = rv.round(4).reset_index().values.tolist()
    # one year of squared increments: the running sum against a straight line
    d25 = y[(y.index >= "2024-12-31") & (y.index <= "2025-12-31")]
    cq = np.concatenate([[0.0], np.cumsum(np.diff(d25.values) ** 2)])
    tt = np.arange(len(cq)) / (len(cq) - 1)
    put("canQvTwentyFive", cq[-1], "{:.3f}")
    put("canQvMaxSq", np.max(np.diff(d25.values) ** 2), "{:.4f}")
    put("canQvMaxShare", 100 * np.max(np.diff(d25.values) ** 2) / cq[-1], "{:.0f}")
    put("canQvLinDev", np.max(np.abs(cq - cq[-1] * tt)) / cq[-1] * 100, "{:.0f}")
    # whole-sample sigma from daily changes
    dy = np.diff(y.values); sig = dy.std(ddof=1) * math.sqrt(252)
    put("canSigma", sig, "{:.2f}")
    # reflection principle: rise or fall of 0.5 points within a year
    a = 0.5; fp = first_passage(y, a)
    p_model = 2 * stats.norm.sf(a / sig); pe_model = stats.norm.sf(a / sig)
    pct("canHitModel", p_model, 1); pct("canEndModel", pe_model, 1)
    hit = 0.5 * (fp["up"] + fp["dn"]); hitm = 0.5 * (fp["upm"] + fp["dnm"]); end = 0.5 * (fp["upe"] + fp["dne"])
    pct("canHitEmp", hit, 1); pct("canHitMonthEmp", hitm, 1); pct("canEndEmp", end, 1)
    pct("canHitUp", fp["up"], 1); pct("canHitDn", fp["dn"], 1)
    put("canHitRatio", hit / end); put("canStarts", fp["n"], "{:,d}")
    # either level, model: P(sup |W| >= b) over one year, b = a / sigma (series for the exit time of a strip)
    b = a / sig; k = np.arange(0, 60)
    stay = 4 / math.pi * np.sum((-1.0) ** k / (2 * k + 1) * np.exp(-(2 * k + 1) ** 2 * math.pi ** 2 / (8 * b * b)))
    pct("canHitEitherModel", 1 - stay, 1); pct("canHitEitherEmp", fp["eit"], 1)
    put("canYearsIndep", fp["n"] / 252, "{:.0f}")
    # Ito's lemma: convexity drift of a five year zero, P = exp(-5 y)
    D = 5.0; s = sig / 100
    pct("canConvex", 0.5 * D ** 2 * s ** 2, 3)
    put("canConvexBp", 1e4 * 0.5 * D ** 2 * s ** 2, "{:.1f}")
    # check by simulation of the exact identity on the data: sum of 0.5 D^2 (dy)^2 per year (median year)
    pct("canConvexRealMed", 0.5 * D ** 2 * (rv["daily"].median() / 100) ** 2, 3)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].plot(rv.index, rv["daily"], "o-", ms=3, color=BLUE, lw=1, label="daily changes")
    ax[0].plot(rv.index, rv["weekly"], "s-", ms=2.5, color=SKY, lw=0.8, label="weekly")
    ax[0].plot(rv.index, rv["monthly"], "^-", ms=3, color=RED, lw=0.9, label="month-end")
    ax[0].set_ylabel("realised volatility (points a year)"); ax[0].set_title("(a) By calendar year")
    ax[0].legend(fontsize=6.8)
    ax[1].plot(d25.index, cq, color=BLUE, lw=1.1, label="running sum of squared daily changes")
    ax[1].plot(d25.index, cq[-1] * tt, color=RED, lw=0.9, ls="--", label="straight line to the year's total")
    ax[1].set_ylabel(r"points$^2$"); ax[1].set_title("(b) 2025, day by day"); ax[1].legend(fontsize=6.6, loc="upper left")
    for lab in ax[1].get_xticklabels():
        lab.set_rotation(30)
    fig.tight_layout(); save(fig, "fig14_canada_qv")
    return finish("canadian_rates")


if __name__ == "__main__":
    main()
