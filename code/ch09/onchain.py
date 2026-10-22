"""Section 9.9: the arithmetic of the bitcoin protocol and the regimes of its returns. The issuance
schedule and its limit, the probability that an attacker overtakes the honest chain (gambler's ruin and
Nakamoto's Poisson version, checked against the table of the white paper), the Fisher test of a change
in correlation with equities, and volatility by calendar year from daily prices.
Writes fig14_onchain.png."""
import math
import numpy as np
import pandas as pd
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, put, pct, sci, finish

# Section 11 of the white paper, P for q = 0.1 and z = 0..10, and z for P < 0.001 (bitcoin.org/bitcoin.pdf)
PAPER_Q10 = [1.0, 0.2045873, 0.0509779, 0.0131722, 0.0034552, 0.0009137, 0.0002428, 0.0000647, 0.0000173,
             0.0000046, 0.0000012]
PAPER_Z = {0.10: 5, 0.15: 8, 0.20: 11, 0.25: 15, 0.30: 24, 0.35: 41, 0.40: 89, 0.45: 340}


def main():
    # issuance
    tot = L.total_supply_sats()
    put("ocSupply", tot / 1e8, "{:,.4f}"); put("ocShortfall", (21e14 - tot) / 1e8, "{:.4f}")
    put("ocHalvings", next(h for h in range(64) if (50 * 100_000_000) >> h == 0), "{:.0f}")
    put("ocEpochYears", 210_000 * 10 / (60 * 24 * 365.25), "{:.2f}")
    epochs = np.arange(0, 34)
    issued = np.cumsum([210_000 * ((50 * 100_000_000) >> int(h)) for h in epochs]) / 1e8
    for k, tag in ((1, "One"), (4, "Four"), (5, "Five")):
        pct(f"ocShare{tag}", issued[k - 1] / (tot / 1e8), 2 if k > 1 else 0)
    # catch-up probabilities
    mine = [L.catch_up_nakamoto(0.1, z) for z in range(11)]
    put("ocPaperGap", 1e7 * max(abs(round(a, 7) - b) for a, b in zip(mine, PAPER_Q10)), "{:.0f}")
    zs = {q: next(z for z in range(2000) if L.catch_up_nakamoto(q, z) < 0.001) for q in PAPER_Z}
    put("ocZMatch", sum(zs[q] == PAPER_Z[q] for q in PAPER_Z), "{:.0f}"); put("ocZCount", len(PAPER_Z), "{:.0f}")
    sci("ocRuinSix", L.catch_up_ruin(0.1, 6), 1); put("ocNakSix", L.catch_up_nakamoto(0.1, 6), "{:.5f}")
    sci("ocRuinTen", L.catch_up_ruin(0.1, 10), 1)
    put("ocZThirty", zs[0.30], "{:.0f}"); put("ocZLast", zs[0.45], "{:.0f}"); put("ocQLast", 0.45, "{:.2f}")
    assert zs == PAPER_Z, zs
    # correlation regimes: Fisher test of 2015-2019 against 2020 to August 2026
    d = L.load_crypto(); f = L.load_factors()
    btc = L.month_end(d["CBBTCUSD"]); rb = (btc / btc.shift(1) - 1).dropna()
    e = f["mkt_rf"]; j = rb.index[rb.index <= e.index[-1]]
    early, late = j[j < 202001], j[j >= 202001]
    r1 = np.corrcoef(rb[early], e[early])[0, 1]; r2 = np.corrcoef(rb[late], e[late])[0, 1]
    z, p = L.fisher_test(r1, len(early), r2, len(late))
    put("ocNEarly", len(early), "{:.0f}"); put("ocNLate", len(late), "{:.0f}")
    put("ocFisherZ", z, "{:.2f}"); put("ocFisherP", p, "{:.3f}")
    lo = math.tanh(math.atanh(r2) - 1.96 / math.sqrt(len(late) - 3)); hi = math.tanh(math.atanh(r2) + 1.96 / math.sqrt(len(late) - 3))
    put("ocLateLo", lo); put("ocLateHi", hi)
    lo1 = math.tanh(math.atanh(r1) - 1.96 / math.sqrt(len(early) - 3)); hi1 = math.tanh(math.atanh(r1) + 1.96 / math.sqrt(len(early) - 3))
    put("ocEarlyLo", lo1); put("ocEarlyHi", hi1)
    # volatility by calendar year, daily log returns (bitcoin and ether trade every day; equities on trading days)
    px = d[["CBBTCUSD", "CBETHUSD"]]
    vols = {}
    for col in px:
        lr = np.log(px[col].dropna()).diff().dropna()
        lr = lr[lr.index.year >= 2015]
        vols[col] = lr.groupby(lr.index.year).std() * math.sqrt(365)
    fd = pd.read_csv(R.ROOT / "data" / "ff_daily.csv", index_col=0, parse_dates=True)
    col = [c for c in fd.columns if c.lower().startswith("mkt")][0]
    er = np.log1p(fd[col] / 100); er = er[er.index.year >= 2015]
    vols["EQ"] = er.groupby(er.index.year).std() * math.sqrt(252)
    vb = vols["CBBTCUSD"].loc[:2025]
    pct("ocVolMax", vb.max(), 0); R.N["ocVolMaxYear"] = str(int(vb.idxmax()))
    pct("ocVolMin", vb.min(), 0); R.N["ocVolMinYear"] = str(int(vb.idxmin()))
    full = vb                                      # calendar years 2015 to 2025, all complete
    pct("btVolFirstYear", full.iloc[0], 0); R.N["btVolFirstYearName"] = str(int(full.index[0]))
    pct("btVolLastYear", full.iloc[-1], 0); R.N["btVolLastYearName"] = str(int(full.index[-1]))
    ratio = (vols["CBBTCUSD"] / vols["EQ"]).loc[:2025]
    put("ocRatioMin", ratio.min(), "{:.1f}"); put("ocRatioMax", ratio.max(), "{:.1f}")
    recent = ratio.loc[2022:]                      # the last four complete calendar years
    put("ocRatioRecentMin", recent.min(), "{:.1f}"); put("ocRatioRecentMax", recent.max(), "{:.1f}")
    R.N["ocRatioRecentMaxYear"] = str(int(recent.idxmax()))
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ax[0].step(np.r_[0, epochs + 1], np.r_[0, issued] / 1e6, where="post", color=BLUE)
    ax[0].axhline(21, color=GREY, lw=0.6, ls="--")
    ax[0].set_xlim(0, 12); ax[0].set_xlabel("halving periods of 210,000 blocks"); ax[0].set_ylabel("coins issued, millions")
    yrs = vols["CBBTCUSD"].index
    ax[1].plot(yrs, 100 * vols["CBBTCUSD"], "o-", ms=3, color=BLUE, label="bitcoin")
    ax[1].plot(vols["CBETHUSD"].index, 100 * vols["CBETHUSD"], "o-", ms=3, color=RED, label="ether")
    ax[1].plot(vols["EQ"].index, 100 * vols["EQ"], "o-", ms=3, color=GREY, label="US equities")
    ax[1].set_ylabel("volatility, % a year"); ax[1].set_ylim(0, None); ax[1].legend(loc="upper right")
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    fig.tight_layout(); R.save(fig, "fig14_onchain"); plt.close(fig)
    return finish("onchain")


if __name__ == "__main__":
    print(main()["N"])
