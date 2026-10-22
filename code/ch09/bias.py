"""Section 9.4: survivorship and backfill bias in a simulated fund database. Checks the survivor formula
(the mean of the survivors) and the backfill formula against simulation, and measures how far each
view of the database overstates the true average return. Writes fig05_survivor.png and fig06_backfill.png."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish

MU, SIG, T = 0.006, 0.03, 36
INC = 24


def main():
    rng = R.rng_for(3)
    # Proposition 9.5 on a grid of cut-offs: funds survive if their mean over T months exceeds c
    cuts = np.linspace(-0.01, 0.012, 23); sim, theo, share = [], [], []
    X = MU + SIG * rng.standard_normal((40000, T)); m = X.mean(axis=1)
    for c in cuts:
        k = m > c; sim.append(m[k].mean()); theo.append(L.survivor_mean(MU, SIG, T, c)); share.append(k.mean())
    put("svMu", 100 * MU, "{:.1f}"); put("svSig", 100 * SIG, "{:.0f}"); put("svT", T, "{:.0f}")
    c0 = 0.0; k = m > c0
    pct("svShareZero", k.mean(), 0); pct("svBiasZero", 12 * (m[k].mean() - MU), 1)
    pct("svBiasZeroTh", 12 * (L.survivor_mean(MU, SIG, T, c0) - MU), 1)
    put("svMaxGap", 1e4 * np.max(np.abs(np.array(sim) - np.array(theo))), "{:.1f}")
    # the database
    db = L.simulate_database(rng)
    v = L.database_views(db)
    put("dbFunds", 3000, "{:,.0f}"); put("dbMonths", 240, "{:.0f}"); put("dbInc", INC, "{:.0f}")
    put("dbNdb", v["n_db"], "{:,.0f}"); put("dbNlive", v["n_live"], "{:,.0f}"); put("dbNall", v["n_all"], "{:,.0f}")
    tm = v["truth"].mean()
    pct("dbTruth", 12 * tm, 2)
    pct("dbLive", 12 * v["live"].mean(), 2); pct("dbBack", 12 * v["backfilled"].mean(), 2)
    pct("dbClean", 12 * v["clean"].mean(), 2)
    pct("dbSurvBias", 12 * (v["live"].mean() - tm), 2); pct("dbBackBias", 12 * (v["backfilled"].mean() - v["clean"].mean()), 2)
    pct("dbCleanBias", 12 * (v["clean"].mean() - tm), 2)
    died = db["enters"] & (db["stop"] < 240) & (db["stop"] > 0)
    pct("dbDeadShare", died.sum() / max(1, v["n_db"]), 0)
    # Proposition 9.6: entry only after a good incubation, then iid
    K, M = INC, 60
    X = MU + SIG * rng.standard_normal((60000, K + M))
    inc = np.prod(1 + X[:, :K], axis=1) - 1 > 0
    pct("bfEnterShare", inc.mean(), 0)
    pct("bfIncMean", 12 * X[inc, :K].mean(), 1); pct("bfAfterMean", 12 * X[inc, K:].mean(), 1)
    # theory uses the arithmetic mean cut c=0 as an approximation to the compounded cut
    pct("bfIncTheory", 12 * L.survivor_mean(MU, SIG, K, 0.0), 1)
    pct("bfRecordBias", 12 * (X[inc].mean() - MU), 1)
    pct("bfRecordTheory", 12 * K / (K + M) * (L.survivor_mean(MU, SIG, K, 0.0) - MU), 1)
    # event time profile from the database
    R_ = db["R"]; prof = {e: [] for e in range(-INC, 37)}
    for i in np.where(db["enters"])[0]:
        for e in prof:
            t = db["entry"][i] + e
            if 0 <= t < 240 and not np.isnan(R_[i, t]):
                prof[e].append(R_[i, t])
    put("dbEventLast", 36, "{:.0f}")
    ev = np.array(list(prof)); pm = np.array([np.mean(prof[e]) for e in ev])
    pct("dbPreEntry", pm[ev < 0].mean() * 12, 2); pct("dbPostEntry", pm[ev >= 0].mean() * 12, 2)
    # figures
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(100 * cuts, 1200 * (np.array(theo) - MU), color=RED, label="formula")
    ax.plot(100 * cuts, 1200 * (np.array(sim) - MU), "o", ms=3, color=BLUE, label="simulation")
    ax2 = ax.twinx(); ax2.plot(100 * cuts, 100 * np.array(share), color=GREY, lw=0.9, ls="--")
    ax2.set_ylabel("funds surviving, %", color=GREY); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_xlabel("survival cut-off on the monthly mean, %"); ax.set_ylabel("bias, % a year")
    ax.legend(loc="center left", frameon=True, framealpha=1, edgecolor="none").set_zorder(5)
    fig.tight_layout(); R.save(fig, "fig05_survivor"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.bar(ev, 1200 * pm, color=[ROSE if e < 0 else BLUE for e in ev], width=0.8)
    ax.axhline(1200 * tm, color=GREY, lw=0.9, ls="--")
    ax.axvline(-0.5, color=GREY, lw=0.6)
    ax.set_xlabel("months from entry into the database"); ax.set_ylabel("mean return, % a year")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); R.save(fig, "fig06_backfill"); plt.close(fig)
    return finish("bias")


if __name__ == "__main__":
    print(main()["N"])
