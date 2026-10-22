"""Section 8.5: how much a backtest can tell. Zone probabilities for 250 days at 99%, the exact power
of the Kupiec test over longer records, and the spread of the expected shortfall statistic Z2 when
the model is right. Writes fig05_power.png."""
import math
import numpy as np
from scipy import stats
import numbers_ch08 as R
from numbers_ch08 import L, plt, BLUE, RED, GREY, SKY, ROSE, put, pct, finish, rng_for


def main():
    g, y, r = L.zones(250, 0.01)
    pct("znGreen", g, 1); pct("znYellow", y, 1); pct("znRed", r, 3)
    for q, tg in ((0.02, "Two"), (0.03, "Three")):
        g2, y2, r2 = L.zones(250, q)
        pct(f"znGreen{tg}", g2, 0); pct(f"znRed{tg}", r2, 0)
    put("znExpected", 2.5, "{:.1f}")
    # Kupiec over one year: which counts are rejected at 5%
    acc = [k for k in range(0, 30) if L.kupiec(k, 250, 0.01)[1] >= 0.05]
    put("kpLo", min(acc), "{:.0f}"); put("kpHi", max(acc), "{:.0f}")
    lr, p = L.kupiec(6, 250, 0.01); put("kpSixLR", lr); put("kpSixP", p, "{:.3f}")
    # power for true rates 1.5% and 2% over 1 to 20 years
    years = np.arange(1, 21)
    pw = {q: np.array([L.kupiec_power(250 * k, 0.01, q) for k in years]) for q in (0.015, 0.02)}
    pct("pwOneTwo", pw[0.02][0], 0); pct("pwOneFifteen", pw[0.015][0], 0)
    pct("pwFiveFifteen", pw[0.015][4], 0); pct("pwTenFifteen", pw[0.015][9], 0)
    put("pwYearsFifteen", int(years[np.argmax(pw[0.015] >= 0.8)]), "{:.0f}")
    put("pwYearsTwo", int(years[np.argmax(pw[0.02] >= 0.8)]), "{:.0f}")
    pct("pwSizeOne", L.kupiec_power(250, 0.01, 0.01), 1)
    # null law of Z2 for a case study length record of iid t(5) losses scored with the true VaR and ES:
    # analytic, by the central limit theorem, and by a large simulation in ten independent streams
    T, a, nu = 14897, 0.975, 5
    m1, m2 = L.z2_moments("t", a, nu=nu)
    zmean, zsd, zcrit = L.z2_null(m1, m2, T, a)
    put("ztSdAn", zsd, "{:.4f}"); put("ztCritAn", zcrit, "{:.4f}")
    s = math.sqrt((nu - 2) / nu)
    var = s * stats.t.ppf(a, nu); q = stats.t.ppf(a, nu)
    es = s * stats.t.pdf(q, nu) / (1 - a) * (nu + q * q) / (nu - 1)
    STREAMS, M = 10, 2000
    z2 = np.empty((STREAMS, M))
    for k in range(STREAMS):
        rng = rng_for(50 + k)
        for i in range(M):
            Lr = s * rng.standard_t(nu, T)
            I = Lr > var
            z2[k, i] = np.sum(Lr[I]) / es / (T * (1 - a)) - 1
    crits = np.quantile(z2, 0.95, axis=1)
    put("ztMean", z2.mean(), "{:.4f}"); put("ztSd", z2.std(), "{:.4f}")
    put("ztCrit", np.quantile(z2, 0.95), "{:.4f}"); put("ztCritSe", crits.std(ddof=1) / math.sqrt(STREAMS), "{:.4f}")
    # asymptotic standard error of the pooled 95th percentile, with the normal density at that point
    q95 = 1.6448536269514722; dens = math.exp(-q95 ** 2 / 2) / math.sqrt(2 * math.pi) / z2.std()
    put("ztCritSeAn", math.sqrt(0.05 * 0.95 / z2.size) / dens, "{:.4f}")
    put("ztMeanSe", z2.mean(axis=1).std(ddof=1) / math.sqrt(STREAMS), "{:.4f}")
    put("ztM", STREAMS * M, "{:,.0f}"); put("ztStreams", STREAMS, "{:.0f}")
    put("ztT", T, "{:,.0f}")
    R.RAW["ztSkewSim"] = float(stats.skew(z2.ravel()))
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.8))
    ax[0].plot(years, 100 * pw[0.02], color=RED, label="true rate 2%")
    ax[0].plot(years, 100 * pw[0.015], color=BLUE, label="true rate 1.5%")
    ax[0].axhline(80, color="#999999", lw=0.8, ls=":")
    ax[0].set_xlabel("years of daily data"); ax[0].set_ylabel("rejection probability, %")
    ax[0].legend(loc="lower right"); ax[0].set_ylim(0, 102)
    qs = np.linspace(0.005, 0.04, 71)
    zz = np.array([L.zones(250, q) for q in qs])
    ax[1].plot(100 * qs, 100 * zz[:, 0], color=BLUE, label="green")
    ax[1].plot(100 * qs, 100 * zz[:, 1], color=GREY, label="yellow")
    ax[1].plot(100 * qs, 100 * zz[:, 2], color=RED, label="red")
    ax[1].axvline(1, color="#999999", lw=0.8, ls=":")
    ax[1].set_xlabel("true exception rate, %"); ax[1].set_ylabel("probability over 250 days, %")
    ax[1].legend(loc="upper right")
    fig.tight_layout(); R.save(fig, "fig05_power"); plt.close(fig)
    return finish("backtest_theory")


if __name__ == "__main__":
    print(main()["N"])
