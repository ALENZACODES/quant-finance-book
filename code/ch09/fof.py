"""Section 9.6: funds of funds. Diversification across n managers, the incentive fees paid when gains
are not netted across managers (the netting inequality), and the second layer of fees, checked by
simulation. Writes fig08_fof.png."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, put, pct, finish

MU, SIG, RHO, PERF = 0.08, 0.15, 0.3, 0.20
FMG, FPF = 0.01, 0.10   # fund of funds layer
NS = np.arange(1, 31)


def main():
    pct("ffMu", MU, 0); pct("ffSig", SIG, 0); put("ffRho", RHO, "{:.1f}"); pct("ffPerf", PERF, 0)
    pct("ffFmg", FMG, 0); pct("ffFpf", FPF, 0)
    sep = np.array([PERF * L.call_mean(MU, SIG) for n in NS])
    pooled = np.array([PERF * L.call_mean(MU, L.pool_sd(SIG, n, RHO)) for n in NS])
    vol = np.array([L.pool_sd(SIG, n, RHO) for n in NS])
    for n in (1, 5, 10, 20):
        tag = R.word(n) if n <= 10 else "Twenty"
        pct(f"ffVol{tag}", vol[n - 1], 1); pct(f"ffSep{tag}", sep[n - 1], 2); pct(f"ffPool{tag}", pooled[n - 1], 2)
    pct("ffVolLimit", SIG * math.sqrt(RHO), 1); pct("ffPoolLimit", PERF * L.call_mean(MU, SIG * math.sqrt(RHO)), 2)
    pct("ffRateLimit", PERF * L.call_mean(MU, SIG * math.sqrt(RHO)) / MU, 0)
    pct("ffRateSep", sep[0] / MU, 1)
    # simulation with n = 10, one year, both layers
    rng = R.rng_for(5); n, m = 10, 200000
    C = RHO * np.ones((n, n)) + (1 - RHO) * np.eye(n)
    G = MU + SIG * rng.standard_normal((m, n)) @ np.linalg.cholesky(C).T
    fees1 = PERF * np.maximum(G, 0)
    net1 = (G - fees1).mean(axis=1)
    pct("ffSimSep", fees1.mean(), 2); put("ffSimSepSe", 100 * fees1.mean(axis=1).std() / math.sqrt(m), "{:.3f}")
    pct("ffSimPool", PERF * np.maximum(G.mean(axis=1), 0).mean(), 2)
    fof = net1 - FMG
    fees2 = FPF * np.maximum(fof, 0)
    final = fof - fees2
    pct("ffNetOne", (G - fees1).mean(), 2)  # net of the funds' fees, per fund
    pct("ffNetFof", final.mean(), 2); pct("ffAllFees", MU - final.mean(), 2)
    pct("ffFeeShare", (MU - final.mean()) / MU, 0)
    pct("ffLossYearsPaid", np.mean((G.mean(axis=1) < 0) & (fees1.sum(axis=1) > 0)), 0)
    pct("ffLossYears", np.mean(G.mean(axis=1) < 0), 0)
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    ax[0].plot(NS, 100 * vol, color=BLUE); ax[0].axhline(100 * SIG * math.sqrt(RHO), color=GREY, lw=0.8, ls="--")
    ax[0].set_xlabel("number of managers"); ax[0].set_ylabel("volatility of the pool, %"); ax[0].set_ylim(0.95 * 100 * SIG * math.sqrt(RHO), None)
    ax[1].plot(NS, 100 * sep, color=RED, label="charged fund by fund")
    ax[1].plot(NS, 100 * pooled, color=BLUE, label="if gains were netted")
    ax[1].set_xlabel("number of managers"); ax[1].set_ylabel("incentive fees, % of assets")
    ax[1].legend(loc="center right"); ax[1].set_ylim(0, None)
    fig.tight_layout(); R.save(fig, "fig08_fof"); plt.close(fig)
    return finish("fof")


if __name__ == "__main__":
    print(main()["N"])
