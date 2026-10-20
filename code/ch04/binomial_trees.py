"""Section 4.7 and 4.8: one-period replication, the binomial model, convergence of the Cox-Ross-
Rubinstein and Leisen-Reimer trees to the Black-Scholes price, and American exercise.

Run on its own:  python code/ch04/binomial_trees.py   (writes code/ch04/out/binomial_trees.json)
"""
from __future__ import annotations
import math, runpy, io, contextlib
import numpy as np
from scipy import stats
from numbers_ch04 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

S0, K, R, SIG, T = 100.0, 110.0, 0.05, 0.20, 1.0


def one_and_two_period():
    S, u, d, Rg, Kc = 100.0, 1.2, 0.9, 1.05, 100.0
    q = (Rg - d) / (u - d); put("opQ", q, "{:.2f}")
    Vu, Vd = max(S * u - Kc, 0), max(S * d - Kc, 0)
    delta = (Vu - Vd) / (S * (u - d)); bond = (Vd - delta * S * d) / Rg
    put("opDelta", delta, "{:.4f}"); put("opBond", bond, "{:.4f}"); put("opPrice", delta * S + bond, "{:.4f}")
    put("opPriceQ", (q * Vu + (1 - q) * Vd) / Rg, "{:.4f}")
    # two-period American put, K = 105
    Kp = 105.0; params = (u, d, q, 1 / Rg)
    am = L.tree_american(S, Kp, 2, params, "put"); eu = L.tree_european(S, Kp, 2, params, "put")
    put("tpAmerican", am, "{:.4f}"); put("tpEuropean", eu, "{:.4f}"); put("tpPremium", am - eu, "{:.4f}")
    put("tpContDown", (q * 0 + (1 - q) * max(Kp - S * d * d, 0)) / Rg, "{:.4f}")
    put("tpExDown", Kp - S * d, "{:.2f}")
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    nodes = {}
    for i in range(3):
        for j in range(i + 1):
            Sij = S * u ** j * d ** (i - j); nodes[(i, j)] = (i, j - i / 2, Sij)
    # values by backward induction, recording exercise
    V = {(2, j): max(Kp - S * u ** j * d ** (2 - j), 0) for j in range(3)}; ex = set()
    for i in (1, 0):
        for j in range(i + 1):
            Sij = S * u ** j * d ** (i - j); cont = (q * V[(i + 1, j + 1)] + (1 - q) * V[(i + 1, j)]) / Rg
            V[(i, j)] = max(cont, Kp - Sij)
            if Kp - Sij > cont:
                ex.add((i, j))
    for (i, j), (x, y, Sij) in nodes.items():
        if i < 2:
            for dj in (0, 1):
                x2, y2, _ = nodes[(i + 1, j + dj)]; ax.plot([x, x2], [y, y2], color="#AAAAAA", lw=0.8, zorder=1)
    for (i, j), (x, y, Sij) in nodes.items():
        col = RED if (i, j) in ex else BLUE
        ax.scatter([x], [y], s=26, color=col, zorder=3)
        ax.annotate(f"$S={Sij:.0f}$\n$V={V[(i, j)]:.2f}$", (x, y), xytext=(0, 9), textcoords="offset points",
                    fontsize=7.8, color=col, va="bottom", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_xlim(-0.3, 2.3); ax.set_ylim(-1.25, 1.6); ax.axis("off")
    ax.set_title("Two-period American put, $K=105$ (red: exercise is optimal)")
    fig.tight_layout(); save(fig, "fig07_tree")


def convergence():
    bs = L.bs_price(S0, K, R, SIG, T, "call"); put("bsCall", bs, "{:.4f}")
    ns = np.arange(10, 501); crr = np.array([L.tree_european(S0, K, n, L.crr_params(R, SIG, T, n)) for n in ns])
    no = np.arange(11, 502, 2); lr = np.array([L.tree_european(S0, K, n, L.lr_params(S0, K, R, SIG, T, n)) for n in no])
    e_crr = crr - bs; e_lr = lr - bs
    for n, nm in [(50, "Fifty"), (100, "Hundred"), (101, "HundredOne"), (500, "FiveHundred")]:
        put(f"crr{nm}", crr[n - 10], "{:.4f}")
    put("lrHundredOne", lr[(101 - 11) // 2], "{:.6f}"); put("lrFiveHundredOne", lr[-1], "{:.6f}")
    put("lrErrHundredOne", 1e6 * e_lr[(101 - 11) // 2], "{:.1f}")
    # n times the CRR error stays bounded and oscillates
    tail = ns >= 100
    put("crrNErrMin", (ns * e_crr)[tail].min(), "{:.3f}"); put("crrNErrMax", (ns * e_crr)[tail].max(), "{:.3f}")
    # slope of the error envelope: max |error| over geometric blocks
    edges = np.unique(np.round(np.geomspace(20, 500, 9)).astype(int))
    env_n, env_e = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (ns >= a) & (ns < b); env_n.append(math.sqrt(a * b)); env_e.append(np.abs(e_crr[m]).max())
    put("crrSlope", L.ols_slope(env_n, env_e), "{:.2f}")
    put("lrSlope", L.ols_slope(no[no >= 31], np.abs(e_lr[no >= 31])), "{:.2f}")
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.8))
    sel = ns <= 200
    ax[0].plot(ns[sel], crr[sel], color=SKY, lw=0.8, label="Cox-Ross-Rubinstein")
    ax[0].plot(no[no <= 200], lr[no <= 200], color=RED, lw=1.2, label="Leisen-Reimer (odd $n$)")
    ax[0].axhline(bs, color=BLUE, lw=1.0, ls="--", label=f"Black-Scholes {bs:.4f}")
    ax[0].set_xlabel("steps $n$"); ax[0].set_ylabel("call price"); ax[0].set_title("(a) prices")
    ax[0].legend(fontsize=7.0, loc="upper right"); ax[0].set_ylim(bs - 0.08, bs + 0.10)
    ax[1].loglog(ns, np.abs(e_crr), color=SKY, lw=0.6, label="CRR error")
    ax[1].loglog(env_n, env_e, "o-", ms=3, color=BLUE, lw=1.0, label="CRR envelope")
    ax[1].loglog(no, np.abs(e_lr), color=RED, lw=1.2, label="Leisen-Reimer")
    ax[1].set_xlabel("steps $n$"); ax[1].set_title("(b) absolute error"); ax[1].legend(fontsize=7.0, loc="lower left")
    fig.tight_layout(); save(fig, "fig08_convergence")


def american():
    Kp = 100.0
    eu = L.bs_price(S0, Kp, R, SIG, T, "put"); put("bsPut", eu, "{:.4f}")
    am1, _ = L.tree_american(S0, Kp, 1000, L.crr_params(R, SIG, T, 1000), "put", boundary=True)
    am2, bnd = L.tree_american(S0, Kp, 2000, L.crr_params(R, SIG, T, 2000), "put", boundary=True)
    put("amPutThousand", am1, "{:.4f}"); put("amPutTwoThousand", am2, "{:.4f}")
    put("amPremium", am2 - eu, "{:.4f}")
    eu2 = L.tree_european(S0, Kp, 2000, L.crr_params(R, SIG, T, 2000), "put")
    put("euPutTwoThousand", eu2, "{:.4f}"); put("amPremiumTree", am2 - eu2, "{:.4f}")
    put("perpBoundary", L.perpetual_put_boundary(Kp, R, SIG), "{:.2f}")
    h = T / 2000; tte = T - np.arange(2000) * h
    # a longer put shows the approach to the perpetual boundary
    _, bnd10 = L.tree_american(S0, Kp, 2000, L.crr_params(R, SIG, 10.0, 2000), "put", boundary=True)
    # calls: no early exercise without dividends, early exercise with them
    p = L.crr_params(R, SIG, T, 1000)
    ac = L.tree_american(S0, Kp, 1000, p, "call"); ec = L.tree_european(S0, Kp, 1000, p, "call")
    RAW["amCallGap"] = float(ac - ec); put("amCallNoDiv", ac, "{:.4f}")
    pq = L.crr_params(R, SIG, T, 1000, q=0.04)
    acq = L.tree_american(S0, Kp, 1000, pq, "call"); ecq = L.tree_european(S0, Kp, 1000, pq, "call")
    put("amCallDiv", acq, "{:.4f}"); put("euCallDiv", ecq, "{:.4f}"); put("amCallDivPrem", acq - ecq, "{:.4f}")
    # hedge ratio at the root against N(d1)
    pc = L.crr_params(R, SIG, T, 1000)
    put("treeDelta", L.tree_delta(S0, K, 1000, pc, "call"), "{:.4f}")
    d1 = (math.log(S0 / K) + (R + 0.5 * SIG ** 2) * T) / (SIG * math.sqrt(T)); put("bsDelta", stats.norm.cdf(d1), "{:.4f}")
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    # nodes sit on a grid of prices, so the raw boundary jitters by one node; the figure shows its running
    # minimum taken outward from expiry, which is the monotone boundary the theory describes
    def mono(tt, bb):
        ok = ~np.isnan(bb); tt, bb = tt[ok][::-1], bb[ok][::-1]
        return tt, np.minimum.accumulate(bb)
    tte10 = 10.0 - np.arange(2000) * (10.0 / 2000)
    t1, b1 = mono(tte, bnd); t10, b10 = mono(tte10, bnd10)
    # the boundary depends only on time to expiry; read it one year out from a two year tree with the same step
    _, bnd2 = L.tree_american(S0, Kp, 4000, L.crr_params(R, SIG, 2.0, 4000), "put", boundary=True)
    tte2 = 2.0 - np.arange(4000) * (2.0 / 4000); t2, b2 = mono(tte2, bnd2)
    put("bndStart", np.interp(1.0, t2, b2), "{:.2f}"); put("bndHalf", np.interp(0.5, t2, b2), "{:.2f}")
    put("bndHalfOne", np.interp(0.5, t1, b1), "{:.2f}")
    put("bndTenYear", b10[-1], "{:.2f}")
    # the raw levels on the ten year tree: the earliest step with an exercise node, and the alternation after it
    first = int(np.argmax(~np.isnan(bnd10)))
    put("bndTenFirst", bnd10[first], "{:.2f}"); put("bndTenFirstT", tte10[first], "{:.3f}")
    put("bndTenFirstEnv", b10[-1], "{:.2f}")
    # the step at which the envelope first reaches its ten year value, and the number of steps after the first exercise node
    k_env = int(np.argmax(np.isclose(b10, b10[-1])))
    put("bndTenEnvT", t10[k_env], "{:.3f}")
    # one step after the first exercise node the raw level is already the lower neighbouring node
    assert not np.isnan(bnd10[first + 1]) and bnd10[first + 1] < bnd10[first]
    put("bndTenSecond", bnd10[first + 1], "{:.2f}"); put("bndTenSecondT", tte10[first + 1], "{:.3f}")
    ok = ~np.isnan(bnd)
    ax.plot(*mono(tte, bnd), color=RED, lw=1.3, label="1 year put, $n=2000$")
    ax.plot(*mono(tte10, bnd10), color=BLUE, lw=1.1, label="10 year put, $n=2000$")
    ax.axhline(L.perpetual_put_boundary(Kp, R, SIG), color=GREY, lw=0.9, ls="--", label="perpetual put boundary")
    ax.axhline(Kp, color="#999999", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("time to expiry (years, log scale)"); ax.set_ylabel("exercise below this price")
    ax.set_title("Early exercise boundary of an American put, $K=100$, $r=5\\%$, $\\sigma=20\\%$")
    ax.legend(fontsize=7.6, loc="upper right"); ax.set_ylim(65, 102)
    fig.tight_layout(); save(fig, "fig09_boundary")


def snippet():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(HERE / "snippets" / "crr.py"), run_name="__main__")
    txt = buf.getvalue().strip(); RAW["snippetOut"] = 0.0
    eu, am = [float(t.split()[-1].rstrip(",")) for t in txt.split(", ")]
    put("snipEu", eu, "{:.4f}"); put("snipAm", am, "{:.4f}")


def main():
    one_and_two_period(); convergence(); american(); snippet()
    return finish("binomial_trees")


if __name__ == "__main__":
    main()
