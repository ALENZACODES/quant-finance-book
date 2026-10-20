"""Section 4.3: one-period markets, state prices, detecting arbitrage, and price bounds in an
incomplete market by linear programming.

Run on its own:  python code/ch04/no_arbitrage.py   (writes code/ch04/out/no_arbitrage.json)
"""
from __future__ import annotations
import numpy as np
from numbers_ch04 import N, RAW, put, pct, word, save, finish, rng_for, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, HERE  # noqa: F401

R = 1.02                       # gross riskless return over the period
STOCK = np.array([120.0, 100.0, 80.0])
S0 = 100.0


def complete_market():
    call = np.maximum(STOCK - 100, 0); c0 = 7.00
    A = np.column_stack([np.full(3, R), STOCK, call]); p = np.array([1.0, S0, c0])
    psi = L.state_prices(A, p)
    for k, nm in enumerate(["Up", "Mid", "Down"]):
        put(f"psi{nm}", psi[k], "{:.4f}"); pct(f"qrn{nm}", R * psi[k], 2)
    put("psiSum", psi.sum(), "{:.4f}")
    put("callQuote", c0, "{:.2f}")
    # a dealer bids 13.00 for a put struck at 110: selling it to the dealer and buying the replica is the arbitrage
    putX = np.maximum(110 - STOCK, 0); quote = 13.00
    fair = float(psi @ putX)
    put("putFair", fair, "{:.4f}"); put("putQuote", quote, "{:.2f}")
    # replicate the put with bond, stock and call
    theta = np.linalg.solve(A, putX)
    put("repBond", theta[0], "{:.4f}"); put("repStock", theta[1], "{:.4f}"); put("repCall", theta[2], "{:.4f}")
    assert abs(theta[1] + 1) < 1e-9          # the text says "short one share"
    put("putGap", quote - fair, "{:.4f}")
    ok4, th4 = L.arbitrage_check(np.column_stack([A, putX]), np.r_[p, quote])
    RAW["fourAssetNoArb"] = float(ok4)
    ok3, _ = L.arbitrage_check(A, p)
    RAW["threeAssetNoArb"] = float(ok3)
    # a call quoted below its lower bound gives an arbitrage too
    ok_bad, _ = L.arbitrage_check(A, np.array([1.0, S0, 1.50]))
    RAW["lowCallNoArb"] = float(ok_bad)
    return psi


def incomplete_bounds():
    A = np.column_stack([np.full(3, R), STOCK]); p = np.array([1.0, S0])
    Ks = np.linspace(80, 120, 81)
    lo, hi, sub, sup = [], [], [], []
    for K in Ks:
        b = L.price_bounds(A, p, np.maximum(STOCK - K, 0))
        lo.append(b[0]); hi.append(b[1]); sub.append(b[2]); sup.append(b[3])
    lo, hi, sub, sup = map(np.array, (lo, hi, sub, sup))
    RAW["dualGap"] = float(max(np.abs(lo - sub).max(), np.abs(hi - sup).max()))
    i = int(np.argmin(np.abs(Ks - 100)))
    put("boundLo", lo[i], "{:.4f}"); put("boundHi", hi[i], "{:.4f}")
    # risk-neutral probability interval: q_up + q_mid + q_down = 1, 120 q_u + 100 q_m + 80 q_d = 102
    # q_up - q_down = (R S0 - 100) / 20 and q_mid = 1 - q_up - q_down >= 0
    gap = (R * S0 - 100) / 20
    put("qUpMin", 100 * gap, "{:.0f}"); put("qUpMax", 100 * (1 + gap) / 2, "{:.0f}")
    # the complete-market price of the same call lies inside
    RAW["callInside"] = float(lo[i] < 7.0 < hi[i])
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.fill_between(Ks, lo, hi, color=PALE, lw=0, label="no-arbitrage prices")
    ax.plot(Ks, hi, color=RED, lw=1.3, label="super-replication cost")
    ax.plot(Ks, lo, color=BLUE, lw=1.3, label="sub-replication value")
    ax.plot([100], [7.0], "o", ms=4.5, color=GREY, label="quote used in the text")
    ax.set_xlabel("strike $K$"); ax.set_ylabel("call price")
    ax.set_title("Call price bounds in a one-period, three-state market")
    ax.legend(loc="upper right", fontsize=7.6); ax.set_xlim(80, 120); ax.set_ylim(0, 22)
    fig.tight_layout(); save(fig, "fig02_bounds")


def main():
    complete_market()
    incomplete_bounds()
    return finish("no_arbitrage")


if __name__ == "__main__":
    main()
