"""Section 11.5: discounts for control and marketability. The combined discount on a minority stake,
the protective put of Chaffe, the average-strike put of Finnerty and the lookback value of Longstaff, both against the length of a
restriction, with a simulation of the lookback that draws each path's maximum exactly. Writes
fig14_dlom.png.
Also writes the stake values of the minority table and out/minority_inputs.csv for the Octave listing."""
import json, math
import numpy as np
import numbers_ch11 as R
from numbers_ch11 import L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, put, pct, finish, rng_for

CONTROL_PREMIUM = 0.25
SIG_CASE, T_CASE = 0.45, 2.0
STAKE = 0.10


def main():
    rng = rng_for(5)
    dloc = 1 - 1 / (1 + CONTROL_PREMIUM); pct("dmCp", CONTROL_PREMIUM, 0); pct("dmDloc", dloc, 0)
    prior = json.loads((R.HERE / "out" / "valuation.json").read_text())["RAW"] if "vlEq" not in R.RAW else R.RAW
    rf = prior["wkRf"] / 100            # the Treasury yield of the valuation, run valuation.py first
    chaffe = L.chaffe_dlom(T_CASE, SIG_CASE, r=rf); chaffe0 = L.chaffe_dlom(T_CASE, SIG_CASE)
    ls = L.longstaff_bound(T_CASE, SIG_CASE)
    pct("dmSig", SIG_CASE, 0); put("dmT", T_CASE, "{:.0f}")
    pct("dmChaffe", chaffe, 1); pct("dmChaffeZero", chaffe0, 1); pct("dmLong", ls, 1); put("dmLongOverPut", ls / chaffe, "{:.1f}")
    comb = L.combined_discount(dloc, chaffe); pct("dmComb", comb, 1); pct("dmSum", dloc + chaffe, 1)
    eq = prior["vlEq"]
    put("dmStakeCtl", STAKE * eq, "{:.2f}"); put("dmStake", STAKE * eq * (1 - comb), "{:.2f}"); pct("dmStakeShare", STAKE, 0)
    # simulation of the lookback at three restriction periods and two volatilities
    pts = []
    for sig in (0.30, 0.60):
        for T in (1.0, 2.0, 4.0):
            m, se = L.max_gbm_mc(T, sig, 200_000, 48, rng)
            ex = L.longstaff_bound(T, sig); pts.append((sig, T, m, se, (m - ex) / se))
    R.RAW["dmMcZ"] = [p[4] for p in pts]; put("dmMcMaxZ", max(abs(p[4]) for p in pts), "{:.1f}")
    put("dmMcPaths", 200_000, "{:,.0f}"); put("dmMcSteps", 48, "{:.0f}")
    put("dmMcSixFour", pts[-1][2] * 100, "{:.1f}"); put("dmMcSixFourSE", pts[-1][3] * 100, "{:.1f}")
    pct("dmLongSixFour", L.longstaff_bound(4.0, 0.60), 1)
    for sig, nm in [(0.30, "Thirty"), (0.60, "Sixty")]:
        pct(f"dmCh{nm}One", L.chaffe_dlom(1.0, sig, r=rf), 1); pct(f"dmCh{nm}Four", L.chaffe_dlom(4.0, sig, r=rf), 1)
        pct(f"dmLs{nm}One", L.longstaff_bound(1.0, sig), 1); pct(f"dmLs{nm}Four", L.longstaff_bound(4.0, sig), 1)
    # Finnerty's average-strike put, against a simulation of the same payoff
    fin = L.finnerty_dlom(T_CASE, SIG_CASE); pct("dmFin", fin, 1)
    m, se = L.average_strike_mc(T_CASE, SIG_CASE, 200_000, 250, rng)
    pct("dmFinMc", m, 1); put("dmFinMcSE", 100 * se, "{:.2f}")
    for sig, nm in [(0.30, "Thirty"), (0.60, "Sixty")]:
        pct(f"dmFin{nm}One", L.finnerty_dlom(1.0, sig), 1); pct(f"dmFin{nm}Four", L.finnerty_dlom(4.0, sig), 1)
    m4, se4 = L.average_strike_mc(4.0, 0.60, 200_000, 250, rng)
    pct("dmFinMcSixFour", m4, 1)
    # the closed form is below the simulated value, by more at higher variance
    assert fin < m and L.finnerty_dlom(4.0, 0.60) < m4 and m4 - L.finnerty_dlom(4.0, 0.60) > m - fin
    assert fin < chaffe0 < ls
    put("dmStakeFin", STAKE * eq * (1 - L.combined_discount(dloc, fin)), "{:.2f}")
    grid(dloc, STAKE * eq, rf)
    # the put can never exceed the share price, the lookback value has no ceiling
    Ts = np.linspace(0.05, 5, 100)
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    for sig, col in [(0.30, SKY), (0.60, BLUE)]:
        ax.plot(Ts, [100 * L.chaffe_dlom(T, sig, r=rf) for T in Ts], color=col, ls="--", label=f"protective put, {int(100 * sig)}% volatility")
        ax.plot(Ts, [100 * L.longstaff_bound(T, sig) for T in Ts], color=col if sig < 0.5 else RED, label=f"lookback value, {int(100 * sig)}% volatility")
    for sig, T, m, se, z in pts:
        ax.plot(T, 100 * m, "o", color="black", ms=3)
    ax.plot([], [], "o", color="black", ms=3, label="simulated lookback")
    ax.set_xlabel("restriction period (years)"); ax.set_ylabel("value (% of the share price)")
    ax.set_ylim(0, 1.08 * 100 * L.longstaff_bound(Ts[-1], 0.60)); ax.legend(loc="upper left", fontsize=7.5)
    R.save(fig, "fig14_dlom")
    return finish("discounts")


SIGS = ((0.30, "Thirty"), (0.45, "FortyFive"), (0.60, "Sixty"))
TS = ((1.0, "One"), (2.0, "Two"), (4.0, "Four"))


def grid(dloc, ctl, rf):
    """Value of the stake, in millions, with the DLOC fixed and the DLOM from the protective put or the
    lookback value, for three volatilities and three restriction periods. A lookback value of 100% or
    more of the share price is not a usable discount and is reported as missing."""
    n_missing = 0
    for sig, sn in SIGS:
        for T, tn in TS:
            p = L.chaffe_dlom(T, sig, r=rf); lb = L.longstaff_bound(T, sig)
            put(f"mgPut{sn}{tn}", ctl * (1 - L.combined_discount(dloc, p)), "{:.2f}")
            if lb < 1:
                put(f"mgLb{sn}{tn}", ctl * (1 - L.combined_discount(dloc, lb)), "{:.2f}")
            else:
                R.N[f"mgLb{sn}{tn}"] = "n.a."; n_missing += 1
            R.RAW[f"mgLbRaw{sn}{tn}"] = float(lb)
    put("mgMissing", n_missing, "{:.0f}")
    (R.HERE / "out" / "minority_inputs.csv").write_text(f"control_value,dloc,rf\n{ctl!r},{dloc!r},{rf!r}\n")


if __name__ == "__main__":
    main()
