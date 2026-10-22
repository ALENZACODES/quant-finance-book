"""Section 9.7: fee mechanics. Hurdles (none, hard, soft with catch-up) and their expected fees, the
one-period incentive fee as a call option on the fund's assets, and the high-water mark identity, which
says the incentive fees paid over any horizon are f/(1-f) times the rise in the high-water mark. A
simulation compares the value of ten years of fees with and without a high-water mark.
Writes fig11_fees.png."""
import math
import numpy as np
import numbers_ch09 as R
from numbers_ch09 import L, plt, BLUE, RED, GREY, SKY, put, pct, sci, finish

F, H = 0.20, 0.05          # incentive rate and hurdle
MU, SIG = 0.08, 0.12       # gross annual return of the fund in the hurdle example
RF, YEARS = 0.03, 10       # bill rate and horizon for the option values
VOLS = np.linspace(0.05, 0.30, 11)


def main():
    rng = R.rng_for(11)
    pct("feF", F, 0); pct("feH", H, 0); pct("feMu", MU, 0); pct("feSig", SIG, 0)
    pct("feCatch", H / (1 - F), 2)
    # expected fees under each hurdle rule, closed form and simulation
    x = MU + SIG * rng.standard_normal(1_000_000)
    for kind, tag in (("none", "None"), ("hard", "Hard"), ("soft", "Soft")):
        pct(f"fe{tag}", L.hurdle_fee_mean(MU, SIG, F, H, kind), 2)
        pct(f"fe{tag}Sim", L.hurdle_fee(x, F, H, kind).mean(), 2)
    put("feSimSe", 100 * L.hurdle_fee(x, F, H, "soft").std() / math.sqrt(len(x)), "{:.3f}")
    # one year fee as a call: at the money, and at the money forward (a hurdle at the bill rate)
    pct("feRf", RF, 0); pct("feOptVol", 0.10, 0)
    pct("feOptAtm", L.fee_option(F, 1.0, 0.10, RF, 1.0), 2)
    pct("feOptFwd", L.fee_option(F, math.exp(RF), 0.10, RF, 1.0), 2)
    pct("feOptApprox", F * 0.10 / math.sqrt(2 * math.pi), 2)
    # ten years of annual fees, priced by simulation under the pricing measure: reset versus high-water mark
    n = 40_000
    put("feSims", n, "{:,.0f}")
    Z = rng.standard_normal((n, YEARS))
    reset_v, hwm_v, strip_v = [], [], []  # strip_v: closed form
    disc = np.exp(-RF * np.arange(1, YEARS + 1))
    maxgap = 0.0
    for s in VOLS:
        G = np.exp((RF - s ** 2 / 2) + s * Z) - 1
        fr = np.empty((n, YEARS)); fh = np.empty((n, YEARS)); rise = np.empty(n)
        for i in range(n):
            fr[i], _, _ = L.incentive_path(G[i], F, hwm=False)
            fh[i], _, mark = L.incentive_path(G[i], F, hwm=True)
            rise[i] = mark - 1
        maxgap = max(maxgap, np.max(np.abs(fh.sum(axis=1) - F / (1 - F) * rise)))
        reset_v.append((fr * disc).sum(axis=1).mean()); hwm_v.append((fh * disc).sum(axis=1).mean())
        c = L.fee_option(F, 1.0, s, RF, 1.0)
        strip_v.append(1 - (1 - c) ** YEARS)   # closed form for the fee on each year's gain
        if abs(s - 0.15) < 1e-9:
            put("feTenResetSe", 100 * (fr * disc).sum(axis=1).std() / math.sqrt(n), "{:.2f}")
            pct("feOneCall", c, 2)
    reset_v, hwm_v, strip_v = map(np.array, (reset_v, hwm_v, strip_v))
    sci("feIdGap", max(maxgap, 1e-16), 0)
    RAW_i = int(np.argmin(np.abs(VOLS - 0.15)))
    pct("feTenVol", VOLS[RAW_i], 0); put("feYears", YEARS, "{:.0f}")
    pct("feTenReset", reset_v[RAW_i], 1); pct("feTenHwm", hwm_v[RAW_i], 1); pct("feTenClosed", strip_v[RAW_i], 1)
    pct("feTenRatio", hwm_v[RAW_i] / reset_v[RAW_i], 0)
    # worked high-water mark example: up 20%, down 25%, up 30%
    path = [0.20, -0.25, 0.30]
    fh, nav_h, mark = L.incentive_path(path, F, hwm=True); fr, nav_r, _ = L.incentive_path(path, F, hwm=False)
    pct("feExFeeH", fh.sum(), 2); pct("feExFeeR", fr.sum(), 2); pct("feExMark", mark - 1, 2)
    pct("feExNavH", nav_h - 1, 2); pct("feExNavR", nav_r - 1, 2)
    # figure
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.6))
    g = np.linspace(-0.05, 0.15, 401)
    ax[0].plot(100 * g, 100 * L.hurdle_fee(g, F, H, "soft"), color=RED, label="soft, with catch-up")
    ax[0].plot(100 * g, 100 * L.hurdle_fee(g, F, H, "none"), color=GREY, ls="--", lw=0.9, label="no hurdle")
    ax[0].plot(100 * g, 100 * L.hurdle_fee(g, F, H, "hard"), color=BLUE, label="hard")
    ax[0].set_xlabel("gross return, %"); ax[0].set_ylabel("incentive fee, % of assets"); ax[0].legend(loc="upper left")
    ax[1].plot(100 * VOLS, 100 * reset_v, color=RED, label="fee on each year's gain")
    ax[1].plot(100 * VOLS, 100 * hwm_v, color=BLUE, label="with a high-water mark")
    ax[1].plot(100 * VOLS, 100 * strip_v, color=GREY, ls="--", lw=0.8, label="formula, each year's gain")
    ax[1].set_xlabel("volatility of the fund, %"); ax[1].set_ylabel("value of fees, % of assets")
    ax[1].legend(loc="lower right", frameon=True, framealpha=1, edgecolor="none").set_zorder(5); ax[1].set_ylim(0, None)
    fig.tight_layout(); R.save(fig, "fig11_fees"); plt.close(fig)
    return finish("fees")


if __name__ == "__main__":
    print(main()["N"])
