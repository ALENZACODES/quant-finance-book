"""Section 1.14: case study on daily Government of Canada five year yields.

Run on its own:  python code/ch01/goc_case.py   (writes code/ch01/out/goc_case.json)
run_all.py runs every script of the chapter, in a fixed order with one random generator, and writes
chapters/ch01-numbers.tex.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from numbers_ch01 import N, RAW, put, pct, save, finish, L, plt, BLUE, RED, GREY, SKY, ROSE, PALE, ROOT, SEED  # noqa: F401


SAMPLE_END = "2026-09-30"   # the book's data cut; the archived file itself is used unchanged


# =============================================================== 1.15 case study: GoC 5 year yield
def par_bond_duration_convexity(y, n_years=5, freq=2):
    """Modified duration and convexity of a par bond (coupon = yield), per unit of yield."""
    c = y / freq
    k = np.arange(1, n_years * freq + 1)
    cf = np.full(k.shape, c); cf[-1] += 1.0
    disc = (1 + y / freq) ** (-k)
    P = np.sum(cf * disc)
    t = k / freq
    mac = np.sum(t * cf * disc) / P
    dmod = mac / (1 + y / freq)
    conv = np.sum(cf * disc * t * (t + 1 / freq)) / (P * (1 + y / freq) ** 2)
    return dmod, conv


def goc_case():
    df = pd.read_csv(ROOT / "data" / "daily_policy_bond.csv", parse_dates=["date"])
    N["gocFileEnd"] = df.date.max().strftime("%-d %B %Y")
    # the archived file runs past the book's data cut; the sample ends with September 2026
    df = df[df.date <= SAMPLE_END]
    df = df.rename(columns={"BD.CDN.5YR.DQ.YLD": "y5", "V39079": "policy"}).dropna(subset=["y5"])
    df = df.sort_values("date").reset_index(drop=True)
    put("gocN", len(df), "{:,d}"); N["gocStart"] = df.date.iloc[0].strftime("%-d %B %Y")
    N["gocEnd"] = df.date.iloc[-1].strftime("%-d %B %Y")
    put("gocYmin", df.y5.min()); put("gocYmax", df.y5.max())
    N["gocYminDate"] = df.date[df.y5.idxmin()].strftime("%B %Y"); N["gocYmaxDate"] = df.date[df.y5.idxmax()].strftime("%B %Y")
    dy = df.y5.diff().dropna().values * 100     # basis points
    s = L.sample_moments(dy)
    put("gocDN", s["n"], "{:,d}"); put("gocDmean", s["mean"], "{:.3f}"); put("gocDsd", s["sd"], "{:.2f}")
    put("gocDskew", s["skew"]); put("gocDexk", s["exkurt"])
    jb, pv = L.jarque_bera(dy); put("gocJB", jb, "{:,.0f}"); RAW["gocJBp"] = pv
    rho = L.acf(dy, 5); put("gocRhoOne", rho[1], "{:.3f}"); put("gocBand", 1.96 / np.sqrt(len(dy)), "{:.3f}")
    put("gocDW", L.durbin_watson(dy - dy.mean()), "{:.3f}")
    put("gocMaxUp", dy.max(), "{:.0f}"); put("gocMaxDn", -dy.min(), "{:.0f}")
    N["gocMaxUpDate"] = df.date.iloc[1:].values[np.argmax(dy)].astype("datetime64[D]").item().strftime("%-d %B %Y")
    N["gocMaxDnDate"] = df.date.iloc[1:].values[np.argmin(dy)].astype("datetime64[D]").item().strftime("%-d %B %Y")
    put("gocBeyondFour", np.sum(np.abs(dy - s["mean"]) > 4 * s["sd"]), "{:d}")
    put("gocExpectFour", len(dy) * 2 * stats.norm.sf(4), "{:.2f}")
    # approximate daily total return of a constant maturity 5 year par bond
    y = df.y5.values / 100; dt = np.diff(df.date.values).astype("timedelta64[D]").astype(float) / 365.0
    D, C = np.vectorize(par_bond_duration_convexity)(y[:-1])
    dyy = np.diff(y)
    ret = y[:-1] * dt - D * dyy + 0.5 * C * dyy ** 2
    sr = L.sample_moments(ret)
    pct("gocRmean", sr["mean"] * 252, 2); pct("gocRsd", sr["sd"] * np.sqrt(252), 2)
    put("gocRskew", sr["skew"]); put("gocRexk", sr["exkurt"]); put("gocDur", np.median(D), "{:.2f}")
    jbr, _ = L.jarque_bera(ret); put("gocRJB", jbr, "{:,.0f}")
    # square root of time scaling, non-overlapping sums of h daily changes
    hs = [1, 2, 5, 10, 21, 63, 126]
    ratios, ses = [], []
    for h in hs:
        m = len(dy) // h
        agg = dy[: m * h].reshape(m, h).sum(1)
        ratios.append(agg.std(ddof=1) / (np.sqrt(h) * s["sd"]))
        # the ratio is exactly one at h = 1 by construction, so it has no sampling error there
        ses.append(0.0 if h == 1 else ratios[-1] / np.sqrt(2 * (m - 1)))
    RAW["gocRatios"] = dict(zip(hs, ratios))
    put("gocRatioMonth", ratios[hs.index(21)], "{:.3f}"); put("gocRatioQuarter", ratios[hs.index(63)], "{:.3f}")
    put("gocRatioWeek", ratios[hs.index(5)], "{:.3f}")
    put("gocSeMonth", ses[hs.index(21)], "{:.3f}"); put("gocSeQuarter", ses[hs.index(63)], "{:.3f}")
    # autocorrelations that sum to zero (rho_1 = 0.1, rho_2 = -0.1, none beyond) still bend the rule
    vr = lambda q: 1 + 2 * sum((1 - k / q) * r for k, r in ((1, 0.1), (2, -0.1)) if k < q)
    put("vrExTwo", vr(2), "{:.2f}"); put("vrExThree", vr(3), "{:.2f}"); put("vrExTen", vr(10), "{:.2f}")
    # policy rate context
    pol = df.dropna(subset=["policy"])
    put("polNow", pol.policy.iloc[-1], "{:.2f}")
    N["polStart"] = pol.date.iloc[0].strftime("%-d %B %Y"); put("polFirst", pol.policy.iloc[0], "{:.2f}")
    import hashlib
    digest = hashlib.sha256((ROOT / "data" / "daily_policy_bond.csv").read_bytes()).hexdigest()
    N["gocSha"] = digest[:12]; N["gocShaA"] = digest[:32]; N["gocShaB"] = digest[32:]

    fig, ax = plt.subplots(2, 1, figsize=(6.2, 3.8), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax[0].plot(df.date, df.y5, color=BLUE, lw=1.0, label="5 year GoC benchmark yield")
    ax[0].step(pol.date, pol.policy, color=RED, lw=0.9, where="post", label="target overnight rate")
    ax[0].set_ylabel("percent"); ax[0].legend(loc="upper left", fontsize=7.5)
    ax[1].plot(df.date.iloc[1:], dy, color=GREY, lw=0.4)
    ax[1].set_ylabel("daily change (bp)")
    fig.tight_layout(); save(fig, "fig14_goc_series")

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.7))
    z = (dy - s["mean"]) / s["sd"]
    ax[0].hist(z, bins=120, density=True, color=PALE, edgecolor=BLUE, linewidth=0.3, range=(-6, 6))
    xx = np.linspace(-6, 6, 400); ax[0].plot(xx, stats.norm.pdf(xx), color=RED, lw=1.1, label="standard normal")
    ax[0].set_xlabel("standardised daily change"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=7.5)
    ax[0].set_title("(a) Histogram")
    q = np.sort(z); th = stats.norm.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q))
    ax[1].plot(th, q, "o", ms=1.5, color=BLUE); ax[1].plot([-4, 4], [-4, 4], color=RED, lw=1)
    ax[1].set_xlabel("normal quantile"); ax[1].set_ylabel("sample quantile"); ax[1].set_title("(b) Q-Q plot")
    fig.tight_layout(); save(fig, "fig15_goc_distribution")

    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax[0].errorbar(hs, ratios, yerr=2 * np.array(ses), color=BLUE, marker="o", ms=3.5, capsize=2, lw=1)
    ax[0].axhline(1, color=RED, ls=":", lw=1); ax[0].set_xscale("log")
    ax[0].set_xticks(hs); ax[0].set_xticklabels([str(h) for h in hs])
    ax[0].set_xlabel("horizon $h$ (trading days)"); ax[0].set_ylabel(r"$\hat\sigma_h / (\sqrt{h}\,\hat\sigma_1)$")
    ax[0].set_title(r"(a) Square root of time check")
    k = np.arange(1, 11); ra = L.acf(dy, 10)[1:]
    ax[1].bar(k, ra, color=BLUE, width=0.6); b = 1.96 / np.sqrt(len(dy))
    ax[1].axhspan(-b, b, color=PALE, zorder=0); ax[1].set_xlabel("lag (days)"); ax[1].set_title("(b) Autocorrelation")
    fig.tight_layout(); save(fig, "fig16_goc_scaling")


def main(rng=None):
    goc_case()
    return finish('goc_case')


if __name__ == "__main__":
    main()
