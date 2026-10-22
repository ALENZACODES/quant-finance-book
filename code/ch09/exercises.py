"""Answers to the numerical exercises of Chapter 9."""
import math
import numpy as np
from scipy import integrate, stats
import numbers_ch09 as R
from numbers_ch09 import L, put, pct, finish


def main():
    # 9.1 straddle identity
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.04])
    S = np.cumsum(r); pnl = np.sum(np.concatenate([[0], S[:-1]]) * r)
    put("exAPnl", 1e4 * pnl, "{:.1f}"); put("exAS", 100 * S[-1], "{:.0f}"); put("exAQ", 1e4 * np.sum(r * r), "{:.0f}")
    # 9.2 sign rule
    pct("exBMonth", L.sign_rule_mean(0.05, 0.04), 2); put("exBSR", math.sqrt(12) * math.sqrt(2 / math.pi) * 0.05)
    # 9.3 combined Sharpe
    put("exCSR", L.combined_sharpe(0.2, 20, 0.1)); put("exCLim", 0.2 / math.sqrt(0.1))
    # 9.4 survivorship
    pct("exDMean", 12 * L.survivor_mean(0.005, 0.04, 24, 0.0), 1)
    pct("exDBias", 12 * (L.survivor_mean(0.005, 0.04, 24, 0.0) - 0.005), 1)
    pct("exDShare", stats.norm.sf((0 - 0.005) / (0.04 / math.sqrt(24))), 0)
    # 9.5 backfill
    b = 12 / 60 * (L.survivor_mean(0.005, 0.03, 12, 0.0) - 0.005)
    pct("exEBias", 12 * b, 2); pct("exEInc", 12 * L.survivor_mean(0.005, 0.03, 12, 0.0), 1)
    # 9.6 prediction inflation
    put("exFInf", L.prediction_inflation(10, 36), "{:.2f}"); put("exFT", 10 / 0.1 + 10 + 1, "{:.0f}")
    # 9.7 non-netting
    pct("exGFee", 0.5 * 0.2 * 0.30, 0); pct("exGNet", 0.2 * 0.5 * (0.30 - 0.10), 0)
    pct("exGRate", 0.5 * 0.2 * 0.30 / (0.5 * (0.30 - 0.10)), 0)
    # 9.8 risk share
    a, bb, c = 0.10 ** 2, 0.75 ** 2, 0.3 * 0.10 * 0.75
    pct("exHW", L.weight_for_share(0.25, a, bb, c), 1); pct("exHShare", L.risk_share(0.03, a, bb, c), 0)
    pct("exHWx", L.weight_for_share(0.25, a, bb, c), 3); pct("exHSharex", L.risk_share(0.03, a, bb, c), 2)
    # 9.9 fees on a trend fund, one year
    gross, rf = 0.09, 0.03
    net = (gross - 0.02) * 0.8
    pct("exINet", net, 1); pct("exINetEx", net - rf, 1); pct("exIRatio", (net - rf) / (gross - rf), 0)
    # 9.10 selection among coins: E max of 20 standard normals
    emax = integrate.quad(lambda z: z * 20 * stats.norm.pdf(z) * stats.norm.cdf(z) ** 19, -10, 10)[0]
    put("exJMax", emax, "{:.2f}"); pct("exJBest", 1.0 / math.sqrt(10) * emax, 0)
    # 9.11 high-water mark path
    fh, nav, mark = L.incentive_path([0.25, -0.20, 0.30], 0.20, hwm=True)
    fr, navr, _ = L.incentive_path([0.25, -0.20, 0.30], 0.20, hwm=False)
    pct("exKOne", fh[0], 1); pct("exKThree", fh[2], 2); pct("exKTot", fh.sum(), 2); pct("exKRise", mark - 1, 2)
    pct("exKReset", fr.sum(), 2); pct("exKResetThree", fr[2], 2)
    # 9.12 hurdles
    for g, tag in ((0.10, "Ten"), (0.06, "Six")):
        for kind, k in (("none", "N"), ("hard", "H"), ("soft", "S")):
            pct(f"exL{k}{tag}", float(L.hurdle_fee(g, 0.20, 0.05, kind)), 1)
    # 9.13 currency carry over one year
    s0, s1, ic, iu = 1.35, 1.33, 0.03, 0.045
    pct("exMEx", (1 + ic) * s0 / s1 - (1 + iu), 3); pct("exMSpot", s0 / s1 - 1, 3)
    pct("exMCarry", ic - iu, 1); pct("exMCross", ic * (s0 / s1 - 1), 3)
    # 9.14 catching up
    put("exNRuin", L.catch_up_ruin(0.2, 6), "{:.6f}"); put("exNNak", L.catch_up_nakamoto(0.2, 6), "{:.4f}")
    return finish("exercises")


if __name__ == "__main__":
    print(main()["N"])
