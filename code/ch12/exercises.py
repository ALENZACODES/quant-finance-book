"""Answers to the Chapter 12 exercises, computed from the formulas in the text.

Run on its own:  python code/ch12/exercises.py   (writes code/ch12/out/exercises.json)
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm
from numbers_ch12 import N, RAW, put, pct, save, finish, rng_for, L  # noqa: F401


def main():
    # 12.1 ridge and lasso with an orthonormal design
    put("zaRidge", 0.8 / (1 + 0.25), "{:.2f}"); put("zaLasso", 0.8 - 0.25, "{:.2f}"); put("zaLassoSmall", 0.0, "{:.0f}")
    # 12.2 shrinking a mean
    put("zbC", 1 / 1.5, "{:.3f}"); put("zbMse", 0.5 / 1.5, "{:.3f}")
    # 12.3 a forest
    put("zcVar", 0.3 + 0.7 / 100, "{:.3f}"); put("zcVarTen", 0.3 + 0.7 / 10, "{:.2f}")
    # 12.4 overlapping labels
    put("zdCorr", 8 / 12, "{:.3f}"); put("zdPurged", 11 + 11 + 6, "{:.0f}")
    # 12.5 backpropagation by hand
    a = math.tanh(0.5); yhat = 2 * a
    put("zeYhat", yhat, "{:.4f}"); put("zeWtwo", yhat * a, "{:.4f}"); put("zeWone", yhat * 2 * (1 - a ** 2), "{:.4f}")
    # 12.6 out-of-sample R^2
    pct("zfRtwo", 1 - 0.00190 / 0.00192, 2)
    # 12.7 Campbell and Thompson
    pct("zgGain", L.ct_gain(0.12, 0.005), 1)
    # 12.8 Clark and West
    put("zhGap", 0.00192 - 0.00190, "{:.5f}"); put("zhAdj", 0.00192 - 0.00190 + 0.00004, "{:.5f}")
    # 12.9 expected maximum
    put("ziMax", L.expected_max_approx(100), "{:.3f}"); put("ziSr", L.expected_max_approx(100) / math.sqrt(5), "{:.2f}")
    put("ziExact", L.expected_max_normal(100), "{:.3f}")
    # 12.10 Holm, Bonferroni, Benjamini-Hochberg
    p = np.array([0.001, 0.008, 0.012, 0.04, 0.2])
    put("zjHolm", int(L.holm(p, 0.05).sum()), "{:.0f}"); put("zjBonf", int((p <= 0.01).sum()), "{:.0f}")
    put("zjBh", int(L.bh(p, 0.05).sum()), "{:.0f}")
    # 12.11 probabilistic Sharpe ratio
    se = L.sharpe_se(0.25, 120, -0.5, 1.0)
    put("zkSe", se, "{:.4f}"); put("zkPsr", L.psr(0.25, 120, -0.5, 1.0), "{:.4f}"); put("zkZ", 0.25 / se, "{:.3f}")
    # 12.12 one-period minimum variance hedge
    put("zlH", 0.9 / 1.6, "{:.4f}"); pct("zlCut", 0.81 / (1.6 * 0.6), 1)
    return finish("exercises")


if __name__ == "__main__":
    main()
