import sys
import numpy as np
import pandas as pd


def weights(r, lookback=12, target=0.20, window=24):
    """Positions set at the end of month t, held through t + 1, from data up to t."""
    past = np.expm1(np.log1p(r).rolling(lookback, min_periods=lookback).sum())
    vol = r.rolling(window, min_periods=12).std() * np.sqrt(12)
    sig = np.sign(past)
    ok = sig.notna() & vol.notna() & (vol > 0)
    n = ok.sum(axis=1).replace(0, np.nan)   # N_t, markets with a signal and a vol
    return (sig * target / vol).where(ok).div(n, axis=0).fillna(0.0)


def programme(r, lookback=12, cost=0.0005, start=200101):
    w = weights(r, lookback)
    held = w.shift(1).fillna(0.0)             # the one-month lag: w_t earns r_{t+1}
    traded = (w - w.shift(1).fillna(0.0)).abs().shift(1).fillna(0.0)
    p = (held * r.fillna(0.0) - cost * traded).sum(axis=1)
    return p[p.index >= start]


def sharpe(x):
    return np.sqrt(12) * x.mean() / x.std(ddof=1)


if __name__ == "__main__":
    sys.path.insert(0, "code/ch09")
    from ch09lib import load_markets          # the seven excess return series
    r, _ = load_markets()
    for k in (1, 3, 12):
        sr = sharpe(programme(r, k))
        print(f"lookback {k:2d} months: Sharpe ratio {sr:.2f}")
