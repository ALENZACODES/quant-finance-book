"""Tests for case_fx_hedge.py: the data file, the bookkeeping, and a synthetic check."""
import math
import numpy as np
import pandas as pd
from conftest import close, HERE
import case_fx_hedge as C

ROOT = HERE.parents[1]


def test_data_file():
    df = pd.read_csv(ROOT / "data" / "usdcad_rates.csv", parse_dates=["date"])
    assert df.date.is_monotonic_increasing and df.date.is_unique and len(df) > 4800
    assert set(df.usdcad_source) == {"noon", "average"}
    assert df.usdcad.between(0.9, 1.7).all()
    assert (df.date[df.usdcad_source == "noon"] < "2017-01-01").all()


def test_case_numbers(RAW):
    cs = pd.read_csv(HERE / "out" / "case_months.csv")
    assert len(cs) == RAW["fxN"]
    close(cs.pnl.mean(), RAW["fxPnlMean"], 1e-9); close(cs.pnl.sum(), RAW["fxCum"], 1e-6)
    assert RAW["fxCorrGT"] > 0.9 and 0 < RAW["fxSlope"] < 1.2
    assert RAW["fxNNoon"] + RAW["fxNAvg"] == RAW["fxN"]
    assert RAW["acRetAvg"] > RAW["acRetNoon"]


def test_synthetic_constant_volatility():
    """On simulated GBM with constant volatility and rates, the trailing estimate is right on average
    and the hedge error is small relative to the premium."""
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2010-01-01", "2015-12-31")
    sig = 0.08; dt = 1 / 260
    S = 1.2 * np.exp(np.cumsum(sig * math.sqrt(dt) * rng.standard_normal(len(dates)) - 0.5 * sig ** 2 * dt))
    df = pd.DataFrame(dict(usdcad=S, cad_tbill_1m=1.0, us_tbill_1m=1.0, usdcad_source="noon"), index=dates)
    d = df.index; me = df.groupby([d.year, d.month]).tail(1).index; pos = [d.get_loc(x) for x in me]
    rows = [C.run_month(df, a, b) for a, b in zip(pos[:-1], pos[1:]) if a >= C.WINDOW]
    cs = pd.DataFrame(rows)
    assert abs(cs.pnl.mean()) < 4 * cs.pnl.std() / math.sqrt(len(cs))
    assert np.corrcoef(cs.pnl, cs.gts)[0, 1] > 0.9 and cs.pnl.std() < 0.5 * cs.prem.mean()


def test_only_complete_months(NUM):
    """The last option closes on the last print of a finished month, never on a part month: the file
    is cut at the end of September 2026, so its last date closes the last option."""
    df = C.load(); end = df.index[-1]
    last_close = pd.Timestamp(NUM["fxLastClose"]); last_sale = pd.Timestamp("1 " + NUM["fxLast"])
    assert end + pd.offsets.MonthEnd(0) <= pd.Timestamp(C.DATA_END) and last_close == end
    assert last_sale.month != last_close.month and last_sale < last_close


def test_data_file_rebuilds_from_raw_downloads():
    import subprocess, sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_usdcad.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0 and "identical" in r.stdout


def test_month_count_matches_output(RAW):
    import pandas as pd
    from pathlib import Path
    cs = pd.read_csv(Path(__file__).parent / "out" / "case_months.csv")
    assert len(cs) == RAW["fxN"] == 230
