"""Loaders for the data files that ship with the code, used by the computing projects at the end of
each chapter. Every series ends at September 2026. None of these files needs a pull script."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
END = "2026-09-30"


def _read(name):
    d = pd.read_csv(DATA / name, index_col=0, parse_dates=True)
    return d.loc[:END]


def month_end(s):
    s = s.dropna()
    return s.groupby(s.index.to_period("M")).last()


def par_bond_return(y0, y1, m=10.0):
    """Monthly return of a par bond bought at yield y0 (decimal, compounded twice a year) with m years
    to run, priced a month later at y1, including the month's accrued coupon."""
    n = 2 * (m - 1 / 12); v = (1 + y1 / 2) ** (-n)
    return y0 / y1 * (1 - v) + v - 1 + y0 / 12


def goc():
    """Daily Government of Canada bill and bond yields, in per cent."""
    return _read("goc_bills_bonds.csv")


def fred_markets():
    """Daily US 10 year yield, WTI crude and three exchange rates."""
    return _read("fred_markets_daily.csv")


def usdcad():
    """Daily USD/CAD (Canadian dollars per US dollar) with one month bill rates in both countries."""
    return _read("usdcad_rates.csv")


def eia():
    """Monthly averages of the first four crude oil and natural gas futures, with inventories."""
    return _read("eia_curves_monthly.csv")


def real_assets_monthly():
    return _read("real_assets_monthly.csv")


def real_assets_quarterly():
    return _read("real_assets_quarterly.csv")


def market_returns():
    """Monthly excess returns over bills of six markets from month-end prices: WTI crude and the
    euro, yen and Canadian dollar against the US dollar are spot returns (no carry or roll); the US and
    Canadian 10 year bonds are par bond returns. The same construction as code/ch12/snooping.py."""
    d, g, ra = fred_markets(), goc(), real_assets_monthly()
    rf = ra["tbill"] / 1200; rf.index = rf.index.to_period("M")
    y, yc = month_end(d["DGS10"]) / 100, month_end(g["BD.CDN.10YR.DQ.YLD"]) / 100
    bc = month_end(g["TB.CDN.90D.MID"]) / 100
    oil, eur, jpy, cad = (month_end(d[c]) for c in ("DCOILWTICO", "DEXUSEU", "DEXJPUS", "DEXCAUS"))
    out = pd.DataFrame({
        "OIL": oil / oil.shift(1) - 1 - rf.shift(1),
        "UST": par_bond_return(y.shift(1), y) - rf.shift(1),
        "GOC": par_bond_return(yc.shift(1), yc) - bc.shift(1) / 12,
        "EUR": eur / eur.shift(1) - 1, "JPY": jpy.shift(1) / jpy - 1, "CAD": cad.shift(1) / cad - 1})
    return out[out.index <= pd.Period("2026-09", "M")].dropna()
