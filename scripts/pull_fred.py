"""Pull the daily FRED series used in Chapter 9 and build the chapter's daily market file.

Usage: python scripts/pull_fred.py [END]
Saves each series, cut at END, as data/raw/fred_<ID>.csv, then writes
  data/fred_markets_daily.csv  US 10 year Treasury yield, WTI crude oil, EUR, JPY and CAD rates (public domain series)
  data/crypto_daily.csv        Coinbase bitcoin and ether prices (not included in the public copy)
both ending at END (default 2026-09-30), and records the access date in data/SOURCES.md.
"""
import io, sys, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKETS = ["DGS10", "DCOILWTICO", "DEXUSEU", "DEXJPUS", "DEXCAUS"]
CRYPTO = ["CBBTCUSD", "CBETHUSD"]
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"


def pull(s, end):
    raw = ROOT / "data" / "raw" / f"fred_{s}.csv"
    txt = urllib.request.urlopen(URL.format(s=s), timeout=60).read().decode()
    head, *rows = txt.strip().split("\n")
    txt = "\n".join([head] + [r for r in rows if r[:10] <= end]) + "\n"  # nothing after END is kept
    raw.write_text(txt)
    df = pd.read_csv(io.StringIO(txt), na_values=["."])
    df.columns = ["date", s]
    return df.assign(date=pd.to_datetime(df["date"])).set_index("date")[s]


def build(series, start, end, path):
    df = pd.concat([pull(s, end) for s in series], axis=1).loc[start:end].dropna(how="all")
    df.index.name = "date"
    df.to_csv(path, float_format="%.6g")
    return df


def main(end="2026-09-30"):
    m = build(MARKETS, "1999-01-01", end, ROOT / "data" / "fred_markets_daily.csv")
    c = build(CRYPTO, "2014-12-01", end, ROOT / "data" / "crypto_daily.csv")
    today = dt.date.today().isoformat()
    notes = {
        "fred_markets_daily.csv": (f"FRED, Federal Reserve Bank of St. Louis: {', '.join(MARKETS)} (market yield on 10 year "
                                   "Treasury securities, WTI crude oil spot price from the US Energy Information "
                                   "Administration, and the Federal Reserve H.10 noon buying rates for EUR, JPY and CAD), "
                                   f"daily, {m.index[0].date()} to {m.index[-1].date()}, accessed {today}. Built by "
                                   "`scripts/pull_fred.py`, which also saves each series, cut at the end date, in `raw/`."),
        "crypto_daily.csv": ("(not included in the public copy; `python scripts/pull_fred.py` rebuilds it) FRED series "
                             f"{', '.join(CRYPTO)}, Coinbase bitcoin and ether prices in US dollars, daily, "
                             f"{c.index[0].date()} to {c.index[-1].date()}, accessed {today}."),
    }
    p = ROOT / "data" / "SOURCES.md"
    lines = [l for l in p.read_text().splitlines(True) if not any(l.startswith(f"- `{k}`") for k in notes)]
    p.write_text("".join(lines).rstrip("\n") + "\n" + "".join(f"- `{k}`: {v}\n" for k, v in notes.items()))
    print(len(m), len(c))


if __name__ == "__main__":
    main(*sys.argv[1:])
