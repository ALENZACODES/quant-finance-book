"""Build data/usdcad_rates.csv from the three raw downloads in data/raw/.

  boc_IEXE0101.csv                 Bank of Canada Valet, legacy noon rate IEXE0101 (USD in CAD)
  boc_FXUSDCAD_TB.CDN.30D.MID.csv  Bank of Canada Valet, daily average FXUSDCAD and 1-month bill TB.CDN.30D.MID
  fred_DGS1MO.csv                  FRED, US 1-month constant maturity Treasury yield DGS1MO

The exchange rate is the noon rate up to 31 December 2016 and the daily average from 2017.
Rows are the dates with an exchange rate; a missing bill yield is left empty (the chapter's code
carries the last value forward). Run:  python scripts/build_usdcad.py [--check]
"""
from __future__ import annotations
import io, sys
from pathlib import Path
import pandas as pd

END = "2026-09-30"
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data" / "raw"; OUT = ROOT / "data" / "usdcad_rates.csv"


def valet(path):
    text = path.read_text()
    return pd.read_csv(io.StringIO(text[text.index('"OBSERVATIONS"') + len('"OBSERVATIONS"'):].lstrip()),
                       parse_dates=["date"]).set_index("date")


def build():
    noon = valet(RAW / "boc_IEXE0101.csv")["IEXE0101"]
    fx = valet(RAW / "boc_FXUSDCAD_TB.CDN.30D.MID.csv")
    us = pd.read_csv(RAW / "fred_DGS1MO.csv", parse_dates=["observation_date"], na_values=["."]) \
        .set_index("observation_date")["DGS1MO"]
    a = pd.DataFrame(dict(usdcad=noon[noon.index < "2017-01-01"], usdcad_source="noon"))
    b = pd.DataFrame(dict(usdcad=fx["FXUSDCAD"][fx.index >= "2017-01-01"], usdcad_source="average"))
    df = pd.concat([a, b]).dropna(subset=["usdcad"])
    df["cad_tbill_1m"] = fx["TB.CDN.30D.MID"].reindex(df.index)
    df["us_tbill_1m"] = us.reindex(df.index)
    df = df[df.index <= END]                    # the book uses data to the end of September 2026 only
    df.index.name = "date"
    df["usdcad"] = df.usdcad.map("{:.4f}".format)
    for c in ("cad_tbill_1m", "us_tbill_1m"):
        df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return df


if __name__ == "__main__":
    df = build()
    s = df.to_csv(date_format="%Y-%m-%d")
    if "--check" in sys.argv:
        print("identical" if s == OUT.read_text() else "differs"); sys.exit(0 if s == OUT.read_text() else 1)
    OUT.write_text(s)
