"""Build the two monthly files used in Chapter 7 from the Kenneth R. French Data Library downloads
kept in data/raw/french/ (the CSV files inside the published zip archives, unchanged).

  F-F_Research_Data_5_Factors_2x3.csv  market excess return, SMB, HML, RMW, CMA and the one month bill rate
  F-F_Momentum_Factor.csv              the momentum factor
  10_Industry_Portfolios.csv           value-weighted returns of ten industry portfolios, number of firms
                                       and average firm size
  F-F_Research_Data_Factors_daily.csv  daily market excess return and bill rate (Chapter 8)

Writes data/ff_factors_monthly.csv (month, mkt_rf, smb, hml, rmw, cma, mom, rf) and
data/industry10_monthly.csv (month, the ten value-weighted returns, ew_<industry> for the equal-weighted
returns, then cap_<industry> = number of firms times average
firm size in millions of dollars at the start of the month), both in percent per month, and
data/ff_daily.csv (date, mkt_rf, rf, mkt = mkt_rf + rf), in percent per day, 1 July 1963 to 31 August
2026.
Months run from July 1963, the first month of the five factors, to August 2026 (END below), the last
month of the release used in the book.

The raw files are not redistributed with the public copy of the book. Fetch them first with
  python scripts/build_french.py --download
which saves the CSV files from the library's zip archives into data/raw/french/, then build with
  python scripts/build_french.py
The library revises past months from time to time, so a later download can differ slightly from the
release used here (built from the CRSP database of August 2026).
Run:  python scripts/build_french.py [--download] [--check]
"""
from __future__ import annotations
import io, sys, urllib.request, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data" / "raw" / "french"
F_OUT = ROOT / "data" / "ff_factors_monthly.csv"; I_OUT = ROOT / "data" / "industry10_monthly.csv"
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FILES = ["F-F_Research_Data_5_Factors_2x3", "F-F_Momentum_Factor", "10_Industry_Portfolios",
         "F-F_Research_Data_Factors", "F-F_Research_Data_Factors_daily"]
D_OUT = ROOT / "data" / "ff_daily.csv"
D_START, D_END = 19630701, 20260831
END = 202608


def download():
    """Save the CSV file inside each zip archive, unchanged, into data/raw/french/."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        with urllib.request.urlopen(URL.format(name), timeout=60) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        member = next(m for m in z.namelist() if m.lower().endswith(".csv"))
        (RAW / f"{name}.csv").write_bytes(z.read(member))
        print("saved", RAW / f"{name}.csv")


def block(path, title=None):
    """The first monthly table of a French CSV file, or the table that follows the line `title`."""
    lines = path.read_text().splitlines()
    start = 0
    if title is not None:
        start = next(i for i, l in enumerate(lines) if l.strip() == title) + 1
    while not lines[start].lstrip().startswith(","):
        start += 1
    end = start + 1
    while end < len(lines) and lines[end].strip()[:6].isdigit() and len(lines[end].split(",")[0].strip()) == 6:
        end += 1
    df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), skipinitialspace=True)
    df = df.rename(columns={df.columns[0]: "month"})
    df["month"] = df["month"].astype(int)
    return df.set_index("month")


def build():
    f5 = block(RAW / "F-F_Research_Data_5_Factors_2x3.csv")
    mom = block(RAW / "F-F_Momentum_Factor.csv")
    f = f5.join(mom, how="inner")
    f.columns = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]
    f = f[["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]]
    ind = block(RAW / "10_Industry_Portfolios.csv", "Average Value Weighted Returns -- Monthly")
    nf = block(RAW / "10_Industry_Portfolios.csv", "Number of Firms in Portfolios")
    sz = block(RAW / "10_Industry_Portfolios.csv", "Average Firm Size")
    ew = block(RAW / "10_Industry_Portfolios.csv", "Average Equal Weighted Returns -- Monthly").add_prefix("ew_")
    cap = (nf * sz).add_prefix("cap_")
    i = ind.join(ew, how="inner").join(cap, how="inner")
    months = f.index.intersection(i.index)
    months = months[(months >= 196307) & (months <= END)]
    f, i = f.loc[months], i.loc[months]
    assert (f.abs() < 99).all().all() and (i.iloc[:, :20] > -99).all().all()
    return f, i


def daily():
    """Daily market excess return and bill rate (Chapter 8), percent per day, with the market return."""
    lines = (RAW / "F-F_Research_Data_Factors_daily.csv").read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(","))
    rows = [l for l in lines[start + 1:] if l.strip()[:8].isdigit() and len(l.split(",")[0].strip()) == 8]
    d = pd.read_csv(io.StringIO("\n".join([lines[start]] + rows)), skipinitialspace=True)
    d = d.rename(columns={d.columns[0]: "date", "Mkt-RF": "mkt_rf", "RF": "rf"})[["date", "mkt_rf", "rf"]]
    d = d[(d["date"] >= D_START) & (d["date"] <= D_END)].copy()
    d["mkt"] = d["mkt_rf"] + d["rf"]
    d["date"] = pd.to_datetime(d["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    assert d["mkt_rf"].abs().max() < 30
    return d.set_index("date").round(4)


def main(check=False):
    missing = [n for n in FILES if n != "F-F_Research_Data_Factors" and not (RAW / f"{n}.csv").exists()]
    if missing:
        sys.exit("raw files missing from data/raw/french/; run python scripts/build_french.py --download")
    f, i = build()
    if check:
        old_f = pd.read_csv(F_OUT, index_col=0); old_i = pd.read_csv(I_OUT, index_col=0)
        assert (old_f.round(4).values == f.round(4).values).all() and (old_i.round(4).values == i.round(4).values).all()
        assert (pd.read_csv(D_OUT, index_col=0).values == daily().values).all()
        print("data files match the raw downloads"); return
    f.round(4).to_csv(F_OUT); i.round(4).to_csv(I_OUT)
    d = daily(); d.to_csv(D_OUT)
    print(f"{len(d)} trading days, {d.index[0]} to {d.index[-1]}")
    print(f"{len(f)} months, {f.index[0]} to {f.index[-1]}")


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    main("--check" in sys.argv)
