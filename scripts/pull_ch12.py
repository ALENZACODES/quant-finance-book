"""Pull the public data used in Chapter 12 and build the chapter's data file.

Usage: python scripts/pull_ch12.py
  data/goyal_welch_monthly.csv  the monthly sheet of the equity premium prediction data of Amit Goyal and
                                Ivo Welch, updated to December 2025: S&P 500 index, twelve month dividends
                                and earnings, book to market, Treasury bill and long yields, AAA and BAA
                                yields, net equity expansion, the risk free rate, inflation, long bond and
                                corporate bond returns, stock variance and the CRSP value weighted S&P 500
                                return with dividends, from January 1926
The file is posted for research use on Amit Goyal's site but its return series come from CRSP and other
licensed sources, so the raw download (data/raw/goyal_welch_*) and this file are kept out of the public
copy; the script rebuilds them. Reading the .xlsx file needs openpyxl. Records the access date in
data/SOURCES.md.
"""
import re, sys, time, datetime as dt, urllib.request, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PAGE = "https://sites.google.com/view/agoyal145"
# "Updated data (up to 2025)" under Goyal and Welch (2008) on the page above, a Google Drive file
FILE_ID = "1qwpl2R_DNujpU5YUkk8lacP1tTeMb9iJ"
URL = f"https://docs.google.com/uc?export=download&id={FILE_ID}"
COLS = {"yyyymm": "month", "Index": "index", "D12": "d12", "E12": "e12", "b/m": "bm", "tbl": "tbl", "AAA": "aaa",
        "BAA": "baa", "lty": "lty", "ntis": "ntis", "Rfree": "rf", "infl": "infl", "ltr": "ltr", "corpr": "corpr",
        "svar": "svar", "CRSP_SPvw": "ret"}


def get(url, tries=4):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    raw = RAW / "goyal_welch_predictors.xlsx"
    if "--offline" not in sys.argv:
        raw.write_bytes(get(URL))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.read_excel(raw, sheet_name="Monthly")
    d = d[list(COLS)].rename(columns=COLS)
    d = d[(d.month >= 192601) & (d.month <= 202512)]
    d.to_csv(ROOT / "data" / "goyal_welch_monthly.csv", index=False)
    today = dt.date.today().isoformat()
    src = ROOT / "data" / "SOURCES.md"; s = src.read_text()
    s = re.sub(r"\n- `goyal_welch_monthly\.csv`.*", "", s)
    s = s.rstrip("\n") + (
        f"\n- `goyal_welch_monthly.csv`: Amit Goyal and Ivo Welch, equity premium prediction data, monthly sheet of "
        f"the update to December 2025 (Google Drive file {FILE_ID}, linked from {PAGE} as \"Updated data (up to "
        f"2025)\"), accessed {today}, January 1926 to December 2025. Not redistributed; built by "
        f"`scripts/pull_ch12.py`.\n")
    src.write_text(s)
    print(len(d), "months", d.month.min(), "to", d.month.max())


if __name__ == "__main__":
    main()
