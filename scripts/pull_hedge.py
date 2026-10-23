"""Pull the public hedge fund data used in Chapter 9 and build the chapter's two data files.

Usage: python scripts/pull_hedge.py [--offline]
  data/ofr_hedge_funds_quarterly.csv  aggregated statistics of Qualifying Hedge Funds from SEC Form PF filings,
                                      published by the US Office of Financial Research in its Hedge Fund
                                      Monitor: net asset-weighted gross and net quarterly returns and
                                      leverage by strategy, investor and portfolio liquidity, the share of
                                      net assets that can be gated, gross and net assets and fund counts,
                                      quarterly from 2013Q1. A work of the US government; the OFR asks only
                                      for credit (https://www.financialresearch.gov/legal-notices/), so the
                                      file is shipped.
  data/fung_hsieh_ptfs.csv            the five trend-following risk factors of Fung and Hsieh (2001), returns
                                      of primitive trend-following strategies built from lookback straddles
                                      on bonds, currencies, commodities, short-term interest rates and stock
                                      indices, monthly from January 1994, from David A. Hsieh's data library.
                                      The data are made available for academic research on condition that
                                      users cite Fung and Hsieh (2001) and the URL of the file; the page does
                                      not grant redistribution, so this file and the raw download stay out
                                      of the public copy and this script rebuilds them.
Both series are cut at 30 September 2026. Records the access dates in data/SOURCES.md.
"""
import json, re, sys, time, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
END = pd.Timestamp("2026-09-30")
OFR_API = "https://data.financialresearch.gov/hf/v1/series/full?mnemonic="
OFR_PAGE = "https://www.financialresearch.gov/hedge-fund-monitor/datasets/fpf/"
HSIEH_URL = "https://people.duke.edu/~dah7/DataLibrary/TF-Fac.xls"
HSIEH_PAGE = "https://people.duke.edu/~dah7/HFRFData.htm"
STRATEGIES = ["CREDIT", "EQUITY", "EVENT", "FUTURES", "MACRO", "MULTI", "OTHER", "RV"]
SERIES = ([f"FPF-STRATEGY_{s}_{k}_NAVWMEAN" for s in STRATEGIES for k in ("GROSSRETURN", "NETRETURN", "LEVERAGERATIO")]
          + [f"FPF-ALLQHF_INVESTORLIQUIDITY{b}_PERCENT" for b in ("LE7", "GT7LE90", "GT90")]
          + [f"FPF-ALLQHF_PORTFOLIOLIQUIDITY{b}_PERCENT" for b in ("LE1", "GT1LE7", "GT7LE90", "GT90")]
          + ["FPF-ALLQHF_CANGATE_PERCENT", "FPF-ALLQHF_CURRENTLYGATES_PERCENT", "FPF-ALLQHF_GAV_SUM",
             "FPF-ALLQHF_NAV_SUM", "FPF-ALLQHF_COUNT"])


def get(url, tries=4):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def column(mnemonic: str) -> str:
    return mnemonic.replace("FPF-", "").replace("_NAVWMEAN", "").replace("_PERCENT", "_pct").lower()


def ofr(offline: bool):
    raw = RAW / "ofr_fpf_series.json"
    if not offline:
        blob = {}
        for m in SERIES:
            r = json.loads(get(OFR_API + m))
            blob[m] = r[m] if m in r else r
        raw.write_text(json.dumps(blob, indent=0))
    blob = json.loads(raw.read_text())
    cols = {}
    for m, v in blob.items():
        obs = v["timeseries"]["aggregation"]
        s = pd.Series({pd.Timestamp(d): (float(x) if x is not None else float("nan")) for d, x in obs})
        cols[column(m)] = s
    d = pd.DataFrame(cols).sort_index()
    d = d[d.index <= END]
    d.index.name = "date"
    d.to_csv(ROOT / "data" / "ofr_hedge_funds_quarterly.csv", float_format="%.6g")
    return d


def hsieh(offline: bool):
    raw = RAW / "hsieh_TF-Fac.xls"
    if not offline:
        raw.write_bytes(get(HSIEH_URL))
    x = pd.read_excel(raw, header=None)
    row = x.index[x[0].astype(str).str.strip() == "yyyymm"][0]
    t = x.iloc[row + 1:, :6].dropna(subset=[0])
    t.columns = ["yyyymm", "ptfsbd", "ptfsfx", "ptfscom", "ptfsir", "ptfsstk"]
    t = t[t.yyyymm.astype(str).str.fullmatch(r"\d{6}")]
    t.index = pd.to_datetime(t.yyyymm.astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    t = t.drop(columns="yyyymm").astype(float)
    t = t[t.index <= END]
    t.index.name = "date"
    t.to_csv(ROOT / "data" / "fung_hsieh_ptfs.csv", float_format="%.6g")
    return t


def main():
    offline = "--offline" in sys.argv
    RAW.mkdir(parents=True, exist_ok=True)
    d, t = ofr(offline), hsieh(offline)
    today = dt.date.today().isoformat()
    src = ROOT / "data" / "SOURCES.md"; s = src.read_text()
    s = re.sub(r"\n- `(ofr_hedge_funds_quarterly|fung_hsieh_ptfs)\.csv`.*", "", s)
    s = s.rstrip("\n") + (
        f"\n- `ofr_hedge_funds_quarterly.csv`: US Office of Financial Research, Hedge Fund Monitor, Form PF "
        f"Statistics ({OFR_PAGE}), aggregated SEC Form PF responses of Qualifying Hedge Funds through the OFR "
        f"API ({OFR_API.split('?')[0]}): gross and net quarterly returns and leverage by strategy (net "
        f"asset-weighted means), investor and portfolio liquidity, gates, gross and net assets and fund "
        f"counts, quarterly, {d.index[0].date()} to {d.index[-1].date()}, accessed {today}. A US government "
        f"work, reproduced with credit as https://www.financialresearch.gov/legal-notices/ requests. Built by "
        f"`scripts/pull_hedge.py`, which also saves the API responses in `raw/`.\n"
        f"- `fung_hsieh_ptfs.csv`: (not included in the public copy; `python scripts/pull_hedge.py` rebuilds it) "
        f"David A. Hsieh's data library ({HSIEH_PAGE}), the trend-following risk factors of Fung and Hsieh "
        f"(2001), file {HSIEH_URL}, monthly returns of the PTFS bond, currency, commodity, short-term interest "
        f"rate and stock index lookback straddles, {t.index[0]:%Y-%m} to {t.index[-1]:%Y-%m}, accessed "
        f"{today}. Made available for academic research with citation of Fung and Hsieh (2001) and of the "
        f"file's URL; not redistributed.\n")
    src.write_text(s)
    print(f"OFR: {d.shape[1]} series, {d.index[0].date()} to {d.index[-1].date()}; "
          f"Fung-Hsieh: {t.index[0]:%Y-%m} to {t.index[-1]:%Y-%m}")


if __name__ == "__main__":
    main()
