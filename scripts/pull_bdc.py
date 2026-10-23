"""Pull the quarterly accounts of nineteen listed business development companies from SEC EDGAR and build
data/bdc_quarterly.csv for the private credit section of Chapter 6.

Usage: SEC_USER_AGENT="Your Name your@address" python scripts/pull_bdc.py [--offline]

The SEC asks every automated client to declare a name and a contact address in its User-Agent header
(https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data), so the script reads one
from SEC_USER_AGENT and refuses to run without it. It downloads the XBRL company facts of each company
(https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json), keeps the facts from 10-Q and 10-K filings
for the concepts below, and saves them in data/raw/sec_bdc_facts.json. --offline rebuilds the table from
that file. Information on sec.gov is public and may be copied or further distributed without the SEC's
permission (https://www.sec.gov/privacy), so both files are shipped.

The table has one row per company and calendar quarter end up to 30 September 2026: net asset value per
share, distributions declared per share in the quarter, investments at fair value and at amortised cost,
total investment income, net investment income, the net increase in net assets resulting from
operations (net investment income plus realised and unrealised gains), the management and incentive fees for the quarter,
total assets, total liabilities and net assets. Prospect Capital, which has a large issue of preferred stock, is left out so that net
assets and the result of operations belong to the common shareholders alone. Quarterly flows are reported directly in most 10-Q
filings; where a filing gives only a year to date figure, the quarter is the difference of two year to
date figures with the same start, and the fourth quarter is the annual figure less nine months.
Distributions per share are kept as tagged, but several companies tag only part of their distributions
in some quarters, so the chapter measures returns from the result of operations over net assets.
"""
import json, os, sys, time, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sec_bdc_facts.json"
OUT = ROOT / "data" / "bdc_quarterly.csv"
END = pd.Timestamp("2026-09-30")
TICKERS = {"ARCC": 1287750, "BXSL": 1736035, "MAIN": 1396440, "OBDC": 1655888, "GBDC": 1476765,
           "HTGC": 1280784, "FSK": 1422183, "TSLX": 1508655, "CSWC": 17313, "OCSL": 1414932,
           "GSBD": 1572694, "BBDC": 1379785, "FDUS": 1513363, "CGBD": 1544206,
           "PFLT": 1504619, "NMFC": 1496099, "SLRC": 1418076, "TCPC": 1370755, "GAIN": 1321741}
# column: concepts tried in order (the first one a company reports is used for that quarter)
INSTANT = {"nav_ps": ["NetAssetValuePerShare"], "fv": ["InvestmentOwnedAtFairValue"],
           "cost": ["InvestmentOwnedAtCost"], "assets": ["Assets"], "liabilities": ["Liabilities"],
           "net_assets": ["StockholdersEquity", "AssetsNet"]}
FLOW = {"dist_ps": ["CommonStockDividendsPerShareDeclared", "InvestmentCompanyDistributionToShareholdersPerShare"],
        "income": ["GrossInvestmentIncomeOperating"],
        "nii": ["NetInvestmentIncome", "InvestmentIncomeOperatingAfterExpenseAndTax"],
        "operations": ["NetIncomeLoss", "ProfitLoss"],
        "mgmt_fee": ["ManagementFeeExpense"],
        "incentive_fee": ["IncentiveFeeExpense", "IncentiveFeeExpenseIncomeBased"]}
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}


def get(url, ua, tries=4):
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120).read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def download():
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        sys.exit("set SEC_USER_AGENT to a name and contact e-mail address, as the SEC requires")
    keep = {c for v in list(INSTANT.values()) + list(FLOW.values()) for c in v}
    blob = {}
    for t, cik in TICKERS.items():
        d = json.loads(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", ua))
        g = d["facts"].get("us-gaap", {})
        blob[t] = {"cik": cik, "name": d["entityName"], "facts": {
            c: [{k: x.get(k) for k in ("start", "end", "val", "form", "filed", "accn")}
                for u in g[c]["units"].values() for x in u if x.get("form") in FORMS]
            for c in keep if c in g}}
        time.sleep(0.2)  # the SEC allows ten requests a second
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(dict(retrieved=dt.date.today().isoformat(), companies=blob), indent=0))


def latest(facts, key):
    """One value per key, from the latest filing that reports it."""
    out = {}
    for x in sorted(facts, key=lambda x: x["filed"]):
        out[key(x)] = x["val"]
    return out


def instants(facts):
    v = latest([x for x in facts if "start" not in x or x["start"] is None], lambda x: x["end"])
    return {pd.Timestamp(e): val for e, val in v.items() if pd.Timestamp(e).is_quarter_end}


def quarters(facts):
    """Quarterly values from quarter facts, or from differences of year to date facts with one start."""
    v = latest([x for x in facts if x.get("start")], lambda x: (x["start"], x["end"]))
    q = {}
    for (s, e), val in v.items():
        n = (pd.Timestamp(e) - pd.Timestamp(s)).days
        if 80 <= n <= 100 and pd.Timestamp(e).is_quarter_end:
            q[pd.Timestamp(e)] = val
    by_start = {}
    for (s, e), val in v.items():
        by_start.setdefault(s, []).append((pd.Timestamp(e), val))
    for s, xs in by_start.items():
        xs = dict(xs)
        for e, val in xs.items():
            if e in q or not e.is_quarter_end:
                continue
            prev = (e - pd.offsets.QuarterEnd(1)).normalize()
            if prev in xs and (prev - pd.Timestamp(s)).days >= 80:
                q[e] = val - xs[prev]
    return q


def build():
    blob = json.loads(RAW.read_text())["companies"]
    rows = []
    for t, c in blob.items():
        f = c["facts"]; cols = {}
        for col, names in INSTANT.items():
            s = {}
            for n in names:
                for k, val in instants(f.get(n, [])).items():
                    s.setdefault(k, val)
            cols[col] = s
        for col, names in FLOW.items():
            s = {}
            for n in names:
                for k, val in quarters(f.get(n, [])).items():
                    s.setdefault(k, val)
            cols[col] = s
        df = pd.DataFrame(cols)
        df.index.name = "quarter_end"
        df = df[df.index <= END].sort_index().reset_index()
        df.insert(0, "ticker", t)
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["quarter_end"] = out.quarter_end.dt.strftime("%Y-%m-%d")
    out.to_csv(OUT, index=False, float_format="%.10g")
    print(f"{OUT.name}: {len(out)} rows, {out.ticker.nunique()} companies, "
          f"{out.quarter_end.min()} to {out.quarter_end.max()}")


if __name__ == "__main__":
    if "--offline" not in sys.argv:
        download()
    build()
