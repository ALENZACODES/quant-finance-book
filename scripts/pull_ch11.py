"""Pull the public data used in Chapter 11 and build the chapter's data files.

Usage: python scripts/pull_ch11.py
  data/calpers_pe_funds.csv     CalPERS Private Equity Program fund performance review, all active
                                partnerships as of 31 March 2026: vintage, commitment, cash in, cash out,
                                cash out plus remaining value, net IRR and multiple as published, with the
                                "not meaningful" flag CalPERS sets for funds of vintage 2021 or later
  data/damodaran_industry.csv   Aswath Damodaran's US industry averages, January 2026 update: EV/EBITDA and
                                EV/EBIT(1-t), beta, cost of equity and capital, debt ratio, tax rate,
                                after-tax return on capital, reinvestment rate and expected EBIT growth
  data/damodaran_erp.csv        the implied equity risk premium of the S&P 500 at each year end, 1960 to 2025,
                                with the ten year Treasury yield used beside it
Both sources are free to read but are not public domain, so the raw downloads (data/raw/calpers_*,
data/raw/damodaran_*) and these files are kept out of the public copy; the script rebuilds them.
Reading the .xls files needs xlrd. Records the access date in data/SOURCES.md.
"""
import io, re, sys, time, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CALPERS = "https://www.calpers.ca.gov/investments/about-investment-office/investment-organization/pep-fund-performance-print"
DAMODARAN = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/{f}.xls"


def get(url, tries=4):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def money(x):
    return float(str(x).replace("$", "").replace(",", "")) if pd.notna(x) else float("nan")


def calpers(html):
    t = pd.read_html(io.StringIO(html))[0]
    t = t[t["Fund"] != "Fund"].copy()
    asof = re.search(r"As of ([A-Z][a-z]+ \d+, \d{4})", html).group(1)
    out = pd.DataFrame({"fund": t["Fund"].str.strip(), "vintage": t["Vintage Year"].astype(int),
                        "committed": t["Capital Committed"].map(money), "cash_in": t["Cash In"].map(money),
                        "cash_out": t["Cash Out"].map(money),
                        "total_value": t["Cash Out & Remaining Value"].map(money)})
    irr = t["Net IRR"].astype(str); mult = t["Investment Multiple"].astype(str)
    out["not_meaningful"] = irr.str.contains("N/M") | irr.str.strip().str.endswith(" 1") | irr.str.contains(r"%\s+1")
    out["net_irr"] = pd.to_numeric(irr.str.extract(r"(-?[\d.]+)%")[0], errors="coerce") / 100
    out["multiple"] = pd.to_numeric(mult.str.extract(r"([\d.]+)x")[0], errors="coerce")
    out["cash_adjusted"] = irr.str.contains(r"%\s*2") | mult.str.contains(r"x\s*2")
    return out, asof


def damodaran():
    def sheet(f, header_row_name):
        x = pd.read_excel(RAW / f"damodaran_{f}.xls", sheet_name="Industry Averages", header=None)
        h = x.index[x.iloc[:, 0].astype(str).str.strip() == header_row_name][0]
        d = x.iloc[h + 1:].copy(); d.columns = [str(c).strip() for c in x.iloc[h]]
        return d[d.iloc[:, 0].notna()].set_index(d.columns[0])
    v = sheet("vebitda", "Industry Name"); w = sheet("wacc", "Industry Name"); g = sheet("fundgrEB", "Industry Name")
    v = v.iloc[:, [0, 2, 4]]; v.columns = ["n_firms", "ev_ebitda_pos", "ev_ebit1t_pos"]   # positive EBITDA firms
    w = w[["Beta", "Cost of Equity", "E/(D+E)", "Tax Rate", "After-tax Cost of Debt", "D/(D+E)", "Cost of Capital"]]
    w.columns = ["beta", "cost_equity", "equity_weight", "tax_rate", "cost_debt_after_tax", "debt_weight", "wacc"]
    g = g[["ROC", "Reinvestment Rate", "Expected Growth in EBIT"]]; g.columns = ["roc", "reinvestment", "growth"]
    d = v.join(w, how="inner").join(g, how="inner")
    d.index.name = "industry"
    return d.apply(pd.to_numeric, errors="coerce")


def erp():
    x = pd.read_excel(RAW / "damodaran_histimpl.xls", sheet_name=0, header=None)
    h = x.index[x.iloc[:, 0].astype(str).str.strip() == "Year"][0]
    d = x.iloc[h + 1:].copy(); d.columns = [str(c).strip() for c in x.iloc[h]]
    d = d[pd.to_numeric(d["Year"], errors="coerce").notna()]
    col = [c for c in d.columns if c.startswith("Implied Premium (FCFE)") or c == "Implied ERP (FCFE)"][0]
    out = pd.DataFrame({"year": d["Year"].astype(int), "erp_fcfe": pd.to_numeric(d[col], errors="coerce"),
                        "tbond": pd.to_numeric(d["T.Bond Rate"], errors="coerce")})
    return out[out.year <= 2025]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    html = get(CALPERS).decode("utf-8", "replace")
    (RAW / "calpers_pep_fund_performance.html").write_text(html)
    for f in ("vebitda", "wacc", "fundgrEB", "histimpl"):
        (RAW / f"damodaran_{f}.xls").write_bytes(get(DAMODARAN.format(f=f)))
    c, asof = calpers(html)
    c.to_csv(ROOT / "data" / "calpers_pe_funds.csv", index=False)
    damodaran().to_csv(ROOT / "data" / "damodaran_industry.csv")
    erp().to_csv(ROOT / "data" / "damodaran_erp.csv", index=False)
    today = dt.date.today().isoformat()
    src = ROOT / "data" / "SOURCES.md"; s = src.read_text()
    s = re.sub(r"\n- `calpers_pe_funds\.csv`.*", "", s); s = re.sub(r"\n- `damodaran_.*", "", s)
    s = s.rstrip("\n") + (
        f"\n- `calpers_pe_funds.csv`: CalPERS, Private Equity Program Fund Performance Review, all active partnerships "
        f"as of {asof}, {CALPERS}, accessed {today}. Not redistributed; built by `scripts/pull_ch11.py`."
        f"\n- `damodaran_industry.csv`, `damodaran_erp.csv`: Aswath Damodaran, data sets at "
        f"https://pages.stern.nyu.edu/~adamodar/ (US industry averages, January 2026 update: vebitda, wacc, fundgrEB; "
        f"historical implied equity risk premiums, 1960 to 2025: histimpl), accessed {today}. Not redistributed; built "
        f"by `scripts/pull_ch11.py`.\n")
    src.write_text(s)
    print(len(c), "funds as of", asof)


if __name__ == "__main__":
    main()
