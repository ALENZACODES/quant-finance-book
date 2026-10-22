"""Pull the public series used in Chapter 10 and build the chapter's three data files.

Usage: python scripts/pull_ch10.py [END]
Every series here is in the public domain (US federal agencies, published through FRED or by the US Energy
Information Administration), so the files are shipped with the code. Each download is saved, cut at END
(default 2026-09-30), in data/raw/, and the script writes
  data/real_assets_quarterly.csv  Z.1 commercial real estate price index, equity REIT market value and its
                                  revaluation, FHFA all-transactions house price index (quarterly)
  data/real_assets_monthly.csv    CPI, 3 month bill and 10 year yields, PPI farm products, PPI logs and timber,
                                  PPI softwood lumber, WTI and Henry Hub spot prices (monthly)
  data/eia_curves_monthly.csv     NYMEX crude oil and natural gas futures, contracts 1 to 4 (monthly averages
                                  of daily closes, published by the EIA to April 2024), US crude oil stocks
                                  excluding the SPR and working gas in underground storage (monthly)
and records the access date in data/SOURCES.md.
"""
import io, re, sys, time, datetime as dt, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
QUARTERLY = {"BOGZ1FL075035503Q": "cre", "BOGZ1FL123164105Q": "reit_mv", "BOGZ1FR123164105Q": "reit_reval",
             "USSTHPI": "hpi"}
MONTHLY = {"CPIAUCSL": "cpi", "TB3MS": "tbill", "GS10": "gs10", "WPU01": "ppi_farm", "WPU0851": "ppi_timber",
           "WPU0811": "ppi_lumber", "MCOILWTICO": "wti", "MHHNGSP": "gas"}
EIA = {"pet/hist/LeafHandler.ashx?n=PET&s=RCLC1&f=M": "cl1", "pet/hist/LeafHandler.ashx?n=PET&s=RCLC2&f=M": "cl2",
       "pet/hist/LeafHandler.ashx?n=PET&s=RCLC3&f=M": "cl3", "pet/hist/LeafHandler.ashx?n=PET&s=RCLC4&f=M": "cl4",
       "ng/hist/rngc1m.htm": "ng1", "ng/hist/rngc2m.htm": "ng2", "ng/hist/rngc3m.htm": "ng3", "ng/hist/rngc4m.htm": "ng4",
       "pet/hist/LeafHandler.ashx?n=PET&s=MCESTUS1&f=M": "crude_stocks", "ng/hist/n5020us2m.htm": "gas_storage"}


def get(url, tries=4):
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8.5.0"})   # FRED stalls browser-like agents
    for k in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=60).read().decode("latin1")
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def fred(s, end):
    txt = get(FRED.format(s=s))
    head, *rows = txt.strip().split("\n")
    txt = "\n".join([head] + [r for r in rows if r[:10] <= end]) + "\n"   # nothing after END is kept
    (ROOT / "data" / "raw" / f"fred_{s}.csv").write_text(txt)
    df = pd.read_csv(io.StringIO(txt), na_values=["."])
    return pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]), name=s)


def eia(code, end):
    """EIA history pages are year-by-month tables; flatten them to a monthly series."""
    tab = max(pd.read_html(io.StringIO(get("https://www.eia.gov/dnav/" + code))), key=lambda d: d.size)
    tab = tab.dropna(subset=["Year"])
    long = tab.melt(id_vars="Year", var_name="month", value_name="v").dropna()
    months = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                          "Nov", "Dec"], 1)}
    idx = pd.to_datetime({"year": long["Year"].astype(int), "month": long["month"].map(months), "day": 1})
    s = pd.Series(pd.to_numeric(long["v"]).values, index=idx).sort_index()
    s = s[s.index <= end]
    name = re.sub(r"[^A-Za-z0-9]+", "_", code.split("s=")[-1] if "s=" in code else code.split("/")[-1])
    s.rename("value").to_csv(ROOT / "data" / "raw" / f"eia_{name}.csv", index_label="date")
    return s


def main(end="2026-09-30"):
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    q = pd.concat({v: fred(k, end) for k, v in QUARTERLY.items()}, axis=1, sort=True)
    q = q[q.index >= "1970-01-01"]
    q.index.name = "date"; q.to_csv(ROOT / "data" / "real_assets_quarterly.csv", float_format="%.10g")
    m = pd.concat({v: fred(k, end) for k, v in MONTHLY.items()}, axis=1, sort=True)
    m = m[m.index >= "1970-01-01"]
    m.index.name = "date"; m.to_csv(ROOT / "data" / "real_assets_monthly.csv", float_format="%.10g")
    e = pd.concat({v: eia(k, end) for k, v in EIA.items()}, axis=1, sort=True)
    e.index.name = "date"; e.to_csv(ROOT / "data" / "eia_curves_monthly.csv", float_format="%.10g")
    today = dt.date.today().isoformat()
    span = lambda df: f"{df.index[0].date()} to {df.dropna(how='all').index[-1].date()}"
    notes = {
        "real_assets_quarterly.csv": (
            "FRED, Federal Reserve Bank of St. Louis, public domain series: BOGZ1FL075035503Q (Financial Accounts "
            "of the United States, Z.1, commercial real estate price index; from 1996 the Federal Reserve chains "
            "the CoStar U.S. Composite Index excluding multifamily, value weighted, and before 1996 it averages "
            "three appraisal-based price series), BOGZ1FL123164105Q and BOGZ1FR123164105Q (Z.1, equity REITs, "
            "corporate equities, liability, level and revaluation) and USSTHPI (US Federal Housing Finance "
            f"Agency, all-transactions house price index), quarterly, {span(q)}, accessed {today}. Built by "
            "`scripts/pull_ch10.py`, which also saves each series, cut at the end date, in `raw/`."),
        "real_assets_monthly.csv": (
            "FRED, public domain series: CPIAUCSL (US Bureau of Labor Statistics, CPI for all urban consumers, "
            "seasonally adjusted), TB3MS and GS10 (Federal Reserve H.15, 3 month Treasury bill secondary market "
            "rate and 10 year constant maturity yield, monthly averages), WPU01, WPU0851 and WPU0811 (BLS producer "
            "price indexes for farm products, for logs, bolts, timber, pulpwood and wood chips, and for softwood "
            "lumber), MCOILWTICO and MHHNGSP (US Energy Information Administration, WTI and Henry Hub spot prices, "
            f"monthly averages), {span(m)}, accessed {today}. Built by `scripts/pull_ch10.py`."),
        "eia_curves_monthly.csv": (
            "US Energy Information Administration, public domain: NYMEX crude oil (Cushing, OK) and natural gas "
            "(Henry Hub) futures, contracts 1 to 4, monthly averages of daily closing prices, which the EIA published "
            "until April 2024; US ending stocks of crude oil excluding the Strategic Petroleum Reserve (thousand "
            "barrels) and US working gas in underground storage (million cubic feet), monthly, from the EIA "
            f"history pages at https://www.eia.gov/dnav/, {span(e)}, accessed {today}. Built by "
            "`scripts/pull_ch10.py`."),
    }
    p = ROOT / "data" / "SOURCES.md"
    lines = [l for l in p.read_text().splitlines(True) if not any(l.startswith(f"- `{k}`") for k in notes)]
    p.write_text("".join(lines).rstrip("\n") + "\n" + "".join(f"- `{k}`: {v}\n" for k, v in notes.items()))
    print(q.dropna(how="all").tail(2), m.tail(2), e.dropna(how="all").tail(2), sep="\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
