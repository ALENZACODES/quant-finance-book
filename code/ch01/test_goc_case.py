"""Independent checks of the Chapter 1 numbers produced by goc_case.py (pass (b) of the production process).

Run from the repository root:  python -m pytest -q code/ch01
The R cross-check (code/ch01/crosscheck.R) must have been run first; run_all.py writes numbers.json.
"""
import csv, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import numpy_financial as npf
import pytest
from scipy import stats
from statsmodels.stats.stattools import jarque_bera as sm_jb, durbin_watson as sm_dw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import ch01lib as L  # noqa: E402

RAW = json.loads((HERE / "out" / "numbers.json").read_text())["RAW"]
rng = np.random.default_rng(7)


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def test_goc_case_claims():
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "daily_policy_bond.csv").dropna(subset=["BD.CDN.5YR.DQ.YLD"])
    dy = df["BD.CDN.5YR.DQ.YLD"].diff().dropna().values * 100
    close(RAW["gocDN"], len(dy), 0)
    assert RAW["gocJBp"] < 1e-10
    ac = L.acf(dy, 10)[1:]
    band = 1.96 / np.sqrt(len(dy))
    outside = [k + 1 for k, v in enumerate(ac) if abs(v) > band]
    assert outside == [10], outside            # text: only the tenth lag falls outside the band
    r = {int(k): v for k, v in RAW["gocRatios"].items()}
    assert all(abs(r[h] - 1) < 0.1 for h in (5, 21, 63))


def test_data_snapshot_is_the_archived_file():
    import hashlib
    import pandas as pd
    NUM = json.loads((HERE / "out" / "numbers.json").read_text())["N"]
    h = hashlib.sha256((ROOT / "data" / "daily_policy_bond.csv").read_bytes()).hexdigest()
    assert h.startswith(NUM["gocSha"]) and h == NUM["gocShaA"] + NUM["gocShaB"]
    df = pd.read_csv(ROOT / "data" / "daily_policy_bond.csv", parse_dates=["date"]).dropna(subset=["V39079"])
    assert df.date.min() == pd.Timestamp(NUM["polStart"]) and abs(df.V39079.iloc[0] - RAW["polFirst"]) < 1e-12


UPSTREAM = ("https://raw.githubusercontent.com/ALENZACODES/canada-policy-rate-bond-yield-mortgage-rate/"
            "815cb7b/data/daily_policy_bond.csv")


def test_digest_listed_in_sources_matches_the_copy():
    h = hashlib.sha256((ROOT / "data" / "daily_policy_bond.csv").read_bytes()).hexdigest()
    src = (ROOT / "data" / "SOURCES.md").read_text()
    line = next(l for l in src.splitlines() if l.startswith("- `daily_policy_bond.csv`"))
    assert h in line


def test_copy_matches_the_file_at_the_cited_commit():
    import urllib.request
    try:
        upstream = urllib.request.urlopen(UPSTREAM, timeout=20).read()
    except Exception as e:  # no network: the digest test above still runs
        pytest.skip(f"upstream not reachable: {e}")
    assert hashlib.sha256(upstream).hexdigest() == hashlib.sha256((ROOT / "data" / "daily_policy_bond.csv").read_bytes()).hexdigest()
    assert UPSTREAM in (ROOT / "data" / "SOURCES.md").read_text()
