"""Shared fixtures for the Chapter 12 tests. Run from the repository root:  python -m pytest -q code/ch12
run_all.py, crosscheck.R and octave/ch12_check.m must have been run first (make code CH=12 does this)."""
import json, sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GW = HERE.parents[1] / "data" / "goyal_welch_monthly.csv"


@pytest.fixture(scope="session")
def RAW():
    return json.loads((HERE / "out" / "numbers.json").read_text())["RAW"]


@pytest.fixture(scope="session")
def NUM():
    return json.loads((HERE / "out" / "numbers.json").read_text())["N"]


def pytest_collection_modifyitems(config, items):
    # The Goyal and Welch file is not shipped with the public copy; tests marked "needs_data" skip without it.
    if not GW.exists():
        skip = pytest.mark.skip(reason="data missing: run python scripts/pull_ch12.py")
        for item in items:
            if "code/ch12" in str(item.fspath) and ("needs_data" in item.keywords or "crosscheck" in item.name):
                item.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: needs data/goyal_welch_monthly.csv")


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)
