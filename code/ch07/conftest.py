"""Shared fixtures for the Chapter 7 tests. Run from the repository root:  python -m pytest -q code/ch07
run_all.py, crosscheck.R and octave/ch07_check.m must have been run first (make code CH=07 does this)."""
import json, sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


@pytest.fixture(scope="session")
def RAW():
    return json.loads((HERE / "out" / "numbers.json").read_text())["RAW"]


@pytest.fixture(scope="session")
def NUM():
    return json.loads((HERE / "out" / "numbers.json").read_text())["N"]


DATA = [HERE.parents[1] / "data" / f for f in ("ff_factors_monthly.csv", "industry10_monthly.csv")]


def pytest_collection_modifyitems(config, items):
    # The French library files are not shipped with the public copy; skip rather than fail without them.
    if not all(p.exists() for p in DATA):
        skip = pytest.mark.skip(reason="data missing: run python scripts/build_french.py --download")
        for item in items:
            if "code/ch07" in str(item.fspath):
                item.add_marker(skip)


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)
