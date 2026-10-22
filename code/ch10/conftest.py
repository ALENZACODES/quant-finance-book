"""Shared fixtures for the Chapter 10 tests. Run from the repository root:  python -m pytest -q code/ch10
run_all.py, crosscheck.R and octave/ch10_check.m must have been run first (make code CH=10 does this)."""
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


DATA = [HERE.parents[1] / "data" / f for f in ("ff_factors_monthly.csv", "real_assets_quarterly.csv",
                                                "real_assets_monthly.csv", "eia_curves_monthly.csv")]


def pytest_collection_modifyitems(config, items):
    # The French library files are not shipped with the public copy; skip rather than fail without them.
    if not all(p.exists() for p in DATA):
        skip = pytest.mark.skip(reason="data missing: run python scripts/build_french.py --download and python scripts/pull_ch10.py")
        for item in items:
            if "code/ch10" in str(item.fspath):
                item.add_marker(skip)


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)
