"""Shared fixtures for the Chapter 3 tests. Run from the repository root:  python -m pytest -q code/ch03
run_all.py, crosscheck.R and octave/ch03_check.m must have been run first (make code CH=03 does this)."""
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


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)
