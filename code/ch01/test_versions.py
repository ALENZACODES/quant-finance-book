"""The package versions cited in references.bib for Chapter 1 are the ones installed and pinned in
code/requirements-lock.txt, and each is dated by the release year of that version on PyPI."""
import re
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIB = (ROOT / "references.bib").read_text()
# PyPI upload date of each pinned version, checked on 19 October 2026:
# numpy 2.2.6 on 2025-05-17, numpy-financial 1.1.0 on 2026-09-15 (1.0.0 was 2019-10-18), scipy 1.18.1 on
# 2026-08-21, statsmodels 0.15.0 on 2026-08-27, matplotlib 3.11.2 on 2026-09-11, pandas 3.0.6 on 2026-09-17.
RELEASE = {"numpy": ("numpy", "2.2.6", 2025), "scipy": ("scipy", "1.18.1", 2026),
           "pandas": ("pandas", "3.0.6", 2026), "matplotlib": ("matplotlib", "3.11.2", 2026),
           "statsmodels": ("statsmodels", "0.15.0", 2026),
           "numpy-financial": ("numpyfinancial", "1.1.0", 2026)}


def entry(key):
    m = re.search(r"@software\{" + key + r",(.*?)\n\}", BIB, re.S)
    assert m, key
    return m.group(1)


def test_cited_versions_are_installed_and_dated():
    for pkg, (key, pinned, year) in RELEASE.items():
        e = entry(key)
        v = re.search(r"version\s*=\s*\{([^}]*)\}", e).group(1)
        assert v == pinned == version(pkg), (pkg, v, version(pkg))
        assert int(re.search(r"year\s*=\s*\{(\d{4})\}", e).group(1)) == year, pkg


def test_lock_file_pins_the_cited_versions():
    lock = (ROOT / "code" / "requirements-lock.txt").read_text()
    for pkg, (_, pinned, _) in RELEASE.items():
        assert f"{pkg}=={pinned}" in lock, pkg
