"""The package versions cited in references.bib are the ones installed in the run environment, and
each is dated by the release year of that version."""
import re
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIB = (ROOT / "references.bib").read_text()
# PyPI release year of each pinned version, checked when the pin was set
RELEASE_YEAR = {"numpy": ("2.2.6", 2025), "scipy": ("1.18.1", 2026), "pandas": ("3.0.6", 2026),
                "statsmodels": ("0.15.0", 2026), "matplotlib": ("3.11.2", 2026),
                "numpy-financial": ("1.1.0", 2026)}
KEYS = {"numpy": "numpy", "scipy": "scipy", "pandas": "pandas", "statsmodels": "statsmodels",
        "matplotlib": "matplotlib", "numpy-financial": "numpyfinancial"}


def entry(key):
    m = re.search(r"@software\{" + key + r",(.*?)\n\}", BIB, re.S)
    assert m, key
    return m.group(1)


def test_cited_versions_are_installed():
    for pkg, key in KEYS.items():
        e = entry(key)
        v = re.search(r"version\s*=\s*\{([^}]*)\}", e).group(1)
        assert v == version(pkg), (pkg, v, version(pkg))
        assert v == RELEASE_YEAR[pkg][0]
        assert int(re.search(r"year\s*=\s*\{(\d{4})\}", e).group(1)) == RELEASE_YEAR[pkg][1], pkg


def test_lock_file_matches():
    lock = (ROOT / "code" / "requirements-lock.txt").read_text()
    for pkg in KEYS:
        assert f"{pkg}=={version(pkg)}" in lock, pkg
