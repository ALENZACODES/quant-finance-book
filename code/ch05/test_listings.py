"""Every listing printed in Chapter 5 runs exactly as printed: the printed lines, in order, as a fresh
file, and no line is wider than the listing column, where the page would wrap it."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import listings as L  # noqa: E402

PRINTED = L.printed("05")


def test_chapter_prints_listings():
    assert PRINTED


@pytest.mark.parametrize("path,lang,text", PRINTED, ids=[p for p, _, _ in PRINTED])
def test_listing_runs_as_printed(path, lang, text):
    assert not L.too_wide(text), L.too_wide(text)
    r = L.run(lang, text, L.SUFFIX[lang], path)
    if r is None:
        pytest.skip(f"{lang} is not installed")
    assert r[0] == 0, r[1]
