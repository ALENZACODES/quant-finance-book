"""Every random draw in the Chapter 1 scripts comes from NumPy's default generator with seed 101, as the
keybox says: one shared default_rng(101) passed down by run_all.py, a fresh default_rng(101) for the
listing, and default_rng([101, 60]) for the n = 60 run. Test files may use their own seeds."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALLOWED = {"SEED", "R.SEED", "101", "[101, 60]"}


def scripts():
    return [p for p in sorted(HERE.glob("*.py")) + sorted((HERE / "snippets").glob("*.py"))
            if not p.name.startswith("test_") and p.name != "conftest.py"]


def test_every_generator_uses_the_chapter_seed():
    seen = []
    for p in scripts():
        src = p.read_text()
        for arg in re.findall(r"default_rng\(([^()]*)\)", src):
            assert arg.strip() in ALLOWED, (p.name, arg)
            seen.append(arg.strip())
    assert {"101", "[101, 60]"} <= set(seen)


def test_no_other_source_of_randomness():
    bad = re.compile(r"np\.random\.(?!default_rng\b)\w+|\bimport random\b|\bfrom random import|"
                     r"RandomState|SeedSequence|\.spawn\(")
    for p in scripts():
        hits = bad.findall(p.read_text())
        assert not hits, (p.name, hits)


def test_seed_constant_is_101():
    import numbers_ch01
    assert numbers_ch01.SEED == 101
