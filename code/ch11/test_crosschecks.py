"""Cross-language checks (R and GNU Octave) and checks of the text."""
import csv, json, re, shutil, subprocess
import pytest
from conftest import close, HERE

ROOT = HERE.parents[1]


def test_r_crosscheck(RAW):
    rows = list(csv.DictReader(open(HERE / "out" / "r_check.csv")))
    assert len(rows) >= 40
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], float(r["tol"]) * max(1.0, abs(v)))


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_crosscheck(RAW):
    subprocess.run(["octave-cli", "--no-gui", str(HERE / "octave" / "ch11_check.m")], check=True,
                   cwd=ROOT, capture_output=True)
    rows = list(csv.DictReader(open(HERE / "out" / "octave_check.csv")))
    assert len(rows) >= 10
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], float(r["tol"]) * max(1.0, abs(v)))


def test_every_macro_used_is_defined():
    tex = (ROOT / "chapters" / "ch11.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch11-numbers.tex").read_text()))
    sty = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "preamble" / "quantbook.sty").read_text()))
    used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    known = {"lstinputlisting", "setcounter", "includegraphics", "printbibliography", "textcite", "parencite",
             "mathrm", "operatorname", "mathbb", "mathcal", "varepsilon", "Longrightarrow", "mathbf",
             "boldsymbol", "displaystyle", "textstyle", "underbrace", "overline", "widehat", "addlinespace"}
    missing = sorted(u for u in used if u not in defined | sty | known)
    assert not missing, missing


def test_macros_do_not_clash_with_other_chapters():
    mine = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch11-numbers.tex").read_text()))
    for f in sorted((ROOT / "chapters").glob("ch[01][0-9]-numbers.tex")):
        if f.name == "ch11-numbers.tex":
            continue
        other = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read_text()))
        assert not mine & other, (f.name, sorted(mine & other))


def test_no_dashes_in_prose():
    tex = (ROOT / "chapters" / "ch11.tex").read_text()
    assert "---" not in tex and "\u2014" not in tex and "\u2013" not in tex
    body = re.sub(r"\$[^$]*\$", "", tex)
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
    assert not re.search(r"[A-Za-z]--[A-Za-z ]", body) and " -- " not in body

