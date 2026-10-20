"""Cross-language checks (R and GNU Octave) and checks of the text and the exercise answers."""
import csv, math, re, shutil, subprocess
import numpy as np
import pytest
from scipy import stats
from conftest import close, HERE
import ch05lib as L  # noqa: F401

ROOT = HERE.parents[1]


def test_r_crosscheck(RAW):
    rows = list(csv.DictReader(open(HERE / "out" / "r_check.csv")))
    assert len(rows) >= 60
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], float(r["tol"]) * max(1.0, abs(v)))


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_crosscheck(RAW):
    subprocess.run(["octave-cli", "--no-gui", str(HERE / "octave" / "ch05_check.m")], check=True,
                   cwd=ROOT, capture_output=True)
    rows = list(csv.DictReader(open(HERE / "out" / "octave_check.csv")))
    assert len(rows) >= 15
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], float(r["tol"]) * max(1.0, abs(v)))


def test_every_macro_used_is_defined():
    tex = (ROOT / "chapters" / "ch05.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch05-numbers.tex").read_text()))
    sty = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "preamble" / "quantbook.sty").read_text()))
    used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    known = {"lstinputlisting", "setcounter", "includegraphics", "printbibliography", "textcite", "parencite",
             "mathrm", "operatorname", "mathbb", "mathcal", "varepsilon", "lVert", "rVert", "Longrightarrow",
             "Longleftrightarrow", "xrightarrow", "mathbf", "boldsymbol", "lfloor", "rfloor", "displaystyle",
             "mathbbm", "textstyle", "underbrace", "overline", "widehat", "leqslant", "geqslant", "lVertt"}
    missing = sorted(u for u in used if u not in defined | sty | known)
    assert not missing, missing


def test_no_dashes_in_prose():
    tex = (ROOT / "chapters" / "ch05.tex").read_text()
    assert "---" not in tex and "\u2014" not in tex and "\u2013" not in tex
    body = re.sub(r"\$[^$]*\$", "", tex)
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
    assert not re.search(r"[A-Za-z]--[A-Za-z ]", body) and " -- " not in body


def test_exercise_answers(RAW):
    close(RAW["xvOneCall"] - RAW["xvOnePut"], RAW["xvOneParity"], 1e-10)
    assert abs(RAW["xvTwoApprox"] - RAW["xvTwoExact"]) < abs(RAW["xvTwoDeltaOnly"] - RAW["xvTwoExact"]) / 10
    close(RAW["xvThreeS"], 55 * math.exp(-(0.04 + 1.5 * 0.0625) * 0.5), 1e-9)
    assert abs(RAW["xvFourCall"] - RAW["xvFourApprox"]) < 0.01
    close(RAW["xvSevenSteps"], 900, 0); assert abs(RAW["xvEightCheck"]) < 1e-10
    close(RAW["xvTenGap"], float(L.bs(100, 100, 0.05, 0.25, 1) - L.bs(100, 100, 0.05, 0.2, 1)), 1e-12)
