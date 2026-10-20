"""Cross-language checks (R and GNU Octave) and checks of the text and the exercise answers."""
import csv, math, re, shutil, subprocess
import numpy as np
import pytest
from scipy import stats
from conftest import close, HERE
import ch04lib as L  # noqa: F401

ROOT = HERE.parents[1]


def test_r_crosscheck(RAW):
    rows = list(csv.DictReader(open(HERE / "out" / "r_check.csv")))
    assert len(rows) >= 30
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], 1e-8 * max(1.0, abs(v)))


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_crosscheck(RAW):
    subprocess.run(["octave-cli", "--no-gui", str(HERE / "octave" / "ch04_check.m")], check=True,
                   cwd=ROOT, capture_output=True)
    rows = list(csv.DictReader(open(HERE / "out" / "octave_check.csv")))
    assert len(rows) >= 15
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], float(r["tol"]) * max(1.0, abs(v)))


def test_every_macro_used_is_defined():
    tex = (ROOT / "chapters" / "ch04.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch04-numbers.tex").read_text()))
    sty = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "preamble" / "quantbook.sty").read_text()))
    used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    known = {"lstinputlisting", "setcounter", "includegraphics", "printbibliography", "textcite", "parencite",
             "mathrm", "operatorname", "mathbb", "mathcal", "varepsilon", "lVert", "rVert", "Longrightarrow",
             "Longleftrightarrow", "xrightarrow", "mathbf", "boldsymbol", "lfloor", "rfloor", "displaystyle",
             "mathbbm", "textstyle", "underbrace", "overline", "widehat", "leqslant", "geqslant", "lVertt"}
    missing = sorted(u for u in used if u not in defined | sty | known)
    assert not missing, missing


def test_no_dashes_in_prose():
    tex = (ROOT / "chapters" / "ch04.tex").read_text()
    assert "---" not in tex and "\u2014" not in tex and "\u2013" not in tex
    body = re.sub(r"\$[^$]*\$", "", tex)
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
    assert not re.search(r"[A-Za-z]--[A-Za-z ]", body) and " -- " not in body


def test_exercise_answers(RAW):
    close(RAW["exOneDTwo"], (1 - 0.035 / 1.03) / 1.035, 1e-6)
    close(RAW["exOneFwd"], 100 * ((1 / 1.03) / ((1 - 0.035 / 1.03) / 1.035) - 1), 1e-3)
    close(RAW["exTwoF"], 50 * math.exp(0.03 * 0.75), 1e-4)
    close(RAW["exFourF"], 1.35 * 1.015 / 1.025, 1e-4); assert RAW["exFourF"] < 1.34 and RAW["exFourProfit"] > 0
    close(RAW["exFiveCall"], 5, 1e-12); close(RAW["exFivePut"], 7.5, 1e-12)
    close(RAW["exSixY"], 100 * (0.06 + math.log(70 / 66)), 1e-9)
    close(RAW["exSevenP"], 0.4 * 5 / 1.02, 1e-4)
    close(RAW["exEightAm"], (0.6 * RAW["exEightContU"] + 0.4 * 7) / 1.02, 1e-4)
    assert RAW["exEightContD"] < RAW["exEightExD"]
    close(RAW["exNineEx"], RAW["exNineSpot"] + RAW["exNineRoll"], 2e-3)
    close(RAW["exNineRoll"], RAW["exNineBasisOut"] - RAW["exNineBasisIn"], 2e-3)

