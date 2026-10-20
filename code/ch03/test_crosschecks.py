"""Cross-language checks (R and GNU Octave) and checks of the text and the exercise answers."""
import csv, math, re, shutil, subprocess
import numpy as np
import pytest
from scipy import stats
from conftest import close, HERE
import ch03lib as L

ROOT = HERE.parents[1]


def test_r_crosscheck(RAW):
    rows = list(csv.DictReader(open(HERE / "out" / "r_check.csv")))
    assert len(rows) >= 35
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], 1e-8 * max(1.0, abs(v)))


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_crosscheck(RAW):
    subprocess.run(["octave-cli", "--no-gui", str(HERE / "octave" / "ch03_check.m")], check=True,
                   cwd=ROOT, capture_output=True)
    rows = list(csv.DictReader(open(HERE / "out" / "octave_check.csv")))
    assert len(rows) >= 18
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], float(r["tol"]) * max(1.0, abs(v)))


def test_every_macro_used_is_defined():
    tex = (ROOT / "chapters" / "ch03.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch03-numbers.tex").read_text()))
    sty = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "preamble" / "quantbook.sty").read_text()))
    used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    known = {"lstinputlisting", "setcounter", "includegraphics", "printbibliography", "textcite", "parencite",
             "mathrm", "operatorname", "mathbb", "mathcal", "varepsilon", "lVert", "rVert", "Longrightarrow",
             "Longleftrightarrow", "xrightarrow", "mathbf", "boldsymbol", "lfloor", "rfloor", "displaystyle",
             "mathbbm", "textstyle", "underbrace", "overline", "widehat", "leqslant", "geqslant", "lVertt"}
    missing = sorted(u for u in used if u not in defined | sty | known)
    assert not missing, missing


def test_no_dashes_in_prose():
    tex = (ROOT / "chapters" / "ch03.tex").read_text()
    assert "---" not in tex and "\u2014" not in tex and "\u2013" not in tex
    body = re.sub(r"\$[^$]*\$", "", tex)
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
    assert not re.search(r"[A-Za-z]--[A-Za-z ]", body) and " -- " not in body


def test_exercise_answers(RAW):
    close(RAW["exOneVar"], 5, 0); close(RAW["exOneCov"], 1, 0)
    close(RAW["exTwo"], 12, 0)
    rng = np.random.default_rng(71)
    W2 = rng.standard_normal(400000) * math.sqrt(2)
    close(np.mean(W2 ** 4), 12, 4 * np.std(W2 ** 4) / math.sqrt(4e5))
    close(RAW["exThreeMean"], 100 * math.exp(0.16), 1e-9)
    close(RAW["exThreeLoss"], 100 * stats.norm.cdf(-0.04875 * math.sqrt(2) / 0.25), 1e-9)
    close(RAW["exFourHalf"], 2 * math.log(2), 1e-12); close(RAW["exFourSd"], 2.0, 1e-12)
    close(RAW["exFive"], 9, 1e-12)
    # int_0^3 W dt = int_0^3 (3 - t) dW has variance 9 too
    _, W = L.bm_paths(600, 20000, 3.0, rng)
    close(np.var(W[:, :-1].sum(axis=1) * 3 / 600), 9.0, 4 * 9 * math.sqrt(2 / 20000) + 0.05)
    close(RAW["exSixTauZero"], 0.5, 1e-12)
    close(RAW["exSeven"], 1, 0)
    close(RAW["exEightB"], 1.2 ** 2 + 0.18, 1e-12)
