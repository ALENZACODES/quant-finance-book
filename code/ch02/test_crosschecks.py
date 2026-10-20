"""Cross-language checks (R and GNU Octave) and checks of the text and the exercise answers."""
import csv, re, shutil, subprocess
from pathlib import Path
import numpy as np
import pytest
from conftest import close, HERE

ROOT = HERE.parents[1]


def _csv(name):
    return {r["name"]: float(r["value"]) for r in csv.DictReader(open(HERE / "out" / name))}


def test_r_crosscheck(RAW):
    rows = _csv("r_check.csv")
    assert len(rows) >= 20
    for k, v in rows.items():
        close(v, RAW[k], 1e-6)


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_crosscheck(RAW):
    subprocess.run(["octave-cli", "--no-gui", str(HERE / "octave" / "ch02_check.m")], check=True,
                   cwd=ROOT, capture_output=True)
    rows = _csv("octave_check.csv")
    assert len(rows) >= 15
    for k, v in rows.items():
        close(v, RAW[k], 1e-6)


def test_every_macro_used_is_defined():
    tex = (ROOT / "chapters" / "ch02.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch02-numbers.tex").read_text()))
    used = set(re.findall(r"\\((?:b|sp|carry|mc|hf|mod|opt|share|ann|fof|net|re|ta|prog|ex|alt)[A-Z][A-Za-z]*)", tex))
    assert used <= defined, used - defined


def test_no_dashes_in_prose():
    tex = (ROOT / "chapters" / "ch02.tex").read_text()
    assert "---" not in tex and "\u2014" not in tex and "\u2013" not in tex
    body = re.sub(r"\$[^$]*\$", "", tex)
    assert " -- " not in body


def test_exercise_answers(RAW):
    close(RAW["exOneA"], 29.0, 1e-12)
    close(RAW["exThreeGP"], 10.0, 1e-9)
    close(RAW["exThreeGPHard"], 0.2 * (150 - 100 * 1.08 ** 4), 1e-9)
    close(RAW["exFourIncOne"], 3.7, 1e-9)
    close(RAW["exFourCloseTwo"], 114.8 * 0.85 - 0.015 * 114.8, 1e-9)
    close(RAW["exFourIncThree"], 0.0, 1e-12)
    phi, G = 0.015 + 0.15 * 0.065, 0.85 * 0.065
    close(phi, 0.02475, 1e-12); close(G, 0.05525, 1e-12)
    q = 1.05525 / 1.08
    close(RAW["exSixPV"], phi * 500 / 1.08 * (1 - q ** 5) / (1 - q), 1e-9)
    close(RAW["exSevenRatio"], np.exp(0.05), 1e-12)
