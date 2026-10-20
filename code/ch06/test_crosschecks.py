"""Cross-language checks (R, GNU Octave and the screener's own JavaScript) and checks of the text."""
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
    subprocess.run(["octave-cli", "--no-gui", str(HERE / "octave" / "ch06_check.m")], check=True,
                   cwd=ROOT, capture_output=True)
    rows = list(csv.DictReader(open(HERE / "out" / "octave_check.csv")))
    assert len(rows) >= 10
    for r in rows:
        v = float(r["value"]); close(v, RAW[r["name"]], float(r["tol"]) * max(1.0, abs(v)))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_screener_javascript_agrees():
    """The Python port of the screener gives the same grade and metrics as its calc.js."""
    js = r"""
const C = require(process.argv[1]); const D = JSON.parse(process.argv[2]); const o = {};
for (const k in D) { const s = C.score(D[k]); o[k] = {total: s.total, grade: s.grade, dscr: s.metrics.dscr,
  exitLtv: s.metrics.exitLtv, ltv: s.metrics.ltv, dy: s.metrics.debtYield}; }
console.log(JSON.stringify(o));"""
    from private_credit import DEALS
    res = subprocess.run(["node", "-e", js, str(HERE / "vendor" / "calc.js"), json.dumps(DEALS)],
                         capture_output=True, text=True, check=True)
    js_out = json.loads(res.stdout); py = json.loads((HERE / "out" / "screener_py.json").read_text())
    for k in DEALS:
        assert js_out[k]["total"] == py[k]["total"] and js_out[k]["grade"] == py[k]["grade"]
        for a, b in [("dscr", "dscr"), ("exitLtv", "exitLtv"), ("ltv", "ltv"), ("dy", "dy")]:
            close(js_out[k][a], py[k][b], 1e-12)


def test_every_macro_used_is_defined():
    tex = (ROOT / "chapters" / "ch06.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch06-numbers.tex").read_text()))
    sty = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "preamble" / "quantbook.sty").read_text()))
    used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    known = {"lstinputlisting", "setcounter", "includegraphics", "printbibliography", "textcite", "parencite",
             "mathrm", "operatorname", "mathbb", "mathcal", "varepsilon", "Longrightarrow", "mathbf",
             "boldsymbol", "displaystyle", "textstyle", "underbrace", "overline", "widehat", "addlinespace"}
    missing = sorted(u for u in used if u not in defined | sty | known)
    assert not missing, missing


def test_macros_do_not_clash_with_other_chapters():
    mine = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch06-numbers.tex").read_text()))
    for f in sorted((ROOT / "chapters").glob("ch0[1-5]-numbers.tex")):
        other = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", f.read_text()))
        assert not mine & other, (f.name, sorted(mine & other))


def test_no_dashes_in_prose():
    tex = (ROOT / "chapters" / "ch06.tex").read_text()
    assert "---" not in tex and "\u2014" not in tex and "\u2013" not in tex
    body = re.sub(r"\$[^$]*\$", "", tex)
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
    assert not re.search(r"[A-Za-z]--[A-Za-z ]", body) and " -- " not in body
