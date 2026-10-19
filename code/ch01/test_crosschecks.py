"""Chapter 1 cross-checks: Octave, exercise answers, macro hygiene and the propositions of the chapter.

Run from the repository root:  python -m pytest -q code/ch01
The R cross-check (code/ch01/crosscheck.R) must have been run first; run_all.py writes numbers.json.
"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
import numpy_financial as npf
import pytest
from scipy import stats
from statsmodels.stats.stattools import jarque_bera as sm_jb, durbin_watson as sm_dw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import ch01lib as L  # noqa: E402

RAW = json.loads((HERE / "out" / "numbers.json").read_text())["RAW"]
rng = np.random.default_rng(7)


def close(a, b, tol):
    assert abs(a - b) <= tol, (a, b)


def test_exercise_answers():
    close((1.005 ** 12 - 1) * 100, 6.17, 0.005); close(12 * np.log(1.005) * 100, 5.985, 0.0005)
    close(np.log(0.95) * 100, -5.129, 0.0005)
    assert L.irr_all([-1, 3, -2], lo=-0.5) == pytest.approx([0.0, 1.0], abs=1e-9)
    close(L.irr([-100, 0, 160]), np.sqrt(1.6) - 1, 1e-10)
    close(L.irr([-100, 130]), 0.30, 1e-10)
    close(L.npv(160 / 130 - 1, [-100, 0, 160]), L.npv(160 / 130 - 1, [-100, 130]), 1e-9)
    close(L.npv(0.10, [-100, 0, 160]), 32.23, 0.005); close(L.npv(0.10, [-100, 130]), 18.18, 0.005)
    for r in (0.05, 0.20, 0.23):            # below the crossover the two-year project has the higher NPV
        assert L.npv(r, [-100, 0, 160]) > L.npv(r, [-100, 130])
    for r in (0.24, 0.26):                  # between crossover and its IRR both are accepted, one-year wins
        assert 0 < L.npv(r, [-100, 0, 160]) < L.npv(r, [-100, 130])
    close(L.irr([-50, 30, 22.5]) * 100, 3.48, 0.005)
    close((np.sqrt(0.99) - 1) * 100, -0.50, 0.005)
    close(1 + 2 * 0.5 * (-0.3), 0.7, 1e-12)
    close(60 / 6 * (0.36 + 1.44 / 4), 7.2, 1e-12)


# ---------------------------------------------------------------- Octave cross-check
import re, shutil  # noqa: E402


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_crosscheck():
    subprocess.run(["octave-cli", "--no-gui", str(HERE / "octave" / "ch01_check.m")], check=True,
                   cwd=ROOT, capture_output=True)
    rows = list(csv.DictReader((HERE / "out" / "octave_check.csv").open()))
    assert len(rows) >= 40
    for row in rows:
        k, v = row["name"], float(row["value"])
        if k == "fundIRRcount":
            assert v == 1
            continue
        close(v, RAW[k], 1e-8 * max(1.0, abs(RAW[k])))


# ---------------------------------------------------------------- text hygiene
def test_every_macro_used_is_defined():
    tex = (ROOT / "chapters" / "ch01.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (ROOT / "chapters" / "ch01-numbers.tex").read_text()))
    preamble = set(re.findall(r"\\(?:newcommand|DeclareMathOperator)\*?\{\\([A-Za-z]+)\}",
                              (ROOT / "preamble" / "quantbook.sty").read_text()))
    used = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    known_latex = {"lstinputlisting", "setcounter", "includegraphics", "addbibresource", "printbibliography",
                   "newrefsection", "begin", "end", "textcite", "parencite", "mathrm", "operatorname"}
    missing = sorted(u for u in used if u not in defined | preamble | known_latex)
    assert not missing, missing


def test_no_dashes_in_prose():
    tex = (ROOT / "chapters" / "ch01.tex").read_text()
    body = re.sub(r"\$[^$]*\$", "", tex)
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("%"))
    assert "---" not in body and "\u2014" not in body and "\u2013" not in body
    assert not re.search(r"[A-Za-z]--[A-Za-z ]", body)


# ---------------------------------------------------------------- the propositions of the chapter
def _conventional(r):
    k = int(r.integers(0, 4)); T = k + int(r.integers(1, 6))
    cf = np.zeros(T + 1)
    cf[: k + 1] = -r.uniform(0, 10, k + 1) * (r.uniform(size=k + 1) < 0.7)
    cf[int(r.integers(0, k + 1))] = -r.uniform(1, 10)
    cf[k + 1:] = r.uniform(0, 10, T - k) * (r.uniform(size=T - k) < 0.7)
    cf[int(r.integers(k + 1, T + 1))] = r.uniform(1, 10)
    return cf


def test_prop_unique_irr_for_conventional_flows():
    r = np.random.default_rng(11)
    for _ in range(300):
        cf = _conventional(r)
        roots = L.irr_all(cf, lo=-0.95, hi=20.0)
        assert len(roots) == 1, cf
        x = roots[0]
        assert np.sign(x) == np.sign(cf.sum()) or abs(cf.sum()) < 1e-12
        assert L.npv(x - 0.01, cf) > 0 > L.npv(x + 0.01, cf)


def test_prop_mirr_npv_identity():
    r = np.random.default_rng(12)
    for _ in range(300):
        cf = r.normal(0, 10, int(r.integers(2, 10)))
        if not ((cf < 0).any() and (cf > 0).any()):
            continue
        rate = r.uniform(-0.2, 0.5); T = len(cf) - 1
        m = L.mirr(cf, rate, rate)
        pvn = -np.sum(np.minimum(cf, 0) / (1 + rate) ** np.arange(T + 1))
        close(L.npv(rate, cf), pvn * (((1 + m) / (1 + rate)) ** T - 1), 1e-9 * max(1, pvn))
        assert np.sign(m - rate) == np.sign(L.npv(rate, cf))


def test_prop_pme_equals_index_discounted_npv():
    r = np.random.default_rng(13)
    for _ in range(200):
        T = int(r.integers(2, 12))
        C = r.uniform(0, 10, T + 1) * (r.uniform(size=T + 1) < 0.6); C[0] += 1
        D = r.uniform(0, 10, T + 1) * (r.uniform(size=T + 1) < 0.5); nav = r.uniform(0, 30)
        idx = 100 * np.cumprod(np.r_[1, 1 + r.normal(0.07, 0.15, T)])
        npv_i = np.sum((D - C) / idx) + nav / idx[-1]
        close(L.ks_pme(C, D, nav, idx) - 1, npv_i / np.sum(C / idx), 1e-12)
        g = r.uniform(-0.05, 0.25); idx_g = 100 * (1 + g) ** np.arange(T + 1)
        net = D - C; net[-1] += nav
        assert (L.ks_pme(C, D, nav, idx_g) > 1) == (L.npv(g, net) > 0)


def test_prop_lognormal_moments_algebra():
    # the factorisations used in the proof, as polynomial identities in w
    P = np.polynomial.Polynomial
    w = P([0, 1])
    assert np.allclose((w ** 3 - 3 * w + 2).coef, ((w - 1) ** 2 * (w + 2)).coef)
    assert np.allclose((w ** 6 - 4 * w ** 3 + 6 * w - 3).coef, ((w - 1) ** 2 * (w ** 4 + 2 * w ** 3 + 3 * w ** 2 - 3)).coef)
    for mu, s in [(0.06, 0.18), (0.3, 0.805), (-0.1, 0.5)]:
        d = stats.lognorm(s=s, scale=np.exp(mu))
        m, v, sk, ek = d.stats(moments="mvsk")
        m2, sd2, sk2, ek2 = L.lognormal_moments(mu, s)
        close(m, m2, 1e-12); close(np.sqrt(v), sd2, 1e-12); close(sk, sk2, 1e-9); close(ek, ek2, 1e-8)
        close(d.median(), np.exp(mu), 1e-12)
        xs = np.linspace(np.exp(mu - s ** 2) * 0.9, np.exp(mu - s ** 2) * 1.1, 20001)
        close(xs[np.argmax(d.pdf(xs))], np.exp(mu - s ** 2), 2e-5 * np.exp(mu))


def test_prop_ols_slope_is_beta():
    mkt = np.array(RAW["exMkt"]); fnd = np.array(RAW["exFnd"])
    b, a = np.polyfit(mkt, fnd, 1)
    close(b, RAW["exBeta"], 1e-10); close(a, fnd.mean() - b * mkt.mean(), 1e-10)


def test_prop_jb_asymptotics():
    r = np.random.default_rng(14)
    n, reps = 4000, 4000
    z = r.standard_normal((reps, n))
    d = z - z.mean(1, keepdims=True); m2 = (d ** 2).mean(1)
    S = (d ** 3).mean(1) / m2 ** 1.5; K = (d ** 4).mean(1) / m2 ** 2 - 3
    vs, vk = n * S.var(), n * K.var()
    assert abs(vs - 6) < 0.4 and abs(vk - 24) < 2.0, (vs, vk)
    assert abs(np.corrcoef(S, K)[0, 1]) < 0.06
    # the moments used in the proof
    mom = {k: stats.norm.moment(k) for k in range(1, 9)}
    close(mom[6] - 6 * mom[4] + 9 * mom[2], 6, 1e-9)
    close(mom[8] - 12 * mom[6] + 42 * mom[4] - 36 * mom[2] + 9, 24, 1e-9)
    close(RAW["jbCrit"], -2 * np.log(0.05), 1e-12)
