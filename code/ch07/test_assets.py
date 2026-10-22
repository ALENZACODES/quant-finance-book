"""Tests of the listings in code/ch07/snippets and of the numbers printed beside the later figures."""
import re, shutil, subprocess, sys
import numpy as np
import pytest
from conftest import close, HERE
import ch07lib as L

ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "snippets"))
import brinson_carino as BC  # noqa: E402


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def floats(text):
    return [float(v) for v in re.findall(r"-?\d+\.\d+", text)]


def py(name):
    return run([sys.executable, str(HERE / "snippets" / name)])


def fmt(v, d):
    return float(f"{v:.{d}f}")


def test_closed_form_listing(RAW):
    v = floats(py("closed_form.py"))
    assert v[:2] == [fmt(RAW["frGmvMu"], 2), fmt(RAW["frGmvVol"], 1)]
    assert v[3:6] == [fmt(RAW["frTanMu"], 2), fmt(RAW["frTanVol"], 1), fmt(RAW["frTanSR"], 2)]
    assert v[6:9] == [fmt(RAW["frGmvMu"], 2), fmt(RAW["frTanMu"], 2), fmt(RAW["frTanSR"], 2)]


def test_hac_listing_matches_table(RAW):
    row = next(r for r in RAW["alphaTable"] if r["name"] == "Hlth")
    v = floats(py("hac_alpha.py"))
    assert v == [fmt(row["a1"], 2), fmt(row["t1ols"], 2), fmt(row["t1"], 2),
                 fmt(row["a6"], 2), fmt(row["t6ols"], 2), fmt(row["t6"], 2)]


def test_brinson_carino_listing(NUM, RAW):
    a, s, i = BC.brinson([0.7, 0.3], [0.6, 0.4], [0.12, 0.01], [0.10, 0.02])
    close(a.sum() + s.sum() + i.sum(), 0.7 * 0.12 + 0.3 * 0.01 - 0.068, 1e-15)
    v = floats(py("brinson_carino.py"))
    assert v[:3] == [float(NUM["xrSixAlloc"]), float(NUM["xrSixSel"]), float(NUM["xrSixInt"])]
    assert v[3] == v[4] and v[5:] == [float(NUM["xrSevenKone"]), float(NUM["xrSevenKtwo"])]
    # same formulas as the library, and the linked effects add to the active return
    rng = np.random.default_rng(5)
    Rp, Rb = rng.normal(0.01, 0.04, 8), rng.normal(0.01, 0.04, 8)
    k, P, B = BC.carino(Rp, Rb); kl, Pl, Bl = L.carino(Rp, Rb)
    close(np.max(np.abs(k - kl)), 0, 1e-14); close(np.sum(k * (Rp - Rb)), P - B, 1e-14)
    w, b = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5)); rp, rb = rng.normal(0, .05, (2, 5))
    for x, y in zip(BC.brinson(w, b, rp, rb), L.brinson(w, b, rp, rb)):
        close(np.max(np.abs(x - y)), 0, 1e-15)


def test_rolling_gmv_listing(RAW):
    assert floats(py("rolling_gmv.py")) == [fmt(RAW["csGmvSR"], 2), fmt(RAW["csLwSR"], 2)]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_grs_listing(RAW):
    v = floats(run(["Rscript", str(HERE / "snippets" / "grs.R")]))
    assert v == [fmt(RAW["grsCapm"], 4), fmt(RAW["grsPCapm"], 4), fmt(RAW["grsSix"], 4), fmt(RAW["grsPSix"], 4)]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_bh_listing(RAW):
    out = run(["Rscript", str(HERE / "snippets" / "bh.R")])
    n = [int(x) for x in re.findall(r"(\d+) flagged, (\d+) without", out)[0] + re.findall(r"(\d+) flagged, (\d+) without", out)[1]]
    assert n == [RAW["mgBh"], RAW["mgBhFalse"], RAW["mgLongBh"], RAW["mgLongBhFalse"]]
    assert floats(out)[0] == fmt(RAW["mgPiZero"], 2)


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_long_only_listing(RAW):
    out = run(["octave-cli", "--no-gui", str(HERE / "snippets" / "lo_tangency.m")])
    assert floats(out) == [fmt(RAW["frLoSR"], 2)] and f"with {int(RAW['frLoN'])} industries" in out


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_erc_listing(RAW, NUM):
    out = run(["octave-cli", "--no-gui", str(HERE / "snippets" / "erc.m")])
    v = floats(out)
    assert v[0] == v[1] == float(NUM["xrNineW"]) / 100
    assert v[2:8] == [fmt(100 * w, 1) for w in RAW["rpWerc"]] and v[8] == 1.0


def test_frontier_weights_numbers(RAW, NUM):
    assert RAW["fwAffineErr"] < 1e-12 and RAW["fwGmvErr"] < 1e-12 and RAW["fwTanErr"] < 1e-12
    close(RAW["fwDoubleMu"], 2 * RAW["frTanMu"], 1e-9)
    assert NUM["fwDoubleShortName"] == "Telcm" and RAW["fwDoubleGross"] > RAW["frGross"]


def test_convergence_numbers(RAW):
    assert RAW["cvMaxRelErr"] < 0.02                    # simulated in-sample values on the exact curve
    out = np.array(RAW["cvOut"]); assert np.all(np.diff(out) > 0) and out[-1] < RAW["frTanSR"]
    assert RAW["cvTs"][0] == 36 and RAW["cvTs"][-1] == RAW["cvTmax"] == 600
    close(RAW["cvShareMax"], 100 * out[-1] / RAW["frTanSR"], 1e-6)


def test_tangency_sml_and_corr_gap(RAW, NUM):
    assert RAW["cpTanBetaErr"] < 1e-12 and RAW["cpTanBetaMin"] < RAW["cpTanBetaMax"]
    close(abs(RAW["fmGapMax"]), RAW["fmCorrErr"], 1e-12)  # the largest gap is the "within" number
    assert RAW["fmGapPairs"] == 45 and 0 < RAW["fmGapPos"] < 45 and NUM["fmGapPair"] == "Enrgy and Shops"


def test_appraisal_drawdowns(RAW):
    f, vw, ew, cap = L.load()
    r = (vw["Other"] - f.rf).values / 100 if vw["Other"].abs().max() > 1 else (vw["Other"] - f.rf).values
    idx = f.index.values; win = (idx >= 200710) & (idx <= 200912)
    for phi, tag in ((None, "True"), (0.5, "Half"), (0.8, "Eight")):
        x = r if phi is None else L.smooth_ar1(r, phi)
        W = np.concatenate([[1.0], np.cumprod(1 + x[win])])
        close(100 * (1 - W / np.maximum.accumulate(W)).max(), RAW[f"smoDd{tag}"], 1e-9)
    assert RAW["smoDdTrue"] > RAW["smoDdHalf"] > RAW["smoDdEight"]


def test_waterfall_adds_up(RAW):
    parts = [RAW[k] for k in ("faMkt", "faSmb", "faHml", "faRmw", "faCma", "faMom", "faAlpha")]
    close(sum(parts), RAW["faMean"], 1e-9); assert RAW["faWaterErr"] < 1e-12
    close(RAW["faRoundedSum"], sum(fmt(v, 2) for v in parts), 1e-9)
    assert fmt(RAW["faRoundedSum"], 2) == 9.22 and fmt(RAW["faMean"], 2) == 9.20


def test_round3_printed_values(RAW, NUM):
    """The Carino factor in the equal-return case, the range of z statistics quoted against
    equal weights, the smoothing reconciliation and the put-writing intercept."""
    import importlib, sys as _s
    _s.path.insert(0, str(HERE / "snippets")); bc = importlib.import_module("brinson_carino")
    f, P, B = bc.carino([0.01, 0.02], [0.01, 0.02])                    # P == B: k = 1 / (1 + P)
    assert np.allclose(f, (1 + P) / (1 + np.array([0.01, 0.02])))
    zs = {k: RAW[k] for k in ("csCapZ", "csGmvZ", "csLwZ", "csLogmvZ")}
    assert min(zs, key=zs.get) == "csCapZ" and max(zs, key=zs.get) == "csLwZ" and max(map(abs, zs.values())) < 1.96
    assert abs(RAW["smoVolExact"] - RAW["smoVolHalf"]) < 0.1 < RAW["smoVolHalf"] - RAW["smoVolFormula"]
    assert abs(RAW["smoSRFormula"] * RAW["smoVolFormula"] / RAW["smoVolHalf"] * RAW["smoMeanKept"] / 100
               - RAW["smoSRHalf"]) < 0.005
    assert RAW["poTwentyHmA"] > RAW["poTwentyPrem"] > RAW["poTwentyFlat"]
    four = -sum(float(f"{RAW[k]:.2f}") for k in ("faHml", "faRmw", "faCma", "faMom"))
    assert abs(four - RAW["faFourRoundedSum"]) < 1e-9
