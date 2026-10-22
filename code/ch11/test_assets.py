"""Tests of the listings in code/ch11/snippets and of the numbers printed beside the later figures."""
import math, re, shutil, subprocess, sys
import numpy as np
import pytest
from conftest import close, HERE
import ch11lib as L

ROOT = HERE.parents[1]
SN = HERE / "snippets"


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def nums(text):
    return re.findall(r"-?\d+\.\d+", text)


def test_waterfall_listing(NUM):
    out = run([sys.executable, str(SN / "waterfall.py")])
    w = L.european_waterfall([0, 3, 6], [100, 0, 0], [0, 50, 140], 0.08, 0.20, 1.0)
    rows = out.strip().splitlines()
    for j, line in enumerate(rows[:3]):
        assert nums(line) == [f"{w[k][j]:.2f}" for k in ("roc", "pref", "catch", "gp_split")]
    assert nums(rows[3]) == [NUM["wlGp"], NUM["wlLp"], "0.20"] == ["18.00", "172.00", "0.20"]
    assert NUM["wlGp"] != NUM["yaGp"]                  # the listing no longer prints Exercise 11.1


def test_lbo_listing(NUM):
    out = run([sys.executable, str(SN / "lbo.py")])
    assert nums(out) == [NUM[k] for k in ("lxEt", "lxEvt", "lxNdt", "lxEqt", "lxMoic", "lxIrr", "lxMoicU", "lxIrrU")]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_implied_growth_listing(NUM):
    out = run(["Rscript", str(SN / "implied_growth.R")])
    assert nums(out) == [NUM["mkWmed"], NUM["mkRocMed"], NUM["mkObsMed"], NUM["mkGstar"], NUM["mkGstar"]]


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_minority_listing(NUM):
    out = run(["octave-cli", "--no-gui", str(SN / "minority.m")])
    (ROOT / "octave-workspace").unlink(missing_ok=True)
    got = re.findall(r"put (\d+\.\d+) lookback (\S+)", out)
    want = [(NUM[f"mgPut{s}{t}"], NUM[f"mgLb{s}{t}"]) for s in ("Thirty", "FortyFive", "Sixty")
            for t in ("One", "Two", "Four")]
    assert got == want
    assert NUM["mgPutFortyFiveTwo"] == NUM["dmStake"]               # the case of the text


def test_minority_grid(RAW, NUM):
    assert sum(NUM[k] == "n.a." for k in NUM if k.startswith("mgLb")) == RAW["mgMissing"] == 1
    assert RAW["mgLbRawSixtyFour"] >= 1 > RAW["mgLbRawSixtyTwo"]
    for s in ("Thirty", "FortyFive", "Sixty"):                      # value falls with the period
        assert RAW[f"mgPut{s}One"] > RAW[f"mgPut{s}Two"] > RAW[f"mgPut{s}Four"]


def test_catchup_rates(RAW):
    assert RAW["ckFullOne"] < RAW["ckFullEight"] < RAW["ckFullHalf"]
    assert RAW["ckFullOne"] > RAW["wfMoicAtHurdle"]
    assert RAW["ckShareZero"] < 100 * 0.20 and RAW["ckZeroMid"] > RAW["wfNiMid"]


def test_carry_paths(RAW):
    close(RAW["ctIdErr"], 0, 1e-12)                     # Equation (timing) on the toy fund
    close(RAW["ctAPeak"] - RAW["ctClaw"], RAW["ctFinal"], 1e-12)
    assert RAW["ctPvGap"] > 0


def test_transitions(RAW, NUM):
    for nm in ("Three", "Five"):
        M = np.array(RAW[f"trMat{nm}"])
        close(np.abs(M.sum(axis=1) - 1).max(), 0, 1e-12)
        assert NUM[f"trTopTop{nm}"] == NUM[f"jcTop{nm}"] and RAW[f"trMiss{nm}"] == 0


def test_duration_example(RAW):
    close(RAW["duEx"], math.log(1.2) / math.log(1.25), 1e-12)


def test_driver_curves(RAW):
    assert RAW["drFiveTwenty"] > RAW["drFiveTwelve"] > RAW["drZero"] > RAW["drFiveSix"]


def test_wacc_rules(RAW):
    close(RAW["wfTargetGap"], 0, 1e-12)                 # the three rules at the target are the text's
    # Harris and Pringle with a riskless debt beta: W = rU + L (rD - rf - tau rD)
    rU, rd, rf = RAW["wkRu"] / 100, RAW["wkRd"] / 100, RAW["wkRf"] / 100
    close(RAW["wrHpSixty"] / 100, rU + 0.6 * (rd - rf - 0.25 * rd), 1e-12)
    close(RAW["wrMeZero"], RAW["wkRu"], 1e-12)


def test_dcf_table(RAW):
    yrs = ("One", "Two", "Three", "Four", "Five")
    close(sum(RAW[f"vdPv{y}"] for y in yrs) + RAW["vdPvTv"], RAW["vlEv"], 1e-9)
    for y in yrs:
        close(RAW[f"vdNopat{y}"] + RAW[f"vdDa{y}"] - RAW[f"vdCapex{y}"] - RAW[f"vdDwc{y}"], RAW[f"vdFcf{y}"], 1e-12)
        close(RAW[f"vdFcf{y}"], RAW[f"vlFcf{y}"], 1e-12)


def test_round3_printed_values(RAW, NUM):
    """The medians typed into the R listing match the data, the transition rows add to 100,
    and the 45%, four year put cell rounds from its unrounded value."""
    import csv
    row = next(csv.DictReader(open(HERE / "out" / "multiple_medians.csv")))
    assert f"{float(row['multiple']):.2f}" == "25.03"
    assert f"{float(row['wacc']):.5f}" == "0.07134" and f"{float(row['roc']):.5f}" == "0.15596"
    for nm in ("Three", "Five"):
        P = np.array(RAW[f"trInt{nm}"]); M = 100 * np.array(RAW[f"trMat{nm}"])
        assert (P.sum(axis=1) == 100).all() and np.abs(P - M).max() < 1
    assert f"{RAW['mgPutFortyFiveFour']:.2f}" == NUM["mgPutFortyFiveFour"] == "10.97"
