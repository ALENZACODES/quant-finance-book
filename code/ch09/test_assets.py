"""Tests of the listings in code/ch09/snippets and of the numbers printed beside the later figures."""
import math, re, shutil, subprocess, sys
import numpy as np
import pandas as pd
import pytest
from conftest import close, HERE
import ch09lib as L

ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "snippets"))
import tsmom  # noqa: E402


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def floats(text):
    return [float(v) for v in re.findall(r"-?\d+\.\d+", text)]


def fmt(v, d):
    return float(f"{v:.{d}f}")


def test_tsmom_listing_matches_library(RAW):
    r, _ = L.load_markets()
    from trend import program
    for k in (1, 3, 6, 12, 24):
        a, b = tsmom.programme(r, k), program(r, k)[2]
        assert a.index.equals(b.index); close(np.max(np.abs(a.values - b.values)), 0, 1e-15)
    out = run([sys.executable, str(HERE / "snippets" / "tsmom.py")])
    assert floats(out) == [fmt(RAW["lbSROne"], 2), fmt(RAW["lbSRThree"], 2), fmt(RAW["lbSRTwelve"], 2)]


def test_permutation_listing(RAW):
    out = run([sys.executable, str(HERE / "snippets" / "permutation.py")])
    assert floats(out) == [fmt(RAW["tfSR"], 2), fmt(RAW["pmP"], 3)]
    assert f"{int(RAW['pmCount'])} of {int(RAW['pmN'])} shuffles" in out


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_fee_listing(RAW):
    v = floats(run(["Rscript", str(HERE / "snippets" / "fees.R")]))
    assert v[:4] == [fmt(RAW["tfFeeMgmt"], 2), fmt(RAW["tfFeeInc"], 2), fmt(RAW["tfSR"], 2), fmt(RAW["tfSRnet"], 2)]
    assert v[4] == v[5] > 0                              # fees paid equal f / (1 - f) times the rise in the mark


def test_rolldown_listing(RAW):
    v = floats(run([sys.executable, str(HERE / "snippets" / "rolldown.py")]))
    assert v == [fmt(RAW[k], 2) for k in ("gcCarry", "gcRoll", "gcReal", "gcRest")]


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_barrier_listing(RAW):
    v = floats(run(["octave-cli", "--no-gui", str(HERE / "snippets" / "barrier.m")]))
    assert v == [fmt(RAW["clSurvFive"], 1)] * 2 + [fmt(RAW["clSurvBgkFive"], 1)] + \
        [fmt(RAW["clSurvTen"], 1)] * 2 + [fmt(RAW["clSurvBgkTen"], 1)]


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_issuance_listing(RAW):
    out = run(["octave-cli", "--no-gui", str(HERE / "snippets" / "issuance.m")])
    assert out.startswith(f"{L.total_supply_sats()} satoshis")
    assert f"{RAW['ocShortfall']:.4f} short" in out and f"after {int(RAW['ocHalvings'])} halvings" in out
    assert L.total_supply_sats() < 2 ** 53


def test_decile_numbers(RAW):
    m = np.array(RAW["dqMeans"]); assert len(m) == 10
    close(RAW["dqBottom"], m[0], 1e-12); close(RAW["dqTop"], m[9], 1e-12)
    assert RAW["dqTailShare"] < 0 and abs(RAW["dqRho"]) < 0.05 and RAW["dqObs"] > 2000


def test_positions_and_sleeves(RAW):
    r, _ = L.load_markets()
    held = L.tsmom_weights(r).shift(1)
    close(held.loc[202002, "EQ"], RAW["psEqFebTwenty"], 1e-12); assert RAW["psEqAprTwenty"] < 0 < RAW["psEqMarTwenty"]
    assert RAW["psTwoTwoShortUst"] == RAW["psTwoTwoShortGoc"] == 12
    assert RAW["scMax"] == RAW["scBonds"] and RAW["scPairs"] == 21 and RAW["scMonths"] <= RAW["tfMonths"]


def test_clone_coefficients(RAW):
    assert RAW["cfFitErr"] < 1e-12 and RAW["cfChangeA"] > RAW["cfChangeB"] and RAW["cfMonths"] == RAW["rpMonths"]


def test_risk_share_halves(RAW):
    for tag in ("Early", "Late"):
        assert 0 < RAW[f"rsWTen{tag}"] < RAW[f"rsWQuarter{tag}"] < RAW[f"rsWHalf{tag}"] < 20
    assert RAW["rsCorrLate"] > RAW["rsCorrEarly"] and RAW["rsWHalfLate"] > RAW["rsWHalfEarly"]
    assert RAW["rsVolBtcLate"] < RAW["rsVolBtcEarly"] and RAW["rsVolSixtyLate"] > RAW["rsVolSixtyEarly"]
    # the share at the found weight is the target share (inverse of L.risk_share)
    a, b, c = 1.0, 40.0, 2.0
    for s in (0.1, 0.25, 0.5):
        close(L.risk_share(L.weight_for_share(s, a, b, c), a, b, c), s, 1e-12)


def test_wealth_gap_and_selection(RAW):
    pr = pd.read_csv(HERE / "out" / "program.csv")
    A = 0.6 * pr["eq"].values + 0.4 * pr["ust"].values; B = 0.9 * A + 0.1 * pr["net"].values
    lw = lambda x: np.sum(np.log1p(x + pr["rf"].values))
    close(100 * (lw(B) - lw(A)), RAW["cwEnd"], 1e-9)
    close(RAW["cwFeeWedge"], RAW["cwEndGross"] - RAW["cwEnd"], 1e-9)
    assert RAW["sxMedian"] < RAW["sxQ"] < RAW["sxActual"] and RAW["sxAbove"] < 5
    q = RAW["sxTail"]; n = math.log(0.5) / math.log(1 - q)
    assert abs(n - RAW["sxNeeded"]) <= 50 and 1 - (1 - q) ** RAW["sxNeeded"] == pytest.approx(0.5, abs=0.02)


def test_canadian_bond_entry(RAW, NUM):
    """The Canadian bond's first signal is a month before its first position, and the months in which all
    seven sleeves hold positions run from that first position to the end of the record."""
    import pandas as pd
    first = pd.Period(pd.Timestamp(NUM["tfGocFirstPos"]), "M"); sig = pd.Period(pd.Timestamp(NUM["tfGocFirst"]), "M")
    last = pd.Period(pd.Timestamp(NUM["tfLast"]), "M")
    assert first == sig + 1 and RAW["scMonths"] == (last - first).n + 1
