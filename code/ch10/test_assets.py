"""Tests of the listings in code/ch10/snippets and of the numbers printed beside the later figures."""
import re, shutil, subprocess, sys
import numpy as np
import pytest
from conftest import close, HERE
import ch10lib as L

ROOT = HERE.parents[1]
SN = HERE / "snippets"


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout


def tokens(text):
    return re.findall(r"-?\d[\d,]*\.\d+|\d{4}Q\d", text)


def test_dcf_listing(NUM):
    out = run([sys.executable, str(SN / "dcf.py")])
    assert tokens(out) == [NUM[k] for k in ("pfValue", "pfValue", "pfExit", "pfValueDcf", "pfIrr", "pfIrrSame")]


def test_ltv_cycle_listing(NUM):
    out = run([sys.executable, str(SN / "ltv_cycle.py")])
    assert out.startswith(f"{NUM['lcN']} vintages; worst {NUM['lcWorst']} peaks at {NUM['lcWorstPeak']}%; "
                          f"{NUM['lcShareEighty']}% of vintages")


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_unsmooth_listing(NUM):
    out = run(["Rscript", str(SN / "unsmooth.R")])
    assert tokens(out) == [NUM[k].replace("\\ensuremath{-}", "-") for k in
                           ("smArPreX", "smVolPre", "smUvolPre", "smRatioPre", "smTheoryPre", "smUarPre")]


@pytest.mark.skipif(shutil.which("octave-cli") is None, reason="GNU Octave not installed")
def test_octave_rotation_listing(NUM):
    out = run(["octave-cli", "--no-gui", str(SN / "rotation.m")])
    (ROOT / "octave-workspace").unlink(missing_ok=True)
    want = []
    for tag in ("Three", "Four", "Six"):
        want += [NUM[f"tbF{tag}"], NUM[f"tbS{tag}"]]
    want.append(NUM["tbFPriceGrowth"])
    assert tokens(out) == want
    lev = [int(v) for v in re.findall(r"land (\d+)", out)]
    assert lev == [round(L.land_value(L.faustmann_rotation(40.0, r, 1500.0), 40.0, r, 1500.0))
                   for r in (0.03, 0.04, 0.06)]
    assert f"{lev[1]:,}" == NUM["tbLevFour"]


def test_storage_lp_matches_pairs(NUM):
    sys.path.insert(0, str(SN))
    from storage_lp import intrinsic_lp
    F, t = [3.00, 3.60, 3.20], [1 / 12, 4 / 12, 7 / 12]
    v, *_ = intrinsic_lp(F, t, 0.04, 0.05, 0.05)
    close(v, L.intrinsic_storage(F, t, 0.04, 0.05, 0.05)[0], 1e-9)
    assert f"{v:.3f}" == NUM["xoV"]
    m = np.arange(12); G = 3.20 + 0.60 * np.cos(2 * np.pi * (m - 4) / 12)
    v1, *_ = intrinsic_lp(G, (m + 1) / 12, 0.04, 0.05, 0.05)
    v3, *_ = intrinsic_lp(G, (m + 1) / 12, 0.04, 0.05, 0.05, rate=1 / 3)
    assert v1 >= L.intrinsic_storage(G, (m + 1) / 12, 0.04, 0.05, 0.05)[0] - 1e-9 and v3 <= v1


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="R not installed")
def test_r_inflation_listing(NUM):
    out = run(["Rscript", str(SN / "inflation.R")])
    assert out.startswith(f"{NUM['ihNCre']} years") and tokens(out) == [NUM["ihBCre"], NUM["ihSeCre"]]


def test_loan_rate_figure(RAW):
    # the DSCR binds above the rate at which cap / (DSCR k) equals the LTV limit
    from leverage import LTV_MAX, DSCR_MIN, AMORT
    for c, tag in ((0.05, "Low"), (0.065, "High")):
        close(c / (DSCR_MIN * L.mortgage_constant(RAW[f"lrSwitch{tag}"] / 100, AMORT)), LTV_MAX, 1e-9)
    assert RAW["lrSwitchLow"] < RAW["lvRateSwitch"] < RAW["lrSwitchHigh"]
    close(RAW["lrLtvThree"], 100 * LTV_MAX, 1e-12)


def test_smoothing_simulation(RAW):
    close(RAW["ssIdErr"], 0, 1e-12)                     # Proposition 10.7(a) path by path
    assert RAW["ssVolRep"] < RAW["ssVolTrue"]
    assert RAW["ssQ"] == 40


def test_marginal_rule(RAW):
    close(RAW["mgFocErr"], 0, 1e-8)
    close(RAW["mgRent"], 0.04 * float(RAW["tbLevFour"]), 1e-9)


def test_curves(RAW):
    assert RAW["cvBackSpread"] < 0 < RAW["cvAprSpread"]
    assert abs(RAW["cvMedSpread"]) < 1
    assert RAW["cvAprCarry"] < 1                         # the bill was near zero in April 2020


def test_scatter(RAW):
    assert RAW["scYears"] == RAW["ihNCre"] == 55
    assert abs(RAW["scBigRes"]) > 3 * RAW["scResSd"]


def test_lag_simulation(RAW):
    # Exercise 10.16: b_k = 1 - 1/k; the simulated means sit near it and the Newey-West
    # rejection rates exceed the nominal 5% more as the horizon lengthens
    for k, kw in ((1, "One"), (5, "Five"), (10, "Ten")):
        assert abs(RAW[f"lsMean{kw}"] - (1 - 1 / k)) < 0.1
    assert 5 < RAW["lsRejOne"] < RAW["lsRejFive"] < RAW["lsRejTen"]
    assert RAW["lsSeTen"] < RAW["lsSdTen"]


def test_nw_matches_statsmodels():
    from inflation import nw_slope
    rng = np.random.default_rng(1)
    x = rng.standard_normal(55); y = 0.5 * x + rng.standard_normal(55)
    for k in (1, 5, 10):
        sy = np.convolve(y, np.ones(k), "valid"); sx = np.convolve(x, np.ones(k), "valid")
        b, se = nw_slope(sy, sx, k)
        b0, se0, _ = L.horizon_beta(y, x, k)
        close(b, b0, 1e-12); close(se, se0, 1e-12)


def test_round3_printed_values(NUM):
    """The storage listing's seasonal values, the model's reported volatility and the
    residual spread set against the spread of inflation."""
    assert NUM["xoSeasOne"] == "0.748" and NUM["xoSeasThird"] == "0.408"
    out = run([sys.executable, str(SN / "storage_lp.py")])
    assert f"{NUM['xoSeasOne']} filling at once, {NUM['xoSeasThird']} at a third a month" in out
    assert NUM["ssVolFactor"] == "0.29"
    assert abs(float(NUM["ssVolModel"]) - float(NUM["ssVolTrue"]) * 0.2860) < 0.06
    assert round(float(NUM["scResSd"]) / float(NUM["ihInflSd"])) == 3


def test_exit_cap_conventions_and_residual_sd(RAW):
    """The exit-cap example uses the log convention of Answer 10.2, and the residual standard deviation
    of the property scatter cannot exceed the real volatility of Table 10.3 under one ddof convention."""
    import math
    assert abs(RAW["pfTenthLog"] - RAW["pfSaleShare"] * math.log(1.1)) < 1e-9
    assert abs(RAW["pfTenthLogVal"] - 100 * (1 - math.exp(-RAW["pfTenthLog"] / 100))) < 1e-9
    assert RAW["pfTenthExact"] < RAW["pfTenthLogVal"] < RAW["pfTenthLog"]
    assert RAW["scResSd"] <= RAW["ihVolCre"] + 1e-12
