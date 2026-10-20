"""Checks for promote_pacing.py: promote tiers and the Takahashi-Alexander model."""
import numpy as np
import ch02lib as L
from conftest import close


def test_promote_conservation(RAW):
    close(RAW["reInvTot"] + RAW["reSpoTot"], RAW["reDist"], 1e-9)
    tot = sum(RAW["reTier" + n + "Inv"] + RAW["reTier" + n + "Spo"] for n in ("One", "Two", "Three"))
    close(tot, RAW["reDist"], 1e-9)
    # tier 1 is pro rata, tier 2 gives the sponsor 0.2 + 0.8 x 0.1 = 28% of each dollar
    close(RAW["reTierOneSpo"] / (RAW["reTierOneSpo"] + RAW["reTierOneInv"]), 0.10, 1e-9)
    close(RAW["reTierTwoSpo"] / (RAW["reTierTwoSpo"] + RAW["reTierTwoInv"]), 0.28, 1e-9)
    assert 8 < RAW["reInvIrr"] < 13 + 1e-9


def test_promote_hurdles_hit_exactly():
    c_t, c = [0.0], [10.0]
    dt = np.arange(1, 6, dtype=float); d = np.array([0.55 * 1.02 ** (t - 1) for t in range(1, 6)]); d[-1] += 15
    tiers = [("irr", 0.08, 0.0), ("irr", 0.12, 0.20), ("irr", None, 0.30)]
    inv, spo, by = L.tiered_waterfall(c_t, c, dt, d, tiers)
    # investor IRR on tier 1 and tier 2 payments alone hits 8% and 12% exactly at the boundary:
    # check with a one-distribution case where the boundary can be computed
    inv1, spo1, _ = L.tiered_waterfall([0], [10], [1], [9 * 1.08 / 0.9], tiers)
    close(inv1[0], 9 * 1.08, 1e-9)


def test_ta_closed_forms(RAW):
    C, D, NAV = L.takahashi_alexander(years=15)
    U = 1 - np.cumsum(C)
    close(U[1], 0.75, 1e-12); close(U[2], 0.5, 1e-12)
    for t in range(3, 16):
        close(U[t], 0.25 * 0.5 ** (t - 3), 1e-12)
    close(NAV[12], C[12], 1e-12)                                   # RD_L = 1: only the year's call remains
    C30, D30, NAV30 = L.takahashi_alexander(years=30)
    close(RAW["progNavSteady"], NAV30.sum(), 1e-9)                 # Proposition 2.12
    close(RAW["progNetCash"], D30[1:].sum() - C30[1:].sum(), 1e-6)
    close(RAW["progUncalledSteady"], 1 + 0.75 + 0.5 + 0.25 * 2, 1e-6)
    close(RAW["progCommitForHundred"] * RAW["progNavSteady"], 100, 1e-9)


def test_pacing_sensitivity(RAW):
    assert RAW["taPeakBowLow"] < RAW["taPeakBowMid"] < RAW["taPeakBowHigh"]
    assert RAW["taTVPIGLow"] < RAW["taTVPIGMid"] < RAW["taTVPIGHigh"]


# ---------------------------------------------------------------- corrected tier balance, Eq. (2.15)
def _tiers_by_definition(c_t, c, d_t, d, tiers, s=0.9, net_same_date=True):
    """Independent implementation of the tier balance of the definition:
    B_j(t) = sum_{t_i<=t} s C_i (1+h_j)^(t-t_i) - sum_{u<t} L_u (1+h_j)^(t-u) - l_{<j}(t)
    (multiple form: mu_j s sum C_i - sum_{u<t} L_u - l_{<j}(t)), where l_{<j}(t) is what the investor
    has already taken from tiers 1..j-1 of the distribution at t. net_same_date=False drops l_{<j}(t),
    the earlier version of the formula."""
    by = np.zeros((len(tiers), 2)); past = []           # past: (date, investor amount)
    for t, x in sorted(zip(d_t, d)):
        taken = 0.0                                     # l_{<j}(t)
        for j, (kind, lev, pi) in enumerate(tiers):
            frac = (1 - pi) * s
            if lev is None:
                y = x
            else:
                if kind == "irr":
                    B = (sum(s * ci * (1 + lev) ** (t - ti) for ti, ci in zip(c_t, c) if ti <= t)
                         - sum(a * (1 + lev) ** (t - u) for u, a in past))
                else:
                    B = lev * s * sum(ci for ti, ci in zip(c_t, c) if ti <= t) - sum(a for _, a in past)
                if net_same_date:
                    B -= taken
                y = min(x, max(B, 0.0) / frac)
            by[j] += (frac * y, (1 - frac) * y); taken += frac * y; x -= y
        past.append((t, taken))
    return by


def _example():
    dt = np.arange(1, 6, dtype=float); d = np.array([0.55 * 1.02 ** (t - 1) for t in range(1, 6)]); d[-1] += 15
    return [0.0], [10.0], dt, d


IRR_T = [("irr", 0.08, 0.0), ("irr", 0.12, 0.20), ("irr", None, 0.30)]
MUL_T = [("mult", 1.0, 0.0), ("mult", 1.6, 0.20), ("mult", None, 0.30)]


def test_corrected_tier_formula_reproduces_example(RAW):
    c_t, c, dt, d = _example()
    by = _tiers_by_definition(c_t, c, dt, d, IRR_T)
    close(by[1, 0], RAW["reTierTwoInv"], 1e-9); close(by[2, 0], RAW["reTierThreeInv"], 1e-9)
    close(by[:, 1].sum(), RAW["reSpoTot"], 1e-9)
    close(round(by[1, 0], 2), 2.39, 1e-12); close(round(by[2, 0], 2), 0.21, 1e-12)
    close(round(by[:, 1].sum(), 2), 2.47, 1e-12)
    _, _, lib = L.tiered_waterfall(c_t, c, dt, d, IRR_T)
    assert np.allclose(lib, by, atol=1e-12)
    # the multiple form gives the library's answer too
    assert np.allclose(L.tiered_waterfall(c_t, c, dt, d, MUL_T)[2], _tiers_by_definition(c_t, c, dt, d, MUL_T), atol=1e-12)
    close(_tiers_by_definition(c_t, c, dt, d, MUL_T)[:, 1].sum(), RAW["reMulSpoTot"], 1e-9)


def test_uncorrected_formula_double_counts_within_a_distribution():
    """Without l_{<j}(t) the tiers overlap within one distribution: 2.63 / 0 / 2.44."""
    c_t, c, dt, d = _example()
    bad = _tiers_by_definition(c_t, c, dt, d, IRR_T, net_same_date=False)
    close(round(bad[1, 0], 2), 2.63, 1e-12); close(round(bad[2, 0], 2), 0.0, 1e-12)
    close(round(bad[:, 1].sum(), 2), 2.44, 1e-12)


def test_tier_hurdles_met_exactly_after_correction():
    """After tier j the investor's own flows have IRR exactly h_j (or multiple mu_j)."""
    c_t, c, dt, d = _example()
    inv, _, by = L.tiered_waterfall(c_t, c, dt, d, IRR_T)
    for j, h in ((0, 0.08), (1, 0.12)):
        last = inv[-1] - by[j + 1:, 0].sum()           # investor's take at year 5 up to the end of tier j
        flows = np.r_[-9.0, inv[:-1], last]
        close(L.irr(np.r_[0, dt], flows), h, 1e-10)
    inv, _, by = L.tiered_waterfall(c_t, c, dt, d, MUL_T)
    close(inv.sum() - by[2, 0], 1.6 * 9, 1e-10)


def test_worked_tier_allocation(RAW):
    close(RAW["reTierOneSize"] + RAW["reTierTwoSize"] + RAW["reTierThreeSize"], RAW["reYrFive"], 1e-9)
    close(RAW["reEarlyInv"] + RAW["reBalOne"], RAW["reTierOneInv"], 1e-9)
    close(RAW["reBalTwo"], RAW["reTierTwoInv"], 1e-9)
