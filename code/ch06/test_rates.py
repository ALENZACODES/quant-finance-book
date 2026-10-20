"""Bonds, short-rate models, swaps and caps."""
import math
import numpy as np
import pytest
from conftest import close
import ch06lib as L


@pytest.mark.parametrize("y,c,T", [(0.04, 0.04, 10), (0.03, 0.05, 7), (0.06, 0.02, 3)])
def test_derivatives_match_differences(y, c, T):
    p0, p1, p2, p3 = L.bond_derivs(y, c, T); h = 1e-5
    close(p1, (L.bond_price(y + h, c, T) - L.bond_price(y - h, c, T)) / (2 * h), 1e-5)
    close(p2, (L.bond_price(y + h, c, T) - 2 * p0 + L.bond_price(y - h, c, T)) / h ** 2, 1e-1)
    assert p1 < 0 < p2 and p3 < 0


@pytest.mark.parametrize("y,T", [(0.01, 2), (0.04, 10), (0.08, 30)])
def test_par_duration_closed_form(y, T):
    close(L.par_duration(y, T), L.mod_duration(y, y, T), 1e-12)


def test_remainder_sign_on_data(RAW):
    assert RAW["bdSignFive"] == 1.0 and RAW["bdSignTen"] == 1.0
    assert RAW["bdExUpTwo"] < RAW["bdQdUpTwo"] and RAW["bdExDnTwo"] > RAW["bdQdDnTwo"]


def test_barbell_immunises(RAW):
    assert RAW["imMinBar"] > -1e-10
    z = 0.035; a1, a2 = L.barbell(z, 7, 2, 15)
    for s in np.linspace(-0.2, 0.2, 81):
        assert L.surplus_shift(s, z, [2, 15], [a1, a2], [7], [100]) >= -1e-10


def test_vasicek_price_matches_gaussian_moments():
    m, C = L.vas_moments(0.025, 5, 0.3, 0.03, 0.01)
    close(L.vas_P(0.025, 5, 0.3, 0.03, 0.01), math.exp(-m[1] + 0.5 * C[1, 1]), 1e-14)


def test_cir_solves_riccati():
    k, rb, s = 0.5, 0.04, 0.08; tau = np.linspace(0.01, 10, 50); h = 1e-6
    A, B = L.cir_AB(tau, k, rb, s); A1, B1 = L.cir_AB(tau + h, k, rb, s)
    dB = (B1 - B) / h; dlnA = (np.log(A1) - np.log(A)) / h
    assert np.max(np.abs(dB - (1 - k * B - 0.5 * s * s * B * B))) < 1e-5
    assert np.max(np.abs(dlnA + k * rb * B)) < 1e-6
    assert np.all(A <= 1) and np.all(B >= 0)


def test_cir_simulation(RAW):
    assert abs(RAW["srMcMonthZ"]) < 4 and abs(RAW["srMcWeekZ"]) < 4


def test_holee_identity():
    for T1, T2 in [(1, 1.25), (5, 5.25), (2, 7)]:
        close(((T2 ** 3 - T1 ** 3) - (T2 - T1) ** 3) / (6 * (T2 - T1)), 0.5 * T1 * T2, 1e-12)


def test_spot_swap_equals_par_yield(RAW):
    assert abs(RAW["swParGap"]) < 1e-14


def test_caplet_simulation_and_parity(RAW):
    assert abs(RAW["cpMcZ"]) < 4
    r0, k, rb, s = RAW["fitNowvasicek"]
    P = lambda t: float(L.vas_P(r0, t, k, rb, s)); T, dl = 2.0, 0.25; F = (P(T) / P(T + dl) - 1) / dl
    for K in (0.02, 0.035, 0.05):
        cap = L.vas_caplet(r0, T, dl, K, k, rb, s)
        zbc = L.vas_zbp(r0, T, T + dl, 1 / (1 + dl * K), k, rb, s) + P(T + dl) - P(T) / (1 + dl * K)
        floor_ = (1 + dl * K) * zbc
        close(cap - floor_, dl * P(T + dl) * (F - K), 1e-14)


def test_fits_are_close_to_quotes(RAW):
    assert RAW["srVasNowRms"] < 10 and RAW["srCirNowRms"] < 10 and RAW["srVasInvRms"] < 15


def test_feller_from_printed_inputs(RAW):
    """the ratio printed can be reproduced from the printed kappa, rbar and sigma."""
    k, rbar, sig = round(RAW["srCirNowKap"], 3), round(RAW["srCirNowRbar"], 2) / 100, round(RAW["srSigC"], 5)
    assert f"{2 * k * rbar / sig ** 2:.1f}" == f"{RAW['srFeller']:.1f}"


def test_signed_gaps_read_as_words(RAW):
    """losses and distances print as positive numbers."""
    assert RAW["imShoLoss"] > 0 and RAW["srMcWeekZabs"] >= 0 and RAW["cpMcZabs"] >= 0


def test_printed_vasicek_sigma_reproduces_text(RAW):
    """the sigma printed to four decimals gives the printed futures adjustments and caplet prices."""
    s = round(RAW["srSigV"], 4) / 100
    for T1, nm in [(1, "One"), (2, "Two"), (5, "Five"), (10, "Ten")]:
        assert f"{1e4 * L.holee_adjustment(s, T1, T1 + 0.25):.1f}" == f"{RAW['hlAdj' + nm]:.1f}"
    r0, k, rb, _ = RAW["fitNowvasicek"]
    P = lambda t: float(L.vas_P(r0, t, k, rb, s)); dl = 0.25
    for T, nm in [(1, "One"), (2, "Two"), (5, "Five")]:
        F = (P(T) / P(T + dl) - 1) / dl
        assert f"{1e4 * L.vas_caplet(r0, T, dl, F, k, rb, s):.2f}" == f"{RAW['cpAtm' + nm]:.2f}"


def test_cir_gaps_from_printed_values(RAW):
    """the standard-error gaps quoted reproduce from the six-decimal prices printed."""
    ex = round(RAW["srMcExact"], 6)
    for nm in ("Month", "Week"):
        m, se = round(RAW[f"srMc{nm}"], 6), round(RAW[f"srMc{nm}SE"], 6)
        assert f"{abs(m - ex) / se:.1f}" == f"{RAW[f'srMc{nm}Zabs']:.1f}"
