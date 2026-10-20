"""Checks of the Chapter 3 numbers produced by feynman_kac.py."""
import math
import numpy as np
from conftest import close
import ch03lib as L


def test_three_methods_agree(RAW):
    close(RAW["fkPDE"], RAW["fkBS"], 2e-4)
    assert abs(RAW["fkMC"] - RAW["fkBS"]) < 4 * RAW["fkMCse"]
    close(RAW["fkOUPDE"], RAW["fkOUExact"], 1e-6)
    assert abs(RAW["fkOUMC"] - RAW["fkOUExact"]) < 4 * RAW["fkOUMCse"]
    close(RAW["fkSlopeCall"], -2.0, 0.15); close(RAW["fkSlopeOU"], -2.0, 0.05)


def test_black_scholes_solves_the_pde():
    r, s, K = 0.03, 0.2, 100.0
    S = np.linspace(70, 140, 15); tau, h, k = 0.7, 1e-2, 1e-4
    V = lambda S, tau: L.bs_call(S, K, tau, r, s)
    Vt = -(V(S, tau + k) - V(S, tau - k)) / (2 * k)          # derivative in calendar time t = T - tau
    VS = (V(S + h, tau) - V(S - h, tau)) / (2 * h)
    VSS = (V(S + h, tau) - 2 * V(S, tau) + V(S - h, tau)) / h ** 2
    assert np.max(np.abs(Vt + r * S * VS + 0.5 * s ** 2 * S ** 2 * VSS - r * V(S, tau))) < 1e-4


def test_heat_equation_exercise(RAW):
    x = np.linspace(-6, 6, 601)
    u = L.fk_theta_solve(x, 2.0, lambda z: 0 * z, lambda z: 1 + 0 * z, 0.0, lambda z: z ** 2,
                         lambda tau, xe: xe ** 2 + tau, 200)
    close(np.interp(1.0, x, u), 3.0, 1e-8)
    close(RAW["exEight"], 3.0, 1e-12)
