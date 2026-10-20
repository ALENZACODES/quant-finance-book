"""Tests for finite_diff.py: orders of convergence, stability, and the von Neumann factors."""
import numpy as np
import ch05lib as L


def test_orders(RAW):
    assert abs(RAW["fdOrderExpl"] - 2) < 0.1 and abs(RAW["fdOrderCN"] - 2) < 0.1
    assert abs(RAW["fdOrderCNR"] - 2) < 0.1 and abs(RAW["fdOrderImpl"] - 1) < 0.2
    errs = np.array(RAW["fdErrsCN"]); assert np.all(np.diff(errs) < 0)


def test_explicit_stability(RAW):
    assert RAW["explLamA"] < 1e-3 and RAW["explLamC"] > 1e3


def test_gamma_oscillation(RAW):
    assert RAW["gammaErrCN"] > 1.0 and RAW["gammaErrCNR"] < 1e-3 and RAW["gammaNegCN"] < 0
    assert RAW["gammaNegCNR"] > 0


def test_amplification_factors():
    w = np.linspace(0, np.pi, 1001)
    for lam in (0.1, 0.5, 1, 10, 1000):
        for th in (0.5, 0.75, 1.0):
            assert np.all(np.abs(L.amplification(lam, th, w)) <= 1 + 1e-12)
    assert np.all(np.abs(L.amplification(0.5, 0.0, w)) <= 1 + 1e-12)
    assert np.abs(L.amplification(0.5001, 0.0, w)).max() > 1
    assert abs(L.amplification(1e8, 0.5, np.pi) + 1) < 1e-7      # Crank-Nicolson: g -> -1
    assert abs(L.amplification(1e8, 1.0, np.pi)) < 1e-7          # implicit: g -> 0


def test_scheme_agrees_with_formula_for_puts():
    v = L.fd_theta(100, 110, 0.03, 0.25, 0.5, 800, 400, 0.5, kind="put", rannacher=2)
    assert abs(v - float(L.bs(100, 110, 0.03, 0.25, 0.5, "put"))) < 1e-3


def test_cn_error_is_spatial(RAW):
    # more time steps leave the Crank-Nicolson error at M = 400 almost unchanged
    assert abs(RAW["fdErrCNFine"] / RAW["fdErrCN"] - 1) < 0.02
