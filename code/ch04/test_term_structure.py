"""Tests for term_structure.py and the Curve and bootstrap functions."""
import math
import numpy as np
import numpy_financial as npf
from conftest import close
import ch04lib as L


def test_bootstrap_reprices_inputs():
    c = L.bootstrap({0.5: 0.04, 1.0: 0.042}, {2.0: 0.045, 3.0: 0.047})
    close(float(c.D(0.5)), 1 / 1.02, 1e-14); close(float(c.D(1.0)), 1 / 1.042, 1e-14)
    for c0, T in [(0.045, 2.0), (0.047, 3.0)]:
        close(L.bond_price(c, c0, T), 1.0, 1e-12)
        close(c.par_yield(T), c0, 1e-12)


def test_forward_identity_and_inst_forward():
    c = L.bootstrap({0.5: 0.04, 1.0: 0.042}, {2.0: 0.045, 3.0: 0.047})
    # (1 + f tau) D(t2) = D(t1), and the integral of the instantaneous forward recovers ln D
    f = float(c.fwd_simple(1.0, 2.0)); close((1 + f) * float(c.D(2.0)), float(c.D(1.0)), 1e-14)
    x = np.linspace(0, 3, 30001); fi = c.fwd_inst(x[:-1] + 1e-12)
    close(float(np.sum(fi * np.diff(x))), -math.log(float(c.D(3.0))), 1e-10)
    # zero rate is the average forward rate
    close(float(c.zero(2.0)), float(np.mean(c.fwd_inst(np.linspace(0, 2, 200001)[:-1] + 1e-12))), 1e-6)


def test_annual_bootstrap_matches_irr():
    c = L.bootstrap({1.0: 0.03}, {2.0: 0.035}, m=1)
    # a par bond's IRR equals its coupon
    D = c.D(np.array([1.0, 2.0]))
    price = 3.5 * D[0] + 103.5 * D[1]
    close(npf.irr([-price, 3.5, 103.5]), 0.035, 1e-10)


def test_numbers(RAW):
    close(RAW["hxCheckTwo"], 100.0, 1e-9)
    close(RAW["fraBuy"], RAW["fraSell"], 1e-6)                    # the FRA replication costs nothing
    assert RAW["gocRepriceErr"] < 1e-15
    assert 0 < RAW["gocZeroFive"] < 10 and 0 < RAW["gocFwdOneTwo"] < 10


def test_simple_and_continuous_forwards(RAW):
    """F(t1,t2) simple and ln(D1/D2)/(t2-t1) continuous describe the same discount factors."""
    import math
    for s_, c_, tau in [("gocFwdOneTwo", "gocFwdOneTwoCont", 1.0), ("gocFwdSevenTen", "gocFwdSevenTenCont", 3.0)]:
        simple = RAW[s_] / 100; cont = RAW[c_] / 100
        assert abs(math.log(1 + simple * tau) / tau - cont) < 1e-12
        assert cont < simple
