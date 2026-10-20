"""Tests for binomial_trees.py: replication, convergence and American exercise."""
import math
import numpy as np
from conftest import close
import ch04lib as L


def test_one_period(RAW):
    close(RAW["opPrice"], RAW["opPriceQ"], 1e-9); close(RAW["opPrice"], 0.5 * 20 / 1.05, 1e-9)
    close(RAW["tpAmerican"], 0.5 * 15 / 1.05, 1e-9)


def test_parity_on_tree():
    p = L.crr_params(0.05, 0.2, 1.0, 301)
    c = L.tree_european(100, 105, 301, p, "call"); pu = L.tree_european(100, 105, 301, p, "put")
    close(c - pu, 100 - 105 * math.exp(-0.05), 1e-10)


def test_convergence(RAW):
    close(RAW["crrFiveHundred"], RAW["bsCall"], 0.01)
    close(RAW["lrFiveHundredOne"], RAW["bsCall"], 1e-5)
    assert -1.2 < RAW["crrSlope"] < -0.7 and -2.2 < RAW["lrSlope"] < -1.8
    assert abs(RAW["crrNErrMin"]) < 5 and abs(RAW["crrNErrMax"]) < 5


def test_american(RAW):
    assert abs(RAW["amCallGap"]) < 1e-10
    assert RAW["amCallDiv"] > RAW["euCallDiv"]
    assert RAW["amPutTwoThousand"] > RAW["bsPut"]
    close(RAW["amPutTwoThousand"], RAW["amPutThousand"], 2e-3)
    assert RAW["perpBoundary"] < RAW["bndTenYear"] < RAW["bndStart"] < RAW["bndHalf"] < 100
    close(RAW["treeDelta"], RAW["bsDelta"], 1e-3)
    close(RAW["snipAm"], RAW["amPutThousand"], 1e-4)


def test_snell_envelope_equals_best_stopping_rule():
    # on a 6-step tree, enumerate all paths and check that the backward induction value equals the value
    # of the stopping rule "exercise the first time the node is in the exercise region"
    n, S, K = 6, 100.0, 100.0; p = L.crr_params(0.05, 0.3, 1.0, n); u, d, q, disc = p
    am = L.tree_american(S, K, n, p, "put")
    # exercise region by backward induction
    V = {}; ex = {}
    for j in range(n + 1):
        V[(n, j)] = max(K - S * u ** j * d ** (n - j), 0); ex[(n, j)] = V[(n, j)] > 0
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            s = S * u ** j * d ** (i - j); c = disc * (q * V[(i + 1, j + 1)] + (1 - q) * V[(i + 1, j)])
            V[(i, j)] = max(c, K - s); ex[(i, j)] = K - s > c
    val = 0.0
    for m in range(2 ** n):
        bits = [(m >> k) & 1 for k in range(n)]; prob = 1.0; j = 0; paid = 0.0
        for i in range(n + 1):
            if ex[(i, j)]:
                paid = disc ** i * max(K - S * u ** j * d ** (i - j), 0); break
            if i < n:
                prob *= q if bits[i] else 1 - q; j += bits[i]
        prob = np.prod([q if b else 1 - q for b in bits])
        val += prob * paid
    close(val, am, 1e-10)


def test_boundary_one_year_out_from_longer_tree(RAW):
    """The boundary depends on time to expiry only: a one year put and a two year put at one year from
    expiry share it. Read it from a two year tree with the same step, and check it is a node price."""
    import ch04lib as L, numpy as np
    _, b2 = L.tree_american(100, 100, 4000, L.crr_params(0.05, 0.2, 2.0, 4000), "put", boundary=True)
    tte = 2.0 - np.arange(4000) * 0.0005
    j = int(np.argmin(np.abs(tte - 1.0)))
    assert abs(b2[j] - RAW["bndStart"]) < 0.01
    # the reported level is a node price of that step: 100 u^k for an integer k
    u = np.exp(0.2 * np.sqrt(0.0005)); k = np.log(b2[j] / 100) / np.log(u)
    assert abs(k - round(k)) < 1e-6
    assert RAW["bndTenYear"] < RAW["bndStart"] < RAW["bndHalf"] < 100


def test_boundary_alternation_numbers(RAW):
    # 4.099: the first exercise node, the lower neighbour one step (0.005 years) later, and the envelope
    assert RAW["bndTenSecond"] < RAW["bndTenFirst"]
    assert abs(RAW["bndTenFirstT"] - RAW["bndTenSecondT"] - 0.005) < 1e-9
    assert abs(RAW["bndTenSecond"] - RAW["bndTenFirstEnv"]) < 1e-9
    assert RAW["bndTenEnvT"] <= RAW["bndTenSecondT"]
