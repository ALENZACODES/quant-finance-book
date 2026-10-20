"""Checks for simulated_funds.py: internal consistency of the Monte Carlo summary."""
import numpy as np
import ch02lib as L
from conftest import close


def test_mean_multiple(RAW):
    rng = np.random.default_rng(3)
    m = []
    for _ in range(4000):
        m += [x["proceeds"] / x["cost"] for x in L.simulate_fund(rng)]
    close(np.mean(m), RAW["mcMeanMult"], 0.05)


def test_fees_within_commitment():
    rng = np.random.default_rng(4)
    for _ in range(500):
        deals = L.simulate_fund(rng); _, f = L.fund_fee_flows(deals)
        assert 84 + f.sum() <= 100 + 1e-9


def test_commitment_cap_binds_when_needed():
    # every deal still unrealised in years 5 to 9: uncapped fees would take paid-in to 100.3
    deals = [dict(t_in=0.0, t_out=10.0, cost=8.4, proceeds=8.4) for _ in range(10)]
    _, f0 = L.fund_fee_flows(deals)
    _, f1 = L.fund_fee_flows(deals, cap=True)
    assert abs(84 + f0.sum() - 100.3) < 1e-9 and L.fund_fee_flows.binding
    assert abs(84 + f1.sum() - 100.0) < 1e-9 and np.all(f1 >= 0)


def test_cap_never_binds_in_simulation(RAW):
    assert RAW["mcCapBindN"] == 0 and RAW["mcPaidMax"] <= 100


def test_summary_relations(RAW):
    assert RAW["mcDbdCarry"] > RAW["mcPoolCarry"] > RAW["mcWfCarry"]
    assert RAW["mcOwed"] > RAW["mcPoolOwed"] > 0
    assert RAW["mcPreFeeIrr"] > RAW["mcGrossIrr"] > RAW["mcNetIrr"]
    assert RAW["mcDbdIrrCost"] > 0 and RAW["mcPoolIrrCost"] > 0
    close(RAW["mcPaidIn"], 84 + RAW["mcFeesMean"], 0.051)
    assert RAW["mcWfAboveDbd"] < 2.0
    assert 0 < RAW["mcPoolFeeCluster"] < 50


def test_pooled_can_exceed_pure_on_some_funds():
    """Netting is not pathwise cheaper: a deal below its own hurdle can earn carry inside a pool."""
    rng = np.random.default_rng(5)
    diffs = [L.deal_by_deal_pooled(x).sum() - L.deal_by_deal(x).sum() for x in (L.simulate_fund(rng) for _ in range(300))]
    assert min(diffs) < 0 < max(diffs) and np.mean(diffs) < 0


def test_paid_in_within_commitment(RAW):
    """The text says no simulated fund calls more than its commitment of 100."""
    assert RAW["mcOverCommitN"] == 0 and RAW["mcPaidMax"] <= 100
    assert abs(84 + 10 + 5 * 0.015 * 84 - 100.3) < 1e-12


def test_clawback_table_identity(RAW):
    """Carry received less the mean clawback matches the whole-fund entitlement to the printed two decimals."""
    for paid, owed in [("mcWfPaid", "mcWfOverAll"), ("mcDbdCarry", "mcOwed"), ("mcPoolCarry", "mcPoolOwed")]:
        assert abs(RAW[paid] - RAW[owed] - RAW["mcWfCarry"]) < 0.005
    assert abs(RAW["mcWfPaid"] - RAW["mcWfOverAll"] - RAW["mcWfCarry"]) < 1e-9
