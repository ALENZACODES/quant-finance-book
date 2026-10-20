"""Checks for fees_waterfalls.py: management fees, the waterfall propositions and Fund B."""
import numpy as np
import pytest
from pyxirr import xirr, irr as px_irr
import ch02lib as L
from conftest import close
from fees_waterfalls import DEALS_B, fund_b_flows

rng = np.random.default_rng(11)


def test_management_fee_formula(RAW):
    # equation (2.1): f1 K T_I + f2 sum of unrealised cost
    close(RAW["bFeeIP"], 0.02 * 100 * 5, 1e-12)
    close(RAW["bFeePost"], 0.015 * (30 + 10 + 10), 1e-12)
    close(RAW["bPaidIn"], 85 + RAW["bFeeTotal"], 1e-12)
    close(RAW["altFeeFlat"], 20.0, 1e-12); close(RAW["altFeeCost"], 0.02 * 425, 1e-12)
    close(RAW["altFeeStep"], 10 + 0.015 * (68 + 51 + 34 + 17), 1e-12)
    assert RAW["altFeeFlat"] - RAW["altFeeCost"] > 10          # "more than ten percent of commitments"
    close(RAW["spSplitTotal"], 180 - RAW["spVstarFull"], 1e-9)


def test_conservation_and_shares():
    """LP + GP = distribution, every allocation nonnegative, and carry never exceeds c x profit."""
    for _ in range(300):
        deals = L.simulate_fund(rng)
        ft, f = L.fund_fee_flows(deals)
        ct = np.r_[[x["t_in"] for x in deals], ft]; c = np.r_[[x["cost"] for x in deals], f]
        dt = np.array([x["t_out"] for x in deals]); d = np.array([x["proceeds"] for x in deals])
        for kw in (dict(), dict(catchup=0.8), dict(hard=True)):
            r = L.european_waterfall(ct, c, dt, d, **kw)
            assert np.allclose(r.lp + r.gp, d)
            assert (r.lp >= -1e-12).all() and (r.gp >= -1e-12).all()
            ent = L.carry_entitlement(ct, c, dt, d, **kw)
            assert ent <= 0.2 * max(d.sum() - c.sum(), 0) + 1e-9 and ent <= r.gp.sum() + 1e-12
            if ct.max() <= dt.min():      # all capital called before any exit: no over-distribution
                close(ent, r.gp.sum(), 1e-9)


def test_balance_matches_npv_proposition():
    """Proposition 2.2: B(t) <= 0 iff NPV at h of the LP's flows to date >= 0."""
    for _ in range(200):
        t_c = np.sort(rng.uniform(0, 3, 3)); c = rng.uniform(10, 40, 3)
        t_d = np.sort(rng.uniform(3, 9, 4)); d = rng.uniform(0, 60, 4)
        r = L.european_waterfall(t_c, c, t_d, d)
        for j in range(4):
            t = t_d[j]
            npv = -np.sum(c / 1.08 ** t_c) + np.sum(r.lp[: j + 1] / 1.08 ** t_d[: j + 1])
            close(r.hurdle_balance[j], -npv * 1.08 ** t, 1e-8)


def test_catchup_size_and_alignment():
    """Proposition 2.3: after the catch-up the GP holds exactly c of cumulative profit."""
    for k in (1.0, 0.8, 0.5):
        for V in np.linspace(150, 400, 26):
            r = L.european_waterfall([0], [100], [5], [V], catchup=k)
            vstar = (k * 100 * 1.08 ** 5 - 20) / (k - 0.2)
            if V >= vstar:
                close(r.gp[0], 0.2 * (V - 100), 1e-9)
            close(r.tiers[0, 2] + r.tiers[0, 3], min(max(V - 100 * 1.08 ** 5, 0), 0.2 * (100 * 1.08 ** 5 - 100) / (k - 0.2)), 1e-9)


def test_single_period_cases_and_options(RAW):
    """Propositions 2.4 and 2.5 against the general dated-flow waterfall."""
    H = 100 * 1.08 ** 5
    for V in np.linspace(50, 400, 351):
        for kw in (dict(), dict(catchup=0.8), dict(hard=True)):
            g = L.european_waterfall([0], [100], [5], [V], **kw).gp[0]
            close(g, float(L.carry_payoff(V, 100, H, **kw)), 1e-9)
    close(RAW["spGP"], 16.0, 1e-9); close(RAW["spCatch"], 0.25 * RAW["spPref"], 1e-9)
    close(RAW["spSplitLP"] + RAW["spSplitGP"], 180 - RAW["spVstarFull"], 1e-9)


def test_snippet_matches_library():
    import runpy, io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod = runpy.run_path(str(L.__file__).replace("ch02lib.py", "snippets/waterfall_core.py"), run_name="__main__")
    for V in (90, 140, 150, 160, 180, 300):
        lp, gp = mod["waterfall"](V)
        close(gp, L.european_waterfall([0], [100], [5], [V]).gp[0], 1e-9)
    assert "V = 180: LP 164.00, GP 16.00" in buf.getvalue()
    assert "Hard hurdle, V = 180: GP 6.61" in buf.getvalue()
    for V in (90, 150, 180, 300):
        close(mod["waterfall"](V, hard=True)[1], 0.2 * max(V - 100 * 1.08 ** 5, 0), 1e-12)
        close(mod["waterfall"](V, k=0.8)[1], L.european_waterfall([0], [100], [5], [V], catchup=0.8).gp[0], 1e-9)


def test_subadditivity_prop():
    for _ in range(2000):
        pi = rng.normal(0, 10, rng.integers(2, 8))
        assert np.maximum(pi, 0).sum() >= max(pi.sum(), 0) - 1e-12


def test_fund_b(RAW):
    ft, f, ct, c, dt, d = fund_b_flows()
    # IRRs against pyxirr (independent implementation)
    from datetime import date, timedelta
    d0 = date(2001, 1, 1)
    # annual flows on integer dates: use the periodic irr of pyxirr by aggregating per year
    flows = np.zeros(10)
    for t, x in zip(ct, -c): flows[int(t)] += x
    for t, x in zip(dt, d): flows[int(t)] += x
    close(100 * px_irr(flows), RAW["bGrossIrr"], 1e-6)
    lp = d - np.array([RAW["bWfGP" + n] for n in "ABCD"])
    flows_lp = np.zeros(10)
    for t, x in zip(ct, -c): flows_lp[int(t)] += x
    for t, x in zip(dt, lp): flows_lp[int(t)] += x
    close(100 * px_irr(flows_lp), RAW["bLPirrWf"], 1e-6)
    close(RAW["bDbdCarry"], 0.2 * (45 + 15 + 10), 1e-9)          # every winning deal clears its own hurdle
    close(RAW["bClawback"], RAW["bDbdCarry"] - RAW["bWfCarry"], 1e-12)
    assert RAW["bLPirrDbdNoClaw"] < RAW["bLPirrDbdClaw"] < RAW["bLPirrWf"]
    close(RAW["bWfShare"], 20.0, 0.05)                              # fund passes its catch-up
    pool = [RAW["bPoolEnt" + w] for w in ("One", "Two", "Three", "Four")]
    close(RAW["bPoolCarry"], max(pool[0], pool[1]), 1e-9)
    assert pool[2] < pool[1] and pool[3] < RAW["bPoolCarry"]
    # tiers in Table 2.3 add up
    for n in "ABCD":
        tot = RAW["bWfRoc" + n] + RAW["bWfPref" + n] + RAW["bWfCatch" + n] + RAW["bWfSplit" + n]
        close(tot, RAW["bWfLP" + n] + RAW["bWfGP" + n], 1e-9)


def test_fee_example_text(RAW):
    close(RAW["bFeeYearFive"], 0.45, 1e-12); close(RAW["bFeeYearSix"], 0.15, 1e-12)


def test_flat_fee_overrun(RAW):
    # 85 invested plus 20 of flat fees exceeds the 100 commitment (2.029)
    assert RAW["altFlatTotal"] == 85 + RAW["altFeeFlat"] and RAW["altFlatTotal"] > 100
