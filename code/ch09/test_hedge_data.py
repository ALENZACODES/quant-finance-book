"""Checks of Section 9.6 (hedge_data.py) on the Form PF statistics of the OFR and, when it has been
built, the Fung and Hsieh factor file. Every printed number is recomputed here a second way."""
import math
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from conftest import close
import hedge_data as H

D = H.load_ofr()


def test_sample_ends_by_september_2026_and_buckets_add_up():
    assert D.index[-1] <= pd.Timestamp("2026-09-30") and D.index[0] == pd.Timestamp("2013-03-31")
    inv = D[[f"allqhf_investorliquidity{b}_pct" for b in ("le7", "gt7le90", "gt90")]].sum(axis=1)
    por = D[[f"allqhf_portfolioliquidity{b}_pct" for b in ("le1", "gt1le7", "gt7le90", "gt90")]].sum(axis=1)
    assert np.all(np.abs(inv - 100) < 0.6) and np.all(np.abs(por - 100) < 0.6)
    assert (D.allqhf_gav_sum >= D.allqhf_nav_sum).all()


def test_strategy_table(RAW):
    for s, k in H.NAMES.items():
        g, n = D[f"strategy_{s}_grossreturn"], D[f"strategy_{s}_netreturn"]
        ok = g.notna() & n.notna()
        close(RAW[f"hdMean{k}"], 400 * n[ok].mean() / 100, 1e-9)
        close(RAW[f"hdDrag{k}"], 4 * (g[ok] - n[ok]).mean(), 1e-9)
        close(RAW[f"hdVol{k}"], 2 * n[ok].std(ddof=1), 1e-9)
        x = n[ok].values
        close(RAW[f"hdRho{k}"], pd.Series(x).autocorr(1), 1e-9)
        assert RAW[f"hdN{k}"] == ok.sum()
    assert RAW["hdLevMax"] == max(D[f"strategy_{s}_leverageratio"].dropna().iloc[-1] for s in H.STRATS)


def test_fee_schedule_against_statsmodels(RAW):
    p = H.fee_panel(D, [s for s in H.STRATS if s != "other"])
    X = sm.add_constant(np.maximum(p.gross.values, 0))
    m = sm.OLS(p.drag.values, X).fit(cov_type="HC1")
    close(RAW["hdFeeMgmt"], 4 * m.params[0], 1e-9)
    close(RAW["hdFeeInc"], 100 * m.params[1], 1e-9)
    close(RAW["hdFeeIncSe"], 100 * m.bse[1], 1e-9)
    close(RAW["hdFeeRsq"], m.rsquared, 1e-9)
    # an effective management fee near the 1 to 2% range of current terms, and an incentive share below
    # the headline 20%, as high-water marks and negotiated terms would make it
    assert 0.5 < RAW["hdFeeMgmt"] < 2.5 and 5 < RAW["hdFeeInc"] < 20
    # in quarters with a gross loss there is no incentive fee: the drag is the management fee alone
    assert abs(RAW["hdFeeNegDrag"] - RAW["hdFeeMgmt"]) < 3 * RAW["hdFeeMgmtSe"] + 0.3


def test_liquidity(RAW):
    inv90 = D.allqhf_investorliquidityle7_pct + D.allqhf_investorliquiditygt7le90_pct
    por90 = 100 - D.allqhf_portfolioliquiditygt90_pct
    close(RAW["hdInvNinety"], inv90.iloc[-1], 1e-9)
    close(RAW["hdPorNinety"], por90.iloc[-1], 0.6)
    close(RAW["hdCushionMin"], (D.allqhf_portfolioliquidityle1_pct + D.allqhf_portfolioliquiditygt1le7_pct
                                + D.allqhf_portfolioliquiditygt7le90_pct - inv90).min(), 1e-9)
    assert RAW["hdPorSeven"] > RAW["hdInvSeven"] and RAW["hdPorNinety"] > RAW["hdInvNinety"]


needs_ptfs = pytest.mark.skipif(not H.PTFS.exists(), reason="run python scripts/pull_hedge.py for the factor file")


@needs_ptfs
def test_quarterly_compounding_by_hand():
    f = H.load_ptfs()
    q = H.quarterly_ptfs(f, D.index)
    m = f.loc["2020-01-01":"2020-03-31"]
    assert len(m) == 3
    close(q.loc[pd.Timestamp("2020-03-31"), "ptfsbd"], float(np.prod(1 + m.ptfsbd) - 1), 1e-12)
    assert f.index[-1] <= pd.Timestamp("2026-09-30")


@needs_ptfs
def test_factor_regressions_against_statsmodels(RAW):
    q = H.quarterly_ptfs(H.load_ptfs(), D.index) * 100
    for s in H.STRATS:
        y = D[f"strategy_{s}_netreturn"]
        ok = y.notna() & q.notna().all(axis=1)
        m = sm.OLS(y[ok].values, sm.add_constant(q.loc[ok, H.FACTORS].values)).fit()
        close(RAW[f"hdRsq{H.NAMES[s]}"], m.rsquared, 1e-9)
        if s == "futures":
            close(RAW["hdPtfsAdj"], m.rsquared_adj, 1e-9)
            close(RAW["hdTFx"], m.tvalues[2], 1e-9)
            close(RAW["hdAlphaQ"], m.params[0], 1e-9)


@needs_ptfs
def test_credit_fit_leans_on_the_first_quarter_of_2020(RAW):
    # every trend factor gained in that quarter while credit funds lost, and the fit falls without it
    q = H.quarterly_ptfs(H.load_ptfs(), D.index) * 100
    k = [i for i in D.index if i.normalize() == pd.Timestamp("2020-03-31")][0]
    assert (q.loc[k, H.FACTORS] > 0).all() and D.loc[k, "strategy_credit_netreturn"] < 0
    assert RAW["hdRsqCreditExRaw"] < RAW["hdPtfs"]["credit"]["r2"] / 2
