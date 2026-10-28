"""Offline tests — run with:  pytest -q"""
import logging

import numpy as np
import pandas as pd
import pytest

from bot.allocation import make_strategy
from bot.backtest import walk_forward
from bot.broker import SimBroker
from bot.config import all_symbols, load_config
from bot.data import get_prices
from bot.features import build_features
from bot.hmm_engine import RegimeEngine, RegimeState
from bot.risk import RiskManager, RiskStatus

logging.disable(logging.INFO)


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["data"]["source"] = "synthetic"
    return c


@pytest.fixture(scope="module")
def prices(cfg):
    return get_prices(cfg, all_symbols(cfg))


def test_hmm_finds_bear_regime(cfg, prices):
    true = pd.Series(prices.attrs["true_states"], index=prices.index)
    f = build_features(prices["SPY"], cfg)
    eng = RegimeEngine(cfg).fit(f.iloc[:756])
    pred = eng.filtered_probs(f).idxmax(axis=1)
    bear_days = true.loc[pred.index] == 2
    assert (pred[bear_days] == "bear").mean() > 0.7


def test_filtered_probs_have_no_lookahead(cfg, prices):
    f = build_features(prices["SPY"], cfg)
    eng = RegimeEngine(cfg).fit(f.iloc[:756])
    full = eng.filtered_probs(f.iloc[:900])
    partial = eng.filtered_probs(f.iloc[:800])
    pd.testing.assert_frame_equal(full.iloc[:800], partial)  # future rows don't change the past


def test_allocations_sum_to_at_most_one(cfg):
    st = RegimeState("bull", 0.4, {"bull": 0.4, "neutral": 0.35, "bear": 0.25}, uncertain=True)
    for name in ("regime_weighted", "binary", "probability_blend"):
        w = make_strategy(cfg, name).target_weights(st)
        assert 0 < sum(w.values()) <= 1.0 + 1e-9


def test_risk_caps_and_kill_switch(cfg):
    r = RiskManager(cfg)
    st = r.update_equity(100_000, new_day=True)
    w = r.apply({"SPY": 0.9, "QQQ": 0.4}, {}, None, st)
    assert max(w.values()) <= cfg.risk.max_position_weight + 1e-9
    assert sum(w.values()) <= cfg.risk.max_gross_exposure + 1e-9
    st = r.update_equity(75_000, new_day=True)          # -25% from peak
    assert st.halted and r.apply({"SPY": 0.5}, {}, None, st) == {}


def test_daily_loss_blocks_new_risk(cfg):
    r = RiskManager(cfg)
    r.update_equity(100_000, new_day=True)
    st = r.update_equity(96_000, new_day=False)         # -4% today
    assert st.paused_today
    w = r.apply({"SPY": 0.6}, {"SPY": 0.2}, None, st)
    assert w["SPY"] <= 0.2


def test_sim_broker_rebalance():
    b = SimBroker(10_000)
    b.set_prices({"SPY": 100.0, "TLT": 50.0})
    b.rebalance({"SPY": 0.5, "TLT": 0.3}, min_order=1)
    w = b.weights()
    assert abs(w["SPY"] - 0.5) < 1e-9 and abs(w["TLT"] - 0.3) < 1e-9
    b.rebalance({"TLT": 1.0}, min_order=1)
    assert "SPY" not in b.positions()


def test_walk_forward_runs(cfg, prices):
    res = walk_forward(cfg, prices)
    assert len(res.folds) > 5
    assert res.equity.index.is_monotonic_increasing
    assert res.metrics.loc["strategy", "max_drawdown"] > -cfg.risk.max_drawdown_pct - 0.05
