"""Walk-forward backtester.

Train the HMM on [t - train_window, t), trade the next test_window days using
only forward-filtered regime probabilities, then roll forward and refit. The
stitched out-of-sample equity curve is the honest estimate of performance.

Signals are computed on day t's close and executed at day t+1's returns
(one-bar delay), with commissions and slippage charged on turnover.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .allocation import make_strategy
from .config import Config
from .features import build_features
from .hmm_engine import RegimeEngine
from .risk import RiskManager, RiskStatus

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    weights: pd.DataFrame
    regimes: pd.DataFrame
    folds: pd.DataFrame
    metrics: pd.DataFrame


def metrics(equity: pd.Series, name: str) -> dict:
    r = equity.pct_change().dropna()
    yrs = max(len(r) / 252, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(252)
    downside = r[r < 0].std() * np.sqrt(252)
    dd = (equity / equity.cummax() - 1).min()
    return {"name": name, "total_return": equity.iloc[-1] / equity.iloc[0] - 1, "cagr": cagr,
            "volatility": vol, "sharpe": (r.mean() * 252) / vol if vol else np.nan,
            "sortino": (r.mean() * 252) / downside if downside else np.nan,
            "max_drawdown": dd, "calmar": cagr / abs(dd) if dd else np.nan,
            "win_rate_daily": (r > 0).mean()}


def walk_forward(cfg: Config, prices: pd.DataFrame, strategy: str | None = None) -> BacktestResult:
    bt = cfg.backtest
    sig = cfg.universe.signal_symbol
    feats = build_features(prices[sig], cfg)
    rets = prices.pct_change().fillna(0.0)
    strat = make_strategy(cfg, strategy)
    risk = RiskManager(cfg)
    cost = (bt.commission_bps + bt.slippage_bps) / 1e4

    dates = feats.index
    start = bt.train_window_days
    if len(dates) <= start + bt.test_window_days:
        raise ValueError("not enough history for one walk-forward fold")

    equity = float(bt.initial_capital)
    cur_w: dict[str, float] = {}
    eq_hist, w_hist, reg_hist, fold_rows = {}, {}, [], []

    for fold, i in enumerate(range(start, len(dates) - 1, bt.test_window_days)):
        train = feats.iloc[i - bt.train_window_days:i]
        engine = RegimeEngine(cfg).fit(train)
        test_end = min(i + bt.test_window_days, len(dates) - 1)
        # filter over train+test so the forward recursion is warm; read only test rows
        probs = engine.filtered_probs(feats.iloc[i - bt.train_window_days:test_end])
        fold_start_eq = equity

        for t in range(i, test_end):
            d, d_next = dates[t], dates[t + 1]
            state = engine.to_state(probs.loc[d])
            status = risk.update_equity(equity, new_day=True)
            target = strat.target_weights(state)
            hist = rets.loc[:d].tail(cfg.risk.vol_lookback + 1)
            new_w = risk.apply(target, cur_w, hist, status)

            # skip small rebalances
            syms = set(new_w) | set(cur_w)
            if max((abs(new_w.get(s, 0) - cur_w.get(s, 0)) for s in syms), default=0) < \
                    cfg.allocation.rebalance_threshold and not status.halted:
                new_w = cur_w
            turnover = sum(abs(new_w.get(s, 0) - cur_w.get(s, 0)) for s in syms)
            equity *= 1 - turnover * cost

            # hold new weights over the next bar
            r_next = rets.loc[d_next]
            port_r = sum(wt * r_next.get(s, 0.0) for s, wt in new_w.items())
            equity *= 1 + port_r
            # weights drift with prices
            grown = {s: wt * (1 + r_next.get(s, 0.0)) for s, wt in new_w.items()}
            tot = 1 + port_r
            cur_w = {s: v / tot for s, v in grown.items()} if tot > 0 else {}

            eq_hist[d_next] = equity
            w_hist[d_next] = new_w
            reg_hist.append({"date": d, "regime": state.label, "confidence": state.confidence,
                             **{f"p_{k}": v for k, v in state.probs.items()}})
            if status.halted:
                break

        fold_rows.append({"fold": fold, "train_start": dates[i - bt.train_window_days].date(),
                          "test_start": dates[i].date(), "test_end": dates[test_end].date(),
                          "return": equity / fold_start_eq - 1})
        if risk.halted:
            log.warning("backtest stopped by kill switch at %s", dates[t])
            break

    eq = pd.Series(eq_hist).sort_index()
    eq = pd.concat([pd.Series({dates[start]: float(bt.initial_capital)}), eq])
    bench_px = prices[bt.benchmark].loc[eq.index]
    bench = bench_px / bench_px.iloc[0] * bt.initial_capital

    m = pd.DataFrame([metrics(eq, "strategy"), metrics(bench, f"buy&hold {bt.benchmark}")]).set_index("name")
    return BacktestResult(equity=eq, benchmark=bench,
                          weights=pd.DataFrame(w_hist).T.fillna(0.0).sort_index(),
                          regimes=pd.DataFrame(reg_hist).set_index("date"),
                          folds=pd.DataFrame(fold_rows), metrics=m)
