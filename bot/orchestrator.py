"""Main loop: wake once per trading day, detect regime, size, risk-check, rebalance, log."""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .allocation import make_strategy
from .broker import Broker, SimBroker, make_broker
from .config import ROOT, Config, all_symbols
from .data import get_prices
from .features import build_features
from .hmm_engine import RegimeEngine
from .risk import RiskManager

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, cfg: Config, broker: Broker | None = None):
        self.cfg = cfg
        self.o = cfg.orchestrator
        self.tz = ZoneInfo(self.o.timezone)
        self.broker = broker or make_broker(cfg)
        self.strategy = make_strategy(cfg)
        self.risk = RiskManager(cfg)
        self.state_path = ROOT / self.o.state_file
        self.trade_log = ROOT / self.o.trade_log
        self.state = self._load_state()
        self.risk.peak_equity = self.state.get("peak_equity")
        self.risk.halted = self.state.get("halted", False)
        if isinstance(self.broker, SimBroker) and "sim_account" in self.state:
            acct = self.state["sim_account"]
            self.broker.cash = acct["cash"]
            self.broker.units = acct["units"]

    # ------------------------------------------------------------ state I/O
    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state.update(peak_equity=self.risk.peak_equity, halted=self.risk.halted)
        if isinstance(self.broker, SimBroker):
            self.state["sim_account"] = {"cash": self.broker.cash, "units": self.broker.units}
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    def _log_trades(self, when, regime, orders) -> None:
        self.trade_log.parent.mkdir(parents=True, exist_ok=True)
        new = not self.trade_log.exists()
        with open(self.trade_log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "regime", "confidence", "symbol", "side", "notional"])
            for o in orders:
                w.writerow([when.isoformat(), regime.label, f"{regime.confidence:.3f}",
                            o.symbol, o.side, f"{o.notional:.2f}"])

    # ------------------------------------------------------------- one cycle
    def run_once(self) -> dict:
        now = datetime.now(self.tz)
        prices = get_prices(self.cfg, all_symbols(self.cfg))
        if isinstance(self.broker, SimBroker):
            self.broker.set_prices(prices.iloc[-1].to_dict())

        feats = build_features(prices[self.cfg.universe.signal_symbol], self.cfg)
        train = feats.tail(self.cfg.backtest.train_window_days)
        engine = RegimeEngine(self.cfg).fit(train)
        regime = engine.current_regime(train)
        log.info("regime=%s conf=%.2f probs=%s", regime.label, regime.confidence,
                 {k: round(v, 2) for k, v in regime.probs.items()})

        equity = self.broker.equity()
        new_day = self.state.get("last_day") != now.date().isoformat()
        status = self.risk.update_equity(equity, new_day=new_day)
        if new_day:
            self.state["day_start_equity"] = equity
        else:
            self.risk.day_start_equity = self.state.get("day_start_equity", equity)
            status = self.risk.update_equity(equity, new_day=False)

        if status.halted:
            log.critical("risk halt active — flattening and skipping trading")
            self.broker.close_all()
            self.state.update(last_day=now.date().isoformat(), last_regime=regime.label)
            self._save_state()
            return {"regime": regime.label, "halted": True, "orders": []}

        target = self.strategy.target_weights(regime)
        rets = prices.pct_change().dropna()
        final = self.risk.apply(target, self.broker.weights(), rets, status)
        for n in status.notes:
            log.warning(n)

        cur = self.broker.weights()
        drift = max((abs(final.get(s, 0) - cur.get(s, 0)) for s in set(final) | set(cur)), default=0)
        orders = []
        if drift >= self.cfg.allocation.rebalance_threshold or not cur:
            orders = self.broker.rebalance(final, self.cfg.risk.min_order_value)
            self._log_trades(now, regime, orders)
        else:
            log.info("max drift %.1f%% below threshold — no trades", drift * 100)

        self.state.update(last_day=now.date().isoformat(), last_run=now.isoformat(),
                          last_regime=regime.label, last_confidence=regime.confidence,
                          target_weights=final, equity=equity)
        self._save_state()
        return {"regime": regime.label, "confidence": regime.confidence, "equity": equity,
                "target": final, "orders": [o.__dict__ for o in orders], "notes": status.notes}

    # ------------------------------------------------------------- forever
    def run_forever(self) -> None:
        hh, mm = map(int, self.o.run_time_local.split(":"))
        log.info("orchestrator started — daily run at %02d:%02d %s", hh, mm, self.o.timezone)
        while True:
            try:
                now = datetime.now(self.tz)
                due = (now.hour, now.minute) >= (hh, mm)
                done_today = self.state.get("last_day") == now.date().isoformat()
                if now.weekday() < 5 and due and not done_today:
                    if self.broker.market_open():
                        self.run_once()
                    else:
                        log.info("market closed (holiday?) — skipping today")
                        self.state["last_day"] = now.date().isoformat()
                        self._save_state()
            except KeyboardInterrupt:
                log.info("stopped by user")
                return
            except Exception:
                log.exception("cycle failed — will retry next poll")
            time.sleep(self.o.poll_seconds)
