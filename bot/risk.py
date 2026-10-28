"""Risk management: position caps, volatility targeting, daily-loss halt, drawdown kill switch.

Every target-weight vector passes through RiskManager.apply() before it can
become an order — both in the backtest and in live trading — so the backtest
measures the same risk rules the bot actually runs with.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class RiskStatus:
    halted: bool = False          # kill switch tripped — flatten and stop
    paused_today: bool = False    # daily loss limit hit — no new risk today
    notes: list[str] = field(default_factory=list)


class RiskManager:
    def __init__(self, cfg: Config):
        self.r = cfg.risk
        self.peak_equity: float | None = None
        self.day_start_equity: float | None = None
        self.halted = False

    # -------------------------------------------------- portfolio-level state
    def update_equity(self, equity: float, new_day: bool) -> RiskStatus:
        st = RiskStatus(halted=self.halted)
        if new_day or self.day_start_equity is None:
            self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity or equity, equity)

        dd = equity / self.peak_equity - 1
        if dd <= -self.r.max_drawdown_pct and not self.halted:
            self.halted = st.halted = True
            st.notes.append(f"KILL SWITCH: drawdown {dd:.1%} breached -{self.r.max_drawdown_pct:.0%}")
            log.critical(st.notes[-1])

        day_pnl = equity / self.day_start_equity - 1
        if day_pnl <= -self.r.max_daily_loss_pct:
            st.paused_today = True
            st.notes.append(f"daily loss {day_pnl:.1%} — no new risk today")
        return st

    def reset_kill_switch(self) -> None:
        self.halted = False
        log.warning("kill switch manually reset")

    # ---------------------------------------------------------- weight rules
    def apply(self, target: dict[str, float], current: dict[str, float],
              returns: pd.DataFrame | None, status: RiskStatus) -> dict[str, float]:
        if status.halted:
            return {}
        w = {s: min(v, self.r.max_position_weight) for s, v in target.items()}

        # volatility targeting on the proposed portfolio
        if returns is not None and len(w) and len(returns) >= self.r.vol_lookback:
            cols = [s for s in w if s in returns.columns]
            if cols:
                window = returns[cols].tail(self.r.vol_lookback)
                wv = np.array([w[c] for c in cols])
                port_vol = float(np.sqrt(wv @ window.cov().to_numpy() @ wv) * np.sqrt(252))
                if port_vol > self.r.target_volatility > 0:
                    scale = self.r.target_volatility / port_vol
                    w = {s: v * scale for s, v in w.items()}
                    status.notes.append(f"vol {port_vol:.1%} > target, scaled x{scale:.2f}")

        gross = sum(w.values())
        if gross > self.r.max_gross_exposure:
            w = {s: v * self.r.max_gross_exposure / gross for s, v in w.items()}

        # daily-loss pause: allow reducing exposure, never increasing it
        if status.paused_today:
            w = {s: min(v, current.get(s, 0.0)) for s, v in w.items()}
        return w
