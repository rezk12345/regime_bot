"""Broker layer. One interface, two implementations:

- AlpacaBroker : real Alpaca account (paper by default; live only if mode: live)
- SimBroker    : in-memory simulated fills for dry runs / testing, no network
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Order:
    symbol: str
    side: str          # buy | sell
    notional: float    # dollars
    qty: float | None = None


class Broker(ABC):
    @abstractmethod
    def equity(self) -> float: ...
    @abstractmethod
    def positions(self) -> dict[str, float]: ...          # symbol -> market value
    @abstractmethod
    def submit(self, order: Order) -> str: ...
    @abstractmethod
    def market_open(self) -> bool: ...
    @abstractmethod
    def close_all(self) -> None: ...

    def weights(self) -> dict[str, float]:
        eq = self.equity()
        return {s: v / eq for s, v in self.positions().items()} if eq > 0 else {}

    def rebalance(self, target: dict[str, float], min_order: float) -> list[Order]:
        """Diff target weights vs current holdings; sells first to free cash."""
        eq = self.equity()
        cur = self.positions()
        orders = []
        for sym in set(cur) | set(target):
            delta = target.get(sym, 0.0) * eq - cur.get(sym, 0.0)
            if abs(delta) < min_order:
                continue
            orders.append(Order(sym, "buy" if delta > 0 else "sell", abs(delta)))
        orders.sort(key=lambda o: o.side != "sell")
        for o in orders:
            o_id = self.submit(o)
            log.info("order %s %s $%.2f -> %s", o.side, o.symbol, o.notional, o_id)
        return orders


# ------------------------------------------------------------------ Alpaca
class AlpacaBroker(Broker):
    def __init__(self, cfg: Config):
        from alpaca.trading.client import TradingClient

        a = cfg.alpaca
        if not a.api_key or not a.secret_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing — copy .env.example to .env")
        self.client = TradingClient(a.api_key, a.secret_key, paper=a.paper)
        log.info("connected to Alpaca (%s)", "PAPER" if a.paper else "LIVE")

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def positions(self) -> dict[str, float]:
        return {p.symbol: float(p.market_value) for p in self.client.get_all_positions()}

    def market_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def submit(self, order: Order) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        if order.side == "sell":
            # sell by qty so we can fully exit; notional sells need fractional support
            pos = {p.symbol: p for p in self.client.get_all_positions()}.get(order.symbol)
            if pos is None:
                return "no-position"
            px = float(pos.current_price)
            qty = min(float(pos.qty), round(order.notional / px, 4))
            req = MarketOrderRequest(symbol=order.symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        else:
            req = MarketOrderRequest(symbol=order.symbol, notional=round(order.notional, 2),
                                     side=side, time_in_force=TimeInForce.DAY)
        return str(self.client.submit_order(req).id)

    def close_all(self) -> None:
        self.client.close_all_positions(cancel_orders=True)
        log.warning("all positions closed")


# --------------------------------------------------------------- Simulator
class SimBroker(Broker):
    def __init__(self, cash: float = 100_000.0):
        self.cash = cash
        self.units: dict[str, float] = {}
        self.prices: dict[str, float] = {}
        self.fills: list[dict] = []

    def set_prices(self, prices: dict[str, float]) -> None:
        self.prices.update(prices)

    def equity(self) -> float:
        return self.cash + sum(u * self.prices.get(s, 0) for s, u in self.units.items())

    def positions(self) -> dict[str, float]:
        return {s: u * self.prices[s] for s, u in self.units.items() if abs(u) > 1e-9}

    def market_open(self) -> bool:
        return True

    def submit(self, order: Order) -> str:
        px = self.prices[order.symbol]
        units = order.notional / px * (1 if order.side == "buy" else -1)
        self.units[order.symbol] = self.units.get(order.symbol, 0) + units
        self.cash -= units * px
        self.fills.append({"symbol": order.symbol, "side": order.side, "units": units, "price": px})
        return f"sim-{len(self.fills)}"

    def close_all(self) -> None:
        for s, u in list(self.units.items()):
            self.cash += u * self.prices[s]
        self.units.clear()


def make_broker(cfg: Config) -> Broker:
    if cfg.mode == "dry_run":
        return SimBroker(cfg.backtest.initial_capital)
    if cfg.mode == "live":
        log.warning("LIVE MODE — real money orders will be sent")
    return AlpacaBroker(cfg)
