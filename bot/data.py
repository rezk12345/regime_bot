"""Market data: yfinance, Alpaca, or a synthetic regime-switching generator for offline testing."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Config

log = logging.getLogger(__name__)


def get_prices(cfg: Config, symbols: list[str], end: datetime | None = None) -> pd.DataFrame:
    """Return a DataFrame of adjusted daily closes, columns = symbols."""
    src = cfg.data.source
    end = end or datetime.now()
    start = end - timedelta(days=int(365.25 * cfg.data.lookback_years))

    if src == "synthetic":
        return synthetic_prices(symbols, start, end)

    cache = ROOT / cfg.data.cache_dir / f"{src}_{'_'.join(symbols)}_{end:%Y%m%d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    if src == "yfinance":
        df = _from_yfinance(symbols, start, end)
    elif src == "alpaca":
        df = _from_alpaca(cfg, symbols, start, end)
    else:
        raise ValueError(f"unknown data source: {src}")

    df = df.dropna(how="all").ffill().dropna()
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
    except Exception as e:  # parquet engine optional
        log.debug("cache write skipped: %s", e)
    return df


def _from_yfinance(symbols, start, end) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    if closes.empty:
        raise RuntimeError("yfinance returned no data (network blocked or bad symbols)")
    return closes


def _from_alpaca(cfg: Config, symbols, start, end) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment

    client = StockHistoricalDataClient(cfg.alpaca.api_key, cfg.alpaca.secret_key)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
                           start=start, end=end, adjustment=Adjustment.ALL)
    bars = client.get_stock_bars(req).df.reset_index()
    df = bars.pivot(index="timestamp", columns="symbol", values="close")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


# ---------------------------------------------------------------- synthetic
def synthetic_prices(symbols, start, end, seed: int = 7) -> pd.DataFrame:
    """Markov-switching price paths with bull/neutral/bear regimes.

    Lets the whole pipeline (HMM, backtest, orchestrator) be exercised without
    network access. Equities fall in 'bear', bonds/gold hold up — so a working
    regime model should visibly beat buy-and-hold here.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    n = len(idx)
    trans = np.array([[0.985, 0.012, 0.003],
                      [0.020, 0.965, 0.015],
                      [0.010, 0.030, 0.960]])
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(3, p=trans[states[t - 1]])

    # (daily mean, daily vol) per regime per asset class
    params = {
        "equity":    [(0.0008, 0.008), (0.0002, 0.012), (-0.0015, 0.025)],
        "growth":    [(0.0011, 0.011), (0.0001, 0.016), (-0.0020, 0.032)],
        "bond":      [(0.0000, 0.006), (0.0002, 0.007), (0.0006, 0.009)],
        "gold":      [(0.0001, 0.009), (0.0002, 0.009), (0.0005, 0.012)],
        "cash":      [(0.00008, 0.0005)] * 3,
    }
    klass = {"SPY": "equity", "QQQ": "growth", "TLT": "bond", "GLD": "gold", "SHY": "cash"}
    market_shock = rng.standard_normal(n)
    out = {}
    for s in symbols:
        p = params[klass.get(s, "equity")]
        mu = np.array([p[k][0] for k in states])
        sd = np.array([p[k][1] for k in states])
        beta = 0.7 if klass.get(s, "equity") in ("equity", "growth") else -0.2
        eps = beta * market_shock + np.sqrt(1 - beta**2) * rng.standard_normal(n)
        out[s] = 100 * np.exp(np.cumsum(mu + sd * eps))
    df = pd.DataFrame(out, index=idx)
    df.attrs["true_states"] = states
    return df
