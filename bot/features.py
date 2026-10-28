"""Feature engineering for the HMM: returns, volatility, momentum, drawdown."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

FEATURE_COLS = ["ret", "vol", "mom", "dd"]


def build_features(prices: pd.Series, cfg: Config) -> pd.DataFrame:
    f = cfg.features
    logret = np.log(prices).diff()
    df = pd.DataFrame(index=prices.index)
    df["ret"] = logret.rolling(f.return_window).sum()
    df["vol"] = logret.rolling(f.vol_window).std() * np.sqrt(252)
    df["mom"] = np.log(prices / prices.shift(f.momentum_window))
    df["dd"] = prices / prices.rolling(f.drawdown_window, min_periods=1).max() - 1.0
    return df.dropna()


class Standardizer:
    """Z-score fitted on training data only (no look-ahead into the test window)."""

    def fit(self, X: pd.DataFrame) -> "Standardizer":
        self.mu = X.mean()
        self.sd = X.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return ((X - self.mu) / self.sd).to_numpy()
