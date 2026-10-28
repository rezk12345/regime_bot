"""Load settings.yaml + environment variables into one config object."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids an extra dependency)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


class Config(dict):
    """dict with attribute access: cfg.hmm.n_states"""

    def __getattr__(self, item: str) -> Any:
        try:
            val = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(val) if isinstance(val, dict) else val


def load_config(path: str | Path | None = None) -> Config:
    _load_dotenv(ROOT / ".env")
    path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(path) as f:
        cfg = Config(yaml.safe_load(f))
    cfg["alpaca"] = {
        "api_key": os.getenv("ALPACA_API_KEY", ""),
        "secret_key": os.getenv("ALPACA_SECRET_KEY", ""),
        "paper": cfg.get("mode", "paper") != "live",
    }
    return cfg


def all_symbols(cfg: Config) -> list[str]:
    u = cfg.universe
    syms = [u.signal_symbol, *u.risk_on, *u.defensive]
    if u.get("cash_proxy"):
        syms.append(u.cash_proxy)
    for w in cfg.allocation.regime_weights.values():
        syms.extend(w.keys())
    return sorted(set(syms))
