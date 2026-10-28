"""Allocation strategies: turn a regime reading into target portfolio weights."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .config import Config
from .hmm_engine import RegimeState


class AllocationStrategy(ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.table: dict[str, dict[str, float]] = {k: dict(v) for k, v in
                                                   cfg.allocation.regime_weights.items()}

    @abstractmethod
    def target_weights(self, regime: RegimeState) -> dict[str, float]: ...

    def _risk_on(self) -> set[str]:
        return set(self.cfg.universe.risk_on)


class RegimeWeighted(AllocationStrategy):
    """Use the table for the most likely regime; de-risk when confidence is low."""

    def target_weights(self, regime: RegimeState) -> dict[str, float]:
        w = dict(self.table.get(regime.label, {}))
        if regime.uncertain:
            scale = self.cfg.allocation.uncertain_scale
            w = {s: (v * scale if s in self._risk_on() else v) for s, v in w.items()}
        return _clean(w)


class Binary(AllocationStrategy):
    """All-in risk-on in bull, all defensive otherwise. Simple baseline."""

    def target_weights(self, regime: RegimeState) -> dict[str, float]:
        key = "bull" if regime.label == "bull" and not regime.uncertain else "bear"
        return _clean(dict(self.table[key]))


class ProbabilityBlend(AllocationStrategy):
    """Probability-weighted mix of every regime's table — smooth, fewer whipsaws."""

    def target_weights(self, regime: RegimeState) -> dict[str, float]:
        w: dict[str, float] = {}
        for lab, p in regime.probs.items():
            for sym, wt in self.table.get(lab, {}).items():
                w[sym] = w.get(sym, 0.0) + p * wt
        return _clean(w)


def _clean(w: dict[str, float]) -> dict[str, float]:
    w = {s: max(0.0, float(v)) for s, v in w.items() if v > 1e-6}
    tot = sum(w.values())
    if tot > 1.0:
        w = {s: v / tot for s, v in w.items()}
    return w


STRATEGIES = {"regime_weighted": RegimeWeighted, "binary": Binary,
              "probability_blend": ProbabilityBlend}


def make_strategy(cfg: Config, name: str | None = None) -> AllocationStrategy:
    name = name or cfg.allocation.strategy
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name}; choose from {list(STRATEGIES)}")
    return STRATEGIES[name](cfg)
