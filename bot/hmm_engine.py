"""HMM regime engine.

Fits a Gaussian Hidden Markov Model to market features, then labels the hidden
states as bull / neutral / bear by their mean return (states are unordered in
an HMM, so labelling must be done after every fit).

Inference uses the *filtered* (forward) probabilities only, so the regime on
day t depends on data up to t — never on the future. hmmlearn's predict()
uses Viterbi/smoothing over the whole sequence, which would leak look-ahead
into a backtest; we avoid it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from .config import Config
from .features import FEATURE_COLS, Standardizer

log = logging.getLogger(__name__)

LABELS_BY_N = {2: ["bear", "bull"], 3: ["bear", "neutral", "bull"],
               4: ["bear", "neutral", "neutral", "bull"]}


@dataclass
class RegimeState:
    label: str                 # bull / neutral / bear
    confidence: float          # posterior probability of that label
    probs: dict[str, float]    # probability per label
    uncertain: bool


class RegimeEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg.hmm
        self.model: GaussianHMM | None = None
        self.scaler = Standardizer()
        self.state_labels: dict[int, str] = {}

    # ------------------------------------------------------------ fitting
    def fit(self, feats: pd.DataFrame) -> "RegimeEngine":
        X_df = feats[FEATURE_COLS]
        if len(X_df) < self.cfg.min_train_days:
            raise ValueError(f"need {self.cfg.min_train_days} rows to fit, got {len(X_df)}")
        self.scaler.fit(X_df)
        X = self.scaler.transform(X_df)

        best, best_ll = None, -np.inf
        for i in range(self.cfg.n_restarts):
            m = GaussianHMM(n_components=self.cfg.n_states,
                            covariance_type=self.cfg.covariance_type,
                            n_iter=self.cfg.n_iter, tol=1e-4,
                            random_state=self.cfg.random_state + i)
            try:
                m.fit(X)
                ll = m.score(X)
            except Exception as e:  # degenerate covariance etc.
                log.debug("restart %d failed: %s", i, e)
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError("HMM failed to converge on every restart")
        self.model = best
        self._label_states(X_df)
        log.info("HMM fit: n=%d ll=%.1f labels=%s", len(X), best_ll, self.state_labels)
        return self

    def _label_states(self, X_df: pd.DataFrame) -> None:
        """Rank states by a composite 'health' score in standardised feature space.

        Mean daily return alone is too noisy to separate regimes reliably, so we
        combine it with trend (momentum), distance from the high (drawdown) and
        volatility: high return/trend, shallow drawdown and low vol = bull.
        """
        m = self.model.means_
        ix = {c: FEATURE_COLS.index(c) for c in FEATURE_COLS}
        score = m[:, ix["ret"]] + m[:, ix["mom"]] + m[:, ix["dd"]] - m[:, ix["vol"]]
        order = np.argsort(score)  # worst first
        names = LABELS_BY_N.get(self.cfg.n_states,
                                ["bear"] + ["neutral"] * (self.cfg.n_states - 2) + ["bull"])
        self.state_labels = {int(s): names[rank] for rank, s in enumerate(order)}

    # ---------------------------------------------------------- inference
    def filtered_probs(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Forward-filtered P(state_t | x_1..x_t) for every row. No look-ahead."""
        X = self.scaler.transform(feats[FEATURE_COLS])
        m = self.model
        log_emit = m._compute_log_likelihood(X)
        log_trans = np.log(m.transmat_ + 1e-300)
        log_alpha = np.empty_like(log_emit)
        log_alpha[0] = np.log(m.startprob_ + 1e-300) + log_emit[0]
        log_alpha[0] -= logsumexp(log_alpha[0])
        for t in range(1, len(X)):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0) + log_emit[t]
            log_alpha[t] -= logsumexp(log_alpha[t])
        probs = np.exp(log_alpha)
        out = pd.DataFrame(0.0, index=feats.index, columns=sorted(set(self.state_labels.values())))
        for s, lab in self.state_labels.items():
            out[lab] += probs[:, s]
        return out

    def current_regime(self, feats: pd.DataFrame) -> RegimeState:
        return self.to_state(self.filtered_probs(feats).iloc[-1])

    def to_state(self, row: pd.Series) -> RegimeState:
        label = str(row.idxmax())
        conf = float(row.max())
        return RegimeState(label=label, confidence=conf,
                           probs={k: float(v) for k, v in row.items()},
                           uncertain=conf < self.cfg.confidence_threshold)

    def summary(self) -> pd.DataFrame:
        """Human-readable per-regime stats (unscaled means, persistence)."""
        rows = []
        for s, lab in self.state_labels.items():
            mean = self.model.means_[s] * self.scaler.sd.to_numpy() + self.scaler.mu.to_numpy()
            rows.append({"state": s, "label": lab,
                         **{f"mean_{c}": round(v, 4) for c, v in zip(FEATURE_COLS, mean)},
                         "persistence": round(self.model.transmat_[s, s], 3),
                         "exp_duration_days": round(1 / max(1e-9, 1 - self.model.transmat_[s, s]), 1)})
        order = {"bear": 0, "neutral": 1, "bull": 2}
        return pd.DataFrame(rows).sort_values("label", key=lambda s: s.map(order))
