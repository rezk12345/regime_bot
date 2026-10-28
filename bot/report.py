"""Backtest report: equity curves, drawdowns and regime shading as PNG + CSV + metrics."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REGIME_COLORS = {"bull": "#2e7d32", "neutral": "#9e9e9e", "bear": "#c62828"}


def write_report(results: dict, prices: pd.DataFrame, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3, 1]})
    ax = axes[0]
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity, label=name, lw=1.4)
    ax.plot(first.benchmark.index, first.benchmark, label=first.metrics.index[-1], color="black", lw=1, ls="--")
    ax.set_title("Walk-forward out-of-sample equity")
    ax.set_ylabel("$")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, res in results.items():
        ax.plot(res.equity.index, res.equity / res.equity.cummax() - 1, lw=1, label=name)
    b = first.benchmark
    ax.fill_between(b.index, b / b.cummax() - 1, 0, color="black", alpha=0.12, label="benchmark")
    ax.set_ylabel("drawdown")
    ax.grid(alpha=0.3)

    ax = axes[2]
    reg = first.regimes["regime"]
    for lab, col in REGIME_COLORS.items():
        mask = (reg == lab).astype(float)
        ax.fill_between(reg.index, 0, mask, step="post", color=col, alpha=0.7, lw=0, label=lab)
    ax.set_yticks([])
    ax.set_ylabel("regime")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    png = out / "backtest.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)

    pd.concat({n: r.metrics for n, r in results.items()}).to_csv(out / "metrics.csv")
    for n, r in results.items():
        r.equity.rename("equity").to_csv(out / f"equity_{n}.csv")
        r.folds.to_csv(out / f"folds_{n}.csv", index=False)
    first.regimes.to_csv(out / "regimes.csv")
    return png
