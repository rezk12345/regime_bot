"""Regime Trading Bot — command line entry point.

    python main.py backtest [--strategy NAME] [--compare] [--source synthetic]
    python main.py regime                      # show today's regime + HMM summary
    python main.py once                        # run one trading cycle now
    python main.py run                         # run the daily loop forever
    python main.py reset-kill-switch
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from bot.config import ROOT, all_symbols, load_config
from bot.data import get_prices


def setup_logging(cfg) -> None:
    log_file = ROOT / cfg.orchestrator.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)])
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def cmd_backtest(cfg, args) -> None:
    from bot.backtest import walk_forward
    from bot.report import write_report

    prices = get_prices(cfg, all_symbols(cfg))
    names = ["regime_weighted", "binary", "probability_blend"] if args.compare else [args.strategy]
    results = {}
    for n in names:
        print(f"\n=== walk-forward: {n} ===")
        res = walk_forward(cfg, prices, n)
        results[n] = res
        print(res.metrics.to_string(float_format=lambda v: f"{v:,.3f}"))
        print(f"folds: {len(res.folds)}  |  regime mix: "
              f"{res.regimes['regime'].value_counts(normalize=True).round(2).to_dict()}")
    out = ROOT / "reports"
    path = write_report(results, prices, out)
    print(f"\nreport -> {path}")


def cmd_regime(cfg, _args) -> None:
    from bot.features import build_features
    from bot.hmm_engine import RegimeEngine

    prices = get_prices(cfg, all_symbols(cfg))
    feats = build_features(prices[cfg.universe.signal_symbol], cfg).tail(cfg.backtest.train_window_days)
    eng = RegimeEngine(cfg).fit(feats)
    st = eng.current_regime(feats)
    print(eng.summary().to_string(index=False))
    print(f"\ncurrent regime: {st.label.upper()}  confidence {st.confidence:.0%}"
          f"{'  (uncertain)' if st.uncertain else ''}")
    print("probabilities:", {k: f"{v:.0%}" for k, v in st.probs.items()})


def main() -> None:
    p = argparse.ArgumentParser(description="HMM regime trading bot")
    p.add_argument("command", choices=["backtest", "regime", "once", "run", "reset-kill-switch"])
    p.add_argument("--config", default=None)
    p.add_argument("--strategy", default=None, help="regime_weighted | binary | probability_blend")
    p.add_argument("--compare", action="store_true", help="backtest all strategies side by side")
    p.add_argument("--source", default=None, help="override data.source (yfinance|alpaca|synthetic)")
    p.add_argument("--mode", default=None, help="override mode (paper|live|dry_run)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg["data"]["source"] = args.source
    if args.mode:
        cfg["mode"] = args.mode
    args.strategy = args.strategy or cfg.allocation.strategy
    setup_logging(cfg)

    if args.command == "backtest":
        cmd_backtest(cfg, args)
    elif args.command == "regime":
        cmd_regime(cfg, args)
    elif args.command in ("once", "run", "reset-kill-switch"):
        from bot.orchestrator import Orchestrator
        orch = Orchestrator(cfg)
        if args.command == "once":
            print(json.dumps(orch.run_once(), indent=2, default=str))
        elif args.command == "run":
            orch.run_forever()
        else:
            orch.risk.reset_kill_switch()
            orch._save_state()


if __name__ == "__main__":
    main()
