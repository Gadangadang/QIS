How the pieces work together (detailed)

This project is organized as a layered pipeline: signal generation → execution simulator → orchestration/CLI → diagnostics. The following explains how the main modules interact and how to run experiments or daily paper runs with the current unified CLI and helper scripts.

- Signals (`signals/`)
  - Each signal implements a `generate(df: DataFrame) -> DataFrame` method. The returned DataFrame must contain a `Position` column with values in {1, 0, -1} (long, flat, short). Signals may optionally include per-row overrides such as `Size`, `StopLossPct`, `TakeProfitPct`, or `MaxHoldDays` to influence execution on a per-trade basis.

- Execution / Simulator (`live/paper_trader.py`)
  - `PaperTrader.simulate(df, position_col='Position', ...)` is the execution engine. It expects the price columns (`Open`, `High`, `Low`, `Close`) and a `Position` signal column.
  - Execution rules:
    - `ExecPosition = Position.shift(1)`: executions use the prior day's signal to avoid lookahead.
    - Optional per-row sizing via a `Size` column; otherwise sizing is capped by `max_position_pct` argument.
    - Risk controls: `stop_loss_pct`, `take_profit_pct`, `max_hold_days` applied per trade; signals can override these with `StopLossPct`, `TakeProfitPct`, `MaxHoldDays` columns.
    - `stop_mode` controls intraday behavior when evaluating stops: `close` (evaluate at close), `low` (use daily Low), or `open` (use daily Open) and an intraday-exit approximation is applied when `low`/`open` are used.
    - Transaction costs are applied as a fraction of turnover using `transaction_cost` (basis points).
  - Outputs: per-day `Strategy` returns, `PortfolioValue`, and a `trades` DataFrame with `entry_date`, `exit_date`, `entry_price`, `exit_price`, `side`, `pnl_pct`, and `exit_reason` (attempts to robustly annotate triggered exits; there's a fallback match by entry date to handle off-by-one differences between trigger and recorded exit).

- Backtest / Orchestration (`backtest/`)
  - `backtest/backtest_engine.py` implements walk-forward splitting and `run_walk_forward()` which wires a `signal_factory` into per-fold training and OOS evaluation. New execution parameters (stop-loss, stop-mode, take-profit, max-hold, max-position) are threaded into `PaperTrader.simulate()` so experiments are reproducible.
  - `backtest/runner.py` is the unified CLI entrypoint with subcommands:
    - `walkforward`: run anchored walk-forward (stitches fold results and saves `stitched_equity.csv`, `combined_returns.csv`, `trades_fold_*.csv`).
    - `backtest`: run a single historical backtest and save outputs.
    - `daily`: run the daily/paper runner (delegate to `live/run_daily.py`) — now accepts the same configuration flags as the backtest so you can run the live logic with stop-loss, stop-mode, max-pos, etc.
    - `testfold`: replay the last fold quickly with custom params (fast diagnostics of a single fold).
    - `sweep`: small grid sweep over stop-loss / stop-mode / max-pos on the last fold and save `sweep_results.csv`.
    - `diagnostics`: run diagnostics on an existing `--save-dir`.
  - All commands accept `--save-dir` so artifacts and diagnostics are saved in a single folder for reproducibility.

- Daily runner (`live/run_daily.py` and helper `run_daily_pandas.py`)
  - `live/run_daily.py` was updated to accept an `argparse.Namespace` so the unified `backtest.runner daily` subcommand can pass parameters directly.
  - `run_daily_pandas.py` (repo root) is a minimal script that uses `pandas` to run the pipeline; useful if you prefer to run the daily pipeline directly from a terminal that has pandas installed.

- Diagnostics (`backtest/run_diagnostics.py`)
  - Generates a `diagnostics.txt` summarizing stitched equity, worst strategy days, worst trades, and other quick checks. The CLI runner calls diagnostics automatically after `walkforward`, `backtest`, `testfold`, and `daily` runs so each experiment folder includes the diagnostics output.

Common parameters (available via CLI)
- `--signal`: which strategy to run (momentum, mean_reversion/mr, ensemble, ensemble_new)
- `--transaction-cost`: trade cost in basis points
- `--stop-loss`: global stop-loss percentage (e.g., `0.1` = 10%)
- `--take-profit`: optional take-profit percentage
- `--max-hold`: maximum days to hold a trade
- `--stop-mode`: `close` | `low` | `open` — controls intraday stop detection
- `--max-pos`: maximum position fraction (1.0 = full size, 0.2 = 20% of portfolio)
- `--save-dir`: directory where outputs will be saved (combined_returns, stitched_equity, trades, diagnostics)

Example commands
- Walk-forward (full experiment with diagnostics):
```
python3 -m backtest.runner walkforward --signal momentum --train-frac 0.6 --test-frac 0.2 \
  --lookback 250 --save-dir logs/walkforward_expt --stop-loss 0.1 --stop-mode low --max-pos 0.2
```

- Run the daily/paper trader via the unified CLI (this will also run diagnostics):
```
python3 -m backtest.runner daily --signal ensemble_new --stop-loss 0.1 --stop-mode low --max-pos 0.2 --save-dir logs/daily_test
```

- Direct lightweight script (requires pandas; suitable for running in your terminal):
```
python3 run_daily_pandas.py --signal ensemble_new --stop-loss 0.1 --stop-mode low --max-pos 0.2 --save-dir logs/daily_test
```

Notes & troubleshooting
- If you see `exit_reason` empty for many trades: the code now attempts a fallback match by `entry_date` when annotating trades. If you still see missing exit reasons in older saved CSVs, re-running the pipeline (or using the included post-processor) will populate them for new outputs.
- If `pandas` is not available in your VS Code terminal, create/activate a virtualenv in the project root and install `pandas, numpy, matplotlib` (recommended) or install them into the interpreter your terminal uses:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or `pip install pandas numpy matplotlib`
```

If you'd like, I can add a short `requirements.txt` and a README section that walks through creating the venv and example commands.
