"""Core backtesting engine with walk-forward and train-test functionality."""
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from signals.momentum import MomentumSignal
from signals.mean_reversion import MeanReversionSignal
from signals.ensemble import EnsembleSignal
from analysis.metrics import max_drawdown, sharpe_ratio
from core.paper_trader import PaperTrader
from core.optimizer import ParameterOptimizer


def split_train_test(df: pd.DataFrame, train_frac: float = 0.6, lookback: int = 20):
    """Split dataframe into train and test windows, returning the train slice and a
    test slice that includes `lookback` rows of history before the test start.

    Returns (train_df, test_with_history_df, test_only_df)
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")

    n = len(df)
    train_n = int(n * train_frac)
    train_df = df.iloc[:train_n].copy()
    hist_start = max(0, train_n - lookback)
    test_with_history = df.iloc[hist_start:].copy()
    test_only = df.iloc[train_n:].copy()
    return train_df, test_with_history, test_only


def run_train_test(
    signal_func,
    df: pd.DataFrame,
    train_frac: float = 0.6,
    lookback: int = 20,
    initial_cash: float = 100000,
):
    """Run a train/test split. The signal function is applied to the train slice for analysis,
    then re-applied over the test window including `lookback` history so indicators can initialize.

    Returns a dict with train_df, test_input_df (with history), test_result_df and trader summary.
    """
    train_df, test_with_history, test_only = split_train_test(
        df, train_frac=train_frac, lookback=lookback
    )

    # Generate signals on train and evaluate
    train_signals = signal_func(train_df)
    train_signals["Return"] = train_signals["Close"].pct_change()
    train_signals["Strategy"] = (
        train_signals["Position"].shift(1) * train_signals["Return"]
    )

    # Simple train evaluation
    train_sharpe = sharpe_ratio(train_signals["Strategy"].fillna(0))
    train_mdd = max_drawdown(train_signals["Strategy"].fillna(0))

    print(f"Train period: Sharpe {train_sharpe:.2f} | MaxDD {train_mdd:.2%}")

    # Generate signals on the test window but include history for lookback
    test_signals_all = signal_func(test_with_history)

    # Start date is the first index of test_only
    start_date = test_only.index[0]

    # Run paper trading on the full test_with_history but only count metrics/trades from start_date
    trader = PaperTrader(initial_cash=initial_cash)
    result_df = trader.simulate(
        test_signals_all, position_col="Position", start_date=start_date
    )

    summary = trader.summary()

    trader.print_summary()

    trader.plot()

    return {
        "train_df": train_signals,
        "test_input_df": test_signals_all,
        "test_result_df": result_df,
        "trader": trader,
        "summary": summary,
    }
    
def run_single_fold_backtest(
    signal,
    train_df,
    test_df,  # Changed from test_only
    lookback,
    initial_cash,
    transaction_cost,
    stop_loss_pct,
    take_profit_pct,
    max_hold_days,
    stop_mode,
    max_position_pct, 
):
    """
    Run a single backtest on train/test data (used for optimization).
    Returns dict with metrics: {'sharpe': ..., 'total_return': ..., etc}
    """
    # Generate signals on test data
    hist_start = max(0, len(train_df) - lookback)
    test_with_history = pd.concat([train_df.iloc[hist_start:], test_df])
    test_signals = signal.generate(test_with_history)
    
    # Run paper trader on test window only
    trader = PaperTrader(initial_cash=initial_cash)
    result_df = trader.simulate(
        test_signals,
        position_col="Position",
        start_date=test_df.index[0],
        end_date=test_df.index[-1],
        transaction_cost=transaction_cost,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_days=max_hold_days,
        stop_mode=stop_mode,
        max_position_pct=max_position_pct,
    )
    
    # Return metrics
    summary = trader.summary()
    num_trades = summary.get('n_trades', 0)  # Fixed: use 'n_trades' not 'Num Trades'
    
    # Debug: print if no trades
    if num_trades == 0:
        print(f"  ⚠️  Validation generated 0 trades (window: {test_df.index[0]} to {test_df.index[-1]}, {len(test_df)} days)")
    
    return {
        'sharpe': summary.get('sharpe', 0),
        'total_return': summary.get('total_return_pct', 0),
        'max_drawdown': summary.get('max_drawdown_pct', 0),
        'win_rate': summary.get('win_rate', 0),
        'num_trades': num_trades,
    }


def run_walk_forward(
    signal_factory,
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    lookback: int = 20,
    step: Optional[int] = None,
    initial_cash: float = 100_000,
    transaction_cost: float = 3.0,
    save_dir: str = "logs/walkforward",
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    max_hold_days: Optional[int] = None,
    stop_mode: str = "close",
    max_position_pct: float = 1.0,
    optimize_per_fold: bool = False,
    param_grid: Optional[dict] = None,
    optimization_metric: str = "sharpe",
):
    """Run an anchored walk-forward backtest.

    Args:
        signal_factory: callable that returns a SignalModel instance (e.g. lambda: MomentumSignal(...))
        df: full price DataFrame (chronological, oldest first)
        train_size: number of rows to use for each training window
        test_size: number of rows for each test window
        lookback: history rows to include before test start so indicators can initialize
        step: step to advance the window; if None, defaults to `test_size` (non-overlapping folds)
        initial_cash: starting capital used for stitching equity
        transaction_cost: per-trade cost in bps (e.g. 3 -> 0.03%) passed to PaperTrader
        save_dir: directory to save per-fold trades and stitched equity
        stop_loss_pct:
        take_profit_pct:
        max_hold_days: 
        stop_mode: 
        max_position_pct: 
        optimize_per_fold: 
        param_grid: 
        optimization_metric:

    Returns:
        dict with keys: "stitched_equity" (pd.Series), "combined_returns" (pd.Series), "folds" (list of fold summaries), "overall" (summary dict)
    """
    

    if step is None:
        step = test_size

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize signal columns in df if they don't exist
    for col in ['Position', 'Momentum', 'Z', 'Ensemble', 'TrendFilter']:
        if col not in df.columns:
            df[col] = np.nan

    n = len(df)
    starts = list(range(0, n - train_size - test_size + 1, step))

    stitched_values = []
    stitched_index = []
    combined_returns = []
    fold_summaries = []
    all_trades = []  # NEW: Collect all trades from all folds
    fold_params = {}

    prev_last_value = initial_cash

    for fold_idx, start in enumerate(starts, start=1):
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = train_end + test_size

        train_df = df.iloc[train_start:train_end].copy()
        hist_start = max(0, train_end - lookback)
        test_with_history = df.iloc[hist_start:test_end].copy()
        test_only = df.iloc[test_start:test_end].copy()

        if optimize_per_fold and param_grid:
            # Split training data into train/validation (80/20)
            train_size_inner = int(len(train_df) * 0.8)
            train_inner = train_df.iloc[:train_size_inner].copy()
            val_inner = train_df.iloc[train_size_inner:].copy()
            
            if fold_idx == 1:
                print(f"\nOptimization Setup:")
                print(f"  Training window: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} rows)")
                print(f"  → Train split: {train_inner.index[0].date()} to {train_inner.index[-1].date()} ({len(train_inner)} rows)")
                print(f"  → Val split: {val_inner.index[0].date()} to {val_inner.index[-1].date()} ({len(val_inner)} rows)")
                print(f"  Test window: {test_only.index[0].date()} to {test_only.index[-1].date()} ({len(test_only)} rows)")
            
            # Separate signal params from execution params
            signal_param_names = {'lookback', 'entry_threshold', 'exit_threshold', 'window', 'entry_z', 'exit_z'}
            execution_param_names = {'stop_loss_pct', 'take_profit_pct', 'max_hold_days'}
            
            signal_params = {k: v for k, v in param_grid.items() if k in signal_param_names}
            execution_params = {k: v for k, v in param_grid.items() if k in execution_param_names}
            
            def objective_fn(params):
                # Split params into signal and execution
                sig_params = {k: v for k, v in params.items() if k in signal_param_names}
                exec_params = {k: v for k, v in params.items() if k in execution_param_names}
                
                # Create signal with signal params only
                signal = signal_factory(**sig_params)
                
                # Use execution params in backtest (override defaults)
                current_stop_loss = exec_params.get('stop_loss_pct', stop_loss_pct)
                current_take_profit = exec_params.get('take_profit_pct', take_profit_pct)
                current_max_hold = exec_params.get('max_hold_days', max_hold_days)
                
                # CRITICAL: Test on validation split, NOT test_only
                return run_single_fold_backtest(
                    signal=signal,
                    train_df=train_inner,  # Train on first 80% of training data
                    test_df=val_inner,     # Validate on last 20% of training data
                    lookback=lookback,
                    initial_cash=initial_cash,
                    transaction_cost=transaction_cost,
                    stop_loss_pct=current_stop_loss,
                    take_profit_pct=current_take_profit,
                    max_hold_days=current_max_hold,
                    stop_mode=stop_mode,
                    max_position_pct=max_position_pct,
                )
            
            optimizer = ParameterOptimizer(
                objective_fn=objective_fn,
                param_grid=param_grid,
                metric=optimization_metric,
                maximize=True,
                verbose=True
            )
            
            best_params, best_score, _ = optimizer.grid_search()
            fold_params[fold_idx] = best_params
            
            # Extract signal params for creating the signal
            best_signal_params = {k: v for k, v in best_params.items() if k in signal_param_names}
            best_execution_params = {k: v for k, v in best_params.items() if k in execution_param_names}
            
            sig = signal_factory(**best_signal_params)
            
            # Override execution params for this fold if optimized
            if 'stop_loss_pct' in best_execution_params:
                stop_loss_pct = best_execution_params['stop_loss_pct']
            if 'take_profit_pct' in best_execution_params:
                take_profit_pct = best_execution_params['take_profit_pct']
            if 'max_hold_days' in best_execution_params:
                max_hold_days = best_execution_params['max_hold_days']
            
        else:
            # create a fresh signal instance
            sig = signal_factory() if callable(signal_factory) else signal_factory

        # generate signals on test_with_history (indicators need history)
        test_signals = sig.generate(test_with_history)
        
        # Store the generated signals back into the main df for diagnostics
        # Only update the test window portion
        for col in ['Position', 'Momentum', 'Z', 'Ensemble', 'TrendFilter']:
            if col in test_signals.columns:
                df.loc[test_signals.index, col] = test_signals[col]

        # run paper trader but only count from test_only start
        trader = PaperTrader(initial_cash=initial_cash)
        result_df = trader.simulate(
            test_signals,
            position_col="Position",
            start_date=test_only.index[0],
            end_date=test_only.index[-1],
            transaction_cost=transaction_cost,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_hold_days=max_hold_days,
            stop_mode=stop_mode,
            max_position_pct=max_position_pct,
        )

        # extract portfolio values and scale to continue from prev_last_value
        pv = result_df.loc[test_only.index, "PortfolioValue"].copy()
        # pv starts at initial_cash for each fold; scale so start equals prev_last_value
        scale = prev_last_value / initial_cash if initial_cash != 0 else 1.0
        pv_scaled = pv * scale

        # append to stitched lists
        stitched_values.append(pv_scaled.values)
        stitched_index.extend(pv_scaled.index.tolist())

        # collect strategy returns for combined metrics
        strat = result_df.loc[test_only.index, "Strategy"].copy()
        combined_returns.append(strat)

        # save trades for this fold
        trades = getattr(trader, "trades", None)
        if trades is not None and not trades.empty:
            # Add fold number to trades
            trades_with_fold = trades.copy()
            trades_with_fold['fold'] = fold_idx
            all_trades.append(trades_with_fold)

            # Still save per-fold trades to CSV
            trades_path = Path(save_dir) / f"trades_fold_{fold_idx}.csv"
            trades.to_csv(trades_path, index=False)

        # fold-level metrics
        fold_return = pv_scaled.iloc[-1] / pv_scaled.iloc[0] - 1 if len(pv_scaled) > 0 else 0.0
        fold_sharpe = sharpe_ratio(strat.fillna(0))
        fold_mdd = max_drawdown(strat.fillna(0))
        fold_summary = {
            "fold": fold_idx,
            "start": pv_scaled.index[0] if len(pv_scaled) else None,
            "end": pv_scaled.index[-1] if len(pv_scaled) else None,
            "fold_return_pct": float(fold_return),
            "sharpe": float(fold_sharpe),
            "max_drawdown": float(fold_mdd),
            "n_trades": int(len(trades)) if trades is not None else 0,
        }
        fold_summaries.append(fold_summary)

        prev_last_value = pv_scaled.iloc[-1] if len(pv_scaled) else prev_last_value

    # build stitched equity series
    if len(stitched_values) == 0:
        raise RuntimeError("No folds executed — check train_size/test_size and data length")

    stitched_flat = np.concatenate(stitched_values)
    stitched_series = pd.Series(stitched_flat, index=stitched_index)
    stitched_series = stitched_series[~stitched_series.index.duplicated(keep="first")]

    combined_returns_series = pd.concat(combined_returns)

    overall = {
        "total_return_pct": float(stitched_series.iloc[-1] / initial_cash - 1),
        "sharpe": float(sharpe_ratio(combined_returns_series.fillna(0))),
        "max_drawdown": float(max_drawdown(combined_returns_series.fillna(0))),
        "n_folds": len(fold_summaries),
    }

    # Consolidate all trades
    all_trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # Build results dict
    res = {
        "stitched_equity": stitched_series,
        "combined_returns": combined_returns_series,
        "folds": fold_summaries,
        "overall": overall,
        "trades": all_trades_df,  # NEW: All trades from all folds
        "df": df,  # NEW: Include original market data for context
    }

    # persist stitched equity and combined returns
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        stitched_series.to_csv(save_path / "stitched_equity.csv", header=["PortfolioValue"])
        combined_returns_series.to_csv(save_path / "combined_returns.csv", header=["Strategy"])

        # NEW: Auto-generate report if save_dir provided
        try:
            from analysis.report import BacktestReport
            from analysis.diagnostics import ModelDiagnostics

            report = BacktestReport(res)
            report.save_html(save_path / 'report.html')
            report.summary()  # Print to console
            
            # NEW: Generate diagnostics report
            diag = ModelDiagnostics(res)
            diag.save_report(save_path / 'diagnostics.txt')
            print(f"\n📊 Diagnostics saved to: {save_path / 'diagnostics.txt'}")
            
        except Exception as e:
            print(f"Warning: Could not generate report: {e}")

    return res


if __name__ == "__main__":
    

    df = pd.read_csv("Dataset/spx_full_1990_2025.csv", index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()

    n = len(df)
    print("Rows:", n)

    # Example: 60% train, 20% test per fold (remaining can be sliding); adjust as needed
    train_size = int(n * 0.6)
    test_size = int(n * 0.2)

    print("Using train_size =", train_size, "test_size =", test_size)

    res = run_walk_forward(lambda: MomentumSignal(lookback=120, threshold=0.02),
                        df,
                        train_size=train_size,
                        test_size=test_size,
                        lookback=250,
                        step=None,
                        initial_cash=100000,
                        transaction_cost=3.0)
    
    print("Folds:", res['folds'])
    print("Overall:", res['overall'])
    print(res['stitched_equity'].head(), res['stitched_equity'].tail())
    print(res['combined_returns'].describe())
    print(" ")
    s = res["combined_returns"]
    print(s[s <= -0.10].sort_values().head(20))