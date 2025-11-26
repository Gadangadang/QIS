"""
Backtesting Framework

Provides comprehensive backtesting capabilities including:
- Single-period backtests
- Walk-forward optimization and validation
- Out-of-sample testing
- Parameter grid search
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from itertools import product

from core.portfolio_manager import PortfolioManager, PortfolioConfig, BacktestResult


def run_multi_asset_backtest(
    signals_dict: Dict[str, pd.DataFrame],
    prices_dict: Dict[str, pd.DataFrame],
    config: PortfolioConfig,
    return_pm: bool = False
) -> Tuple['PortfolioManager', pd.DataFrame, pd.DataFrame]:
    """
    Run a multi-asset backtest with portfolio management.
    
    This is the main entry point for backtesting. It orchestrates the backtest
    process and can return either the full PortfolioManager state or a lightweight
    BacktestResult object.
    
    Args:
        signals_dict: {ticker: df_with_Signal_column}
        prices_dict: {ticker: df_with_OHLC}
        config: PortfolioConfig
        return_pm: If True, return full PortfolioManager object
                   If False, return lightweight BacktestResult
        
    Returns:
        (backtest_result, equity_curve_df, trades_df)
        - backtest_result: PortfolioManager or BacktestResult object
        - equity_curve_df: DataFrame with portfolio equity over time
        - trades_df: DataFrame with all trades
    """
    pm = _run_backtest(signals_dict, prices_dict, config)
    equity_curve = pm.get_equity_curve()
    trades = pm.get_trades_df()
    
    # Extract risk metrics if risk manager was used
    risk_metrics = None
    violations = None
    if config.risk_manager:
        risk_metrics = config.risk_manager.get_metrics_dataframe()
        violations = config.risk_manager.get_violations_dataframe()
    
    if return_pm:
        return pm, equity_curve, trades
    else:
        # Return lightweight BacktestResult with risk data
        result = BacktestResult(equity_curve, trades, config, risk_metrics, violations)
        return result, equity_curve, trades


def _run_backtest(
    signals_dict: Dict[str, pd.DataFrame],
    prices_dict: Dict[str, pd.DataFrame],
    config: PortfolioConfig
) -> PortfolioManager:
    """
    Core backtest implementation.
    
    Runs the portfolio through historical data, handling:
    - Signal changes (entries/exits)
    - Position value updates
    - Drift-based rebalancing
    - Transaction cost tracking
    - Risk management (if configured)
    """
    # Initialize portfolio manager
    pm = PortfolioManager(config)
    
    # Get common dates from index
    all_dates = None
    for df in signals_dict.values():
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates &= dates
    
    all_dates = sorted(all_dates)
    
    # Track previous prices for returns calculation (if using risk manager)
    prev_prices = {}
    
    # Initialize on first date
    first_date = all_dates[0]
    init_prices = {ticker: df.loc[first_date, 'Close'] 
                   for ticker, df in prices_dict.items()}
    init_signals = {ticker: df.loc[first_date, 'Signal'] 
                    for ticker, df in signals_dict.items()}
    
    pm.initialize_positions(init_prices, init_signals)
    pm.equity_curve.append(pm.get_portfolio_state(first_date))
    
    # Initialize previous prices
    prev_prices = init_prices.copy()
    
    # Initialize correlation matrix if risk manager exists
    if config.risk_manager:
        # Build initial returns dataframe for correlation
        # Use the first 60+ days of the backtest period for initialization
        returns_data = {}
        for ticker, df in prices_dict.items():
            # Try to get data before first_date (historical warm-up data)
            mask_before = df.index < first_date
            
            if mask_before.sum() >= 60:
                # We have historical data before backtest start - use it!
                hist_data = df.loc[mask_before, 'Close'].tail(60)
                returns_data[ticker] = hist_data.pct_change().dropna()
            else:
                # No historical data - use first 60 days of backtest period
                mask_first = (df.index >= first_date) & (df.index <= df.index[min(59, len(df)-1)])
                if mask_first.sum() >= 20:  # Need at least 20 days
                    hist_data = df.loc[mask_first, 'Close']
                    returns_data[ticker] = hist_data.pct_change().dropna()
        
        if returns_data and len(returns_data) >= 2:  # Need at least 2 assets
            returns_df = pd.DataFrame(returns_data).dropna()
            if len(returns_df) >= 10:  # Minimum 10 returns for correlation
                config.risk_manager.update_correlations(returns_df)
    
    # Track iteration for periodic correlation updates
    iteration_count = 0
    
    # Run through all dates
    for date in all_dates[1:]:
        iteration_count += 1
        
        # Get current prices and signals
        current_prices = {ticker: df.loc[date, 'Close'] 
                         for ticker, df in prices_dict.items()}
        current_signals = {ticker: df.loc[date, 'Signal'] 
                          for ticker, df in signals_dict.items()}
        
        # Update risk manager with daily returns
        if config.risk_manager:
            for ticker in current_prices: # Should be vectorized later
                if ticker in prev_prices and prev_prices[ticker] > 0:
                    daily_return = (current_prices[ticker] / prev_prices[ticker]) - 1
                    config.risk_manager.update_returns(ticker, pd.Timestamp(date), daily_return)
            
            # Update correlations periodically (every 20 days) for better heatmap
            if iteration_count % 20 == 0 and len(config.risk_manager.returns_history) > 0:
                # Build returns dataframe from returns_history buffers
                returns_for_corr = {}
                for ticker, returns_deque in config.risk_manager.returns_history.items():
                    if len(returns_deque) >= 60:
                        returns_for_corr[ticker] = list(returns_deque)
                
                if len(returns_for_corr) >= 2:  # Need at least 2 assets for correlation
                    returns_df_update = pd.DataFrame(returns_for_corr)
                    config.risk_manager.update_correlations(returns_df_update)
        
        # Update position values with current prices
        pm.update_positions(current_prices)
        
        # Check for drawdown stop before trading
        if config.risk_manager:
            current_value = pm.portfolio_value
            equity_series = pd.Series([state['TotalValue'] for state in pm.equity_curve])
            peak = max(equity_series.max(), current_value) if len(equity_series) > 0 else current_value
            current_dd = min(0, (current_value - peak) / peak) if peak > 0 else 0  # Drawdown is always ≤ 0
            
            should_stop, reason = config.risk_manager.check_stop_conditions(
                current_drawdown=current_dd,
                equity_curve=equity_series
            )
            
            if should_stop:
                # Log violation and stop trading
                config.risk_manager._log_violation('PORTFOLIO', 'STOP', reason)
                # Update timestamp with actual date
                if config.risk_manager.violations_history:
                    config.risk_manager.violations_history[-1]['date'] = pd.Timestamp(date)
                    config.risk_manager.violations_history[-1]['timestamp'] = pd.Timestamp(date)
                # Record final state and exit
                pm.equity_curve.append(pm.get_portfolio_state(date))
                break
        
        # Check if signals changed (entries/exits)
        pm.update_signals(current_signals, current_prices, date)
        
        # Update position values again after signal changes
        pm.update_positions(current_prices)
        
        # Check if rebalancing needed (only among active positions)
        if pm.check_rebalance_needed(current_signals):
            pm.rebalance(current_prices, current_signals, date)
            pm.update_positions(current_prices)  # Update after rebalance
        
        # Collect risk metrics if risk manager is configured
        if config.risk_manager:
            current_value = pm.portfolio_value
            
            # Get current positions as shares dict
            current_positions = {ticker: pos['shares'] 
                               for ticker, pos in pm.positions.items() 
                               if pos['shares'] != 0}
            
            # Calculate drawdown
            equity_series = pd.Series([state['TotalValue'] for state in pm.equity_curve])
            peak = max(equity_series.max(), current_value) if len(equity_series) > 0 else current_value
            drawdown = min(0, (current_value - peak) / peak) if peak > 0 else 0  # Drawdown is always ≤ 0
            
            # Log metrics
            config.risk_manager.log_metrics(
                date=pd.Timestamp(date),
                positions=current_positions,
                prices=current_prices,
                portfolio_value=current_value,
                drawdown=drawdown
            )
        
        # Record portfolio state
        pm.equity_curve.append(pm.get_portfolio_state(date))
        
        # Update previous prices for next iteration
        prev_prices = current_prices.copy()
    
    return pm


class WalkForwardEngine:
    """
    Walk-forward optimization and validation engine.
    
    Performs rolling window optimization to:
    1. Find best parameters on training data
    2. Test parameters on out-of-sample data
    3. Aggregate OOS results for true performance estimate
    4. Return optimized parameters for production use
    
    Example:
        >>> engine = WalkForwardEngine(
        ...     signal_class=MomentumSignalV2,
        ...     param_grid={'lookback': [60, 90, 120], 'sma_filter': [150, 200, 250]},
        ...     tickers=['ES', 'GC', 'NQ'],
        ...     start_date='2010-01-01',
        ...     end_date='2024-12-31',
        ...     train_years=3,
        ...     test_years=1
        ... )
        >>> results = engine.run()
        >>> print(f"Best params: {results['best_params']}")
        >>> print(f"Avg OOS Sharpe: {results['oos_sharpe_mean']:.2f}")
    """
    
    def __init__(
        self,
        signal_class: type,
        param_grid: Dict[str, List],
        tickers: List[str],
        start_date: str,
        end_date: str,
        train_years: int = 3,
        test_years: int = 1,
        config: Optional[PortfolioConfig] = None,
        optimization_metric: str = 'Sharpe Ratio',
        verbose: bool = True
    ):
        """
        Initialize walk-forward engine.
        
        Args:
            signal_class: Signal generator class (e.g., MomentumSignalV2)
            param_grid: Dictionary of parameters to test, e.g.,
                       {'lookback': [60, 90, 120], 'sma_filter': [150, 200, 250]}
            tickers: List of tickers to trade
            start_date: Start date for walk-forward (train period begins here)
            end_date: End date for walk-forward (last test period ends here)
            train_years: Years for each training window
            test_years: Years for each test window
            config: PortfolioConfig (uses default if None)
            optimization_metric: Metric to optimize ('Sharpe Ratio', 'Total Return', etc.)
            verbose: Print progress during optimization
        """
        self.signal_class = signal_class
        self.param_grid = param_grid
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.train_years = train_years
        self.test_years = test_years
        self.config = config or PortfolioConfig(initial_capital=100000)
        self.optimization_metric = optimization_metric
        self.verbose = verbose
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        self.param_combinations = [
            dict(zip(param_names, combo))
            for combo in product(*param_values)
        ]
        
        # Generate time windows
        self.splits = self._create_splits()
        
        # Results storage
        self.window_results = []
        
    def _create_splits(self) -> List[Dict]:
        """Create rolling train/test windows."""
        splits = []
        current_train_start = pd.Timestamp(self.start_date)
        final_date = pd.Timestamp(self.end_date)
        
        while True:
            train_end = current_train_start + pd.DateOffset(years=self.train_years)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(years=self.test_years)
            
            if test_end > final_date:
                break
            
            splits.append({
                'train_start': current_train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d')
            })
            
            current_train_start = test_start
        
        return splits
    
    def _optimize_window(
        self,
        train_split: Dict,
        prices_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Optimize parameters on training window.
        
        Returns:
            best_params: Dict with best parameter values
            results_df: DataFrame with all tested parameter combinations
        """
        from core.multi_asset_loader import load_assets
        from core.multi_asset_signal import SingleAssetWrapper
        
        if self.verbose:
            print(f"\n🔍 Optimizing: {train_split['train_start']} to {train_split['train_end']}")
        
        # Filter prices to training period
        train_prices = {}
        for ticker, df in prices_dict.items():
            mask = (df.index >= train_split['train_start']) & (df.index <= train_split['train_end'])
            train_prices[ticker] = df.loc[mask].copy()
        
        results = []
        
        for params in self.param_combinations:
            # Generate signals with these parameters
            signal_gen = self.signal_class(**params)
            multi_signal = SingleAssetWrapper(signal_gen)
            signals = multi_signal.generate(train_prices)
            
            # Run backtest
            result, equity_curve, trades = run_multi_asset_backtest(
                signals, train_prices, self.config
            )
            
            # Calculate metrics
            metrics = result.calculate_metrics()
            
            # Store results
            result_dict = params.copy()
            result_dict.update({
                'sharpe': metrics.get('Sharpe Ratio', 0.0),
                'total_return': metrics.get('Total Return', 0.0),
                'max_dd': metrics.get('Max Drawdown', 0.0),
                'cagr': metrics.get('CAGR', 0.0)
            })
            results.append(result_dict)
        
        # Find best parameters
        results_df = pd.DataFrame(results)
        
        # Sort by optimization metric (handle different metric names)
        metric_col = 'sharpe' if 'Sharpe' in self.optimization_metric else 'total_return'
        results_df = results_df.sort_values(metric_col, ascending=False)
        
        best_params = results_df.iloc[0][list(self.param_grid.keys())].to_dict()
        
        if self.verbose:
            print(f"   ✅ Best: {best_params} | {metric_col.title()}: {results_df.iloc[0][metric_col]:.2f}")
        
        return best_params, results_df
    
    def _test_window(
        self,
        test_split: Dict,
        best_params: Dict,
        prices_dict: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Test parameters on out-of-sample window.
        
        Returns:
            metrics: Dict with OOS performance metrics
        """
        from core.multi_asset_signal import SingleAssetWrapper
        
        if self.verbose:
            print(f"   🧪 Testing: {test_split['test_start']} to {test_split['test_end']}")
        
        # Filter prices to test period
        test_prices = {}
        for ticker, df in prices_dict.items():
            mask = (df.index >= test_split['test_start']) & (df.index <= test_split['test_end'])
            test_prices[ticker] = df.loc[mask].copy()
        
        # Generate signals with best parameters
        signal_gen = self.signal_class(**{k: int(v) if isinstance(v, (float, np.floating)) else v 
                                          for k, v in best_params.items()})
        multi_signal = SingleAssetWrapper(signal_gen)
        signals = multi_signal.generate(test_prices)
        
        # Run backtest
        result, equity_curve, trades = run_multi_asset_backtest(
            signals, test_prices, self.config
        )
        
        # Calculate metrics
        metrics = result.calculate_metrics()
        
        if self.verbose:
            print(f"   📊 OOS Sharpe: {metrics.get('Sharpe Ratio', 0.0):.2f} | "
                  f"Return: {metrics.get('Total Return', 0.0):.2%} | "
                  f"Trades: {len(trades)}")
        
        return metrics
    
    def run(self, prices_dict: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """
        Run full walk-forward optimization.
        
        Args:
            prices_dict: Optional pre-loaded prices dict. If None, will load data.
        
        Returns:
            Dictionary with:
            - 'windows': List of window results
            - 'best_params': Best params from most recent window
            - 'oos_sharpe_mean': Average OOS Sharpe across windows
            - 'oos_return_mean': Average OOS return across windows
            - 'summary_df': DataFrame with all window results
        """
        from core.multi_asset_loader import load_assets
        
        # Load prices if not provided
        if prices_dict is None:
            if self.verbose:
                print(f"\n📊 Loading data for {len(self.tickers)} assets...")
            prices_dict = load_assets(
                self.tickers,
                start_date=self.start_date,
                end_date=self.end_date
            )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"WALK-FORWARD OPTIMIZATION")
            print(f"{'='*60}")
            print(f"Windows: {len(self.splits)}")
            print(f"Train Period: {self.train_years} years")
            print(f"Test Period: {self.test_years} years")
            print(f"Parameter Combinations: {len(self.param_combinations)}")
            print(f"Optimization Metric: {self.optimization_metric}")
        
        # Run each window
        for i, split in enumerate(self.splits, 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"WINDOW {i}/{len(self.splits)}")
                print(f"{'='*60}")
            
            # Optimize on training data
            best_params, train_results = self._optimize_window(split, prices_dict)
            
            # Test on OOS data
            oos_metrics = self._test_window(split, best_params, prices_dict)
            
            # Store results
            window_result = {
                'window': i,
                'train_start': split['train_start'],
                'train_end': split['train_end'],
                'test_start': split['test_start'],
                'test_end': split['test_end'],
                'best_params': best_params,
                'is_sharpe': train_results.iloc[0]['sharpe'],
                'oos_sharpe': oos_metrics.get('Sharpe Ratio', 0.0),
                'oos_return': oos_metrics.get('Total Return', 0.0),
                'oos_max_dd': oos_metrics.get('Max Drawdown', 0.0),
                'oos_cagr': oos_metrics.get('CAGR', 0.0)
            }
            self.window_results.append(window_result)
        
        # Calculate aggregate statistics
        summary_df = pd.DataFrame(self.window_results)
        
        oos_sharpe_mean = summary_df['oos_sharpe'].mean()
        oos_return_mean = summary_df['oos_return'].mean()
        best_params_latest = self.window_results[-1]['best_params']
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"WALK-FORWARD SUMMARY")
            print(f"{'='*60}")
            print(f"Average OOS Sharpe: {oos_sharpe_mean:.2f}")
            print(f"Average OOS Return: {oos_return_mean:.2%}")
            print(f"Best Params (Latest Window): {best_params_latest}")
            print(f"\nParameter Stability:")
            for param_name in self.param_grid.keys():
                param_values = [w['best_params'][param_name] for w in self.window_results]
                print(f"  {param_name}: {param_values}")
        
        return {
            'windows': self.window_results,
            'best_params': best_params_latest,
            'oos_sharpe_mean': oos_sharpe_mean,
            'oos_return_mean': oos_return_mean,
            'oos_return_total': summary_df['oos_return'].sum(),
            'summary_df': summary_df
        }


if __name__ == "__main__":
    # Test the backtesting framework
    from core.multi_asset_loader import load_assets
    from core.multi_asset_signal import SingleAssetWrapper
    from signals.momentum import MomentumSignalV2
    
    print("Testing Backtest Framework...")
    
    # Test 1: Simple backtest
    print("\n" + "="*60)
    print("TEST 1: Simple Backtest")
    print("="*60)
    
    prices = load_assets(['ES', 'GC'], start_date='2020-01-01', end_date='2021-12-31')
    
    signal_gen = MomentumSignalV2(lookback=60, sma_filter=150)
    multi_signal = SingleAssetWrapper(signal_gen)
    signals = multi_signal.generate(prices)
    
    config = PortfolioConfig(initial_capital=100000)
    result, equity_curve, trades = run_multi_asset_backtest(signals, prices, config)
    
    metrics = result.calculate_metrics()
    print(f"\nSharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Total Trades: {len(trades)}")
    
    # Test 2: Walk-forward optimization
    print("\n" + "="*60)
    print("TEST 2: Walk-Forward Optimization (Quick Test)")
    print("="*60)
    
    engine = WalkForwardEngine(
        signal_class=MomentumSignalV2,
        param_grid={
            'lookback': [60, 90],
            'sma_filter': [150, 200]
        },
        tickers=['ES', 'GC'],
        start_date='2015-01-01',
        end_date='2019-12-31',
        train_years=2,
        test_years=1,
        verbose=True
    )
    
    wf_results = engine.run()
    
    print("\n✅ Walk-forward testing complete!")
    print(f"Best parameters: {wf_results['best_params']}")
    print(f"Average OOS Sharpe: {wf_results['oos_sharpe_mean']:.2f}")
