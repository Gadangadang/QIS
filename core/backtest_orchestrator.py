"""
Backtest Orchestrator - High-level API for running backtests.

This module provides a simplified, intuitive interface for common backtesting workflows.
It handles:
- Data loading with validation
- Signal generation for multiple strategies
- Batch backtest execution  
- Results aggregation and formatting
- Integrated visualization

Example usage:
    >>> from core.backtest_orchestrator import BacktestOrchestrator
    >>> 
    >>> orchestrator = BacktestOrchestrator()
    >>> orchestrator.load_data(['ES', 'NQ'], start_date='2020-01-01')
    >>> orchestrator.add_strategy('Momentum', MomentumSignalV2(), ['ES'], capital=100000)
    >>> results = orchestrator.run_backtests()
    >>> orchestrator.plot_results()

Backward compatibility: All existing code continues to work unchanged.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime

from core.multi_asset_loader import load_assets
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from core.portfolio.position_sizers import (
    FixedFractionalSizer, 
    FuturesContractSizer,
    ATRSizer,
    VolatilityScaledSizer,
    KellySizer
)
from core.benchmark import BenchmarkLoader
from utils.plotter import PortfolioPlotter
from utils.formatter import PerformanceSummary
from core.walk_forward_optimizer import WalkForwardOptimizer


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    name: str
    signal_generator: Any  # SignalModel instance
    assets: List[str]
    capital: float
    capital_pct: Optional[float] = None  # Alternative to absolute capital
    max_position_pct: Optional[float] = None
    risk_per_trade: float = 0.02
    transaction_cost_bps: float = 3.0
    position_sizer_type: str = 'fixed'  # 'fixed', 'atr', 'volatility', 'kelly', 'futures'
    position_sizer_params: Optional[Dict[str, Any]] = None  # Additional sizer parameters


class BacktestOrchestrator:
    """
    High-level orchestrator for running multi-strategy backtests.
    
    Simplifies common workflows while maintaining full flexibility.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_futures_sizing: bool = False,
        contract_multipliers: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Optional configuration dictionary with keys:
                - assets: List[str] - Assets to load
                - total_capital: float - Total portfolio capital
                - oos_split: float - Out-of-sample split percentage (e.g., 0.20 for 20%)
                - date_range: Tuple[str, str] - (start_date, end_date)
                - use_futures_sizing: bool - Use futures contract sizing
            use_futures_sizing: If True, use FuturesContractSizer for integer contracts
            contract_multipliers: Dict of ticker -> multiplier (e.g., {'ES': 50, 'CL': 1000})
        """
        self.prices: Dict[str, pd.DataFrame] = {}
        self.strategies: List[StrategyConfig] = []
        self.signals: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.results: Dict[str, Any] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.benchmark_name: Optional[str] = None
        
        # Configuration
        self.config = config or {}
        self.total_capital = self.config.get('total_capital', 0.0)
        self.oos_split = self.config.get('oos_split', 0.0)
        self.allocated_capital = 0.0  # Track allocated capital via capital_pct
        
        # Futures contract sizing
        self.use_futures_sizing = use_futures_sizing or self.config.get('use_futures_sizing', False)
        self.contract_multipliers = contract_multipliers or {}
        
        # OOS data storage
        self.prices_train: Dict[str, pd.DataFrame] = {}
        self.prices_test: Dict[str, pd.DataFrame] = {}
        self.oos_results: Dict[str, Any] = {}
        
        # Track what's been run
        self._data_loaded = False
        self._signals_generated = False
        self._backtests_run = False
        self._oos_run = False
    
    def load_data(
        self,
        tickers: List[str],
        start_date: str = '2015-01-01',
        end_date: Optional[str] = None,
        use_yfinance: bool = True,
        verbose: bool = True
    ) -> 'BacktestOrchestrator':
        """
        Load price data for multiple assets.
        
        Args:
            tickers: List of asset tickers (e.g., ['ES', 'NQ', 'CL'])
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (default: today)
            use_yfinance: Whether to use yfinance for data fetching
            verbose: Print loading summary
        
        Returns:
            self (for method chaining)
        """
        if verbose:
            print(f"📊 Loading data for {len(tickers)} assets...")
        
        self.prices = load_assets(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            use_yfinance=use_yfinance
        )
        
        self._data_loaded = True
        
        if verbose:
            print(f"\n✅ Data loaded successfully:")
            for ticker, df in self.prices.items():
                print(f"   {ticker}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")
                print(f"      Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        
        return self
    
    def load_benchmark(
        self,
        benchmark_ticker: str = 'SPY',
        verbose: bool = True
    ) -> 'BacktestOrchestrator':
        """
        Load benchmark data for comparison.
        
        Args:
            benchmark_ticker: Benchmark ticker (default: SPY)
            verbose: Print loading summary
        
        Returns:
            self (for method chaining)
        """
        if not self._data_loaded:
            raise RuntimeError("Must load data before loading benchmark. Call load_data() first.")
        
        if verbose:
            print(f"\n📈 Loading benchmark: {benchmark_ticker}...")
        
        # Get date range from loaded data
        first_ticker = list(self.prices.keys())[0]
        start_date = self.prices[first_ticker].index[0]
        end_date = self.prices[first_ticker].index[-1]
        
        loader = BenchmarkLoader()
        self.benchmark_data = loader.load_benchmark(
            ticker=benchmark_ticker,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        self.benchmark_name = benchmark_ticker
        
        if verbose:
            print(f"✅ Benchmark loaded: {len(self.benchmark_data)} days")
        
        return self
    
    def add_strategy(
        self,
        name: str,
        signal_generator: Any,
        assets: List[str],
        capital: Optional[float] = None,
        capital_pct: Optional[float] = None,
        max_position_pct: Optional[float] = None,
        risk_per_trade: float = 0.02,
        transaction_cost_bps: float = 3.0,
        position_sizer_type: str = 'fixed',
        position_sizer_params: Optional[Dict[str, Any]] = None
    ) -> 'BacktestOrchestrator':
        """
        Add a strategy to the backtest.
        
        Args:
            name: Strategy name (e.g., 'Momentum_ES')
            signal_generator: Signal generator instance (e.g., MomentumSignalV2())
            assets: List of assets this strategy trades
            capital: Initial capital for this strategy (absolute amount)
            capital_pct: Capital as percentage of total_capital (alternative to capital)
            max_position_pct: Max position size per asset (default: 1/num_assets)
            risk_per_trade: Risk per trade as fraction (default: 0.02 = 2%)
            transaction_cost_bps: Transaction costs in basis points (default: 3)
            position_sizer_type: Type of position sizer ('fixed', 'atr', 'volatility', 'kelly', 'futures')
            position_sizer_params: Additional parameters for position sizer
        
        Returns:
            self (for method chaining)
        """
        if not self._data_loaded:
            raise RuntimeError("Must load data before adding strategies. Call load_data() first.")
        
        # Validate assets
        for asset in assets:
            if asset not in self.prices:
                raise ValueError(f"Asset '{asset}' not found in loaded data. Available: {list(self.prices.keys())}")
        
        # Handle capital allocation
        if capital is not None and capital_pct is not None:
            raise ValueError("Cannot specify both 'capital' and 'capital_pct'. Choose one.")
        
        if capital_pct is not None:
            if self.total_capital == 0:
                raise ValueError("Cannot use capital_pct without setting total_capital in config")
            if not (0 < capital_pct <= 1.0):
                raise ValueError(f"capital_pct must be between 0 and 1, got {capital_pct}")
            
            # Check total allocation
            if self.allocated_capital + capital_pct > 1.0:
                raise ValueError(
                    f"Total capital allocation would exceed 100%: "
                    f"{self.allocated_capital:.1%} + {capital_pct:.1%} = "
                    f"{(self.allocated_capital + capital_pct):.1%}"
                )
            
            capital = self.total_capital * capital_pct
            self.allocated_capital += capital_pct
        elif capital is None:
            raise ValueError("Must specify either 'capital' or 'capital_pct'")
        
        # Auto-calculate max position if not provided
        if max_position_pct is None:
            max_position_pct = 1.0 / len(assets) if len(assets) > 1 else 1.0
        
        config = StrategyConfig(
            name=name,
            signal_generator=signal_generator,
            assets=assets,
            capital=capital,
            capital_pct=capital_pct,
            max_position_pct=max_position_pct,
            risk_per_trade=risk_per_trade,
            transaction_cost_bps=transaction_cost_bps,
            position_sizer_type=position_sizer_type,
            position_sizer_params=position_sizer_params or {}
        )
        
        self.strategies.append(config)
        return self
    
    def generate_signals(self, verbose: bool = True) -> 'BacktestOrchestrator':
        """
        Generate trading signals for all strategies.
        
        Args:
            verbose: Print generation summary
        
        Returns:
            self (for method chaining)
        """
        if not self.strategies:
            raise RuntimeError("No strategies added. Call add_strategy() first.")
        
        if verbose:
            print(f"\n🎯 Generating signals for {len(self.strategies)} strategies...")
        
        for strat in self.strategies:
            if verbose:
                print(f"\n  {strat.name} ({strat.signal_generator.__class__.__name__}):")
            
            strategy_signals = {}
            for asset in strat.assets:
                # Generate signal for this asset
                sig = strat.signal_generator.generate(self.prices[asset].copy())
                strategy_signals[asset] = sig
                
                if verbose:
                    long_pct = (sig['Signal'] == 1).sum() / len(sig) * 100
                    short_pct = (sig['Signal'] == -1).sum() / len(sig) * 100
                    flat_pct = (sig['Signal'] == 0).sum() / len(sig) * 100
                    
                    print(f"    {asset}: {long_pct:.1f}% long, {short_pct:.1f}% short, {flat_pct:.1f}% flat")
            
            self.signals[strat.name] = strategy_signals
        
        self._signals_generated = True
        
        if verbose:
            print(f"\n✅ Signals generated successfully")
        
        return self
    
    def _create_position_sizer(self, strat: StrategyConfig):
        """
        Create position sizer based on strategy configuration.
        
        Args:
            strat: Strategy configuration
            
        Returns:
            PositionSizer instance
        """
        sizer_type = strat.position_sizer_type.lower()
        params = strat.position_sizer_params or {}
        
        if sizer_type == 'futures' or self.use_futures_sizing:
            return FuturesContractSizer(
                contract_multipliers=self.contract_multipliers,
                max_position_pct=strat.max_position_pct,
                risk_per_trade=strat.risk_per_trade,
                min_contracts=params.get('min_contracts', 1)
            )
        elif sizer_type == 'fixed':
            return FixedFractionalSizer(
                risk_per_trade=strat.risk_per_trade,
                max_position_pct=strat.max_position_pct,
                min_trade_value=params.get('min_trade_value', 100.0)
            )
        elif sizer_type == 'atr':
            return ATRSizer(
                risk_per_trade=strat.risk_per_trade,
                atr_multiplier=params.get('atr_multiplier', 2.0),
                max_position_pct=strat.max_position_pct,
                min_trade_value=params.get('min_trade_value', 100.0)
            )
        elif sizer_type == 'volatility':
            return VolatilityScaledSizer(
                target_volatility=params.get('target_volatility', 0.15),
                max_position_pct=strat.max_position_pct,
                min_position_pct=params.get('min_position_pct', 0.05),
                min_trade_value=params.get('min_trade_value', 100.0)
            )
        elif sizer_type == 'kelly':
            return KellySizer(
                max_position_pct=strat.max_position_pct,
                kelly_fraction=params.get('kelly_fraction', 0.5),
                min_trade_value=params.get('min_trade_value', 100.0)
            )
        else:
            raise ValueError(
                f"Unknown position_sizer_type: '{sizer_type}'. "
                f"Valid options: 'fixed', 'atr', 'volatility', 'kelly', 'futures'"
            )
    
    def run_backtests(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run backtests for all strategies.
        
        Args:
            verbose: Print backtest progress
        
        Returns:
            Dictionary of strategy_name -> BacktestResult
        """
        if not self._signals_generated:
            raise RuntimeError("Must generate signals first. Call generate_signals().")
        
        if verbose:
            print(f"\n🔄 Running backtests for {len(self.strategies)} strategies...")
        
        for strat in self.strategies:
            if verbose:
                print(f"\n  {strat.name} (${strat.capital:,})...")
            
            # Get signals and prices for this strategy
            signal_dict = self.signals[strat.name]
            prices_dict = {asset: self.prices[asset] for asset in strat.assets}
            
            # Create position sizer
            position_sizer = self._create_position_sizer(strat)
            
            # Create portfolio manager
            pm = PortfolioManagerV2(
                initial_capital=strat.capital,
                risk_per_trade=strat.risk_per_trade,
                max_position_size=strat.max_position_pct,
                transaction_cost_bps=strat.transaction_cost_bps,
                position_sizer=position_sizer
            )
            
            # Run backtest
            result = pm.run_backtest(signals=signal_dict, prices=prices_dict)
            
            # Store result
            self.results[strat.name] = {
                'result': result,
                'capital': strat.capital,
                'assets': strat.assets
            }
            
            if verbose:
                print(f"    ✅ Total Return: {result.total_return:.2%}")
                print(f"    📊 Sharpe: {result.metrics['Sharpe Ratio']:.2f}")
                print(f"    📉 Max DD: {result.metrics['Max Drawdown']:.2%}")
        
        self._backtests_run = True
        
        if verbose:
            print(f"\n✅ All backtests completed successfully")
        
        return self.results
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all strategy results.
        
        Returns:
            DataFrame with key metrics for each strategy
        """
        if not self._backtests_run:
            raise RuntimeError("Must run backtests first. Call run_backtests().")
        
        summary_data = []
        for name, data in self.results.items():
            result = data['result']
            summary_data.append({
                'Strategy': name,
                'Capital': data['capital'],
                'Assets': ', '.join(data['assets']),
                'Final Value': result.final_equity,
                'Total Return': result.total_return,
                'CAGR': result.metrics['CAGR'],
                'Sharpe': result.metrics['Sharpe Ratio'],
                'Max Drawdown': result.metrics['Max Drawdown'],
                'Win Rate': result.metrics['Win Rate'],
                'Total Trades': result.metrics['Total Trades']
            })
        
        return pd.DataFrame(summary_data)
    
    def print_summary(self):
        """Print formatted summary of all strategies."""
        if not self._backtests_run:
            raise RuntimeError("Must run backtests first. Call run_backtests().")
        
        print("\n" + "="*100)
        print("📊 BACKTEST SUMMARY")
        print("="*100)
        
        summary_df = self.get_summary()
        
        # Format for display
        pd.options.display.float_format = '{:.2f}'.format
        print(summary_df.to_string(index=False))
        
        # Overall portfolio stats
        total_capital = summary_df['Capital'].sum()
        total_final = summary_df['Final Value'].sum()
        portfolio_return = (total_final / total_capital - 1)
        
        print("\n" + "-"*100)
        print(f"Portfolio Total Capital: ${total_capital:,.0f}")
        print(f"Portfolio Final Value:   ${total_final:,.0f}")
        print(f"Portfolio Return:        {portfolio_return:.2%}")
        print("="*100)
    
    def plot_results(
        self,
        show_individual: bool = True,
        show_benchmark: bool = True,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Plot backtest results using PortfolioPlotter.
        
        Args:
            show_individual: Show individual strategy curves
            show_benchmark: Show benchmark comparison (if loaded)
            figsize: Figure size (width, height)
        """
        if not self._backtests_run:
            raise RuntimeError("Must run backtests first. Call run_backtests().")
        
        plotter = PortfolioPlotter(
            strategy_results=self.results,
            benchmark_data=self.benchmark_data,
            benchmark_name=self.benchmark_name
        )
        
        # plot_equity_curves handles benchmark internally when show_benchmark=True
        plotter.plot_equity_curves(
            show_individual=show_individual, 
            show_benchmark=show_benchmark,
            figsize=figsize
        )
    
    def get_performance_summary(self, benchmark_name: Optional[str] = None) -> PerformanceSummary:
        """
        Get detailed performance summary for the portfolio.
        
        Args:
            benchmark_name: Optional benchmark name for comparison
        
        Returns:
            PerformanceSummary object
        """
        if not self._backtests_run:
            raise RuntimeError("Must run backtests first. Call run_backtests().")
        
        # Combine all strategy equity curves
        first_result = list(self.results.values())[0]['result']
        dates = first_result.equity_curve.index
        
        combined_equity = pd.Series(
            sum(data['result'].equity_curve['TotalValue'].values for data in self.results.values()),
            index=dates
        )
        
        # Get benchmark if available
        benchmark_equity = None
        if self.benchmark_data is not None:
            benchmark_equity = self.benchmark_data['Close']
        
        return PerformanceSummary(
            strategy_results=self.results,
            benchmark_data=self.benchmark_data,
            benchmark_name=benchmark_name or self.benchmark_name
        )
    
    def split_train_test_data(self, oos_pct: Optional[float] = None, verbose: bool = True):
        """
        Split data into train/test sets for out-of-sample validation.
        
        Args:
            oos_pct: Percentage for out-of-sample (e.g., 0.20 for 20%). Uses config if not provided.
            verbose: Print split information
        """
        oos_pct = oos_pct or self.oos_split
        if oos_pct == 0:
            raise ValueError("oos_pct must be > 0. Set via oos_pct parameter or config['oos_split']")
        
        if not (0 < oos_pct < 1):
            raise ValueError(f"oos_pct must be between 0 and 1, got {oos_pct}")
        
        if verbose:
            print(f"\n📊 Splitting data: {(1-oos_pct)*100:.0f}% train, {oos_pct*100:.0f}% OOS")
        
        for ticker, df in self.prices.items():
            split_idx = int(len(df) * (1 - oos_pct))
            self.prices_train[ticker] = df.iloc[:split_idx].copy()
            self.prices_test[ticker] = df.iloc[split_idx:].copy()
            
            if verbose:
                print(f"  {ticker}: {len(self.prices_train[ticker])} train days, "
                      f"{len(self.prices_test[ticker])} test days")
                print(f"      Train: {self.prices_train[ticker].index[0].date()} to "
                      f"{self.prices_train[ticker].index[-1].date()}")
                print(f"      Test:  {self.prices_test[ticker].index[0].date()} to "
                      f"{self.prices_test[ticker].index[-1].date()}")
        
        # Update main prices to train data
        self.prices = self.prices_train.copy()
        
        if verbose:
            print(f"✅ Data split complete. Use .prices for train, .prices_test for OOS")
    
    def run_oos_backtest(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run out-of-sample backtest on test data.
        
        Must call split_train_test_data() first.
        
        Args:
            verbose: Print progress
            
        Returns:
            Dictionary of strategy_name -> OOS BacktestResult
        """
        if not self.prices_test:
            raise RuntimeError("Must call split_train_test_data() before running OOS backtest")
        
        if not self._signals_generated:
            raise RuntimeError("Must generate signals on train data first. Call generate_signals()")
        
        if verbose:
            print(f"\n🎯 Running OUT-OF-SAMPLE backtests...")
        
        # Temporarily swap prices to test set
        original_prices = self.prices
        self.prices = self.prices_test
        
        # Generate signals on OOS data
        for strat in self.strategies:
            if verbose:
                print(f"\n  {strat.name} - Generating OOS signals...")
            
            strategy_signals = {}
            for asset in strat.assets:
                sig = strat.signal_generator.generate(self.prices_test[asset].copy())
                strategy_signals[asset] = sig
            
            self.signals[strat.name] = strategy_signals
        
        # Run backtests on OOS data
        for strat in self.strategies:
            if verbose:
                print(f"\n  {strat.name} (${strat.capital:,})...")
            
            signal_dict = self.signals[strat.name]
            prices_dict = {asset: self.prices_test[asset] for asset in strat.assets}
            
            position_sizer = self._create_position_sizer(strat)
            
            pm = PortfolioManagerV2(
                initial_capital=strat.capital,
                risk_per_trade=strat.risk_per_trade,
                max_position_size=strat.max_position_pct,
                transaction_cost_bps=strat.transaction_cost_bps,
                position_sizer=position_sizer
            )
            
            result = pm.run_backtest(signals=signal_dict, prices=prices_dict)
            
            self.oos_results[strat.name] = {
                'result': result,
                'capital': strat.capital,
                'assets': strat.assets
            }
            
            if verbose:
                print(f"    ✅ OOS Return: {result.total_return:.2%}")
                print(f"    📊 OOS Sharpe: {result.metrics['Sharpe Ratio']:.2f}")
                print(f"    📉 OOS Max DD: {result.metrics['Max Drawdown']:.2%}")
        
        # Restore original prices
        self.prices = original_prices
        self._oos_run = True
        
        if verbose:
            print(f"\n✅ OOS backtests completed")
        
        return self.oos_results
    
    def run_walkforward(
        self,
        signal_class,
        param_grid: Dict[str, List[Any]],
        assets: List[str],
        train_pct: float = 0.70,
        test_pct: float = 0.15,
        metric: str = 'sharpe',
        initial_capital: float = 100000,
        verbose: bool = True
    ) -> WalkForwardOptimizer:
        """
        Run walk-forward optimization with parameter tuning.
        
        Args:
            signal_class: Signal generator class (not instance)
            param_grid: Dictionary of parameter names to lists of values
            assets: List of assets to optimize (currently single-asset only)
            train_pct: Training window percentage (default: 0.70 = 70%)
            test_pct: Test window percentage (default: 0.15 = 15%)
            metric: Optimization metric ('sharpe', 'return', 'risk_adjusted')
            initial_capital: Starting capital for optimization
            verbose: Print progress
            
        Returns:
            WalkForwardOptimizer with results
        """
        if len(assets) != 1:
            raise ValueError("Walk-forward optimization currently supports single asset only")
        
        asset = assets[0]
        if asset not in self.prices:
            raise ValueError(f"Asset '{asset}' not in loaded data")
        
        if verbose:
            print(f"\n🔄 Starting walk-forward optimization for {asset}...")
        
        optimizer = WalkForwardOptimizer(
            signal_class=signal_class,
            param_grid=param_grid,
            train_pct=train_pct,
            test_pct=test_pct,
            metric=metric,
            verbose=verbose
        )
        
        optimizer.optimize(
            prices=self.prices[asset],
            asset_name=asset,
            initial_capital=initial_capital
        )
        
        return optimizer
    
    def export_html_dashboard(
        self,
        output_dir: str = 'results/html',
        filename_prefix: str = 'backtest',
        include_oos: bool = False
    ) -> str:
        """
        Export backtest results to HTML dashboard.
        
        Args:
            output_dir: Output directory for HTML file
            filename_prefix: Prefix for filename (default: 'backtest')
            include_oos: Include OOS results if available
            
        Returns:
            Path to generated HTML file
        """
        import os
        from datetime import datetime as dt
        
        if not self._backtests_run:
            raise RuntimeError("Must run backtests first. Call run_backtests()")
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with date
        date_str = dt.now().strftime('%Y-%m-%d')
        filename = f"{filename_prefix}_{date_str}.html"
        filepath = os.path.join(output_dir, filename)
        
        # Build HTML content
        html_parts = []
        html_parts.append("<html><head><style>")
        html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_parts.append("h1 { color: #333; }")
        html_parts.append("h2 { color: #666; }")
        html_parts.append("table { border-collapse: collapse; margin: 20px 0; }")
        html_parts.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html_parts.append("th { background-color: #4CAF50; color: white; }")
        html_parts.append("</style></head><body>")
        html_parts.append(f"<h1>Backtest Results - {date_str}</h1>")
        
        # Add summary table
        summary_df = self.get_summary()
        html_parts.append("<h2>Strategy Performance</h2>")
        html_parts.append(summary_df.to_html(index=False))
        
        # Add OOS results if requested
        if include_oos and self._oos_run:
            html_parts.append("\n<h2>Out-of-Sample Results</h2>")
            
            oos_data = []
            for name, data in self.oos_results.items():
                result = data['result']
                oos_data.append({
                    'Strategy': name,
                    'OOS Return': f"{result.total_return:.2%}",
                    'OOS Sharpe': f"{result.metrics['Sharpe Ratio']:.2f}",
                    'OOS Max DD': f"{result.metrics['Max Drawdown']:.2%}",
                    'OOS Trades': result.metrics['Total Trades']
                })
            
            oos_df = pd.DataFrame(oos_data)
            html_parts.append(oos_df.to_html(index=False))
        
        html_parts.append("</body></html>")
        html_content = "\n".join(html_parts)
        
        # Write HTML file
        with open(filepath, 'w') as f:
            f.write(html_content)
        
        print(f"\n💾 HTML dashboard saved: {filepath}")
        return filepath
    
    # Convenience methods for common workflows
    
    @classmethod
    def quick_backtest(
        cls,
        tickers: List[str],
        signal_generator: Any,
        start_date: str = '2020-01-01',
        capital: float = 100000,
        use_futures_sizing: bool = False,
        contract_multipliers: Optional[Dict[str, float]] = None
    ) -> 'BacktestOrchestrator':
        """
        Quick backtest for a single strategy on multiple assets.
        
        Args:
            tickers: List of assets to trade
            signal_generator: Signal generator instance
            start_date: Start date
            capital: Initial capital
            use_futures_sizing: Use futures contract sizing
            contract_multipliers: Contract multipliers if using futures
        
        Returns:
            Configured and run orchestrator
        """
        orchestrator = cls(
            use_futures_sizing=use_futures_sizing,
            contract_multipliers=contract_multipliers
        )
        
        return (orchestrator
                .load_data(tickers, start_date=start_date)
                .add_strategy('Strategy', signal_generator, tickers, capital)
                .generate_signals()
                .run_backtests())


# Convenience function for backward compatibility
def run_multi_strategy_backtest(
    prices: Dict[str, pd.DataFrame],
    strategies: List[Dict],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run backtests for multiple strategies (backward compatible interface).
    
    Args:
        prices: Dict of ticker -> DataFrame with OHLC data
        strategies: List of strategy configs, each containing:
            - name: str
            - signal_generator: SignalModel
            - assets: List[str]
            - capital: float
        verbose: Print progress
    
    Returns:
        Dict of strategy_name -> {'result': BacktestResult, 'capital': float, 'assets': List[str]}
    """
    orchestrator = BacktestOrchestrator()
    orchestrator.prices = prices
    orchestrator._data_loaded = True
    
    for strat in strategies:
        orchestrator.add_strategy(
            name=strat['name'],
            signal_generator=strat['signal_generator'],
            assets=strat['assets'],
            capital=strat['capital']
        )
    
    orchestrator.generate_signals(verbose=verbose)
    return orchestrator.run_backtests(verbose=verbose)
