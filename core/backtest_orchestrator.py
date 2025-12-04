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
from core.portfolio.position_sizers import FixedFractionalSizer, FuturesContractSizer
from core.benchmark import BenchmarkLoader
from utils.plotter import PortfolioPlotter
from utils.formatter import PerformanceSummary


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""
    name: str
    signal_generator: Any  # SignalModel instance
    assets: List[str]
    capital: float
    max_position_pct: Optional[float] = None
    risk_per_trade: float = 0.02
    transaction_cost_bps: float = 3.0


class BacktestOrchestrator:
    """
    High-level orchestrator for running multi-strategy backtests.
    
    Simplifies common workflows while maintaining full flexibility.
    """
    
    def __init__(
        self,
        use_futures_sizing: bool = False,
        contract_multipliers: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            use_futures_sizing: If True, use FuturesContractSizer for integer contracts
            contract_multipliers: Dict of ticker -> multiplier (e.g., {'ES': 50, 'CL': 1000})
        """
        self.prices: Dict[str, pd.DataFrame] = {}
        self.strategies: List[StrategyConfig] = []
        self.signals: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.results: Dict[str, Any] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.benchmark_name: Optional[str] = None
        
        # Futures contract sizing
        self.use_futures_sizing = use_futures_sizing
        self.contract_multipliers = contract_multipliers or {}
        
        # Track what's been run
        self._data_loaded = False
        self._signals_generated = False
        self._backtests_run = False
    
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
        capital: float,
        max_position_pct: Optional[float] = None,
        risk_per_trade: float = 0.02,
        transaction_cost_bps: float = 3.0
    ) -> 'BacktestOrchestrator':
        """
        Add a strategy to the backtest.
        
        Args:
            name: Strategy name (e.g., 'Momentum_ES')
            signal_generator: Signal generator instance (e.g., MomentumSignalV2())
            assets: List of assets this strategy trades
            capital: Initial capital for this strategy
            max_position_pct: Max position size per asset (default: 1/num_assets)
            risk_per_trade: Risk per trade as fraction (default: 0.02 = 2%)
            transaction_cost_bps: Transaction costs in basis points (default: 3)
        
        Returns:
            self (for method chaining)
        """
        if not self._data_loaded:
            raise RuntimeError("Must load data before adding strategies. Call load_data() first.")
        
        # Validate assets
        for asset in assets:
            if asset not in self.prices:
                raise ValueError(f"Asset '{asset}' not found in loaded data. Available: {list(self.prices.keys())}")
        
        # Auto-calculate max position if not provided
        if max_position_pct is None:
            max_position_pct = 1.0 / len(assets) if len(assets) > 1 else 1.0
        
        config = StrategyConfig(
            name=name,
            signal_generator=signal_generator,
            assets=assets,
            capital=capital,
            max_position_pct=max_position_pct,
            risk_per_trade=risk_per_trade,
            transaction_cost_bps=transaction_cost_bps
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
            if self.use_futures_sizing:
                position_sizer = FuturesContractSizer(
                    contract_multipliers=self.contract_multipliers,
                    max_position_pct=strat.max_position_pct,
                    risk_per_trade=strat.risk_per_trade,
                    min_contracts=1
                )
            else:
                position_sizer = FixedFractionalSizer(
                    risk_per_trade=strat.risk_per_trade,
                    max_position_pct=strat.max_position_pct
                )
            
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
        
        plotter.plot_equity_curves(show_individual=show_individual, figsize=figsize)
        
        if show_benchmark and self.benchmark_data is not None:
            plotter.plot_benchmark_comparison(figsize=figsize)
    
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
            strategy_equity=combined_equity,
            benchmark_equity=benchmark_equity,
            benchmark_name=benchmark_name or self.benchmark_name
        )
    
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
