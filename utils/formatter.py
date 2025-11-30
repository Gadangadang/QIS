"""
Unified text formatting and summary utilities for backtesting analysis.

Provides standardized output formatting to eliminate duplicate print statements
across notebooks.

Usage:
    from utils.formatter import PerformanceSummary
    
    summary = PerformanceSummary(strategy_results, benchmark_data=spy_data)
    summary.print_full_report()
    summary.print_strategy_rankings()
    summary.print_recommendations()
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime


class PerformanceSummary:
    """
    Unified text output formatter for portfolio backtest analysis.
    
    Handles single-strategy and multi-strategy portfolios.
    Eliminates duplicate print statements across notebooks.
    """
    
    def __init__(
        self,
        strategy_results: Dict,
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_name: str = 'SPY',
        period_label: str = 'IN-SAMPLE'
    ):
        """
        Initialize summary formatter with backtest results.
        
        Args:
            strategy_results: Dict of {strategy_name: {'result': BacktestResult, 'capital': float}}
            benchmark_data: Optional benchmark price data (OHLC)
            benchmark_name: Name of benchmark (default 'SPY')
            period_label: Label for period (e.g., 'IN-SAMPLE', 'OUT-OF-SAMPLE')
        """
        self.strategy_results = strategy_results
        self.benchmark_data = benchmark_data
        self.benchmark_name = benchmark_name
        self.period_label = period_label
        
        # Calculate metrics
        self._calculate_portfolio_metrics()
        if benchmark_data is not None:
            self._calculate_benchmark_metrics()
    
    def _calculate_portfolio_metrics(self):
        """Calculate combined portfolio metrics."""
        # Get reference dates
        first_result = list(self.strategy_results.values())[0]['result']
        self.dates = first_result.equity_curve.index
        
        # Calculate combined equity
        self.combined_equity = pd.Series(
            sum(data['result'].equity_curve['TotalValue'].values 
                for data in self.strategy_results.values()),
            index=self.dates
        )
        
        # Total capital and final value
        self.total_capital = sum(data['capital'] for data in self.strategy_results.values())
        self.final_value = self.combined_equity.iloc[-1]
        
        # Returns
        self.total_return = (self.final_value - self.total_capital) / self.total_capital
        
        # Calculate CAGR
        years = (self.dates[-1] - self.dates[0]).days / 365.25
        self.cagr = (self.final_value / self.total_capital) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate Sharpe
        returns = self.combined_equity.pct_change().fillna(0)
        self.sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        running_max = self.combined_equity.expanding().max()
        drawdown = (self.combined_equity - running_max) / running_max
        self.max_drawdown = drawdown.min()
        
        # Number of strategies
        self.n_strategies = len(self.strategy_results)
    
    def _calculate_benchmark_metrics(self):
        """Calculate benchmark metrics."""
        bench_prices = self.benchmark_data['Close'].reindex(self.dates, method='ffill')
        
        # Buy-and-hold equity
        initial_value = self.total_capital
        shares = initial_value / bench_prices.iloc[0]
        bench_equity = shares * bench_prices
        
        # Returns
        self.bench_return = (bench_equity.iloc[-1] - initial_value) / initial_value
        
        # Calculate CAGR
        years = (self.dates[-1] - self.dates[0]).days / 365.25
        self.bench_cagr = (bench_equity.iloc[-1] / initial_value) ** (1 / years) - 1 if years > 0 else 0
        
        # Outperformance
        self.outperformance = self.total_return - self.bench_return
    
    def print_full_report(self, width: int = 120):
        """Print comprehensive performance report."""
        print("=" * width)
        print(f"📋 PERFORMANCE SUMMARY - {self.period_label}")
        print("=" * width)
        
        self.print_portfolio_metrics(width)
        
        if self.benchmark_data is not None:
            self.print_benchmark_comparison(width)
        
        self.print_strategy_rankings(width)
        
        print("\n" + "=" * width)
        print(f"📝 {self.period_label} Analysis Complete")
        print("=" * width)
    
    def print_portfolio_metrics(self, width: int = 80):
        """Print portfolio performance metrics."""
        print(f"\n📊 PORTFOLIO PERFORMANCE ({self.period_label})")
        print("-" * width)
        print(f"Initial Capital:     ${self.total_capital:,.0f}")
        print(f"Final Value:         ${self.final_value:,.0f}")
        print(f"Total Return:        {self.total_return:.2%}")
        print(f"CAGR:                {self.cagr:.2%}")
        print(f"Sharpe Ratio:        {self.sharpe:.2f}")
        print(f"Max Drawdown:        {self.max_drawdown:.2%}")
        print(f"Number of Strategies: {self.n_strategies}")
        
        # Period dates
        start_date = self.dates[0].strftime('%Y-%m-%d')
        end_date = self.dates[-1].strftime('%Y-%m-%d')
        print(f"Period:              {start_date} to {end_date}")
    
    def print_benchmark_comparison(self, width: int = 80):
        """Print benchmark comparison."""
        if self.benchmark_data is None:
            return
        
        print(f"\n🎯 BENCHMARK COMPARISON ({self.benchmark_name})")
        print("-" * width)
        print(f"Portfolio Return:    {self.total_return:.2%}")
        print(f"{self.benchmark_name} Return:        {self.bench_return:.2%}")
        print(f"Outperformance:      {self.outperformance:.2%}")
        print(f"Portfolio CAGR:      {self.cagr:.2%}")
        print(f"{self.benchmark_name} CAGR:          {self.bench_cagr:.2%}")
        
        # Win/loss indicator
        if self.outperformance > 0:
            print(f"\n✅ BEATING {self.benchmark_name} by {self.outperformance:.2%}")
        else:
            print(f"\n❌ LAGGING {self.benchmark_name} by {abs(self.outperformance):.2%}")
    
    def print_strategy_rankings(self, width: int = 80):
        """Print strategy performance rankings."""
        print(f"\n🏆 STRATEGY RANKINGS ({self.period_label})")
        print("-" * width)
        
        # Compile strategy performance
        rankings = []
        for name, data in self.strategy_results.items():
            result = data['result']
            capital = data['capital']
            final_val = result.equity_curve['TotalValue'].iloc[-1]
            
            ret = (final_val - capital) / capital
            sharpe = result.metrics.get('Sharpe Ratio', 0)
            max_dd = result.metrics.get('Max Drawdown', 0)
            
            rankings.append({
                'Strategy': name,
                'Return': ret,
                'Sharpe': sharpe,
                'Max DD': max_dd,
                'Capital': capital
            })
        
        # Sort by return
        rankings.sort(key=lambda x: x['Return'], reverse=True)
        
        # Print rankings
        print(f"{'Rank':<6} {'Strategy':<30} {'Return':>12} {'Sharpe':>8} {'Max DD':>10} {'Capital':>12}")
        print("-" * width)
        
        for i, strat in enumerate(rankings, 1):
            print(f"{i:<6} {strat['Strategy']:<30} {strat['Return']:>11.2%} "
                  f"{strat['Sharpe']:>8.2f} {strat['Max DD']:>10.2%} ${strat['Capital']:>10,.0f}")
    
    def print_trade_statistics(self, strategy_name: Optional[str] = None, width: int = 80):
        """Print trade statistics for a strategy or combined portfolio."""
        if strategy_name and strategy_name in self.strategy_results:
            result = self.strategy_results[strategy_name]['result']
            title = f"TRADE STATISTICS - {strategy_name}"
        else:
            # Aggregate trades from all strategies
            title = "COMBINED TRADE STATISTICS"
            result = self._aggregate_trades()
        
        print(f"\n📈 {title}")
        print("-" * width)
        
        metrics = result.metrics
        
        print(f"Total Trades:        {metrics.get('Total Trades', 0):.0f}")
        print(f"Win Rate:            {metrics.get('Win Rate', 0):.2%}")
        print(f"Avg Win:             {metrics.get('Avg Win', 0):.2%}")
        print(f"Avg Loss:            {metrics.get('Avg Loss', 0):.2%}")
        print(f"Win/Loss Ratio:      {abs(metrics.get('Avg Win', 0) / metrics.get('Avg Loss', -1)) if metrics.get('Avg Loss', 0) != 0 else 0:.2f}")
        print(f"Profit Factor:       {metrics.get('Profit Factor', 0):.2f}")
        
        if hasattr(result, 'closed_positions') and len(result.closed_positions) > 0:
            trades_df = pd.DataFrame(result.closed_positions)
            print(f"\nBest Trade:          {trades_df['pnl'].max():>+.2f} ({trades_df['return'].max():>+.2%})")
            print(f"Worst Trade:         {trades_df['pnl'].min():>+.2f} ({trades_df['return'].min():>+.2%})")
            print(f"Avg Trade Duration:  {trades_df['hold_days'].mean():.1f} days")
    
    def print_recommendations(self, width: int = 120):
        """Print actionable recommendations based on performance."""
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * width)
        
        # Check if beating benchmark
        if self.benchmark_data is not None and self.outperformance > 0:
            print(f"\n✅ STRONG PERFORMANCE - Beating {self.benchmark_name}")
            print("\nNext Steps:")
            print("  1. Generate detailed HTML report for investor presentation")
            print("  2. Consider live paper trading with current allocations")
            print("  3. Monitor concentration risk across strategies")
            print("  4. Test alternative position sizers (Kelly, ATR-based)")
            print("  5. Implement risk parity allocation for better diversification")
        
        elif self.benchmark_data is not None:
            print(f"\n⚠️  UNDERPERFORMANCE - Lagging {self.benchmark_name}")
            print("\nRecommended Actions:")
            print("  1. Optimize signal parameters (lookbacks, thresholds)")
            print("  2. Add regime filters (bull/bear detection)")
            print("  3. Test additional signal types (mean reversion, volume)")
            print("  4. Review and remove underperforming assets")
            print("  5. Increase allocation to high-Sharpe strategies")
        
        # Sharpe-based recommendations
        if self.sharpe < 1.0:
            print("\n⚠️  LOW SHARPE RATIO (<1.0)")
            print("  • Consider more diversified strategies")
            print("  • Tighten risk controls (stop-losses, position limits)")
            print("  • Test lower correlation assets")
        
        elif self.sharpe > 2.0:
            print("\n✅ EXCELLENT SHARPE RATIO (>2.0)")
            print("  • Strong risk-adjusted performance")
            print("  • Validate with out-of-sample testing")
            print("  • Consider increasing capital allocation")
        
        # Drawdown-based recommendations
        if abs(self.max_drawdown) > 0.20:
            print("\n⚠️  LARGE DRAWDOWN (>20%)")
            print("  • Implement stronger position sizing controls")
            print("  • Consider volatility-based sizing (ATR, vol-scaled)")
            print("  • Add portfolio heat limits")
            print("  • Test tighter stop-losses or time-based exits")
        
        print()
    
    def print_comparison_table(self, other_summary: 'PerformanceSummary', width: int = 100):
        """Print side-by-side comparison with another period (e.g., IS vs OOS)."""
        print(f"\n🔍 CONSISTENCY CHECK: {self.period_label} vs {other_summary.period_label}")
        print("=" * width)
        
        print(f"\n{'Metric':<25} {self.period_label:<20} {other_summary.period_label:<20} {'Difference':>15}")
        print("-" * width)
        
        # Return comparison
        diff_return = other_summary.total_return - self.total_return
        print(f"{'Total Return':<25} {self.total_return:>19.2%} {other_summary.total_return:>19.2%} {diff_return:>+14.2%}")
        
        # CAGR comparison
        diff_cagr = other_summary.cagr - self.cagr
        print(f"{'CAGR':<25} {self.cagr:>19.2%} {other_summary.cagr:>19.2%} {diff_cagr:>+14.2%}")
        
        # Sharpe comparison
        diff_sharpe = other_summary.sharpe - self.sharpe
        print(f"{'Sharpe Ratio':<25} {self.sharpe:>19.2f} {other_summary.sharpe:>19.2f} {diff_sharpe:>+14.2f}")
        
        # Drawdown comparison
        diff_dd = other_summary.max_drawdown - self.max_drawdown
        print(f"{'Max Drawdown':<25} {self.max_drawdown:>19.2%} {other_summary.max_drawdown:>19.2%} {diff_dd:>+14.2%}")
        
        # Consistency assessment
        print("\n🎯 CONSISTENCY ASSESSMENT:")
        
        if other_summary.total_return > 0 and self.total_return > 0:
            consistency = 1 - abs(diff_return) / abs(self.total_return)
            print(f"✅ Both periods profitable")
            print(f"   Return consistency: {consistency:.1%} ({'Good' if consistency > 0.5 else 'Variable'})")
        elif self.total_return > 0 and other_summary.total_return <= 0:
            print(f"❌ {self.period_label} profitable but {other_summary.period_label} loses money")
            print("⚠️  WARNING: Potential overfitting detected!")
        else:
            print("⚠️  Strategy needs optimization in both periods")
        
        print()
    
    def _aggregate_trades(self):
        """Aggregate trades from all strategies (placeholder)."""
        # Return first strategy result as placeholder
        # In practice, would aggregate all trades
        return list(self.strategy_results.values())[0]['result']
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export strategy metrics as DataFrame."""
        data = []
        for name, strat_data in self.strategy_results.items():
            result = strat_data['result']
            capital = strat_data['capital']
            final_val = result.equity_curve['TotalValue'].iloc[-1]
            
            metrics = {
                'Strategy': name,
                'Capital': capital,
                'Final Value': final_val,
                'Total Return': (final_val - capital) / capital,
                'Sharpe Ratio': result.metrics.get('Sharpe Ratio', 0),
                'Max Drawdown': result.metrics.get('Max Drawdown', 0),
                'Win Rate': result.metrics.get('Win Rate', 0),
                'Total Trades': result.metrics.get('Total Trades', 0)
            }
            data.append(metrics)
        
        df = pd.DataFrame(data)
        return df
    
    def print_metrics_table(self):
        """Print metrics as formatted table."""
        df = self.to_dataframe()
        
        print(f"\n📊 STRATEGY METRICS TABLE ({self.period_label})")
        print("=" * 120)
        
        # Format and display
        formatted = df.style.format({
            'Capital': '${:,.0f}',
            'Final Value': '${:,.0f}',
            'Total Return': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}',
            'Win Rate': '{:.2%}',
            'Total Trades': '{:.0f}'
        })
        
        try:
            from IPython.display import display
            display(formatted)
        except ImportError:
            print(df.to_string(index=False))


# Convenience functions
def quick_summary(strategy_results: Dict, benchmark_data: Optional[pd.DataFrame] = None, **kwargs):
    """Quick performance summary print."""
    summary = PerformanceSummary(strategy_results, benchmark_data, **kwargs)
    summary.print_full_report()
    return summary


def compare_periods(is_results: Dict, oos_results: Dict, benchmark_data: Optional[pd.DataFrame] = None):
    """Compare in-sample vs out-of-sample performance."""
    is_summary = PerformanceSummary(is_results, benchmark_data, period_label='IN-SAMPLE')
    oos_summary = PerformanceSummary(oos_results, benchmark_data, period_label='OUT-OF-SAMPLE')
    
    is_summary.print_full_report()
    print("\n\n")
    oos_summary.print_full_report()
    print("\n\n")
    is_summary.print_comparison_table(oos_summary)
    
    return is_summary, oos_summary
