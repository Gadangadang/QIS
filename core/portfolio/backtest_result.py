"""
BacktestResult class - Container for backtest results with analysis methods.

Responsibilities:
- Store backtest results (equity curve, trades, metrics)
- Calculate performance metrics
- Generate reports and visualizations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """
    Container for backtest results with analysis methods.
    
    Example:
        result = BacktestResult(equity_curve=equity_df, trades=trades_df)
        result.print_summary()
        result.plot_equity_curve()
        print(f"Sharpe: {result.metrics['Sharpe Ratio']:.2f}")
    """
    equity_curve: pd.DataFrame  # Columns: Date (index), Cash, PositionsValue, TotalValue
    trades: pd.DataFrame  # Columns: ticker, entry_date, exit_date, pnl, return, etc.
    initial_capital: float = 100000.0
    benchmark_equity: Optional[pd.DataFrame] = None  # Optional benchmark for comparison
    benchmark_name: Optional[str] = None
    
    @property
    def final_equity(self) -> float:
        """Final portfolio value."""
        if len(self.equity_curve) == 0:
            return self.initial_capital
        return self.equity_curve['TotalValue'].iloc[-1]
    
    @property
    def total_return(self) -> float:
        """Total return as decimal (e.g., 0.15 = 15%)."""
        if len(self.equity_curve) == 0:
            return 0.0
        initial = self.equity_curve['TotalValue'].iloc[0]
        final = self.equity_curve['TotalValue'].iloc[-1]
        return (final / initial - 1) if initial > 0 else 0.0
    
    @property
    def returns(self) -> pd.Series:
        """Daily returns series."""
        if len(self.equity_curve) == 0:
            return pd.Series()
        return self.equity_curve['TotalValue'].pct_change().dropna()
    
    @property
    def metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dict with Sharpe, CAGR, Max Drawdown, Win Rate, etc.
            If benchmark is provided, also includes beta, alpha, tracking error, etc.
        """
        if len(self.equity_curve) == 0:
            return {
                'Total Return': 0.0,
                'CAGR': 0.0,
                'Sharpe Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Calmar Ratio': 0.0,
                'Win Rate': 0.0,
                'Avg Trade': 0.0,
                'Profit Factor': 0.0,
                'Total Trades': 0
            }
        
        returns = self.returns
        
        metrics = {
            'Total Return': self.total_return,
            'CAGR': self._calculate_cagr(),
            'Sharpe Ratio': self._calculate_sharpe(returns),
            'Sortino Ratio': self._calculate_sortino(returns),
            'Max Drawdown': self._calculate_max_drawdown(),
            'Calmar Ratio': self._calculate_calmar(),
            'Win Rate': self._calculate_win_rate(),
            'Avg Trade': self.trades['pnl'].mean() if len(self.trades) > 0 else 0.0,
            'Profit Factor': self._calculate_profit_factor(),
            'Total Trades': len(self.trades)
        }
        
        # Add benchmark metrics if available
        if self.benchmark_equity is not None and len(self.benchmark_equity) > 0:
            benchmark_metrics = self._calculate_benchmark_metrics()
            metrics.update(benchmark_metrics)
        
        return metrics
    
    def _calculate_sharpe(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        # Annualize assuming 252 trading days
        excess_returns = returns - (rf_rate / 252)
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (like Sharpe but only penalizes downside volatility)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (rf_rate / 252)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.equity_curve) == 0:
            return 0.0
        
        equity = self.equity_curve['TotalValue']
        rolling_max = equity.expanding().max()
        drawdowns = (equity - rolling_max) / rolling_max
        return drawdowns.min()
    
    def _calculate_cagr(self) -> float:
        """Calculate compound annual growth rate."""
        if len(self.equity_curve) == 0:
            return 0.0
        
        initial = self.equity_curve['TotalValue'].iloc[0]
        final = self.equity_curve['TotalValue'].iloc[-1]
        
        # Calculate years (assume 252 trading days per year)
        years = len(self.equity_curve) / 252
        
        if years <= 0 or initial <= 0:
            return 0.0
        
        return (final / initial) ** (1 / years) - 1
    
    def _calculate_calmar(self) -> float:
        """Calculate Calmar ratio (CAGR / Max Drawdown)."""
        cagr = self._calculate_cagr()
        max_dd = abs(self._calculate_max_drawdown())
        
        if max_dd == 0:
            return 0.0
        
        return cagr / max_dd
    
    def _calculate_win_rate(self) -> float:
        """Calculate percentage of winning trades."""
        if len(self.trades) == 0:
            return 0.0
        return (self.trades['pnl'] > 0).sum() / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profits / gross losses)."""
        if len(self.trades) == 0:
            return 0.0
        
        winning_trades = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
        losing_trades = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum())
        
        if losing_trades == 0:
            return 0.0 if winning_trades == 0 else float('inf')
        
        return winning_trades / losing_trades
    
    def _calculate_benchmark_metrics(self) -> Dict:
        """Calculate benchmark-relative metrics."""
        from core.benchmark import BenchmarkComparator
        
        try:
            comparator = BenchmarkComparator()
            bench_metrics = comparator.calculate_metrics(
                self.equity_curve,
                self.benchmark_equity,
                risk_free_rate=0.02
            )
            
            # Add benchmark name
            if self.benchmark_name:
                bench_metrics['Benchmark'] = self.benchmark_name
            
            return bench_metrics
        except Exception as e:
            print(f"⚠️  Could not calculate benchmark metrics: {e}")
            return {}
    
    def print_summary(self):
        """Print comprehensive performance summary."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS SUMMARY")
        print("="*70)
        
        metrics = self.metrics
        
        # Performance metrics
        print(f"\n{'PERFORMANCE METRICS':<35}")
        print(f"{'-'*35}")
        print(f"{'Total Return':<25} {metrics['Total Return']:>12.2%}")
        print(f"{'CAGR':<25} {metrics['CAGR']:>12.2%}")
        print(f"{'Sharpe Ratio':<25} {metrics['Sharpe Ratio']:>12.2f}")
        print(f"{'Sortino Ratio':<25} {metrics['Sortino Ratio']:>12.2f}")
        print(f"{'Max Drawdown':<25} {metrics['Max Drawdown']:>12.2%}")
        print(f"{'Calmar Ratio':<25} {metrics['Calmar Ratio']:>12.2f}")
        
        # Benchmark metrics if available
        if 'Beta (Full Period)' in metrics:
            print(f"\n{'BENCHMARK COMPARISON':<35}")
            print(f"{'-'*35}")
            if 'Benchmark' in metrics:
                print(f"{'Benchmark':<25} {metrics['Benchmark']:>12}")
            if 'Benchmark Return' in metrics:
                print(f"{'Benchmark Return':<25} {metrics['Benchmark Return']:>12.2%}")
            if 'Relative Return' in metrics:
                print(f"{'Relative Return':<25} {metrics['Relative Return']:>12.2%}")
            print(f"{'Beta (Full Period)':<25} {metrics['Beta (Full Period)']:>12.2f}")
            print(f"{'Beta (90-day avg)':<25} {metrics['Beta (90-day avg)']:>12.2f}")
            print(f"{'Beta (1-year avg)':<25} {metrics['Beta (1-year avg)']:>12.2f}")
            print(f"{'Alpha (Annual)':<25} {metrics['Alpha (Annual)']:>12.2%}")
            if 'Tracking Error' in metrics:
                print(f"{'Tracking Error':<25} {metrics['Tracking Error']:>12.2%}")
            print(f"{'Information Ratio':<25} {metrics['Information Ratio']:>12.2f}")
            print(f"{'Correlation':<25} {metrics['Correlation']:>12.2f}")
            if 'Up Capture Ratio' in metrics:
                print(f"{'Up Capture Ratio':<25} {metrics['Up Capture Ratio']:>12.2f}")
            if 'Down Capture Ratio' in metrics:
                print(f"{'Down Capture Ratio':<25} {metrics['Down Capture Ratio']:>12.2f}")
        
        # Trade statistics
        print(f"\n{'TRADE STATISTICS':<35}")
        print(f"{'-'*35}")
        print(f"{'Total Trades':<25} {metrics['Total Trades']:>12}")
        print(f"{'Win Rate':<25} {metrics['Win Rate']:>12.2%}")
        print(f"{'Avg Trade P&L':<25} ${metrics['Avg Trade']:>11,.2f}")
        print(f"{'Profit Factor':<25} {metrics['Profit Factor']:>12.2f}")
        
        # Portfolio values
        print(f"\n{'PORTFOLIO VALUES':<35}")
        print(f"{'-'*35}")
        print(f"{'Initial Capital':<25} ${self.initial_capital:>11,.2f}")
        print(f"{'Final Equity':<25} ${self.final_equity:>11,.2f}")
        print(f"{'Total P&L':<25} ${self.final_equity - self.initial_capital:>11,.2f}")
        
        print("="*70 + "\n")
    
    def plot_equity_curve(self):
        """Plot equity curve with matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        if len(self.equity_curve) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Equity curve
        ax1 = axes[0]
        self.equity_curve['TotalValue'].plot(ax=ax1, linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', 
                   alpha=0.5, label='Initial Capital')
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Plot 2: Drawdown
        ax2 = axes[1]
        equity = self.equity_curve['TotalValue']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        drawdown.plot(ax=ax2, linewidth=2, color='red', label='Drawdown')
        ax2.fill_between(drawdown.index, 0, drawdown, alpha=0.3, color='red')
        ax2.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @property
    def risk_analysis(self) -> Dict:
        """
        Detailed risk analysis.
        
        Returns:
            Dict with VaR, CVaR, volatility, downside deviation, etc.
        """
        if len(self.equity_curve) == 0:
            return {}
        
        returns = self.returns
        
        return {
            'max_drawdown': self._calculate_max_drawdown(),
            'volatility': returns.std() * np.sqrt(252),  # Annualized
            'downside_volatility': returns[returns < 0].std() * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'avg_positive_return': returns[returns > 0].mean(),
            'avg_negative_return': returns[returns < 0].mean()
        }
    
    def generate_html_report(self, save_path: str):
        """
        Generate HTML report using Reporter class.
        
        Args:
            save_path: Path to save HTML file
        """
        try:
            from core.reporter import Reporter
            
            reporter = Reporter()
            
            # Prepare equity DataFrame (needs 'Date' column and 'TotalValue')
            equity_df = self.equity_curve.reset_index()
            if 'Date' not in equity_df.columns and equity_df.columns[0] != 'Date':
                equity_df.rename(columns={equity_df.columns[0]: 'Date'}, inplace=True)
            
            reporter.generate_html_report(
                equity_df=equity_df,
                trades_df=self.trades,
                metrics=self.metrics,
                title="Portfolio Backtest Results",
                save_path=save_path
            )
            
            print(f"\n✅ HTML report saved to: {save_path}")
            
        except Exception as e:
            print(f"⚠️  Could not generate HTML report: {e}")
