"""
Unified plotting utilities for backtesting analysis.

Provides standardized, reusable plotting functions to eliminate code duplication
across notebooks. Supports both matplotlib (for notebooks) and plotly (for HTML reports).

Usage:
    from utils.plotter import PortfolioPlotter
    
    plotter = PortfolioPlotter(strategy_results, benchmark_data=spy_data)
    plotter.plot_equity_curves()
    plotter.plot_drawdown()
    plotter.plot_monthly_returns_heatmap()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 10


class PortfolioPlotter:
    """
    Unified plotting class for portfolio backtest analysis.
    
    Handles single-strategy and multi-strategy portfolios with matplotlib.
    Eliminates hundreds of lines of duplicate plotting code from notebooks.
    """
    
    def __init__(
        self,
        strategy_results: Dict,
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_name: str = 'SPY'
    ):
        """
        Initialize plotter with backtest results.
        
        Args:
            strategy_results: Dict of {strategy_name: {'result': BacktestResult, 'capital': float}}
            benchmark_data: Optional benchmark price data (OHLC)
            benchmark_name: Name of benchmark (default 'SPY')
        """
        self.strategy_results = strategy_results
        self.benchmark_data = benchmark_data
        self.benchmark_name = benchmark_name
        
        # Calculate combined portfolio
        self._prepare_combined_data()
    
    def _prepare_combined_data(self):
        """Prepare combined portfolio equity curve and metrics."""
        if not self.strategy_results:
            raise ValueError("No strategy results provided")
        
        # Get reference dates from first strategy
        first_result = list(self.strategy_results.values())[0]['result']
        self.dates = first_result.equity_curve.index
        
        # Calculate combined equity
        self.combined_equity = pd.Series(
            sum(data['result'].equity_curve['TotalValue'].values 
                for data in self.strategy_results.values()),
            index=self.dates,
            name='TotalValue'
        )
        
        # Calculate total capital
        self.total_capital = sum(data['capital'] for data in self.strategy_results.values())
        
        # Calculate combined returns
        self.combined_returns = self.combined_equity.pct_change().fillna(0)
    
    def plot_equity_curves(
        self,
        show_individual: bool = True,
        show_combined: bool = True,
        show_benchmark: bool = True,
        log_scale: bool = False,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Plot equity curves for strategies, combined portfolio, and benchmark.
        
        Args:
            show_individual: Show individual strategy curves
            show_combined: Show combined portfolio curve
            show_benchmark: Show benchmark curve
            log_scale: Use log scale for y-axis
            figsize: Figure size (width, height)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual strategies
        if show_individual:
            for strategy_name, data in self.strategy_results.items():
                result = data['result']
                equity = result.equity_curve['TotalValue']
                ax.plot(equity.index, equity.values, 
                       label=strategy_name, linewidth=2, alpha=0.7)
        
        # Plot combined portfolio
        if show_combined:
            ax.plot(self.combined_equity.index, self.combined_equity.values,
                   label='Combined Portfolio', linewidth=3, color='darkblue',
                   linestyle='--')
        
        # Plot benchmark
        if show_benchmark and self.benchmark_data is not None:
            bench_equity = self._calculate_benchmark_equity()
            ax.plot(bench_equity.index, bench_equity.values,
                   label=self.benchmark_name, linewidth=2, color='gray',
                   linestyle=':')
        
        # Formatting
        if log_scale:
            ax.set_yscale('log')
            title_suffix = ' (Log Scale)'
        else:
            title_suffix = ''
        
        ax.set_title(f'Portfolio Equity Curves{title_suffix}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_drawdown(
        self,
        show_underwater: bool = True,
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        Plot drawdown chart with underwater equity curve.
        
        Args:
            show_underwater: Show underwater equity curve vs peak
            figsize: Figure size (width, height)
        """
        # Calculate drawdown
        cumulative = self.combined_equity
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig, axes = plt.subplots(2, 1, figsize=figsize) if show_underwater else plt.subplots(1, 1, figsize=figsize)
        axes = [axes] if not isinstance(axes, np.ndarray) else axes
        
        # Plot 1: Drawdown percentage
        axes[0].fill_between(drawdown.index, 0, drawdown.values * 100,
                            color='red', alpha=0.3, label='Drawdown')
        axes[0].plot(drawdown.index, drawdown.values * 100,
                    color='darkred', linewidth=2)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Drawdown (%)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Find max drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        axes[0].annotate(f'Max DD: {max_dd:.2%}',
                        xy=(max_dd_date, max_dd * 100),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Plot 2: Underwater chart
        if show_underwater:
            axes[1].fill_between(cumulative.index, running_max.values, cumulative.values,
                                where=cumulative.values < running_max.values,
                                color='red', alpha=0.3, label='Underwater')
            axes[1].plot(cumulative.index, cumulative.values,
                        color='darkblue', linewidth=2, label='Equity')
            axes[1].plot(running_max.index, running_max.values,
                        color='green', linewidth=1, linestyle='--', label='Peak Equity')
            axes[1].set_title('Underwater Equity Chart', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Date', fontsize=12)
            axes[1].set_ylabel('Equity ($)', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            from matplotlib.ticker import FuncFormatter
            axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_monthly_returns_heatmap(
        self,
        strategy_name: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Plot monthly returns heatmap.
        
        Args:
            strategy_name: Specific strategy to plot (None = combined portfolio)
            figsize: Figure size (width, height)
        """
        # Get returns
        if strategy_name and strategy_name in self.strategy_results:
            returns = self.strategy_results[strategy_name]['result'].equity_curve['TotalValue'].pct_change()
            title = f'Monthly Returns Heatmap - {strategy_name}'
        else:
            returns = self.combined_returns
            title = 'Monthly Returns Heatmap - Combined Portfolio'
        
        # Create monthly returns table
        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.to_frame()
        monthly_returns['Year'] = monthly_returns.index.year
        monthly_returns['Month'] = monthly_returns.index.month
        
        # Pivot to heatmap format
        heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values=monthly_returns.columns[0])
        heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(heatmap_data * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'Return (%)'}, ax=ax,
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_rolling_metrics(
        self,
        window: int = 60,
        metrics: List[str] = ['sharpe', 'volatility'],
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Plot rolling performance metrics.
        
        Args:
            window: Rolling window size in days
            metrics: List of metrics to plot ('sharpe', 'volatility', 'returns')
            figsize: Figure size (width, height)
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
        axes = [axes] if n_metrics == 1 else axes
        
        returns = self.combined_returns
        
        for i, metric in enumerate(metrics):
            if metric == 'sharpe':
                rolling_ret = returns.rolling(window).mean() * 252
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                rolling_sharpe = rolling_ret / rolling_vol
                
                axes[i].plot(rolling_sharpe.index, rolling_sharpe.values,
                           color='blue', linewidth=2)
                axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
                axes[i].fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                                    where=rolling_sharpe.values > 0,
                                    color='green', alpha=0.3)
                axes[i].fill_between(rolling_sharpe.index, 0, rolling_sharpe.values,
                                    where=rolling_sharpe.values < 0,
                                    color='red', alpha=0.3)
                axes[i].set_title(f'Rolling {window}-Day Sharpe Ratio',
                                fontsize=14, fontweight='bold')
                axes[i].set_ylabel('Sharpe Ratio', fontsize=12)
                
            elif metric == 'volatility':
                rolling_vol = returns.rolling(window).std() * np.sqrt(252)
                
                axes[i].plot(rolling_vol.index, rolling_vol.values * 100,
                           color='orange', linewidth=2)
                axes[i].fill_between(rolling_vol.index, 0, rolling_vol.values * 100,
                                    color='orange', alpha=0.3)
                axes[i].set_title(f'Rolling {window}-Day Volatility',
                                fontsize=14, fontweight='bold')
                axes[i].set_ylabel('Volatility (% ann.)', fontsize=12)
                
            elif metric == 'returns':
                rolling_ret = returns.rolling(window).mean() * 252
                
                axes[i].plot(rolling_ret.index, rolling_ret.values * 100,
                           color='green', linewidth=2)
                axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1)
                axes[i].fill_between(rolling_ret.index, 0, rolling_ret.values * 100,
                                    where=rolling_ret.values > 0,
                                    color='green', alpha=0.3)
                axes[i].fill_between(rolling_ret.index, 0, rolling_ret.values * 100,
                                    where=rolling_ret.values < 0,
                                    color='red', alpha=0.3)
                axes[i].set_title(f'Rolling {window}-Day Returns (Annualized)',
                                fontsize=14, fontweight='bold')
                axes[i].set_ylabel('Return (% ann.)', fontsize=12)
            
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_returns_dashboard(
        self,
        in_sample: bool = True,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Plot comprehensive returns dashboard (4 subplots).
        
        Args:
            in_sample: Whether this is in-sample or out-of-sample data
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        title_prefix = "In-Sample" if in_sample else "Out-of-Sample"
        
        # 1. Cumulative returns (log scale)
        for strategy_name, data in self.strategy_results.items():
            equity = data['result'].equity_curve['TotalValue']
            axes[0].plot(equity.index, equity.values, label=strategy_name, linewidth=2)
        
        axes[0].set_yscale('log')
        axes[0].set_title(f'{title_prefix}: Cumulative Returns (Log Scale)',
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Equity ($)')
        axes[0].legend(loc='best', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Rolling 1-year returns
        rolling_returns = self.combined_returns.rolling(252).apply(lambda x: (1 + x).prod() - 1)
        axes[1].plot(rolling_returns.index, rolling_returns.values * 100,
                    color='blue', linewidth=2)
        axes[1].fill_between(rolling_returns.index, 0, rolling_returns.values * 100,
                            where=rolling_returns.values > 0, color='green', alpha=0.3)
        axes[1].fill_between(rolling_returns.index, 0, rolling_returns.values * 100,
                            where=rolling_returns.values < 0, color='red', alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1].set_title(f'{title_prefix}: Rolling 1-Year Returns',
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Return (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Monthly returns heatmap (simplified)
        monthly_returns = self.combined_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_data = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        heatmap = monthly_data.pivot(index='Year', columns='Month', values='Return')
        
        sns.heatmap(heatmap * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, cbar=False, ax=axes[2], linewidths=0.5)
        axes[2].set_title(f'{title_prefix}: Monthly Returns Heatmap',
                         fontsize=12, fontweight='bold')
        
        # 4. Strategy contribution bar chart
        contributions = []
        for name, data in self.strategy_results.items():
            final_val = data['result'].equity_curve['TotalValue'].iloc[-1]
            profit = final_val - data['capital']
            contributions.append({'Strategy': name, 'Profit': profit})
        
        contrib_df = pd.DataFrame(contributions)
        colors = ['green' if x > 0 else 'red' for x in contrib_df['Profit']]
        axes[3].bar(range(len(contrib_df)), contrib_df['Profit'], color=colors)
        axes[3].set_xticks(range(len(contrib_df)))
        axes[3].set_xticklabels(contrib_df['Strategy'], rotation=45, ha='right')
        axes[3].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[3].set_title(f'{title_prefix}: Strategy P&L Contribution',
                         fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Profit ($)')
        axes[3].grid(True, alpha=0.3, axis='y')
        
        from matplotlib.ticker import FuncFormatter
        axes[3].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.suptitle(f'{title_prefix} Returns Dashboard', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_risk_dashboard(
        self,
        in_sample: bool = True,
        figsize: Tuple[int, int] = (16, 10)
    ):
        """
        Plot comprehensive risk dashboard (4 subplots).
        
        Args:
            in_sample: Whether this is in-sample or out-of-sample data
            figsize: Figure size (width, height)
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        title_prefix = "In-Sample" if in_sample else "Out-of-Sample"
        
        # 1. Drawdown chart
        cumulative = self.combined_equity
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        axes[0].fill_between(drawdown.index, 0, drawdown.values * 100,
                            color='red', alpha=0.3)
        axes[0].plot(drawdown.index, drawdown.values * 100,
                    color='darkred', linewidth=2)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[0].set_title(f'{title_prefix}: Drawdown',
                         fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Drawdown (%)')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Rolling 30-day volatility
        rolling_vol = self.combined_returns.rolling(30).std() * np.sqrt(252)
        axes[1].plot(rolling_vol.index, rolling_vol.values * 100,
                    color='orange', linewidth=2)
        axes[1].fill_between(rolling_vol.index, 0, rolling_vol.values * 100,
                            color='orange', alpha=0.3)
        axes[1].set_title(f'{title_prefix}: Rolling 30-Day Volatility',
                         fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Volatility (% ann.)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Risk metrics comparison table (as text)
        axes[2].axis('off')
        metrics_text = self._format_risk_metrics_table()
        axes[2].text(0.1, 0.9, metrics_text, transform=axes[2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].set_title(f'{title_prefix}: Risk Metrics',
                         fontsize=12, fontweight='bold')
        
        # 4. Return distribution histogram
        axes[3].hist(self.combined_returns * 100, bins=50, color='blue',
                    alpha=0.7, edgecolor='black')
        axes[3].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        axes[3].axvline(x=self.combined_returns.mean() * 100, color='green',
                       linestyle='--', linewidth=2, label=f'Mean: {self.combined_returns.mean()*100:.2f}%')
        axes[3].set_title(f'{title_prefix}: Daily Returns Distribution',
                         fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Daily Return (%)')
        axes[3].set_ylabel('Frequency')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'{title_prefix} Risk Dashboard', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_all_dashboards(self, in_sample: bool = True):
        """Plot both returns and risk dashboards."""
        self.plot_returns_dashboard(in_sample=in_sample)
        self.plot_risk_dashboard(in_sample=in_sample)
    
    def _calculate_benchmark_equity(self) -> pd.Series:
        """Calculate benchmark buy-and-hold equity curve."""
        if self.benchmark_data is None:
            return None
        
        # Align benchmark with portfolio dates
        bench_prices = self.benchmark_data['Close'].reindex(self.dates, method='ffill')
        
        # Calculate buy-and-hold equity
        initial_value = self.total_capital
        shares = initial_value / bench_prices.iloc[0]
        bench_equity = shares * bench_prices
        
        return bench_equity
    
    def _format_risk_metrics_table(self) -> str:
        """Format risk metrics as text table."""
        # Calculate key risk metrics
        returns = self.combined_returns
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        sortino = (returns.mean() / returns[returns < 0].std()) * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        
        cumulative = self.combined_equity
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        metrics_text = f"""
Sharpe Ratio:      {sharpe:.2f}
Sortino Ratio:     {sortino:.2f}
Max Drawdown:      {max_dd:.2%}
VaR (95%):         {var_95:.2%}
CVaR (95%):        {cvar_95:.2%}
Daily Volatility:  {returns.std():.2%}
Annual Volatility: {returns.std() * np.sqrt(252):.2%}
        """
        
        return metrics_text.strip()


# Convenience functions for quick plotting
def quick_equity_plot(strategy_results: Dict, **kwargs):
    """Quick equity curve plot."""
    plotter = PortfolioPlotter(strategy_results)
    return plotter.plot_equity_curves(**kwargs)


def quick_drawdown_plot(strategy_results: Dict, **kwargs):
    """Quick drawdown plot."""
    plotter = PortfolioPlotter(strategy_results)
    return plotter.plot_drawdown(**kwargs)


def quick_heatmap(strategy_results: Dict, **kwargs):
    """Quick monthly returns heatmap."""
    plotter = PortfolioPlotter(strategy_results)
    return plotter.plot_monthly_returns_heatmap(**kwargs)
