"""
Reporter Module
Handles performance reporting, visualization, and analysis output.

Features:
- Equity curve plotting with drawdown
- Trade analysis and visualization
- Performance metrics calculation and formatting
- Interactive HTML report generation with Plotly charts
- Monthly returns heatmap
- Regime analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime


class Reporter:
    """
    Generates reports and visualizations for backtest results.
    
    Features:
    - Equity curve charts with drawdown
    - Trade distribution analysis
    - Performance metrics tables
    - Interactive HTML report generation
    - Monthly returns heatmap
    - Regime analysis
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize reporter.
        
        Args:
            output_dir: Directory for saving reports (default: 'reports/')
        """
        self.output_dir = Path(output_dir) if output_dir else Path('reports')
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_html_report(
        self,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: Dict,
        title: str = "Backtest Report",
        save_path: Optional[str] = None,
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate comprehensive HTML report with interactive charts.
        
        Args:
            equity_df: DataFrame with Date and TotalValue columns
            trades_df: DataFrame with trade details
            metrics: Performance metrics dictionary
            title: Report title
            save_path: Path to save HTML file (if None, returns HTML string)
            benchmark_df: Optional benchmark equity curve for comparison
            
        Returns:
            HTML string
        """
        try:
            from plotly import graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
        except ImportError:
            print("Warning: plotly not installed. Install with: pip install plotly")
            return self._generate_basic_html(equity_df, trades_df, metrics, title)
        
        # Prepare data
        equity_df = equity_df.copy()
        if 'Date' in equity_df.columns:
            equity_df['Date'] = pd.to_datetime(equity_df['Date'])
            equity_df.set_index('Date', inplace=True)
        
        if not isinstance(equity_df.index, pd.DatetimeIndex):
            equity_df.index = pd.to_datetime(equity_df.index)
        
        # Create figure
        fig = self._create_report_figure(equity_df, trades_df, benchmark_df)
        
        # Update layout
        fig.update_layout(
            height=1200 if benchmark_df is None else 1600,
            title_text=f"{title}: {equity_df.index[0].date()} to {equity_df.index[-1].date()}",
            showlegend=True,
            legend=dict(x=0.01, y=0.99)
        )
        
        # Build HTML components
        metrics_html = self._metrics_to_html(metrics)
        trades_summary_html = self._trades_summary_to_html(trades_df)
        worst_days_html = self._worst_days_to_html(equity_df, equity_df['TotalValue'].pct_change().fillna(0), n=10)
        
        # Generate full HTML
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px 15px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e8f4f8;
            transition: background-color 0.3s;
        }}
        .metric-value {{
            font-weight: 600;
            color: #2c3e50;
        }}
        .positive {{
            color: #27ae60;
            font-weight: 600;
        }}
        .negative {{
            color: #e74c3c;
            font-weight: 600;
        }}
        .neutral {{
            color: #95a5a6;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .card h3 {{
            margin-top: 0;
            color: #34495e;
            font-size: 16px;
        }}
        .summary-stat {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .summary-stat:last-child {{
            border-bottom: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Period: {equity_df.index[0].date()} to {equity_df.index[-1].date()} | 
            {len(equity_df)} trading days
        </div>

        <h2>📊 Performance Metrics</h2>
        {metrics_html}

        <h2>📈 Interactive Charts</h2>
        {pio.to_html(fig, include_plotlyjs='cdn', full_html=False)}

        <h2>💼 Trade Summary</h2>
        {trades_summary_html}

        <h2>📉 Worst Days (Top 10)</h2>
        {worst_days_html}
    </div>
</body>
</html>
        """
        
        # Save or return
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(full_html)
            print(f"✅ Report saved to: {save_path}")
        
        return full_html

    def _create_report_figure(
        self,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None
    ):
        """Create the Plotly figure for the report."""
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots
        
        equity = equity_df['TotalValue']
        returns = equity.pct_change().fillna(0)
        
        # Calculate drawdown
        cumulative = equity / equity.iloc[0]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        
        # Create subplots
        n_rows = 4 if benchmark_df is not None else 3
        subplot_titles = [
            'Equity Curve', 'Drawdown %',
            'Daily Returns Distribution', 'Trade PnL Distribution',
            'Cumulative Returns', 'Monthly Returns Heatmap'
        ]
        
        if benchmark_df is not None:
            subplot_titles.extend(['Rolling Beta (90-day)', 'Base 100 Comparison'])
        
        specs = [
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'histogram'}, {'type': 'histogram'}],
            [{'type': 'scatter'}, {'type': 'heatmap'}]
        ]
        
        if benchmark_df is not None:
            specs.append([{'type': 'scatter'}, {'type': 'scatter'}])
        
        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=subplot_titles[:n_rows*2],
            specs=specs,
            vertical_spacing=0.10,
            horizontal_spacing=0.1
        )
        
        # 1. Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                name='Portfolio',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add benchmark if provided
        if benchmark_df is not None:
            if 'Date' in benchmark_df.columns:
                benchmark_df.set_index('Date', inplace=True)
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df.index,
                    y=benchmark_df['TotalValue'].values,
                    name='Benchmark',
                    line=dict(color='gray', width=1, dash='dash'),
                    hovertemplate='%{x}<br>Value: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red'),
                hovertemplate='%{x}<br>DD: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns.values * 100,
                name='Daily Returns',
                nbinsx=50,
                marker=dict(color='lightblue'),
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Trade PnL distribution
        if not trades_df.empty and 'Value' in trades_df.columns:
            trade_pnl = trades_df.groupby('Date')['Value'].sum()
            fig.add_trace(
                go.Histogram(
                    x=trade_pnl.values,
                    name='Trade PnL',
                    nbinsx=30,
                    marker=dict(color='lightgreen'),
                    hovertemplate='PnL: $%{x:,.2f}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # 5. Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                name='Cumulative',
                line=dict(color='green', width=2),
                hovertemplate='%{x}<br>Return: %{y:.2f}x<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Monthly returns heatmap
        monthly_returns = self._calculate_monthly_returns(returns)
        if monthly_returns is not None and not monthly_returns.empty:
            fig.add_trace(
                go.Heatmap(
                    z=monthly_returns.values * 100,
                    x=monthly_returns.columns,
                    y=monthly_returns.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    text=monthly_returns.values * 100,
                    texttemplate='%{text:.1f}%',
                    textfont={"size": 10},
                    hovertemplate='%{y}-%{x}<br>Return: %{z:.2f}%<extra></extra>'
                ),
                row=3, col=2
            )
        
        # 7 & 8. Benchmark comparison charts (if benchmark provided)
        if benchmark_df is not None:
            from core.benchmark import BenchmarkComparator
            
            # Calculate benchmark metrics
            comparator = BenchmarkComparator()
            bench_metrics = comparator.calculate_metrics(equity_df, benchmark_df)
            
            # 7. Rolling beta (90-day)
            if 'rolling_beta_90d' in bench_metrics and not bench_metrics['rolling_beta_90d'].empty:
                rolling_beta = bench_metrics['rolling_beta_90d']
                fig.add_trace(
                    go.Scatter(
                        x=rolling_beta.index,
                        y=rolling_beta.values,
                        name='Beta (90d)',
                        line=dict(color='purple', width=2),
                        hovertemplate='%{x}<br>Beta: %{y:.2f}<extra></extra>'
                    ),
                    row=4, col=1
                )
                # Add horizontal line at beta=1
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=4, col=1)
            
            # 8. Base 100 comparison
            port_norm, bench_norm = comparator.format_for_base_100(equity_df, benchmark_df)
            fig.add_trace(
                go.Scatter(
                    x=port_norm.index,
                    y=port_norm['TotalValue'],
                    name='Portfolio (Base 100)',
                    line=dict(color='blue', width=2),
                    hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
                ),
                row=4, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=bench_norm.index,
                    y=bench_norm['TotalValue'],
                    name='Benchmark (Base 100)',
                    line=dict(color='gray', width=2, dash='dash'),
                    hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
                ),
                row=4, col=2
            )
            fig.add_hline(y=100, line_dash="dot", line_color="gray", row=4, col=2)
            
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Return (%)", row=2, col=1)
        fig.update_xaxes(title_text="PnL ($)", row=2, col=2)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_xaxes(title_text="Month", row=3, col=2)
        
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Return", row=3, col=1)
        fig.update_yaxes(title_text="Year", row=3, col=2)
        
        if benchmark_df is not None:
            fig.update_xaxes(title_text="Date", row=4, col=1)
            fig.update_xaxes(title_text="Date", row=4, col=2)
            fig.update_yaxes(title_text="Beta", row=4, col=1)
            fig.update_yaxes(title_text="Value (Base 100)", row=4, col=2)
            
        return fig
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> Optional[pd.DataFrame]:
        """Calculate monthly returns as a pivot table (years x months)."""
        try:
            monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_pivot = pd.DataFrame({
                'Year': monthly.index.year,
                'Month': monthly.index.month,
                'Return': monthly.values
            })
            pivot = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
            
            # Reindex to ensure all 12 months exist (fill missing with NaN)
            pivot = pivot.reindex(columns=range(1, 13))
            
            # Now assign month names safely
            pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            return pivot
        except Exception as e:
            # Log the error for debugging instead of silently returning None
            import warnings
            warnings.warn(f"Error calculating monthly returns: {e}")
            return None
    
    def _metrics_to_html(self, metrics: Dict) -> str:
        """Format metrics dictionary as HTML table."""
        rows = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'Return' in key or 'CAGR' in key or 'Volatility' in key or 'Drawdown' in key:
                    formatted = f"{value:.2%}"
                    css_class = 'positive' if value > 0 else 'negative' if value < 0 else 'neutral'
                elif 'Ratio' in key:
                    formatted = f"{value:.3f}"
                    css_class = 'positive' if value > 0 else 'negative' if value < 0 else 'neutral'
                else:
                    formatted = f"{value:.2f}"
                    css_class = 'metric-value'
            else:
                formatted = str(value)
                css_class = 'metric-value'
            
            rows.append(f'<tr><td>{key}</td><td class="{css_class}">{formatted}</td></tr>')
        
        return f"""
        <table>
            <thead>
                <tr><th>Metric</th><th>Value</th></tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """
    
    def _trades_summary_to_html(self, trades_df: pd.DataFrame) -> str:
        """Generate trades summary HTML."""
        if trades_df.empty:
            return "<p>No trades executed.</p>"
        
        # Calculate trade statistics
        n_trades = len(trades_df)
        
        trade_stats = []
        if 'Type' in trades_df.columns:
            type_counts = trades_df['Type'].value_counts()
            trade_stats.append(f"<div class='summary-stat'><span>By Type:</span><span>{dict(type_counts)}</span></div>")
        
        if 'Ticker' in trades_df.columns:
            ticker_counts = trades_df['Ticker'].value_counts()
            trade_stats.append(f"<div class='summary-stat'><span>By Asset:</span><span>{dict(ticker_counts)}</span></div>")
        
        if 'Value' in trades_df.columns:
            total_value = trades_df['Value'].abs().sum()
            trade_stats.append(f"<div class='summary-stat'><span>Total Traded:</span><span class='metric-value'>${total_value:,.2f}</span></div>")
        
        if 'TransactionCost' in trades_df.columns:
            total_tc = trades_df['TransactionCost'].sum()
            trade_stats.append(f"<div class='summary-stat'><span>Transaction Costs:</span><span class='negative'>${total_tc:,.2f}</span></div>")
        
        stats_html = ''.join(trade_stats)
        
        # Recent trades table
        recent_trades = trades_df.tail(10).copy()
        if 'Date' in recent_trades.columns:
            recent_trades['Date'] = pd.to_datetime(recent_trades['Date']).dt.strftime('%Y-%m-%d')
        
        # Format numeric columns
        for col in ['Value', 'Price', 'TransactionCost']:
            if col in recent_trades.columns:
                recent_trades[col] = recent_trades[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
        
        if 'Shares' in recent_trades.columns:
            recent_trades['Shares'] = recent_trades['Shares'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        
        trades_table = recent_trades.to_html(
            index=False,
            classes='',
            border=0,
            escape=False
        )
        
        return f"""
        <div class="grid-2">
            <div class="card">
                <h3>Trade Statistics</h3>
                <div class='summary-stat'><span>Total Trades:</span><span class='metric-value'>{n_trades}</span></div>
                {stats_html}
            </div>
            <div class="card">
                <h3>Recent Trades (Last 10)</h3>
                {trades_table}
            </div>
        </div>
        """
    
    def _worst_days_to_html(self, equity_df: pd.DataFrame, returns: pd.Series, n: int = 10) -> str:
        """Generate worst days HTML table."""
        worst = returns.nsmallest(n)
        
        if worst.empty:
            return "<p>No negative days found.</p>"
        
        worst_df = pd.DataFrame({
            'Date': worst.index.strftime('%Y-%m-%d'),
            'Return (%)': (worst.values * 100).round(2),
            'Portfolio Value': equity_df.loc[worst.index, 'TotalValue'].values
        })
        
        # Format values
        worst_df['Portfolio Value'] = worst_df['Portfolio Value'].apply(lambda x: f"${x:,.2f}")
        
        return worst_df.to_html(index=False, classes='', border=0)
    
    def _generate_basic_html(
        self,
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics: Dict,
        title: str
    ) -> str:
        """Generate basic HTML report without Plotly (fallback)."""
        metrics_html = self._metrics_to_html(metrics)
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <h2>Performance Metrics</h2>
    {metrics_html}
    <p style="color: #666;">Install plotly for interactive charts: pip install plotly</p>
</body>
</html>
        """
    
    def generate_multi_strategy_report(
        self,
        results: Dict,
        capital_allocation: Dict,
        total_capital: float,
        benchmark_equity: Optional[pd.DataFrame] = None,
        benchmark_name: str = "SPY",
        period_label: str = "Full Period",
        save_path: Optional[str] = None,
        auto_open: bool = True
    ) -> str:
        """
        Generate consolidated multi-strategy portfolio HTML report.
        
        Args:
            results: Dict of {strategy_name: BacktestResult}
            capital_allocation: Dict of {strategy_name: allocation_pct}
            total_capital: Total portfolio capital
            benchmark_equity: Optional DataFrame with benchmark equity curve
            benchmark_name: Name of benchmark (default: SPY)
            period_label: Label for the period (e.g., "In-Sample" or "Out-of-Sample")
            save_path: Path to save HTML file
            auto_open: Automatically open report in browser
            
        Returns:
            HTML string
        """
        from pathlib import Path
        
        # Calculate portfolio-level metrics
        # Get union of all dates from all strategies
        all_dates = pd.DatetimeIndex([])
        for result in results.values():
            all_dates = all_dates.union(result.equity_curve.index)
        all_dates = all_dates.sort_values()
        
        portfolio_equity = pd.DataFrame(index=all_dates)
        portfolio_equity['TotalValue'] = total_capital  # Start with total allocated capital
        
        # Calculate portfolio equity by tracking cumulative P&L from all strategies
        # We need to track the CHANGE in equity for each strategy, not absolute values
        cumulative_pnl = pd.Series(0.0, index=all_dates)
        
        for strategy_name, result in results.items():
            # Get strategy equity curve
            strategy_equity = result.equity_curve['TotalValue'].reindex(all_dates, method='ffill')
            strategy_equity = strategy_equity.fillna(result.initial_capital)
            
            # Calculate P&L: current value - initial capital
            strategy_pnl = strategy_equity - result.initial_capital
            cumulative_pnl += strategy_pnl
        
        # Portfolio equity = initial capital + cumulative P&L from all strategies
        portfolio_equity['TotalValue'] = total_capital + cumulative_pnl
        
        # Portfolio-level calculations
        portfolio_returns = portfolio_equity['TotalValue'].pct_change().dropna()
        portfolio_total_return = (portfolio_equity['TotalValue'].iloc[-1] / total_capital - 1)
        years = len(portfolio_equity) / 252
        portfolio_cagr = (portfolio_equity['TotalValue'].iloc[-1] / total_capital) ** (1/years) - 1 if years > 0 else 0
        portfolio_sharpe = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
        
        portfolio_rolling_max = portfolio_equity['TotalValue'].expanding().max()
        portfolio_drawdown = (portfolio_equity['TotalValue'] - portfolio_rolling_max) / portfolio_rolling_max
        portfolio_max_dd = portfolio_drawdown.min()
        
        all_trades = pd.concat([result.trades for result in results.values()], ignore_index=True)
        portfolio_win_rate = (all_trades['pnl'] > 0).sum() / len(all_trades) if len(all_trades) > 0 else 0
        
        # Calculate benchmark metrics if provided
        benchmark_metrics = {}
        if benchmark_equity is not None and len(benchmark_equity) > 0:
            bench_aligned = benchmark_equity.reindex(portfolio_equity.index, method='ffill').fillna(method='bfill')
            if 'TotalValue' in bench_aligned.columns:
                bench_returns = bench_aligned['TotalValue'].pct_change().dropna()
                benchmark_total_return = (bench_aligned['TotalValue'].iloc[-1] / bench_aligned['TotalValue'].iloc[0] - 1)
                benchmark_cagr = (bench_aligned['TotalValue'].iloc[-1] / bench_aligned['TotalValue'].iloc[0]) ** (1/years) - 1 if years > 0 else 0
                
                # Calculate beta (portfolio vs benchmark)
                common_idx = portfolio_returns.index.intersection(bench_returns.index)
                if len(common_idx) > 1:
                    port_ret_aligned = portfolio_returns.loc[common_idx]
                    bench_ret_aligned = bench_returns.loc[common_idx]
                    covariance = port_ret_aligned.cov(bench_ret_aligned)
                    benchmark_variance = bench_ret_aligned.var()
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                else:
                    beta = 0
                
                benchmark_metrics = {
                    'total_return': benchmark_total_return,
                    'cagr': benchmark_cagr,
                    'beta': beta,
                    'equity': bench_aligned['TotalValue']
                }
            else:
                benchmark_metrics = {}
        
        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Strategy Portfolio Report</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metric-card.positive {{
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }}
        .metric-card.negative {{
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }}
        .metric-label {{
            font-size: 12px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #34495e;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .strategy-section {{
            background-color: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .allocation-bar {{
            height: 30px;
            background-color: #3498db;
            border-radius: 4px;
            margin: 5px 0;
            display: flex;
            align-items: center;
            padding-left: 10px;
            color: white;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏦 Multi-Strategy Portfolio Report</h1>
        <p><strong>Period:</strong> {period_label} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Portfolio Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card positive">
                <div class="metric-label">Total Capital</div>
                <div class="metric-value">${total_capital:,.0f}</div>
            </div>
            <div class="metric-card {'positive' if portfolio_total_return > 0 else 'negative'}">
                <div class="metric-label">Total Return</div>
                <div class="metric-value">{portfolio_total_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">CAGR</div>
                <div class="metric-value">{portfolio_cagr:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{portfolio_sharpe:.2f}</div>
            </div>
            <div class="metric-card negative">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{portfolio_max_dd:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{portfolio_win_rate:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{len(all_trades)}</div>
            </div>
            <div class="metric-card positive">
                <div class="metric-label">Final Value</div>
                <div class="metric-value">${portfolio_equity['TotalValue'].iloc[-1]:,.0f}</div>
            </div>
"""
        
        # Add benchmark comparison if provided
        if benchmark_metrics:
            html_content += f"""
            <div class="metric-card">
                <div class="metric-label">{benchmark_name} Return</div>
                <div class="metric-value">{benchmark_metrics['total_return']:.2%}</div>
            </div>
            <div class="metric-card {'positive' if (portfolio_total_return - benchmark_metrics['total_return']) > 0 else 'negative'}">
                <div class="metric-label">Alpha vs {benchmark_name}</div>
                <div class="metric-value">{(portfolio_total_return - benchmark_metrics['total_return']):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Beta vs {benchmark_name}</div>
                <div class="metric-value">{benchmark_metrics['beta']:.2f}</div>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>💰 Capital Allocation</h2>
"""
        
        # Capital allocation bars
        for strategy_name, allocation in capital_allocation.items():
            allocated_capital = total_capital * allocation
            html_content += f"""
        <div class="allocation-bar" style="width: {allocation*100}%;">
            {strategy_name}: ${allocated_capital:,.0f} ({allocation:.1%})
        </div>
"""
        
        # Strategy comparison table
        html_content += """
        <h2>📈 Strategy Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Capital</th>
                    <th>Final Value</th>
                    <th>Return</th>
                    <th>CAGR</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for strategy_name, result in results.items():
            metrics = result.metrics
            strat_capital = total_capital * capital_allocation[strategy_name]
            html_content += f"""
                <tr>
                    <td><strong>{strategy_name}</strong></td>
                    <td>${strat_capital:,.0f}</td>
                    <td>${result.final_equity:,.0f}</td>
                    <td style="color: {'green' if result.total_return > 0 else 'red'};">{result.total_return:.2%}</td>
                    <td>{metrics['CAGR']:.2%}</td>
                    <td>{metrics['Sharpe Ratio']:.2f}</td>
                    <td style="color: red;">{metrics['Max Drawdown']:.2%}</td>
                    <td>{metrics['Win Rate']:.1%}</td>
                    <td>{metrics['Total Trades']}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <h2>🔍 Individual Strategy Details</h2>
"""
        
        # Individual strategy sections
        for strategy_name, result in results.items():
            metrics = result.metrics
            risk_analysis = result.risk_analysis
            
            html_content += f"""
        <div class="strategy-section">
            <h3>📌 {strategy_name}</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value">{metrics['Sortino Ratio']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Calmar Ratio</div>
                    <div class="metric-value">{metrics['Calmar Ratio']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">{metrics['Profit Factor']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Trade P&L</div>
                    <div class="metric-value">${metrics['Avg Trade']:,.0f}</div>
                </div>
            </div>
            
            <h4>Risk Metrics</h4>
            <table>
                <tr>
                    <td><strong>Annual Volatility:</strong></td>
                    <td>{risk_analysis.get('volatility', 0):.2%}</td>
                    <td><strong>Downside Volatility:</strong></td>
                    <td>{risk_analysis.get('downside_volatility', 0):.2%}</td>
                </tr>
                <tr>
                    <td><strong>VaR (95%):</strong></td>
                    <td style="color: red;">{risk_analysis.get('var_95', 0)*100:.2f}%</td>
                    <td><strong>CVaR (95%):</strong></td>
                    <td style="color: red;">{risk_analysis.get('cvar_95', 0)*100:.2f}%</td>
                </tr>
                <tr>
                    <td><strong>Best Day:</strong></td>
                    <td style="color: green;">{risk_analysis.get('best_day', 0)*100:.2f}%</td>
                    <td><strong>Worst Day:</strong></td>
                    <td style="color: red;">{risk_analysis.get('worst_day', 0)*100:.2f}%</td>
                </tr>
                <tr>
                    <td><strong>Positive Days:</strong></td>
                    <td>{risk_analysis.get('positive_days', 0)}</td>
                    <td><strong>Negative Days:</strong></td>
                    <td>{risk_analysis.get('negative_days', 0)}</td>
                </tr>
            </table>
            
            <h4>Recent Trades (Last 5)</h4>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Asset</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                        <th>Return</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Add last 5 trades
            if len(result.trades) > 0:
                last_trades = result.trades.tail(5)
                for _, trade in last_trades.iterrows():
                    pnl_color = 'green' if trade['pnl'] > 0 else 'red'
                    html_content += f"""
                    <tr>
                        <td>{trade['entry_date']}</td>
                        <td>{trade['ticker']}</td>
                        <td>${trade['entry_price']:.2f}</td>
                        <td>${trade['exit_price']:.2f}</td>
                        <td style="color: {pnl_color};">${trade['pnl']:,.2f}</td>
                        <td style="color: {pnl_color};">{trade['return']:.2%}</td>
                    </tr>
"""
            else:
                html_content += """
                    <tr><td colspan="6">No trades executed</td></tr>
"""
            
            html_content += """
                </tbody>
            </table>
        </div>
"""
        
        # Add Plotly charts
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Portfolio vs Benchmark equity chart
            html_content += """
        <h2>📈 Equity Curves</h2>
"""
            
            fig = go.Figure()
            
            # Add portfolio equity - convert index to list for proper serialization
            fig.add_trace(go.Scatter(
                x=portfolio_equity.index.tolist(),
                y=portfolio_equity['TotalValue'].tolist(),
                name='Portfolio',
                line=dict(color='#3498db', width=3)
            ))
            
            # Add benchmark if available
            if benchmark_metrics and 'equity' in benchmark_metrics:
                fig.add_trace(go.Scatter(
                    x=portfolio_equity.index.tolist(),
                    y=benchmark_metrics['equity'].tolist(),
                    name=benchmark_name,
                    line=dict(color='#95a5a6', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f'Portfolio vs {benchmark_name} Equity Curves' if benchmark_metrics else 'Portfolio Equity Curve',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            html_content += f"""
        <div id="portfolio-equity-chart"></div>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script>
            var portfolioData = {fig.to_json()};
            Plotly.newPlot('portfolio-equity-chart', portfolioData.data, portfolioData.layout);
        </script>
"""
            
            # Individual strategy equity curves
            html_content += """
        <h3>Individual Strategy Equity Curves</h3>
"""
            
            fig_strategies = go.Figure()
            
            for strategy_name, result in results.items():
                fig_strategies.add_trace(go.Scatter(
                    x=result.equity_curve.index.tolist(),
                    y=result.equity_curve['TotalValue'].tolist(),
                    name=strategy_name,
                    mode='lines'
                ))
            
            fig_strategies.update_layout(
                title='All Strategies Equity Curves',
                xaxis_title='Date',
                yaxis_title='Value ($)',
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            html_content += f"""
        <div id="strategies-equity-chart"></div>
        <script>
            var strategiesData = {fig_strategies.to_json()};
            Plotly.newPlot('strategies-equity-chart', strategiesData.data, strategiesData.layout);
        </script>
"""
        
        except ImportError:
            html_content += """
        <p style="color: red;">⚠️ Plotly not available. Install with: pip install plotly</p>
"""
        
        # Footer
        html_content += """
        <hr style="margin-top: 40px;">
        <p style="text-align: center; color: #7f8c8d;">
            Generated by QuantTrading Multi-Strategy Backtest System
        </p>
    </div>
</body>
</html>
"""
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Multi-strategy report saved: {save_path}")
            
            # Auto-open in browser
            if auto_open:
                import webbrowser
                abs_path = save_path.resolve()
                file_url = f"file://{abs_path}"
                webbrowser.open(file_url)
                print(f"🌐 Report opened in browser!")
        
        return html_content
    
    def generate_risk_dashboard(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        show_plots: bool = True
    ):
        """
        Generate comprehensive risk dashboard with visualizations.
        
        Args:
            results: Dict of {strategy_name: BacktestResult}
            save_path: Optional path to save plots
            show_plots: Whether to display plots (default True)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style("whitegrid")
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        print("📊 RETURNS & RISK ANALYSIS DASHBOARD\n")
        print("="*80)
        
        # 1. Four-panel dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot 1: Daily Returns Distribution
        ax1 = axes[0, 0]
        for idx, (strategy_name, result) in enumerate(results.items()):
            if len(result.returns) > 0:
                returns_pct = result.returns * 100
                ax1.hist(returns_pct, bins=50, alpha=0.6, label=strategy_name, color=colors[idx])
        
        ax1.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Daily Return (%)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.legend(loc='upper right')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.grid(alpha=0.3)
        
        # Plot 2: Cumulative Returns
        ax2 = axes[0, 1]
        for idx, (strategy_name, result) in enumerate(results.items()):
            if len(result.equity_curve) > 0:
                cumulative_returns = (result.equity_curve['TotalValue'] / result.initial_capital - 1) * 100
                cumulative_returns.plot(ax=ax2, label=strategy_name, linewidth=2, color=colors[idx])
        
        ax2.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax2.legend(loc='best')
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 3: Rolling Sharpe Ratio (252-day window)
        ax3 = axes[1, 0]
        for idx, (strategy_name, result) in enumerate(results.items()):
            if len(result.returns) > 0:
                rolling_sharpe = result.returns.rolling(252).mean() / result.returns.rolling(252).std() * np.sqrt(252)
                rolling_sharpe.plot(ax=ax3, label=strategy_name, linewidth=2, alpha=0.8, color=colors[idx])
        
        ax3.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=11)
        ax3.set_ylabel('Sharpe Ratio', fontsize=11)
        ax3.legend(loc='best')
        ax3.grid(alpha=0.3)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.3)
        
        # Plot 4: Drawdown Comparison
        ax4 = axes[1, 1]
        for idx, (strategy_name, result) in enumerate(results.items()):
            if len(result.equity_curve) > 0:
                equity = result.equity_curve['TotalValue']
                rolling_max = equity.expanding().max()
                drawdown = (equity - rolling_max) / rolling_max * 100
                drawdown.plot(ax=ax4, label=strategy_name, linewidth=2, color=colors[idx])
        
        ax4.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date', fontsize=11)
        ax4.set_ylabel('Drawdown (%)', fontsize=11)
        ax4.legend(loc='best')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.html', '_dashboard.png'), dpi=300, bbox_inches='tight')
            print(f"✅ Dashboard saved: {save_path.replace('.html', '_dashboard.png')}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # 2. Risk Metrics Summary Table
        print("\n\n📈 DETAILED RISK METRICS\n")
        print("="*100)
        
        risk_data = {}
        for strategy_name, result in results.items():
            risk_analysis = result.risk_analysis
            metrics = result.metrics
            
            risk_data[strategy_name] = {
                'Ann. Volatility': risk_analysis.get('volatility', 0),
                'Downside Vol': risk_analysis.get('downside_volatility', 0),
                'VaR (95%)': risk_analysis.get('var_95', 0) * 100,
                'CVaR (95%)': risk_analysis.get('cvar_95', 0) * 100,
                'Best Day': risk_analysis.get('best_day', 0) * 100,
                'Worst Day': risk_analysis.get('worst_day', 0) * 100,
                'Positive Days': risk_analysis.get('positive_days', 0),
                'Negative Days': risk_analysis.get('negative_days', 0),
                'Avg Positive': risk_analysis.get('avg_positive_return', 0) * 100,
                'Avg Negative': risk_analysis.get('avg_negative_return', 0) * 100,
                'Sortino Ratio': metrics.get('Sortino Ratio', 0),
                'Calmar Ratio': metrics.get('Calmar Ratio', 0)
            }
        
        risk_df = pd.DataFrame(risk_data).T
        print(risk_df.to_string())
        
        # 3. Monthly Returns Heatmaps
        print("\n\n📅 MONTHLY RETURNS HEATMAP\n")
        print("="*100)
        
        for strategy_name, result in results.items():
            if len(result.equity_curve) > 0:
                print(f"\n{strategy_name}:")
                
                # Calculate monthly returns
                equity = result.equity_curve['TotalValue'].copy()
                equity.index = pd.to_datetime(equity.index)
                monthly_returns = equity.resample('ME').last().pct_change() * 100
                
                # Create pivot table for heatmap
                monthly_returns_df = pd.DataFrame({
                    'Year': monthly_returns.index.year,
                    'Month': monthly_returns.index.month,
                    'Return': monthly_returns.values
                })
                
                if len(monthly_returns_df) > 0:
                    pivot = monthly_returns_df.pivot_table(
                        values='Return',
                        index='Year',
                        columns='Month',
                        aggfunc='first'
                    )
                    
                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(14, 6))
                    sns.heatmap(
                        pivot,
                        annot=True,
                        fmt='.1f',
                        cmap='RdYlGn',
                        center=0,
                        cbar_kws={'label': 'Return (%)'},
                        linewidths=0.5,
                        ax=ax
                    )
                    ax.set_title(f'{strategy_name} - Monthly Returns (%)', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Month', fontsize=11)
                    ax.set_ylabel('Year', fontsize=11)
                    
                    # Set month labels
                    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    ax.set_xticklabels(month_labels[:pivot.shape[1]])
                    
                    plt.tight_layout()
                    
                    if save_path:
                        heatmap_path = save_path.replace('.html', f'_{strategy_name}_heatmap.png')
                        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    
                    if show_plots:
                        plt.show()
                    else:
                        plt.close()
        
        print("\n" + "="*100)
        print("✅ Risk dashboard complete")
        print("="*100)


def quick_report(equity_df: pd.DataFrame, trades_df: pd.DataFrame, metrics: Dict):
    """
    Generate a quick console report.
    
    Args:
        equity_df: Equity curve data
        trades_df: Trade data
        metrics: Performance metrics
    """
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Return' in key or 'CAGR' in key or 'Volatility' in key or 'Drawdown' in key:
                print(f"{key}: {value:.2%}")
            elif 'Ratio' in key:
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    if not trades_df.empty:
        print("\n" + "="*60)
        print("TRADES SUMMARY")
        print("="*60)
        print(f"Total trades: {len(trades_df)}")
        
        if 'Type' in trades_df.columns:
            print(f"\nBy type:")
            print(trades_df['Type'].value_counts())
        
        if 'Ticker' in trades_df.columns:
            print(f"\nBy ticker:")
            print(trades_df['Ticker'].value_counts())
