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
            # Ensure Date column is datetime before setting as index
            equity_df['Date'] = pd.to_datetime(equity_df['Date'])
            equity_df.set_index('Date', inplace=True)
        
        # Ensure index is datetime
        if not isinstance(equity_df.index, pd.DatetimeIndex):
            equity_df.index = pd.to_datetime(equity_df.index)
        
        equity = equity_df['TotalValue']
        returns = equity.pct_change().fillna(0)
        
        # Calculate drawdown
        cumulative = equity / equity.iloc[0]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve', 'Drawdown %',
                'Daily Returns Distribution', 'Trade PnL Distribution',
                'Cumulative Returns', 'Monthly Returns Heatmap'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'histogram'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.12,
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
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"{title}: {equity.index[0].date()} to {equity.index[-1].date()}",
            showlegend=True,
            legend=dict(x=0.01, y=0.99)
        )
        
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
        
        # Build HTML components
        metrics_html = self._metrics_to_html(metrics)
        trades_summary_html = self._trades_summary_to_html(trades_df)
        worst_days_html = self._worst_days_to_html(equity_df, returns, n=10)
        
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
            Period: {equity.index[0].date()} to {equity.index[-1].date()} | 
            {len(equity)} trading days
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
    
    def _calculate_monthly_returns(self, returns: pd.Series) -> Optional[pd.DataFrame]:
        """Calculate monthly returns as a pivot table (years x months)."""
        try:
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_pivot = pd.DataFrame({
                'Year': monthly.index.year,
                'Month': monthly.index.month,
                'Return': monthly.values
            })
            pivot = monthly_pivot.pivot(index='Year', columns='Month', values='Return')
            pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            return pivot
        except:
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
