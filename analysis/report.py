"""Unified backtest reporting with metrics, charts, and HTML output."""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from analysis.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    cagr,
    calmar_ratio,
    profit_factor,
    win_rate,
    average_win,
    average_loss,
)


class BacktestReport:
    """Unified diagnostic report for backtest results.

    Provides comprehensive analysis including:
    - Performance metrics (Sharpe, Sortino, Calmar, CAGR, MaxDD)
    - Trade statistics (win rate, profit factor, avg win/loss)
    - Regime analysis (correlation to market, performance in up/down days)
    - Worst days and worst trades identification
    - Interactive HTML report generation with charts

    Args:
        results: dict from run_walk_forward() with keys:
            - stitched_equity: pd.Series (portfolio value over time)
            - combined_returns: pd.Series (daily strategy returns)
            - folds: list of fold summary dicts
            - overall: dict with aggregated metrics
            - trades: pd.DataFrame (all trades from all folds)
            - df: pd.DataFrame (original market data for context)

    Example:
        >>> results = run_walk_forward(...)
        >>> report = BacktestReport(results)
        >>> report.summary()  # Print to console
        >>> report.save_html('logs/report.html')  # Save interactive report
        >>> worst = report.worst_days(10)  # Get 10 worst days
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.equity = results['stitched_equity']
        self.returns = results['combined_returns']
        self.trades = results.get('trades', pd.DataFrame())
        self.market_data = results.get('df', pd.DataFrame())
        self.folds = results.get('folds', [])

        # Calculate all metrics on init
        self.metrics = self._calculate_metrics()

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics.

        Returns:
            dict: All calculated metrics including performance, trade, and regime metrics.
        """
        returns = self.returns.fillna(0)

        # Core performance metrics
        total_return = self.equity.iloc[-1] / self.equity.iloc[0] - 1
        sharpe = sharpe_ratio(returns)
        sortino = sortino_ratio(returns)
        max_dd = max_drawdown(returns)
        calmar = calmar_ratio(returns)
        annualized_return = cagr(returns)

        # Trade-level metrics
        trade_metrics = self._calculate_trade_metrics()

        # Regime analysis
        regime_metrics = self._analyze_regimes()

        return {
            'total_return_pct': float(total_return),
            'cagr_pct': float(annualized_return),
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'calmar': float(calmar),
            'max_drawdown_pct': float(max_dd),
            **trade_metrics,
            **regime_metrics,
        }

    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """Calculate trade-level statistics.

        Returns:
            dict: Trade metrics including count, win rate, profit factor, etc.
        """
        if self.trades.empty:
            return {
                'n_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win_pct': 0.0,
                'avg_loss_pct': 0.0,
                'largest_win_pct': 0.0,
                'largest_loss_pct': 0.0,
            }

        wins = self.trades[self.trades['pnl_pct'] > 0]
        losses = self.trades[self.trades['pnl_pct'] <= 0]

        gross_profit = wins['pnl_pct'].sum() if not wins.empty else 0.0
        gross_loss = abs(losses['pnl_pct'].sum()) if not losses.empty else 0.0
        profit_factor_val = gross_profit / gross_loss if gross_loss != 0 else 0.0

        return {
            'n_trades': int(len(self.trades)),
            'win_rate': float(len(wins) / len(self.trades)),
            'profit_factor': float(profit_factor_val),
            'avg_win_pct': float(wins['pnl_pct'].mean() if not wins.empty else 0.0),
            'avg_loss_pct': float(losses['pnl_pct'].mean() if not losses.empty else 0.0),
            'largest_win_pct': float(wins['pnl_pct'].max() if not wins.empty else 0.0),
            'largest_loss_pct': float(losses['pnl_pct'].min() if not losses.empty else 0.0),
        }

    def _analyze_regimes(self) -> Dict[str, float]:
        """Analyze performance in different market regimes.

        Returns:
            dict: Regime metrics including correlation to market and performance in up/down days.
        """
        if self.market_data.empty or 'Close' not in self.market_data.columns:
            return {}

        # Calculate market returns
        market_returns = self.market_data['Close'].pct_change()

        # Align with strategy returns by date
        aligned = pd.DataFrame({
            'strategy': self.returns,
            'market': market_returns
        }).dropna()

        if len(aligned) < 10:
            return {}

        # Up market vs down market
        up_days = aligned[aligned['market'] > 0]
        down_days = aligned[aligned['market'] < 0]

        return {
            'correlation_to_market': float(aligned['strategy'].corr(aligned['market'])),
            'avg_return_up_days': float(up_days['strategy'].mean() if not up_days.empty else 0.0),
            'avg_return_down_days': float(down_days['strategy'].mean() if not down_days.empty else 0.0),
        }

    def summary(self) -> None:
        """Print human-readable summary to console."""
        m = self.metrics

        print("\n" + "=" * 60)
        print("                BACKTEST REPORT")
        print("=" * 60)
        print(f"Period:           {self.equity.index[0].date()} to {self.equity.index[-1].date()}")
        print(f"Total Return:     {m['total_return_pct']:+.2%}")
        print(f"CAGR:             {m['cagr_pct']:+.2%}")
        print(f"Sharpe Ratio:     {m['sharpe']:+.3f}")
        print(f"Sortino Ratio:    {m['sortino']:+.3f}")
        print(f"Calmar Ratio:     {m['calmar']:+.3f}")
        print(f"Max Drawdown:     {m['max_drawdown_pct']:.2%}")
        print("\n" + "-" * 60)
        print("                  TRADE STATISTICS")
        print("-" * 60)
        print(f"Number of Trades: {m['n_trades']}")
        if m['n_trades'] > 0:
            print(f"Win Rate:         {m['win_rate']:.1%}")
            print(f"Profit Factor:    {m['profit_factor']:.2f}")
            print(f"Avg Win:          {m['avg_win_pct']:+.2%}")
            print(f"Avg Loss:         {m['avg_loss_pct']:+.2%}")
            print(f"Largest Win:      {m['largest_win_pct']:+.2%}")
            print(f"Largest Loss:     {m['largest_loss_pct']:+.2%}")

        if 'correlation_to_market' in m:
            print("\n" + "-" * 60)
            print("                REGIME ANALYSIS")
            print("-" * 60)
            print(f"Market Correlation: {m['correlation_to_market']:+.3f}")
            print(f"Avg Return (Up Days):   {m['avg_return_up_days']:+.4%}")
            print(f"Avg Return (Down Days): {m['avg_return_down_days']:+.4%}")

        print("=" * 60 + "\n")

    def worst_days(self, n: int = 10) -> pd.DataFrame:
        """Return n worst strategy days with market context.

        Args:
            n: Number of worst days to return.

        Returns:
            DataFrame with columns: date, strategy_return, market_return, market_close
        """
        if self.market_data.empty:
            # No market context available
            worst = self.returns.nsmallest(n)
            return pd.DataFrame({
                'date': worst.index,
                'strategy_return': worst.values,
            })

        market_returns = self.market_data['Close'].pct_change()

        df = pd.DataFrame({
            'strategy_return': self.returns,
            'market_return': market_returns,
            'market_close': self.market_data['Close']
        }).dropna()

        worst = df.nsmallest(n, 'strategy_return')
        worst = worst.reset_index()
        worst.rename(columns={'index': 'date'}, inplace=True)

        return worst

    def worst_trades(self, n: int = 10) -> pd.DataFrame:
        """Return n worst trades across all folds.

        Args:
            n: Number of worst trades to return.

        Returns:
            DataFrame with trade details.
        """
        if self.trades.empty:
            return pd.DataFrame()

        worst = self.trades.nsmallest(n, 'pnl_pct')
        columns = ['entry_date', 'exit_date', 'side', 'entry_price', 'exit_price', 'pnl_pct']

        # Add exit_reason if available
        if 'exit_reason' in worst.columns:
            columns.append('exit_reason')

        # Add fold if available
        if 'fold' in worst.columns:
            columns.append('fold')

        return worst[columns]

    def save_html(self, path: str) -> None:
        """Save interactive HTML report with plotly charts.

        Creates a standalone HTML file with:
        - Summary metrics table
        - Interactive equity curve
        - Drawdown chart
        - Trade distribution histogram
        - Worst days/trades tables

        Args:
            path: Output file path for HTML report.
        """
        try:
            from plotly import graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
        except ImportError:
            print("Warning: plotly not installed. Install with: pip install plotly")
            print("Falling back to basic HTML report without charts.")
            self._save_basic_html(path)
            return

        # Create multi-panel figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curve', 'Drawdown',
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
            go.Scatter(x=self.equity.index, y=self.equity.values,
                      name='Equity', line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # 2. Drawdown
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      name='Drawdown', fill='tozeroy', line=dict(color='red')),
            row=1, col=2
        )

        # 3. Returns distribution
        fig.add_trace(
            go.Histogram(x=self.returns.values, name='Daily Returns',
                        nbinsx=50, marker=dict(color='lightblue')),
            row=2, col=1
        )

        # 4. Trade PnL distribution
        if not self.trades.empty:
            fig.add_trace(
                go.Histogram(x=self.trades['pnl_pct'].values, name='Trade PnL',
                            nbinsx=30, marker=dict(color='lightgreen')),
                row=2, col=2
            )

        # 5. Cumulative returns
        fig.add_trace(
            go.Scatter(x=cumulative.index, y=cumulative.values,
                      name='Cumulative', line=dict(color='green', width=2)),
            row=3, col=1
        )

        # 6. Monthly returns heatmap
        monthly_returns = self._calculate_monthly_returns()
        if monthly_returns is not None:
            fig.add_trace(
                go.Heatmap(
                    z=monthly_returns.values,
                    x=monthly_returns.columns,
                    y=monthly_returns.index,
                    colorscale='RdYlGn',
                    zmid=0,
                ),
                row=3, col=2
            )

        fig.update_layout(
            height=1200,
            title_text=f"Backtest Report: {self.equity.index[0].date()} to {self.equity.index[-1].date()}",
            showlegend=False
        )

        # Build HTML with metrics table + plotly chart
        metrics_html = self._metrics_to_html()
        worst_days_html = self._worst_days_to_html()
        worst_trades_html = self._worst_trades_to_html()
        folds_html = self._folds_to_html()
        
        # Generate diagnostic charts if diagnostics module available
        diagnostics_html = self._generate_diagnostics_html()

        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .metric-value {{ font-weight: bold; color: #333; }}
        .positive {{ color: #4CAF50; }}
        .negative {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Backtest Report</h1>
        <p style="color: #666; font-size: 14px;">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Performance Metrics</h2>
        {metrics_html}

        <h2>Walk-Forward Folds</h2>
        {folds_html}

        <h2>Interactive Charts</h2>
        {pio.to_html(fig, include_plotlyjs='cdn', full_html=False)}

        {diagnostics_html}

        <h2>Worst Days (Top 10)</h2>
        {worst_days_html}

        <h2>Worst Trades (Top 10)</h2>
        {worst_trades_html}
    </div>
</body>
</html>
        """

        Path(path).write_text(full_html)
        print(f"HTML report saved to: {path}")

    def _save_basic_html(self, path: str) -> None:
        """Save basic HTML report without plotly charts."""
        metrics_html = self._metrics_to_html()
        worst_days_html = self._worst_days_to_html()
        worst_trades_html = self._worst_trades_to_html()
        folds_html = self._folds_to_html()

        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
    <h2>Performance Metrics</h2>
    {metrics_html}

    <h2>Walk-Forward Folds</h2>
    {folds_html}

    <h2>Worst Days</h2>
    {worst_days_html}

    <h2>Worst Trades</h2>
    {worst_trades_html}
</body>
</html>
        """

        Path(path).write_text(full_html)
        print(f"Basic HTML report saved to: {path}")

    def _calculate_monthly_returns(self) -> Optional[pd.DataFrame]:
        """Calculate monthly returns for heatmap."""
        try:
            returns_series = self.returns.copy()
            returns_series.index = pd.to_datetime(returns_series.index)

            monthly = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            monthly_pivot = monthly.to_frame('return')
            monthly_pivot['year'] = monthly_pivot.index.year
            monthly_pivot['month'] = monthly_pivot.index.month

            pivot = monthly_pivot.pivot(index='year', columns='month', values='return')
            pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            return pivot
        except Exception:
            return None

    def _metrics_to_html(self) -> str:
        """Convert metrics dict to HTML table."""
        rows = []
        for k, v in self.metrics.items():
            # Format value
            if isinstance(v, float):
                if 'pct' in k or 'rate' in k:
                    v_str = f"{v:.2%}"
                else:
                    v_str = f"{v:.3f}"

                # Color code positive/negative
                css_class = 'positive' if v > 0 else 'negative' if v < 0 else ''
                v_str = f"<span class='metric-value {css_class}'>{v_str}</span>"
            else:
                v_str = f"<span class='metric-value'>{str(v)}</span>"

            label = k.replace('_', ' ').title()
            rows.append(f"<tr><td>{label}</td><td>{v_str}</td></tr>")

        return f"<table>{''.join(rows)}</table>"

    def _worst_days_to_html(self) -> str:
        """Convert worst days DataFrame to HTML table."""
        df = self.worst_days(10)
        if df.empty:
            return "<p>No data available</p>"
        return df.to_html(index=False, float_format=lambda x: f'{x:.4f}')

    def _worst_trades_to_html(self) -> str:
        """Convert worst trades DataFrame to HTML table."""
        df = self.worst_trades(10)
        if df.empty:
            return "<p>No trades available</p>"
        return df.to_html(index=False, float_format=lambda x: f'{x:.4f}')

    def _folds_to_html(self) -> str:
        """Convert folds summary to HTML table."""
        if not self.folds:
            return "<p>No fold data available</p>"

        df = pd.DataFrame(self.folds)
        return df.to_html(index=False, float_format=lambda x: f'{x:.4f}')

    def plot_equity(self):
        """Interactive equity curve with plotly (for notebook use).

        Returns:
            plotly Figure object or None if plotly not installed.
        """
        try:
            from plotly import graph_objects as go
        except ImportError:
            print("plotly not installed; install with: pip install plotly")
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.equity.index,
            y=self.equity.values,
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title='Strategy Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            height=500
        )

        return fig

    def _generate_diagnostics_html(self) -> str:
        """Generate diagnostic charts HTML section.
        
        Returns HTML with regime breakdown, trade anatomy, and signal quality visualizations.
        """
        try:
            from analysis.diagnostics import ModelDiagnostics
            from plotly import graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
        except ImportError:
            return ""
        
        try:
            diag = ModelDiagnostics(self.results)
            
            html_sections = []
            html_sections.append('<h2>Model Diagnostics</h2>')
            
            # Signal Quality Section
            signal_quality = diag.signal_quality_report()
            if 'error' not in signal_quality:
                signal_html = '<h3>Signal Quality Analysis</h3><table>'
                for key, value in signal_quality.items():
                    label = key.replace('_', ' ').title()
                    if isinstance(value, float):
                        val_str = f"{value:.4f}"
                    else:
                        val_str = str(value)
                    signal_html += f"<tr><td>{label}</td><td class='metric-value'>{val_str}</td></tr>"
                signal_html += '</table>'
                html_sections.append(signal_html)
            
            # Regime Breakdown Section
            regime_stats = diag.regime_breakdown()
            if not regime_stats.empty and 'regime' in regime_stats.columns:
                regime_df = regime_stats.set_index('regime')
                
                html_sections.append('<h3>Regime Performance</h3>')
                html_sections.append(regime_df.to_html(float_format=lambda x: f'{x:.4f}'))
                
                # Create regime charts
                fig_regime = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Sharpe by Regime', 'Total Return by Regime', 
                                  'Max Drawdown by Regime', 'Days in Each Regime')
                )
                
                # Sharpe
                fig_regime.add_trace(
                    go.Bar(x=regime_df.index, y=regime_df['sharpe'], name='Sharpe', 
                          marker_color='lightblue'),
                    row=1, col=1
                )
                
                # Total Return
                fig_regime.add_trace(
                    go.Bar(x=regime_df.index, y=regime_df['total_return'] * 100, name='Total Return (%)',
                          marker_color='lightgreen'),
                    row=1, col=2
                )
                
                # Max Drawdown
                fig_regime.add_trace(
                    go.Bar(x=regime_df.index, y=regime_df['max_drawdown'] * 100, name='Max DD (%)',
                          marker_color='lightcoral'),
                    row=2, col=1
                )
                
                # Days
                fig_regime.add_trace(
                    go.Bar(x=regime_df.index, y=regime_df['n_days'], name='Days',
                          marker_color='lightyellow'),
                    row=2, col=2
                )
                
                fig_regime.update_layout(height=600, showlegend=False, title_text="Regime Analysis")
                html_sections.append(pio.to_html(fig_regime, include_plotlyjs=False, full_html=False))
            
            # Trade Anatomy Section
            trade_anatomy = diag.trade_anatomy()
            if 'error' not in trade_anatomy:
                html_sections.append('<h3>Trade Anatomy</h3>')
                
                # Trade stats table
                trades_df = self.results.get('trades', pd.DataFrame())
                long_trades = len(trades_df[trades_df['side'] == 'long']) if 'side' in trades_df.columns else 0
                short_trades = len(trades_df[trades_df['side'] == 'short']) if 'side' in trades_df.columns else 0
                
                stats_html = '<table>'
                stats_html += f"<tr><td>Total Trades</td><td class='metric-value'>{trade_anatomy['total_trades']}</td></tr>"
                stats_html += f"<tr><td>Win Rate</td><td class='metric-value'>{trade_anatomy['win_rate']:.1%}</td></tr>"
                stats_html += f"<tr><td>Avg Hold (Winners)</td><td class='metric-value'>{trade_anatomy['avg_hold_days_wins']:.1f} days</td></tr>"
                stats_html += f"<tr><td>Avg Hold (Losers)</td><td class='metric-value'>{trade_anatomy['avg_hold_days_losses']:.1f} days</td></tr>"
                stats_html += f"<tr><td>Long Trades</td><td class='metric-value'>{long_trades}</td></tr>"
                stats_html += f"<tr><td>Long Win Rate</td><td class='metric-value'>{trade_anatomy['long_win_rate']:.1%}</td></tr>"
                stats_html += f"<tr><td>Short Trades</td><td class='metric-value'>{short_trades}</td></tr>"
                stats_html += f"<tr><td>Short Win Rate</td><td class='metric-value'>{trade_anatomy['short_win_rate']:.1%}</td></tr>"
                stats_html += '</table>'
                html_sections.append(stats_html)
                
                # Trade anatomy charts
                fig_trades = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Exit Reasons', 'Hold Period: Winners vs Losers',
                                  'Win Rate: Long vs Short', 'Trade Count: Long vs Short'),
                    specs=[[{'type': 'pie'}, {'type': 'bar'}],
                          [{'type': 'bar'}, {'type': 'bar'}]]
                )
                
                # Exit reasons pie
                if trade_anatomy['exit_reasons']:
                    labels = list(trade_anatomy['exit_reasons'].keys())
                    values = list(trade_anatomy['exit_reasons'].values())
                    fig_trades.add_trace(
                        go.Pie(labels=labels, values=values, name='Exit Reasons'),
                        row=1, col=1
                    )
                
                # Hold period comparison
                fig_trades.add_trace(
                    go.Bar(x=['Winners', 'Losers'], 
                          y=[trade_anatomy['avg_hold_days_wins'], trade_anatomy['avg_hold_days_losses']],
                          marker_color=['green', 'red']),
                    row=1, col=2
                )
                
                # Win rates
                fig_trades.add_trace(
                    go.Bar(x=['Long', 'Short'],
                          y=[trade_anatomy['long_win_rate'] * 100, trade_anatomy['short_win_rate'] * 100],
                          marker_color=['green', 'red']),
                    row=2, col=1
                )
                
                # Trade counts
                fig_trades.add_trace(
                    go.Bar(x=['Long', 'Short'],
                          y=[long_trades, short_trades],
                          marker_color=['green', 'red']),
                    row=2, col=2
                )
                
                fig_trades.update_layout(height=700, showlegend=False, title_text="Trade Analysis")
                fig_trades.update_yaxes(title_text="Days", row=1, col=2)
                fig_trades.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
                fig_trades.update_yaxes(title_text="Count", row=2, col=2)
                
                html_sections.append(pio.to_html(fig_trades, include_plotlyjs=False, full_html=False))
            
            # Execution Leakage Section
            leakage = diag.execution_leakage()
            html_sections.append('<h3>Execution Leakage Analysis</h3>')
            leak_html = '<table>'
            leak_html += f"<tr><td>Theoretical Sharpe (perfect execution)</td><td class='metric-value'>{leakage['theoretical_sharpe']:.4f}</td></tr>"
            leak_html += f"<tr><td>Actual Sharpe (with stops/costs)</td><td class='metric-value'>{leakage['actual_sharpe']:.4f}</td></tr>"
            leak_html += f"<tr><td>Execution Leakage</td><td class='metric-value'>{leakage['leakage_pct']:.1f}%</td></tr>"
            leak_html += '</table>'
            
            if leakage['leakage_pct'] > 20:
                leak_html += '<p style="color: orange;">HIGH LEAKAGE: Execution costs/stops significantly hurt performance</p>'
            elif leakage['leakage_pct'] < -10:
                leak_html += '<p style="color: green;">POSITIVE IMPACT: Risk controls improved risk-adjusted returns</p>'
            else:
                leak_html += '<p style="color: green;">LOW LEAKAGE: Execution close to theoretical performance</p>'
            
            html_sections.append(leak_html)
            
            return '\n'.join(html_sections)
            
        except Exception as e:
            return f'<p>⚠️ Could not generate diagnostics: {str(e)}</p>'
