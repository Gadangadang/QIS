"""
Multi-Strategy Reporter Module
Generates comprehensive HTML reports for multi-strategy portfolios with benchmark comparison.

Features:
- All-in-one equity curves (strategies + composite + benchmark)
- Performance comparison tables
- Individual strategy deep dives
- Benchmark analysis with rolling beta
- Aggregated trade analysis
- Interactive Plotly visualizations
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class MultiStrategyReporter:
    """
    Generates unified HTML reports for multi-strategy portfolios.
    
    Handles:
    - Multiple strategy equity curves
    - Combined portfolio performance
    - Benchmark comparison
    - Individual strategy metrics
    - Interactive drill-down sections
    """
    
    def __init__(self):
        """Initialize multi-strategy reporter."""
        pass
    
    def generate_report(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_name: str = "Benchmark",
        strategy_metrics: Optional[Dict] = None,
        combined_metrics: Optional[Dict] = None,
        title: str = "Multi-Strategy Portfolio Report"
    ) -> str:
        """
        Generate comprehensive multi-strategy HTML report.
        
        Args:
            strategy_results: Dict of {strategy_name: {'result': BacktestResult, 'capital': float, 'assets': list}}
            combined_equity: DataFrame with combined portfolio equity curve
            benchmark_data: Optional benchmark equity DataFrame (scaled to initial capital)
            benchmark_name: Name of benchmark (e.g., 'SPY', 'VT')
            strategy_metrics: Dict of {strategy_name: benchmark_metrics_dict}
            combined_metrics: Benchmark metrics for combined portfolio
            title: Report title
            
        Returns:
            HTML string
        """
        # Build HTML sections
        html_sections = []
        
        # Header
        html_sections.append(self._generate_header(title))
        
        # Executive Summary
        html_sections.append(self._generate_executive_summary(
            strategy_results, combined_equity, benchmark_data, 
            benchmark_name, combined_metrics
        ))
        
        # Main Equity Curve Chart
        html_sections.append(self._generate_main_chart(
            strategy_results, combined_equity, benchmark_data, benchmark_name
        ))
        
        # Performance Comparison Table
        html_sections.append(self._generate_comparison_table(
            strategy_results, combined_equity, benchmark_data,
            benchmark_name, strategy_metrics, combined_metrics
        ))
        
        # Benchmark Analysis Section (if benchmark provided)
        if benchmark_data is not None and combined_metrics is not None:
            html_sections.append(self._generate_benchmark_analysis(
                combined_equity, benchmark_data, combined_metrics, 
                strategy_metrics, benchmark_name
            ))
        
        # Individual Strategy Deep Dives
        html_sections.append(self._generate_strategy_sections(
            strategy_results, strategy_metrics
        ))
        
        # Trade Analysis
        html_sections.append(self._generate_trade_analysis(strategy_results))
        
        # Footer
        html_sections.append(self._generate_footer())
        
        return '\n'.join(html_sections)
    
    def _generate_header(self, title: str) -> str:
        """Generate HTML header with styling."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .timestamp {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 0.9em;
        }}
        .section {{
            padding: 40px;
            border-bottom: 1px solid #eee;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section-title {{
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 20px;
            color: #667eea;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-card .label {{
            font-size: 0.85em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: 600;
            margin-top: 5px;
            color: #333;
        }}
        .metric-card.positive {{
            border-left-color: #28a745;
        }}
        .metric-card.negative {{
            border-left-color: #dc3545;
        }}
        .chart-container {{
            margin: 30px 0;
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .strategy-section {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        .strategy-header {{
            font-size: 1.3em;
            font-weight: 600;
            color: #667eea;
            margin-bottom: 15px;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>📊 {title}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
"""
    
    def _generate_executive_summary(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        benchmark_name: str,
        combined_metrics: Optional[Dict]
    ) -> str:
        """Generate executive summary with key metrics."""
        total_initial = sum(data['capital'] for data in strategy_results.values())
        total_final = combined_equity['TotalValue'].iloc[-1]
        total_return = (total_final - total_initial) / total_initial
        
        # Calculate number of trades
        total_trades = sum(
            len(data['result'].trades) 
            for data in strategy_results.values()
        )
        
        metrics_html = f"""
    <div class="section">
        <h2 class="section-title">Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Initial Capital</div>
                <div class="value">${total_initial:,.0f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Final Equity</div>
                <div class="value">${total_final:,.0f}</div>
            </div>
            <div class="metric-card {'positive' if total_return > 0 else 'negative'}">
                <div class="label">Total Return</div>
                <div class="value">{total_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Number of Strategies</div>
                <div class="value">{len(strategy_results)}</div>
            </div>
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value">{total_trades}</div>
            </div>
"""
        
        if benchmark_data is not None and combined_metrics:
            bench_return = combined_metrics.get('Benchmark Return', 0)
            beta = combined_metrics.get('Beta (Full Period)', 0)
            alpha = combined_metrics.get('Alpha (Annual)', 0)
            
            metrics_html += f"""
            <div class="metric-card">
                <div class="label">{benchmark_name} Return</div>
                <div class="value">{bench_return:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Portfolio Beta</div>
                <div class="value">{beta:.3f}</div>
            </div>
            <div class="metric-card {'positive' if alpha > 0 else 'negative'}">
                <div class="label">Alpha (Annual)</div>
                <div class="value">{alpha:.2%}</div>
            </div>
"""
        
        metrics_html += """
        </div>
    </div>
"""
        return metrics_html
    
    def _generate_main_chart(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        benchmark_name: str
    ) -> str:
        """Generate main equity curve chart with all strategies + composite + benchmark."""
        fig = go.Figure()
        
        # Individual strategies (dashed lines)
        for strategy_name, data in strategy_results.items():
            equity = data['result'].equity_curve
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity['TotalValue'],
                name=strategy_name,
                mode='lines',
                line=dict(width=2, dash='dash'),
                opacity=0.7
            ))
        
        # Combined portfolio (solid line, thicker)
        fig.add_trace(go.Scatter(
            x=combined_equity.index,
            y=combined_equity['TotalValue'],
            name='Combined Portfolio',
            mode='lines',
            line=dict(color='rgb(31, 119, 180)', width=3),
            yaxis='y'
        ))
        
        # Benchmark (if provided)
        if benchmark_data is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data['TotalValue'],
                name=benchmark_name,
                mode='lines',
                line=dict(color='rgb(255, 65, 54)', width=3),
                opacity=0.8
            ))
        
        fig.update_layout(
            title='Portfolio Equity Curves: Strategies + Composite + Benchmark',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            hovermode='x unified',
            height=600,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        
        chart_html = f"""
    <div class="section">
        <h2 class="section-title">Portfolio Performance</h2>
        <div class="chart-container">
            <div id="main-equity-chart"></div>
        </div>
    </div>
    <script>
        var mainData = {fig.to_json()};
        Plotly.newPlot('main-equity-chart', mainData.data, mainData.layout, {{responsive: true}});
    </script>
"""
        return chart_html
    
    def _generate_comparison_table(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        benchmark_name: str,
        strategy_metrics: Optional[Dict],
        combined_metrics: Optional[Dict]
    ) -> str:
        """Generate performance comparison table."""
        rows = []
        
        # Individual strategies
        for strategy_name, data in strategy_results.items():
            result = data['result']
            metrics = result.metrics
            
            strat_return = (result.equity_curve['TotalValue'].iloc[-1] / 
                           result.equity_curve['TotalValue'].iloc[0] - 1)
            
            beta = strategy_metrics.get(strategy_name, {}).get('Beta (Full Period)', 0) if strategy_metrics else 0
            alpha = strategy_metrics.get(strategy_name, {}).get('Alpha (Annual)', 0) if strategy_metrics else 0
            corr = strategy_metrics.get(strategy_name, {}).get('Correlation', 0) if strategy_metrics else 0
            
            rows.append(f"""
                <tr>
                    <td><strong>{strategy_name}</strong></td>
                    <td>Strategy</td>
                    <td>${data['capital']:,.0f}</td>
                    <td>{strat_return:.2%}</td>
                    <td>{metrics.get('Sharpe Ratio', 0):.2f}</td>
                    <td>{metrics.get('Max Drawdown', 0):.2%}</td>
                    <td>{beta:.3f}</td>
                    <td>{alpha:.2%}</td>
                    <td>{corr:.3f}</td>
                    <td>{metrics.get('Total Trades', 0):.0f}</td>
                </tr>
            """)
        
        # Combined portfolio
        total_initial = sum(data['capital'] for data in strategy_results.values())
        portfolio_return = (combined_equity['TotalValue'].iloc[-1] / combined_equity['TotalValue'].iloc[0] - 1)
        
        # Calculate portfolio Sharpe
        returns = combined_equity['TotalValue'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Calculate max drawdown
        cumulative = combined_equity['TotalValue'] / combined_equity['TotalValue'].iloc[0]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        beta = combined_metrics.get('Beta (Full Period)', 0) if combined_metrics else 0
        alpha = combined_metrics.get('Alpha (Annual)', 0) if combined_metrics else 0
        corr = combined_metrics.get('Correlation', 0) if combined_metrics else 0
        total_trades = sum(len(data['result'].trades) for data in strategy_results.values())
        
        rows.append(f"""
            <tr style="background: #e3f2fd; font-weight: 600;">
                <td><strong>Combined Portfolio</strong></td>
                <td>Composite</td>
                <td>${total_initial:,.0f}</td>
                <td>{portfolio_return:.2%}</td>
                <td>{sharpe:.2f}</td>
                <td>{max_dd:.2%}</td>
                <td>{beta:.3f}</td>
                <td>{alpha:.2%}</td>
                <td>{corr:.3f}</td>
                <td>{total_trades:.0f}</td>
            </tr>
        """)
        
        # Benchmark
        if benchmark_data is not None and combined_metrics:
            bench_return = combined_metrics.get('Benchmark Return', 0)
            rows.append(f"""
                <tr style="background: #ffebee;">
                    <td><strong>{benchmark_name}</strong></td>
                    <td>Benchmark</td>
                    <td>${total_initial:,.0f}</td>
                    <td>{bench_return:.2%}</td>
                    <td>-</td>
                    <td>-</td>
                    <td>1.000</td>
                    <td>0.00%</td>
                    <td>1.000</td>
                    <td>-</td>
                </tr>
            """)
        
        table_html = f"""
    <div class="section">
        <h2 class="section-title">Performance Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Capital</th>
                    <th>Return</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Beta</th>
                    <th>Alpha</th>
                    <th>Correlation</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
"""
        return table_html
    
    def _generate_benchmark_analysis(
        self,
        combined_equity: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        combined_metrics: Dict,
        strategy_metrics: Optional[Dict],
        benchmark_name: str
    ) -> str:
        """Generate benchmark analysis section with rolling beta and base 100 comparison."""
        # Rolling beta chart
        if 'rolling_beta_90d' in combined_metrics:
            rolling_beta = combined_metrics['rolling_beta_90d']
            
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=rolling_beta.index,
                y=rolling_beta.values,
                name='90-Day Rolling Beta',
                mode='lines',
                line=dict(color='rgb(102, 126, 234)', width=2),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig_beta.add_hline(y=1.0, line_dash="dash", line_color="red", 
                             annotation_text="Market Beta = 1.0")
            
            fig_beta.update_layout(
                title=f'Rolling Beta (90-Day Window) vs {benchmark_name}',
                xaxis_title='Date',
                yaxis_title='Beta',
                height=400,
                template='plotly_white'
            )
            
            beta_chart = f"""
            <div class="chart-container">
                <div id="rolling-beta-chart"></div>
            </div>
            <script>
                var betaData = {fig_beta.to_json()};
                Plotly.newPlot('rolling-beta-chart', betaData.data, betaData.layout, {{responsive: true}});
            </script>
"""
        else:
            beta_chart = ""
        
        # Base 100 comparison
        portfolio_base100 = (combined_equity['TotalValue'] / combined_equity['TotalValue'].iloc[0]) * 100
        benchmark_base100 = (benchmark_data['TotalValue'] / benchmark_data['TotalValue'].iloc[0]) * 100
        
        fig_base100 = go.Figure()
        fig_base100.add_trace(go.Scatter(
            x=combined_equity.index,
            y=portfolio_base100,
            name='Portfolio',
            mode='lines',
            line=dict(color='rgb(31, 119, 180)', width=3)
        ))
        fig_base100.add_trace(go.Scatter(
            x=benchmark_data.index,
            y=benchmark_base100,
            name=benchmark_name,
            mode='lines',
            line=dict(color='rgb(255, 65, 54)', width=3)
        ))
        
        fig_base100.update_layout(
            title=f'Normalized Returns (Base 100) vs {benchmark_name}',
            xaxis_title='Date',
            yaxis_title='Value (Base 100)',
            height=400,
            template='plotly_white',
            hovermode='x unified'
        )
        
        base100_chart = f"""
            <div class="chart-container">
                <div id="base100-chart"></div>
            </div>
            <script>
                var base100Data = {fig_base100.to_json()};
                Plotly.newPlot('base100-chart', base100Data.data, base100Data.layout, {{responsive: true}});
            </script>
"""
        
        # Metrics summary
        metrics_html = f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Beta (Full Period)</div>
                <div class="value">{combined_metrics.get('Beta (Full Period)', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Beta (90-day avg)</div>
                <div class="value">{combined_metrics.get('Beta (90-day avg)', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Beta (1-year avg)</div>
                <div class="value">{combined_metrics.get('Beta (1-year avg)', 0):.3f}</div>
            </div>
            <div class="metric-card {'positive' if combined_metrics.get('Alpha (Annual)', 0) > 0 else 'negative'}">
                <div class="label">Alpha (Annual)</div>
                <div class="value">{combined_metrics.get('Alpha (Annual)', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Correlation</div>
                <div class="value">{combined_metrics.get('Correlation', 0):.3f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Tracking Error</div>
                <div class="value">{combined_metrics.get('Tracking Error', 0):.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Information Ratio</div>
                <div class="value">{combined_metrics.get('Information Ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Up Capture</div>
                <div class="value">{combined_metrics.get('Up Capture Ratio', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Down Capture</div>
                <div class="value">{combined_metrics.get('Down Capture Ratio', 0):.2f}</div>
            </div>
        </div>
"""
        
        section_html = f"""
    <div class="section">
        <h2 class="section-title">Benchmark Analysis vs {benchmark_name}</h2>
        {metrics_html}
        {beta_chart}
        {base100_chart}
    </div>
"""
        return section_html
    
    def _generate_strategy_sections(
        self,
        strategy_results: Dict,
        strategy_metrics: Optional[Dict]
    ) -> str:
        """Generate individual strategy deep dive sections."""
        sections = ['<div class="section"><h2 class="section-title">Individual Strategy Analysis</h2>']
        
        for strategy_name, data in strategy_results.items():
            result = data['result']
            metrics = result.metrics
            
            # Strategy metrics
            strat_html = f"""
        <div class="strategy-section">
            <div class="strategy-header">{strategy_name}</div>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Capital Allocated</div>
                    <div class="value">${data['capital']:,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Final Equity</div>
                    <div class="value">${result.equity_curve['TotalValue'].iloc[-1]:,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Total Return</div>
                    <div class="value">{metrics.get('Total Return', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value">{metrics.get('Sharpe Ratio', 0):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Max Drawdown</div>
                    <div class="value">{metrics.get('Max Drawdown', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Win Rate</div>
                    <div class="value">{metrics.get('Win Rate', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Total Trades</div>
                    <div class="value">{metrics.get('Total Trades', 0):.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Profit Factor</div>
                    <div class="value">{metrics.get('Profit Factor', 0):.2f}</div>
                </div>
            </div>
"""
            
            if strategy_metrics and strategy_name in strategy_metrics:
                strat_metrics = strategy_metrics[strategy_name]
                strat_html += f"""
            <p style="margin-top: 20px; font-weight: 600;">Benchmark Comparison:</p>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Beta</div>
                    <div class="value">{strat_metrics.get('Beta (Full Period)', 0):.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Alpha (Annual)</div>
                    <div class="value">{strat_metrics.get('Alpha (Annual)', 0):.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Correlation</div>
                    <div class="value">{strat_metrics.get('Correlation', 0):.3f}</div>
                </div>
            </div>
"""
            
            strat_html += """
        </div>
"""
            sections.append(strat_html)
        
        sections.append('</div>')
        return '\n'.join(sections)
    
    def _generate_trade_analysis(self, strategy_results: Dict) -> str:
        """Generate aggregated trade analysis."""
        all_trades = []
        for strategy_name, data in strategy_results.items():
            trades = data['result'].trades.copy()
            if len(trades) > 0:
                trades['Strategy'] = strategy_name
                all_trades.append(trades)
        
        if not all_trades:
            return '<div class="section"><h2 class="section-title">Trade Analysis</h2><p>No trades executed.</p></div>'
        
        combined_trades = pd.concat(all_trades, ignore_index=True)
        
        total_trades = len(combined_trades)
        winning_trades = (combined_trades['pnl'] > 0).sum()
        losing_trades = (combined_trades['pnl'] < 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = combined_trades[combined_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = combined_trades[combined_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # PnL distribution chart
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=combined_trades['pnl'],
            nbinsx=50,
            name='Trade PnL',
            marker_color='rgb(102, 126, 234)'
        ))
        
        fig.update_layout(
            title='Trade PnL Distribution',
            xaxis_title='PnL ($)',
            yaxis_title='Count',
            height=400,
            template='plotly_white'
        )
        
        trade_html = f"""
    <div class="section">
        <h2 class="section-title">Trade Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Total Trades</div>
                <div class="value">{total_trades}</div>
            </div>
            <div class="metric-card positive">
                <div class="label">Winning Trades</div>
                <div class="value">{winning_trades}</div>
            </div>
            <div class="metric-card negative">
                <div class="label">Losing Trades</div>
                <div class="value">{losing_trades}</div>
            </div>
            <div class="metric-card">
                <div class="label">Win Rate</div>
                <div class="value">{win_rate:.2%}</div>
            </div>
            <div class="metric-card positive">
                <div class="label">Avg Win</div>
                <div class="value">${avg_win:,.2f}</div>
            </div>
            <div class="metric-card negative">
                <div class="label">Avg Loss</div>
                <div class="value">${avg_loss:,.2f}</div>
            </div>
        </div>
        <div class="chart-container">
            <div id="trade-pnl-chart"></div>
        </div>
    </div>
    <script>
        var tradeData = {fig.to_json()};
        Plotly.newPlot('trade-pnl-chart', tradeData.data, tradeData.layout, {{responsive: true}});
    </script>
"""
        return trade_html
    
    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return """
    <div class="footer">
        <p>Generated by QuantTrading Multi-Strategy Reporter</p>
        <p>© 2025 | Powered by Python, Plotly, and Pandas</p>
    </div>
</div>
</body>
</html>
"""
