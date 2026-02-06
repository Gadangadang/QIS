"""
Risk Dashboard Module
Generates comprehensive risk analysis HTML dashboards for portfolios.

Features:
- Value at Risk (VaR) and Conditional VaR
- Drawdown analysis with underwater charts
- Strategy correlation matrix
- Rolling risk metrics (volatility, Sharpe, beta)
- Position concentration analysis
- Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats


class RiskDashboard:
    """
    Generates risk-focused HTML dashboards for portfolio analysis.
    
    Provides:
    - Comprehensive risk metrics
    - Correlation analysis
    - Drawdown visualization
    - Rolling risk measures
    - VaR/CVaR calculations
    """
    
    def __init__(self, output_dir: str = 'reports/risk'):
        """
        Initialize risk dashboard.
        
        Args:
            output_dir: Directory to save risk dashboard HTML files
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_dashboard(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_name: str = "Benchmark",
        title: str = "Portfolio Risk Dashboard"
    ) -> str:
        """
        Generate comprehensive risk analysis HTML dashboard.
        
        Args:
            strategy_results: Dict of {strategy_name: {'result': BacktestResult, 'capital': float}}
            combined_equity: DataFrame with combined portfolio equity curve
            benchmark_data: Optional benchmark equity DataFrame
            benchmark_name: Name of benchmark
            title: Dashboard title
            
        Returns:
            HTML string
        """
        html_sections = []
        
        # Header
        html_sections.append(self._generate_header(title))
        
        # Risk Summary
        html_sections.append(self._generate_risk_summary(
            strategy_results, combined_equity
        ))
        
        # Drawdown Analysis
        html_sections.append(self._generate_drawdown_analysis(
            strategy_results, combined_equity
        ))
        
        # Correlation Matrix
        html_sections.append(self._generate_correlation_matrix(
            strategy_results, benchmark_data, benchmark_name
        ))
        
        # Rolling Risk Metrics
        html_sections.append(self._generate_rolling_risk_metrics(
            combined_equity, benchmark_data, benchmark_name
        ))
        
        # VaR and CVaR Analysis
        html_sections.append(self._generate_var_analysis(
            strategy_results, combined_equity
        ))
        
        # Individual Strategy Risk Profiles
        html_sections.append(self._generate_strategy_risk_profiles(
            strategy_results
        ))
        
        # Footer
        html_sections.append(self._generate_footer())
        
        return '\n'.join(html_sections)
    
    def generate_multi_strategy_risk_dashboard(
        self,
        results: Dict,
        capital_allocation: Dict[str, float],
        total_capital: float,
        benchmark_data: Optional[pd.DataFrame] = None,
        benchmark_name: str = "Benchmark",
        save_path: Optional[str] = None,
        auto_open: bool = True
    ) -> str:
        """
        Generate comprehensive risk dashboard for multi-strategy portfolio.
        
        This method formats the results dictionary from a multi-strategy backtest
        into the format expected by generate_dashboard() and creates an aggregated
        portfolio equity curve.
        
        Args:
            results: Dict of {strategy_name: BacktestResult}
            capital_allocation: Dict of {strategy_name: allocation_percentage}
            total_capital: Total portfolio capital
            benchmark_data: Optional benchmark equity DataFrame
            benchmark_name: Name of benchmark
            save_path: Optional path to save HTML file. If None, uses output_dir with timestamp
            auto_open: If True, automatically open the dashboard in default browser
            
        Returns:
            Path to generated HTML file
        """
        import os
        import webbrowser
        from datetime import datetime
        
        # Format strategy results for generate_dashboard()
        strategy_results_formatted = {}
        for strategy_name, result in results.items():
            allocation_pct = capital_allocation.get(strategy_name, 0)
            strategy_capital = total_capital * allocation_pct
            strategy_results_formatted[strategy_name] = {
                'result': result,
                'capital': strategy_capital
            }
        
        # Create combined equity curve (aggregate all strategies)
        combined_equity = None
        for strategy_name, result in results.items():
            if len(result.equity_curve) > 0:
                if combined_equity is None:
                    combined_equity = result.equity_curve[['TotalValue']].copy()
                    combined_equity.columns = [strategy_name]
                else:
                    # Align on date index and add
                    strategy_equity = result.equity_curve[['TotalValue']].copy()
                    strategy_equity.columns = [strategy_name]
                    combined_equity = combined_equity.join(strategy_equity, how='outer')
        
        # Sum across strategies to get total portfolio value
        if combined_equity is not None:
            combined_equity = combined_equity.ffill().fillna(0)
            combined_equity['TotalValue'] = combined_equity.sum(axis=1)
            combined_equity = combined_equity[['TotalValue']]
        else:
            # Empty portfolio
            combined_equity = pd.DataFrame({'TotalValue': [total_capital]})
        
        # Generate dashboard HTML
        dashboard_html = self.generate_dashboard(
            strategy_results=strategy_results_formatted,
            combined_equity=combined_equity,
            benchmark_data=benchmark_data,
            benchmark_name=benchmark_name,
            title=f"Multi-Strategy Portfolio Risk Dashboard (${total_capital:,.0f})"
        )
        
        # Determine save path
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f'risk_dashboard_{timestamp}.html')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Write HTML file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"✅ Risk dashboard saved to: {save_path}")
        
        # Auto-open in browser
        if auto_open:
            abs_path = os.path.abspath(save_path)
            webbrowser.open(f'file://{abs_path}')
            print(f"🌐 Opening dashboard in browser...")
        
        return save_path
    
    def _generate_header(self, title: str) -> str:
        """Generate HTML header."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 1.1em;
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
            color: #f5576c;
            border-left: 4px solid #f5576c;
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
            border-left: 4px solid #f5576c;
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
        .metric-card.good {{
            border-left-color: #28a745;
        }}
        .metric-card.warning {{
            border-left-color: #ffc107;
        }}
        .metric-card.bad {{
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
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            font-weight: 500;
        }}
        .alert-warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }}
        .alert-danger {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            color: #721c24;
        }}
        .alert-success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
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
        <h1>⚠️ {title}</h1>
        <div class="subtitle">Comprehensive Risk Analysis & Metrics</div>
        <div style="margin-top: 10px; opacity: 0.9; font-size: 0.9em;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
"""
    
    def _generate_risk_summary(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame
    ) -> str:
        """Generate risk metrics summary."""
        # Calculate portfolio metrics
        returns = combined_equity['TotalValue'].pct_change().dropna()
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe & Sortino (using 2% risk-free rate)
        rf_rate = 0.02
        excess_returns = returns - (rf_rate / 252)
        sharpe = (np.sqrt(252) * excess_returns.mean() / returns.std()) if returns.std() > 0 else 0
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = combined_equity['TotalValue'] / combined_equity['TotalValue'].iloc[0]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()
        
        # VaR (95% and 99%)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # CVaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()
        
        # Calmar Ratio
        total_return = (combined_equity['TotalValue'].iloc[-1] / combined_equity['TotalValue'].iloc[0] - 1)
        years = len(returns) / 252
        cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Skewness and Kurtosis
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)
        
        # Risk assessment
        risk_level = "Low" if annual_vol < 0.15 else "Medium" if annual_vol < 0.25 else "High"
        risk_class = "good" if annual_vol < 0.15 else "warning" if annual_vol < 0.25 else "bad"
        
        summary_html = f"""
    <div class="section">
        <h2 class="section-title">Risk Metrics Summary</h2>
        
        <div class="alert alert-{'success' if risk_level == 'Low' else 'warning' if risk_level == 'Medium' else 'danger'}">
            <strong>Overall Risk Level: {risk_level}</strong> - Annual volatility of {annual_vol:.2%}
        </div>
        
        <div class="metric-grid">
            <div class="metric-card {risk_class}">
                <div class="label">Annual Volatility</div>
                <div class="value">{annual_vol:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{sharpe:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Sortino Ratio</div>
                <div class="value">{sortino:.2f}</div>
            </div>
            <div class="metric-card {'bad' if max_dd < -0.20 else 'warning' if max_dd < -0.10 else 'good'}">
                <div class="label">Max Drawdown</div>
                <div class="value">{max_dd:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Calmar Ratio</div>
                <div class="value">{calmar:.2f}</div>
            </div>
            <div class="metric-card bad">
                <div class="label">VaR (95%)</div>
                <div class="value">{var_95:.2%}</div>
            </div>
            <div class="metric-card bad">
                <div class="label">VaR (99%)</div>
                <div class="value">{var_99:.2%}</div>
            </div>
            <div class="metric-card bad">
                <div class="label">CVaR (95%)</div>
                <div class="value">{cvar_95:.2%}</div>
            </div>
            <div class="metric-card bad">
                <div class="label">CVaR (99%)</div>
                <div class="value">{cvar_99:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Skewness</div>
                <div class="value">{skew:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="label">Kurtosis</div>
                <div class="value">{kurt:.2f}</div>
            </div>
        </div>
    </div>
"""
        return summary_html
    
    def _generate_drawdown_analysis(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame
    ) -> str:
        """Generate drawdown analysis with underwater chart."""
        # Calculate drawdowns for combined portfolio
        cumulative = combined_equity['TotalValue'] / combined_equity['TotalValue'].iloc[0]
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        
        # Create underwater chart
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=('Portfolio Equity with Peaks', 'Underwater Drawdown Chart'),
            vertical_spacing=0.12
        )
        
        # Equity curve with running maximum
        fig.add_trace(
            go.Scatter(x=combined_equity.index.tolist(), y=combined_equity['TotalValue'].tolist(),
                      name='Equity', line=dict(color='rgb(31, 119, 180)', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=combined_equity.index.tolist(), y=(peak * combined_equity['TotalValue'].iloc[0]).tolist(),
                      name='Peak', line=dict(color='rgba(255, 65, 54, 0.5)', dash='dash')),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(x=combined_equity.index.tolist(), y=(drawdown * 100).tolist(),
                      name='Drawdown', fill='tozeroy',
                      line=dict(color='rgb(220, 53, 69)', width=2),
                      fillcolor='rgba(220, 53, 69, 0.3)'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Equity ($)', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        
        fig.update_layout(height=700, template='plotly_white', showlegend=True)
        
        # Calculate drawdown statistics
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        start = None
        
        for i, (date, val) in enumerate(zip(drawdown.index, in_drawdown)):
            if val and start is None:
                start = date
            elif not val and start is not None:
                drawdown_periods.append((start, drawdown.index[i-1]))
                start = None
        
        if start is not None:
            drawdown_periods.append((start, drawdown.index[-1]))
        
        avg_dd_duration = np.mean([(end - start).days for start, end in drawdown_periods]) if drawdown_periods else 0
        
        dd_html = f"""
    <div class="section">
        <h2 class="section-title">Drawdown Analysis</h2>
        
        <div class="metric-grid">
            <div class="metric-card bad">
                <div class="label">Maximum Drawdown</div>
                <div class="value">{max_dd:.2%}</div>
            </div>
            <div class="metric-card">
                <div class="label">Max DD Date</div>
                <div class="value">{max_dd_date.strftime('%Y-%m-%d')}</div>
            </div>
            <div class="metric-card">
                <div class="label">Drawdown Periods</div>
                <div class="value">{len(drawdown_periods)}</div>
            </div>
            <div class="metric-card">
                <div class="label">Avg DD Duration</div>
                <div class="value">{avg_dd_duration:.0f} days</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="drawdown-chart-{int(__import__('time').time())}"></div>
        </div>
    </div>
    <script>
        var ddData = {fig.to_json()};
        Plotly.newPlot('drawdown-chart-{int(__import__('time').time())}', ddData.data, ddData.layout, {{responsive: true}});
    </script>
"""
        return dd_html
    
    def _generate_correlation_matrix(
        self,
        strategy_results: Dict,
        benchmark_data: Optional[pd.DataFrame],
        benchmark_name: str
    ) -> str:
        """Generate correlation matrix heatmap."""
        # Build returns DataFrame
        returns_dict = {}
        
        for strategy_name, data in strategy_results.items():
            equity = data['result'].equity_curve['TotalValue']
            returns_dict[strategy_name] = equity.pct_change()
        
        if benchmark_data is not None:
            returns_dict[benchmark_name] = benchmark_data['TotalValue'].pct_change()
        
        returns_df = pd.DataFrame(returns_dict).dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values.tolist(),
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.tolist(),
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Strategy Correlation Matrix',
            height=500,
            template='plotly_white'
        )
        
        # Diversification analysis
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        diversification_level = "Excellent" if avg_corr < 0.3 else "Good" if avg_corr < 0.6 else "Moderate" if avg_corr < 0.8 else "Poor"
        
        corr_html = f"""
    <div class="section">
        <h2 class="section-title">Correlation & Diversification Analysis</h2>
        
        <div class="alert alert-{'success' if diversification_level in ['Excellent', 'Good'] else 'warning'}">
            <strong>Diversification Level: {diversification_level}</strong> - Average correlation of {avg_corr:.2f}
        </div>
        
        <div class="chart-container">
            <div id="correlation-chart"></div>
        </div>
        
        <p style="margin-top: 20px; color: #666;">
            <strong>Interpretation:</strong> Lower correlation (closer to 0 or negative) indicates better diversification. 
            Strategies with correlation < 0.5 provide meaningful diversification benefits.
        </p>
"""
        
        # Add covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        fig_cov = go.Figure(data=go.Heatmap(
            z=cov_matrix.values.tolist(),
            x=cov_matrix.columns.tolist(),
            y=cov_matrix.columns.tolist(),
            colorscale='Viridis',
            text=cov_matrix.values.tolist(),
            texttemplate='%{text:.4f}',
            textfont={"size": 10},
            colorbar=dict(title="Covariance")
        ))
        
        fig_cov.update_layout(
            title='Strategy Covariance Matrix (Annualized)',
            height=500,
            template='plotly_white'
        )
        
        corr_html += """
        <h3 style="margin-top: 30px;">Covariance Matrix</h3>
        <div class="chart-container">
            <div id="covariance-chart"></div>
        </div>
        <p style="margin-top: 10px; color: #666;">
            <strong>Interpretation:</strong> Covariance measures how strategies move together. Higher values indicate greater joint variability.
        </p>
"""
        
        # Add beta calculations if benchmark is present
        if benchmark_name in returns_df.columns:
            corr_html += """
        <h3 style="margin-top: 30px;">Beta vs Benchmark</h3>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Beta</th>
                    <th>Interpretation</th>
                </tr>
            </thead>
            <tbody>
"""
            
            benchmark_returns = returns_df[benchmark_name]
            benchmark_var = benchmark_returns.var()
            
            for col in returns_df.columns:
                if col != benchmark_name:
                    cov = returns_df[col].cov(benchmark_returns)
                    beta = cov / benchmark_var if benchmark_var > 0 else 0
                    interpretation = "More volatile" if beta > 1 else "Less volatile" if beta < 1 else "Same volatility"
                    color = "#f39c12" if beta > 1 else "#27ae60"
                    
                    corr_html += f"""
                <tr>
                    <td><strong>{col}</strong></td>
                    <td style="color: {color};">{beta:.2f}</td>
                    <td>{interpretation} than {benchmark_name}</td>
                </tr>
"""
            
            corr_html += """
            </tbody>
        </table>
        <p style="margin-top: 10px; color: #666;">
            <strong>Beta > 1:</strong> Strategy is more volatile than benchmark<br>
            <strong>Beta < 1:</strong> Strategy is less volatile than benchmark<br>
            <strong>Beta = 1:</strong> Strategy moves in line with benchmark
        </p>
"""
        
        corr_html += f"""
    </div>
    <script>
        var corrData = {fig.to_json()};
        Plotly.newPlot('correlation-chart', corrData.data, corrData.layout, {{responsive: true}});
        
        var covData = {fig_cov.to_json()};
        Plotly.newPlot('covariance-chart', covData.data, covData.layout, {{responsive: true}});
    </script>
"""
        return corr_html
    
    def _generate_rolling_risk_metrics(
        self,
        combined_equity: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame],
        benchmark_name: str
    ) -> str:
        """Generate rolling risk metrics charts."""
        returns = combined_equity['TotalValue'].pct_change().dropna()
        
        # Rolling volatility (30, 60, 90 day)
        rolling_vol_30 = returns.rolling(30).std() * np.sqrt(252)
        rolling_vol_60 = returns.rolling(60).std() * np.sqrt(252)
        rolling_vol_90 = returns.rolling(90).std() * np.sqrt(252)
        
        # Rolling Sharpe (90 day)
        rolling_sharpe = (returns.rolling(90).mean() / returns.rolling(90).std()) * np.sqrt(252)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Rolling Volatility (Annualized)', 'Rolling Sharpe Ratio (90-Day)'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Rolling volatility
        fig.add_trace(
            go.Scatter(x=rolling_vol_30.index.tolist(), y=(rolling_vol_30 * 100).tolist(),
                      name='30-Day', line=dict(width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_vol_60.index.tolist(), y=(rolling_vol_60 * 100).tolist(),
                      name='60-Day', line=dict(width=1.5)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=rolling_vol_90.index.tolist(), y=(rolling_vol_90 * 100).tolist(),
                      name='90-Day', line=dict(width=2)),
            row=1, col=1
        )
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index.tolist(), y=rolling_sharpe.tolist(),
                      name='Sharpe 90D', line=dict(color='rgb(102, 126, 234)', width=2),
                      fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)'),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Volatility (%)', row=1, col=1)
        fig.update_yaxes(title_text='Sharpe Ratio', row=2, col=1)
        
        fig.update_layout(height=700, template='plotly_white', showlegend=True)
        
        rolling_html = f"""
    <div class="section">
        <h2 class="section-title">Rolling Risk Metrics</h2>
        <div class="chart-container">
            <div id="rolling-risk-chart-{int(__import__('time').time())}"></div>
        </div>
    </div>
    <script>
        var rollingData = {fig.to_json()};
        Plotly.newPlot('rolling-risk-chart-{int(__import__('time').time())}', rollingData.data, rollingData.layout, {{responsive: true}});
    </script>
"""
        return rolling_html
    
    def _generate_var_analysis(
        self,
        strategy_results: Dict,
        combined_equity: pd.DataFrame
    ) -> str:
        """Generate VaR and CVaR analysis."""
        returns = combined_equity['TotalValue'].pct_change().dropna()
        
        # Calculate VaR at different confidence levels
        confidence_levels = [90, 95, 99]
        var_results = []
        
        for conf in confidence_levels:
            var = np.percentile(returns, 100 - conf)
            cvar = returns[returns <= var].mean()
            var_results.append({
                'Confidence': f'{conf}%',
                'VaR (Daily)': f'{var:.2%}',
                'CVaR (Daily)': f'{cvar:.2%}',
                'VaR (Portfolio)': f'${combined_equity["TotalValue"].iloc[-1] * var:,.2f}',
                'CVaR (Portfolio)': f'${combined_equity["TotalValue"].iloc[-1] * cvar:,.2f}'
            })
        
        # Create table HTML
        table_rows = '\n'.join([
            f"""
            <tr>
                <td>{r['Confidence']}</td>
                <td>{r['VaR (Daily)']}</td>
                <td>{r['CVaR (Daily)']}</td>
                <td>{r['VaR (Portfolio)']}</td>
                <td>{r['CVaR (Portfolio)']}</td>
            </tr>
            """ for r in var_results
        ])
        
        # Returns distribution with VaR lines
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=(returns * 100).tolist(),
            nbinsx=100,
            name='Returns Distribution',
            marker_color='rgb(102, 126, 234)',
            opacity=0.7
        ))
        
        # Add VaR lines
        for conf in confidence_levels:
            var = np.percentile(returns, 100 - conf)
            fig.add_vline(x=var * 100, line_dash="dash", 
                         annotation_text=f'VaR {conf}%',
                         line_color='red' if conf == 99 else 'orange')
        
        fig.update_layout(
            title='Returns Distribution with VaR Levels',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            height=400,
            template='plotly_white'
        )
        
        var_html = f"""
    <div class="section">
        <h2 class="section-title">Value at Risk (VaR) & Conditional VaR Analysis</h2>
        
        <p style="color: #666; margin-bottom: 20px;">
            <strong>VaR</strong> estimates the maximum loss at a given confidence level. 
            <strong>CVaR</strong> (Expected Shortfall) is the average loss when VaR is exceeded.
        </p>
        
        <table>
            <thead>
                <tr>
                    <th>Confidence Level</th>
                    <th>VaR (Daily %)</th>
                    <th>CVaR (Daily %)</th>
                    <th>VaR ($)</th>
                    <th>CVaR ($)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div class="chart-container">
            <div id="var-chart-{int(__import__('time').time())}"></div>
        </div>
    </div>
    <script>
        var varData = {fig.to_json()};
        Plotly.newPlot('var-chart-{int(__import__('time').time())}', varData.data, varData.layout, {{responsive: true}});
    </script>
"""
        return var_html
    
    def _generate_strategy_risk_profiles(self, strategy_results: Dict) -> str:
        """Generate individual strategy risk profiles."""
        sections = ['<div class="section"><h2 class="section-title">Individual Strategy Risk Profiles</h2>']
        
        for strategy_name, data in strategy_results.items():
            result = data['result']
            equity = result.equity_curve['TotalValue']
            returns = equity.pct_change().dropna()
            
            # Risk metrics (using 2% risk-free rate)
            annual_vol = returns.std() * np.sqrt(252)
            rf_rate = 0.02
            excess_returns = returns - (rf_rate / 252)
            sharpe = (np.sqrt(252) * excess_returns.mean() / returns.std()) if returns.std() > 0 else 0
            
            downside_returns = returns[returns < 0]
            sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            cumulative = equity / equity.iloc[0]
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min()
            
            risk_level = "Low" if annual_vol < 0.15 else "Medium" if annual_vol < 0.25 else "High"
            
            sections.append(f"""
        <div style="background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 15px 0;">
            <h3 style="color: #f5576c; margin-bottom: 15px;">{strategy_name}</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Risk Level</div>
                    <div class="value">{risk_level}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Annual Volatility</div>
                    <div class="value">{annual_vol:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value">{sharpe:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sortino Ratio</div>
                    <div class="value">{sortino:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Max Drawdown</div>
                    <div class="value">{max_dd:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">VaR (95%)</div>
                    <div class="value">{var_95:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="label">CVaR (95%)</div>
                    <div class="value">{cvar_95:.2%}</div>
                </div>
            </div>
        </div>
""")
        
        sections.append('</div>')
        return '\n'.join(sections)
    
    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        return """
    <div class="footer">
        <p>⚠️ Risk Dashboard - QuantTrading</p>
        <p>© 2025 | Comprehensive Risk Analysis & Monitoring</p>
    </div>
</div>
</body>
</html>
"""
