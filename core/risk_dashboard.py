"""
Risk Dashboard Module
Creates interactive visualizations for risk management metrics.

This module provides comprehensive risk visualization including:
- Leverage tracking
- Position sizing over time
- Correlation heatmaps
- Risk contribution analysis
- Drawdown monitoring
- Violation alerts
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class RiskDashboard:
    """
    Creates interactive risk management dashboards.
    
    Features:
    - Real-time risk metrics tracking
    - Interactive Plotly charts
    - Violation alerts and warnings
    - Correlation analysis
    - Position sizing visualization
    """
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize risk dashboard.
        
        Args:
            output_dir: Directory to save HTML reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_PLOTLY:
            print("Warning: Plotly not installed. Install with: pip install plotly")
    
    def generate_dashboard(
        self,
        risk_metrics_df: pd.DataFrame,
        violations_df: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        equity_df: Optional[pd.DataFrame] = None,
        title: str = "Risk Management Dashboard",
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive risk dashboard.
        
        Args:
            risk_metrics_df: DataFrame with risk metrics history
            violations_df: DataFrame with violation history (optional)
            correlation_matrix: Correlation matrix (optional)
            equity_df: Equity curve dataframe (optional)
            title: Dashboard title
            save_path: Path to save HTML file (optional)
        
        Returns:
            HTML string of the dashboard
        """
        if not HAS_PLOTLY:
            return self._generate_basic_dashboard(risk_metrics_df, violations_df)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Leverage Over Time',
                'Number of Positions',
                'Max Position Weight',
                'Portfolio Volatility',
                'Drawdown',
                'Correlation Heatmap'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # Plot 1: Leverage over time
        if 'leverage' in risk_metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=risk_metrics_df['date'],
                    y=risk_metrics_df['leverage'],
                    mode='lines',
                    name='Leverage',
                    line=dict(color='blue', width=2),
                    hovertemplate='%{x}<br>Leverage: %{y:.2f}x<extra></extra>'
                ),
                row=1, col=1
            )
            # Add threshold line at 1.0x
            fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                         annotation_text="Max Leverage", row=1, col=1)
        
        # Plot 2: Number of positions
        if 'num_positions' in risk_metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=risk_metrics_df['date'],
                    y=risk_metrics_df['num_positions'],
                    mode='lines+markers',
                    name='# Positions',
                    line=dict(color='green', width=2),
                    marker=dict(size=4),
                    hovertemplate='%{x}<br>Positions: %{y}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Max position weight
        if 'max_position_weight' in risk_metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=risk_metrics_df['date'],
                    y=risk_metrics_df['max_position_weight'] * 100,
                    mode='lines',
                    name='Max Weight',
                    line=dict(color='orange', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 165, 0, 0.2)',
                    hovertemplate='%{x}<br>Max Weight: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Portfolio volatility
        if 'portfolio_volatility' in risk_metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=risk_metrics_df['date'],
                    y=risk_metrics_df['portfolio_volatility'] * 100,
                    mode='lines',
                    name='Vol',
                    line=dict(color='purple', width=2),
                    hovertemplate='%{x}<br>Vol: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Plot 5: Drawdown
        if 'drawdown' in risk_metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=risk_metrics_df['date'],
                    y=risk_metrics_df['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    hovertemplate='%{x}<br>DD: %{y:.1f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Plot 6: Correlation heatmap
        if correlation_matrix is not None and not correlation_matrix.empty:
            fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=correlation_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    hovertemplate='%{x} vs %{y}<br>Corr: %{z:.2f}<extra></extra>',
                    showscale=True,
                    colorbar=dict(title="Correlation")
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            height=1200,
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Leverage", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Weight (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
        
        # Generate HTML
        html_content = self._wrap_in_html(
            fig.to_html(include_plotlyjs='cdn', full_html=False),
            title,
            risk_metrics_df,
            violations_df
        )
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(html_content)
            print(f"✅ Dashboard saved to: {save_path}")
        
        return html_content
    
    def _wrap_in_html(
        self,
        plotly_html: str,
        title: str,
        risk_metrics_df: pd.DataFrame,
        violations_df: Optional[pd.DataFrame]
    ) -> str:
        """Wrap Plotly charts in complete HTML with styling."""
        
        # Generate summary stats
        summary_html = self._generate_summary_stats(risk_metrics_df)
        
        # Generate violations table
        violations_html = ""
        if violations_df is not None and not violations_df.empty:
            violations_html = self._generate_violations_table(violations_df)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
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
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    margin-bottom: 10px;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-size: 14px;
                    margin-bottom: 20px;
                }}
                .summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .summary-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .summary-card h3 {{
                    margin: 0 0 10px 0;
                    font-size: 14px;
                    opacity: 0.9;
                }}
                .summary-card .value {{
                    font-size: 28px;
                    font-weight: bold;
                    margin: 0;
                }}
                .violations {{
                    margin: 30px 0;
                }}
                .violations h2 {{
                    color: #e74c3c;
                    margin-bottom: 15px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #34495e;
                    color: white;
                    font-weight: 600;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .chart-container {{
                    margin: 30px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                
                {summary_html}
                {violations_html}
                
                <div class="chart-container">
                    {plotly_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_summary_stats(self, risk_metrics_df: pd.DataFrame) -> str:
        """Generate summary statistics HTML."""
        if risk_metrics_df.empty:
            return ""
        
        # Calculate summary stats
        avg_leverage = risk_metrics_df['leverage'].mean() if 'leverage' in risk_metrics_df.columns else 0
        max_leverage = risk_metrics_df['leverage'].max() if 'leverage' in risk_metrics_df.columns else 0
        avg_positions = risk_metrics_df['num_positions'].mean() if 'num_positions' in risk_metrics_df.columns else 0
        avg_vol = risk_metrics_df['portfolio_volatility'].mean() if 'portfolio_volatility' in risk_metrics_df.columns else 0
        max_dd = risk_metrics_df['drawdown'].min() if 'drawdown' in risk_metrics_df.columns else 0
        
        return f"""
        <div class="summary">
            <div class="summary-card">
                <h3>Avg Leverage</h3>
                <p class="value">{avg_leverage:.2f}x</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>Max Leverage</h3>
                <p class="value">{max_leverage:.2f}x</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>Avg Positions</h3>
                <p class="value">{avg_positions:.1f}</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>Avg Volatility</h3>
                <p class="value">{avg_vol*100:.1f}%</p>
            </div>
            <div class="summary-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <h3>Max Drawdown</h3>
                <p class="value">{max_dd*100:.1f}%</p>
            </div>
        </div>
        """
    
    def _generate_violations_table(self, violations_df: pd.DataFrame) -> str:
        """Generate violations table HTML."""
        if violations_df.empty:
            return ""
        
        # Get recent violations
        recent = violations_df.tail(20).copy()
        
        rows_html = ""
        for _, row in recent.iterrows():
            rows_html += f"""
            <tr>
                <td>{row.get('timestamp', 'N/A')}</td>
                <td>{row.get('ticker', 'N/A')}</td>
                <td>{row.get('type', 'N/A')}</td>
                <td>{row.get('reason', 'N/A')}</td>
            </tr>
            """
        
        return f"""
        <div class="violations">
            <h2>⚠️ Risk Violations ({len(violations_df)} total)</h2>
            <p>Showing most recent 20 violations:</p>
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Ticker</th>
                        <th>Type</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_basic_dashboard(
        self,
        risk_metrics_df: pd.DataFrame,
        violations_df: Optional[pd.DataFrame]
    ) -> str:
        """Generate basic HTML dashboard without Plotly."""
        
        summary_html = self._generate_summary_stats(risk_metrics_df)
        violations_html = ""
        if violations_df is not None and not violations_df.empty:
            violations_html = self._generate_violations_table(violations_df)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Risk Dashboard (Basic)</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .warning {{ color: #e74c3c; background: #fee; padding: 10px; }}
            </style>
        </head>
        <body>
            <h1>Risk Management Dashboard</h1>
            <div class="warning">
                <strong>Note:</strong> Install Plotly for interactive charts: pip install plotly
            </div>
            {summary_html}
            {violations_html}
        </body>
        </html>
        """
    
    def plot_position_sizing_comparison(
        self,
        methods_results: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ) -> str:
        """
        Compare different position sizing methods.
        
        Args:
            methods_results: Dict of {method_name: equity_df}
            save_path: Optional path to save HTML
        
        Returns:
            HTML string
        """
        if not HAS_PLOTLY:
            return "<html><body>Plotly required for comparison charts</body></html>"
        
        fig = go.Figure()
        
        for method_name, equity_df in methods_results.items():
            fig.add_trace(go.Scatter(
                x=equity_df['Date'] if 'Date' in equity_df.columns else equity_df.index,
                y=equity_df['TotalValue'] if 'TotalValue' in equity_df.columns else equity_df['Equity'],
                mode='lines',
                name=method_name,
                hovertemplate=f'{method_name}<br>%{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Position Sizing Methods Comparison",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html_content)
        
        return html_content
